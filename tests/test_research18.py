from dataclasses import asdict
from datetime import date

import numpy as np
import pytest

from qlab.research15.setup import epoch_15m
from qlab.research16 import engine as G
from qlab.research17 import forward as F, setup as T, variants as V
from qlab.research18 import evaluate as E, rules as H, setup as S18

N = 96 * 330
T_START = epoch_15m(date(2026, 1, 1))


def _market(seed=5, drift=2e-6):
    """BTC with a slow uptrend (above its 200-day mean most of the time) and an altcoin; BTC jumps every 150 candles
    from 2026-09-15 on. The altcoin lags: after odd jumps it falls slightly, after even ones it creeps up."""
    rng = np.random.default_rng(seed)
    t = T_START + np.arange(N)

    def panel(vol, mu):
        c = 100 * np.exp(np.cumsum(rng.normal(mu, vol, N)))
        return {"t": t, "o": c, "h": c * 1.001, "l": c * 0.999, "c": c, "real": np.ones(N, bool),
                "funding": np.full(N, np.nan)}

    btc, alt = panel(0.001, drift), panel(0.002, 0.0)
    first = int(epoch_15m(date(2026, 9, 15)) - T_START)
    for j, k in enumerate(range(first, N - 40, 150)):
        btc["c"][k:] *= 1.02
        alt["c"][k:] *= 0.9985 if j % 2 else 1.001
    for p in (btc, alt):
        p["o"], p["h"], p["l"] = p["c"], p["c"] * 1.001, p["c"] * 0.999
    return btc, alt


def _same_fields(a, b):
    da, db = asdict(a), asdict(b)
    for k in ("name", "group", "desc"):
        da.pop(k), db.pop(k)
    return da == db


def test_hypotheses_are_pinned_to_the_variants_they_were_picked_from():
    specs = {s.name: s for s in V.SPECS}
    assert _same_fields(H.H1, specs["V05_lag0"]) and _same_fields(H.H2, specs["V01_base"])
    assert H.HYPOTHESES["H1"] == (H.H1, False) and H.HYPOTHESES["H2"] == (H.H2, True)


def test_ma_gate_is_point_in_time_and_false_before_the_window():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    g = H.ma_gate(c, window=50)
    assert not g[:49].any()
    for k in (49, 120, 399):
        assert g[k] == (c[k] > c[k - 49:k + 1].mean())
    c2 = c.copy()
    c2[200:] *= 3.0                                                  # change of the future must not touch the past
    assert (H.ma_gate(c2, window=50)[:200] == g[:200]).all()
    assert not H.ma_gate(c[:30], window=50).any()


def test_holm_and_event_bootstrap():
    assert E.holm({"a": 0.01, "b": 0.02}, 0.025) == {"a": True, "b": True}       # 0.01 < 0.0125, 0.02 < 0.025
    assert E.holm({"a": 0.013, "b": 0.02}, 0.025) == {"a": False, "b": False}    # the smaller one fails -> both fail
    assert E.holm({"a": 0.01, "b": 0.03}, 0.025) == {"a": True, "b": False}
    assert E.holm({"only": 0.024}, 0.025) == {"only": True}
    rng = np.random.default_rng(1)
    ev = np.repeat(np.arange(40), 3)                                 # 40 events with 3 trades each
    pos = E.event_boot(ev, rng.normal(0.01, 0.01, 120), n_boot=2000)
    neg = E.event_boot(ev, rng.normal(-0.01, 0.01, 120), n_boot=2000)
    assert pos["n_events"] == 40 and pos["p"] < 0.01 and pos["lo"] > 0 and neg["p"] > 0.99
    r = rng.normal(0.01, 0.01, 120)
    assert E.event_boot(ev, r, n_boot=500, seed=3) == E.event_boot(ev, r, n_boot=500, seed=3)       # seeded
    assert E.event_boot(np.zeros(0, np.int64), np.zeros(0))["p"] is None


def test_event_bootstrap_resamples_events_not_trades():
    ev = np.array([0] * 20 + [1, 2, 3, 4])                           # one event holds 20 of 24 trades
    r = np.array([-0.01] * 20 + [0.05] * 4)
    b = E.event_boot(ev, r, n_boot=4000)
    assert b["mean"] == pytest.approx(r.mean()) and b["n_events"] == 5
    assert b["p"] > 0.05 and b["lo"] < 0                             # trade-level iid would be far more confident


def _m(sharpe=0.5, mean=0.003, p=0.01, ev=100, s15=0.2):
    return {"sharpe": sharpe, "mean_trade": mean, "p": p, "n_events": ev, "sharpe_cost15": s15}


def test_decision_rules():
    both = ["H1", "H2"]
    d = E.decide({"H1": _m(ev=50, p=0.005), "H2": _m(ev=100, p=0.02)}, both, 1)
    assert d == {"H1": "success", "H2": "success"}
    d = E.decide({"H1": _m(ev=50, p=0.013), "H2": _m(ev=100, p=0.02)}, both, 1)            # Holm: smaller > 0.0125
    assert d == {"H1": "inconclusive", "H2": "inconclusive"}
    d = E.decide({"H1": _m(ev=50, p=0.005), "H2": _m(ev=100, mean=-0.001, p=0.7)}, both, 1)
    assert d == {"H2": "rejected", "H1": "success"}                  # rejected leaves the family: H1 tested at 0.025
    d = E.decide({"H1": _m(ev=50, p=0.02), "H2": _m(ev=100, mean=-0.001, p=0.7)}, both, 1)
    assert d["H1"] == "success"
    assert E.decide({"H1": _m(ev=29, p=0.0001), "H2": _m()}, both, 1)["H1"] == "inconclusive"     # < min events
    assert E.decide({"H1": _m(ev=29, mean=-1.0, sharpe=-1.0), "H2": _m()}, both, 1)["H1"] == "inconclusive"
    assert E.decide({"H1": _m(s15=-0.1), "H2": _m()}, both, 1)["H1"] == "inconclusive"             # 15 bps criterion
    assert E.decide({"H1": _m(sharpe=-0.1), "H2": _m()}, both, 1)["H1"] == "rejected"
    d = E.decide({"H1": _m(p=0.2)}, ["H1"], 2)
    assert d == {"H1": "not_confirmed"}                              # second look: no further extension
    assert E.decide({"H2": _m(p=0.024)}, ["H2"], 2) == {"H2": "success"}


def test_run_gate_all_true_equals_no_gate_and_gate_false_blocks_everything():
    btc, alt = _market()
    a = G.Asset.build(alt)
    ld = V.leader_moves(btc)
    members = {m: ["ALTUSDT"] for m in ("2026-09", "2026-10", "2026-11")}
    with E.window(date(2026, 11, 20)):
        s0 = int(epoch_15m(S18.T0))
        base = H.run(H.H2, {"ALTUSDT": a}, ld, members, s0)
        allow = H.run(H.H2, {"ALTUSDT": a}, ld, members, s0, np.ones(len(btc["c"]), bool))
        block = H.run(H.H2, {"ALTUSDT": a}, ld, members, s0, np.zeros(len(btc["c"]), bool))
        ref = V.run_spec(V.SPECS[0], {"ALTUSDT": a}, ld, members, s0, {"ALTUSDT": {}})
    assert len(base["rets"]) > 0 and np.array_equal(base["rets"], allow["rets"])
    assert np.allclose(base["daily"], ref["daily"]) and len(base["rets"]) == ref["n_trades"]
    assert len(block["rets"]) == 0 and not block["daily"].any()


def test_gate_applies_at_entry_only():
    btc, alt = _market()
    a = G.Asset.build(alt)
    ld = V.leader_moves(btc)
    members = {m: ["ALTUSDT"] for m in ("2026-09", "2026-10", "2026-11")}
    with E.window(date(2026, 11, 20)):
        s0 = int(epoch_15m(S18.T0))
        base = H.run(H.H2, {"ALTUSDT": a}, ld, members, s0)
        assert len(base["events"]) >= 4
        k0 = int(base["events"][0])
        gate = np.ones(len(btc["c"]), bool)
        gate[k0 - btc["t"][0] + 1:] = False                          # BTC drops below MA200 right after the 1st entry
        part = H.run(H.H2, {"ALTUSDT": a}, ld, members, s0, gate)
    assert part["events"].tolist() == [k0] and part["rets"][0] == pytest.approx(base["rets"][0])   # still exits on time


def test_evaluate_end_to_end_on_a_synthetic_market(tmp_path):
    btc, alt = _market()
    members = {"2026-09": ["ALTUSDT"], "2026-10": ["ALTUSDT"], "2026-11": ["ALTUSDT"]}
    asof = date(2026, 11, 25)
    out = E.evaluate({"BTCUSDT@spot": btc, "ALTUSDT": alt}, members, asof)
    m1, m2 = out["metrics"]["H1"], out["metrics"]["H2"]
    assert out["period"] == ["2026-10-20", "2026-11-25"] and out["days"] == 37
    assert m2["n_trades"] > 0 and m1["n_trades"] > 0 and m1["n_trades"] <= m2["n_trades"]     # rA < 0 implies rA/rB < 0.5
    assert m1["p"] is not None and 0 < m1["p"] <= 1 and m1["n_events"] == len({t["entry"] for t in out["trades"]["H1"]})
    assert out["descriptive"]["V01_base_no_gate"]["n_trades"] >= m2["n_trades"]
    assert "H2_minus_V01" in out["descriptive"] and set(out["provisional_decision"]) == {"H1", "H2"}
    assert len(out["daily"]["dates"]) == 37 and out["daily"]["dates"][0] == "2026-10-20"
    assert T.START == date(2020, 5, 1) and T.END == date(2026, 9, 30)                          # window restored


def test_event_bootstrap_and_membership_horizon():
    ms = F.forward_months(F.MEMBERS_END)
    assert ms[0] == date(2026, 10, 1) and ms[-1] == date(2028, 10, 1) and len(ms) == 25
    assert F.forward_months()[-1] == date(2027, 3, 1)               # research 17 unchanged
    assert S18.END[1] == date(2027, 10, 19) and S18.END[2] == date(2028, 10, 19) and S18.T0 == date(2026, 10, 20)


# ---------------------------------------------------------------- dashboard

def _status():
    btc, alt = _market()
    members = {"2026-09": ["ALTUSDT"], "2026-10": ["ALTUSDT"], "2026-11": ["ALTUSDT"]}
    return E.evaluate({"BTCUSDT@spot": btc, "ALTUSDT": alt}, members, date(2026, 11, 25)) | {
        "commit": "abc", "rule_hash": "r" * 64, "data_hash": "d" * 64}


def test_dashboard_renders_status_escapes_html_and_has_no_scripts():
    from datetime import datetime, timezone

    from qlab.research18 import dashboard as D
    st = _status()
    st["trades"]["H1"][-1] = {**st["trades"]["H1"][-1], "sym": "<script>alert(1)</script>"}
    page = D.render(st, {"error": "boom <b>", "last_run": "2026-11-26T06:30:00+00:00"}, datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert "<script" not in page and "&lt;script&gt;alert" in page and "boom &lt;b&gt;" in page
    assert "H1 · LEAD_LAG0" in page and "H2 · LEAD_K3_BTC200" in page and "<svg" in page and "href=\"index.html\"" in page
    assert "Stav kritérií" in page and "práh 0.0125" in page and "práh 0.0250" in page      # Holm thresholds, both alive
    assert "NaN" not in page and "None" not in page
    assert str(st["metrics"]["H1"]["n_events"]) in page and "2026-11-25" in page


def test_dashboard_waiting_before_the_first_status_and_error_state():
    from datetime import datetime, timezone

    from qlab.research18 import dashboard as D
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    waiting = D.render(None, {"error": None}, now)
    assert "ČEKÁ NA ZAČÁTEK" in waiting and "2026-10-20" in waiting
    failed = D.render(None, {"error": "rule files changed since the freeze"}, now)
    assert "CHYBA" in failed and "rule files changed" in failed


def test_dashboard_links_between_the_two_pages():
    from datetime import datetime, timezone

    from qlab.research17 import dashboard
    page = dashboard.render(None, {"download": {}, "status": {}, "alerts": []}, datetime(2026, 10, 20, tzinfo=timezone.utc),
                            date(2026, 10, 1), date(2027, 3, 31), 30)
    assert 'href="research18.html"' in page and 'href="backtest.html"' in page
