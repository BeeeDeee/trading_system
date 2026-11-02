"""Research 20: forward data construction."""

import numpy as np
import pytest

from qlab.research20 import data as D


def fake_chart(rows, divs=(), splits=()):
    """rows: [(date 'YYYY-MM-DD', o, h, l, c)]; UTC midnight timestamps, gmtoffset 0."""
    ts = [int(np.datetime64(d, "s").astype(int)) for d, *_ in rows]
    q = {k: [r[i + 1] for r in rows] for i, k in enumerate(("open", "high", "low", "close"))}
    ev = {"dividends": {str(t): {"date": t, "amount": a} for t, a in
                        ((int(np.datetime64(d, "s").astype(int)), a) for d, a in divs)},
          "splits": {str(t): {"date": t, "numerator": n, "denominator": m} for t, (n, m) in
                     ((int(np.datetime64(d, "s").astype(int)), x) for d, x in splits)}}
    return {"chart": {"result": [{"meta": {"gmtoffset": 0}, "timestamp": ts, "indicators": {"quote": [q]}, "events": ev}]}}


def test_tr_with_dividend_matches_sharadar_convention():
    rows = [("2026-01-02", 100, 101, 99, 100), ("2026-01-05", 99.5, 101, 99, 100.5), ("2026-01-06", 100.5, 102, 100, 101)]
    p = D.parse_yahoo(fake_chart(rows, divs=[("2026-01-05", 1.0)]))
    b = D.tr_bars(p)
    assert b["ret_co"][0] == 0.0
    np.testing.assert_allclose(b["tr"][1], (100.5 + 1.0) / 100 - 1)        # (close + D) / prev close - 1
    np.testing.assert_allclose(b["ret_oc"][1], 100.5 / 99.5 - 1)            # dividend not in the intraday part
    np.testing.assert_allclose(b["tr"][2], 101 / 100.5 - 1)                 # no dividend: plain price return
    np.testing.assert_allclose((1 + b["ret_co"]) * (1 + b["ret_oc"]) - 1, b["tr"])


def test_parse_drops_incomplete_rows_and_reads_splits():
    j = fake_chart([("2026-01-02", 10, 11, 9, 10), ("2026-01-05", 5, 6, 4, 5.5)], splits=[("2026-01-05", (1, 2))])
    j["chart"]["result"][0]["indicators"]["quote"][0]["open"][0] = None
    p = D.parse_yahoo(j)
    assert len(p["dates"]) == 1 and p["split"][0] == 0.5


def test_gate_stats_counts_missing_dates_and_basis_points():
    d1 = np.array(["2026-01-02", "2026-01-05", "2026-01-06"], dtype="datetime64[D]")
    d2 = d1[:2]
    s = D.gate_stats(d1, np.array([0.01, 0.02, 0.03]), d2, np.array([0.01, 0.0201]))
    assert s["n_common"] == 2 and s["n_only_yahoo"] == 1 and s["n_only_sharadar"] == 0
    assert s["max_abs_bps"] == pytest.approx(1.0)


def test_gate_two_day_statistic_cancels_dividend_timing_but_not_isolated_errors():
    d = np.array(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], dtype="datetime64[D]")
    base = np.array([0.001, 0.002, -0.001, 0.0005, 0.001])
    a = base.copy(); b = base.copy()
    a[1] += 0.006                                      # source A books the dividend on day 1, B on day 2
    b[2] += 0.006
    s = D.gate_stats(d, a, d, b)
    assert s["max_abs_bps"] > 50 and s["max_abs2_bps"] < 0.5
    c = base.copy(); c[3] += 0.006                     # an isolated error is not cancelled
    assert D.gate_stats(d, base, d, c)["max_abs2_bps"] > 50


# ---------------------------------------------------------------- forward panel, statistics, decisions
from datetime import date  # noqa: E402

from qlab.data.normalize import normalize_prices  # noqa: E402
from qlab.data.panel import panel_from_bars  # noqa: E402
from qlab.data.synthetic import generate_market  # noqa: E402
from qlab.research19 import setup as S19  # noqa: E402
from qlab.research20 import evaluate as E  # noqa: E402
from qlab.research20 import forward as F  # noqa: E402
from qlab.research20 import setup as S  # noqa: E402
from qlab.validation.leakage import check_point_in_time  # noqa: E402


def test_holm_hand_example():
    assert E.holm({"a": 0.004, "b": 0.015}, 0.0167) == {"a": True, "b": True}      # 0.004 < 0.00835, 0.015 < 0.0167
    assert E.holm({"a": 0.009, "b": 0.001}, 0.0167) == {"a": True, "b": True}      # 0.001 < 0.00835, 0.009 < 0.0167
    assert E.holm({"a": 0.012, "b": 0.013}, 0.0167) == {"a": False, "b": False}    # smaller fails -> stop
    assert E.holm({"a": 0.016}, 0.0167) == {"a": True}


def base(**kw):
    return {"n_enough": True, "main_x1": 0.5, "up90": 0.8, "p": 0.001, "plan_bar": True, "point_ok": True, **kw}


def test_states():
    assert E.decide(1, {"F1": base(), "F2": base(p=0.5)}) == {"F1": "potvrzena", "F2": "konzistentni"}
    assert E.decide(1, {"F1": base(plan_bar=False)})["F1"] == "neprukazne"
    assert E.decide(1, {"F1": base(n_enough=False)})["F1"] == "malo_dat"
    assert E.decide(1, {"F1": base(up90=-0.1, main_x1=-0.5, point_ok=False)})["F1"] == "zamitnuta"
    # a point estimate <= 0 rejects only at the last decision
    assert E.decide(1, {"F1": base(point_ok=False, main_x1=-0.1, plan_bar=False)})["F1"] == "neprukazne"
    assert E.decide(3, {"F1": base(point_ok=False, main_x1=-0.1)})["F1"] == "zamitnuta"
    assert E.decide(3, {"F1": base(p=0.5), "F2": base(n_enough=False)}) == {"F1": "nepotvrzeno", "F2": "nepotvrzeno"}
    # a rejected hypothesis leaves the Holm family: the other is then tested at the full alpha
    out = E.decide(1, {"F1": base(up90=-1.0, point_ok=False), "F2": base(p=0.015)})
    assert out == {"F1": "zamitnuta", "F2": "potvrzena"}
    assert E.decide(1, {"F1": base(p=0.012), "F2": base(p=0.013)}) == {"F1": "konzistentni", "F2": "konzistentni"}


def test_look_dates_are_last_trading_day_before_anniversary():
    cal = np.arange(np.datetime64("2026-11-03"), np.datetime64("2029-12-31"))
    cal = cal[np.is_busday(cal)]
    d = E.look_dates(date(2026, 11, 3), cal)
    assert d == [date(2027, 11, 2), date(2028, 11, 2), date(2029, 11, 2)]
    cal2 = cal[cal != np.datetime64("2027-11-02")]                       # closed on that day
    assert E.look_dates(date(2026, 11, 3), cal2)[0] == date(2027, 11, 1)


def test_pvalue_and_upper_bound():
    boots = np.r_[np.full(50, -1.0), np.full(9950, 1.0)]
    assert E.p_value(boots) == pytest.approx(51 / 10001)
    r = np.random.default_rng(0).normal(0.0004, 0.005, 600)
    b = E.boot_sharpe(r, 0.0, reps=300)
    assert E.upper90(b) > np.median(b)
    net = np.r_[np.full(30, 0.02), np.full(30, -0.01)]
    bm = E.boot_trade_means(net, np.repeat(np.arange(20), 3), reps=200)
    assert -0.01 - 1e-12 <= bm.min() and bm.max() <= 0.02 + 1e-12


@pytest.fixture(scope="module")
def etf_like():
    m = generate_market(n_assets=26, seed=20, start="2010-01-01", end="2012-12-31", p_halt=0, p_no_open=0, p_delist=0)
    p = panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)
    return p, np.full(p.shape[0], 0.00008), list(S19.ALL_ETF)


def test_append_days_extends_without_touching_history(etf_like):
    p, cash, tickers = etf_like
    last = p.dates[-1]
    days = [last + np.timedelta64(1, "D"), last + np.timedelta64(2, "D")]
    led = {tk: {d: (10.0, 10.1, 0.001 * (j % 3), 0.002) for d in days} for j, tk in enumerate(tickers)}
    p2, c2 = F.append_days(p, cash, tickers, led, days)
    assert p2.shape[0] == p.shape[0] + 2 and len(c2) == len(p2.dates)
    np.testing.assert_array_equal(p2.ret_oc[:p.shape[0]], p.ret_oc)
    np.testing.assert_array_equal(c2[:p.shape[0]], cash)
    assert (p2.ret_oc[-2:] == 0.002).all() and p2.tradable[-1].all() and not p2.delisting[-2:].any()
    assert F.complete_days({"A": {days[0]: 1, days[1]: 1}, "B": {days[0]: 1}}) == [days[0]]    # stops at the first gap


def test_forward_metrics_start_flat_and_are_point_in_time(etf_like):
    p, cash, tickers = etf_like
    t0 = 520
    m = F.forward_metrics(p, cash, tickers, t0, p.shape[0])
    assert m["F1"]["n_days"] == p.shape[0] - t0 and 0.0 <= m["F1"]["p"] <= 1.0
    assert m["F2"]["n_trades"] > 0 and m["F2"]["sharpe_x1.5"] <= m["F2"]["sharpe_x1"] + 1e-9
    # nothing is held before T0: a panel cut at T0 gives flat returns of zero excess (cash only)
    r = F.run_f1(p.slice(t0 + 1), cash[:t0 + 1], tickers, t0)
    assert len(r) == 1
    nav = lambda q: F.run_f1(q, cash[:q.shape[0]], tickers, t0)  # noqa: E731
    f = lambda q: np.concatenate([np.full(t0, np.nan), nav(q)])[:, None]  # noqa: E731
    assert check_point_in_time(f, p, "forward_f1", n_cuts=5, min_day=t0 + 5).ok


def test_t0_is_the_day_after_the_commit():
    assert S.T0 == date(2026, 11, 3) and S.ALPHA_LOOK == pytest.approx(0.05 / 3)
