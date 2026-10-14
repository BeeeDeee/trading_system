import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from qlab.research8 import download as dl
from qlab.research15.setup import GRID_START, epoch_15m, panel_dir
from qlab.research17 import forward as F, setup as T


# ---------------------------------------------------------------- helpers

def _kline_zip(path: Path, rows: list[tuple[int, float]], unit: int = 1):
    """Binance kline CSV zip; rows = (open time in 15m index, close); o=c-1, h=c+1, l=c-2."""
    lines = ["open_time,open,high,low,close,volume,close_time,qv,n,tb,tq,ignore"]
    lines += [f"{q * 900_000 * unit},{c - 1},{c + 1},{c - 2},{c},10,0,0,0,0,0,0" for q, c in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", "\n".join(lines))
    return path


Q0 = epoch_15m(date(2026, 9, 1))                      # multiple of 96


def test_build_panel_handles_gaps_duplicates_microseconds_and_funding(tmp_path):
    f1 = _kline_zip(tmp_path / "a.zip", [(Q0 + 3, 100.0), (Q0 + 4, 101.0), (Q0 + 6, 103.0)])
    f2 = _kline_zip(tmp_path / "b.zip", [(Q0 + 6, 999.0), (Q0 + 7, 104.0)], unit=1000)   # duplicate, microseconds
    p = F.build_panel([f1, f2], funding=[((Q0 + 4) * 900_000 + 12, 3e-4), ((Q0 + 99) * 900_000, 1.0)])
    assert p["t"][0] == Q0 and len(p["t"]) == 8 and p["t"][-1] == Q0 + 7
    assert p["real"].tolist() == [False] * 3 + [True, True, False, True, True]
    assert p["c"][6] == 103.0 and p["c"][5] == 101.0                    # first duplicate wins; close carried
    assert p["o"][5] == p["h"][5] == p["l"][5] == 101.0
    assert np.isnan(p["c"][:3]).all()
    assert p["funding"][4] == 3e-4 and np.isfinite(p["funding"]).sum() == 1   # out-of-range record ignored


@pytest.mark.skipif(not (F.BASE_RAW / "perp_15m" / "BTCUSDT").exists(), reason="no data/ snapshot")
def test_panel_builder_reproduces_the_snapshot_panel():
    for market, sym in (("perp", "BTCUSDT"), ("spot", "BTCUSDT")):
        files = sorted((F.BASE_RAW / f"{market}_15m" / sym).glob(f"{sym}-15m-*.zip"))
        fund = F.funding_records(sym) if market == "perp" else None
        p = F.build_panel(files, fund, first_grid=epoch_15m(GRID_START))
        with np.load(panel_dir(F.BASE_SNAPSHOT, market) / f"{sym}.npz") as z:
            assert int(z["t0"]) == int(p["t"][0]) and len(z["c"]) == len(p["c"])
            for k in ("o", "h", "l", "c", "funding"):
                np.testing.assert_array_equal(np.nan_to_num(z[k], nan=-1.0), np.nan_to_num(p[k], nan=-1.0), err_msg=k)
            np.testing.assert_array_equal(z["real"], p["real"])


def _fake_net(monkeypatch, files: dict[str, bytes], bad_checksum: set[str] = frozenset()):
    calls = []

    def get(url, tries=5):
        calls.append(url)
        key = url.removeprefix(dl.HTTP + "/")
        base = key.removesuffix(".CHECKSUM")
        if base not in files:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        if key.endswith(".CHECKSUM"):
            return (("0" * 64 if base in bad_checksum else hashlib.sha256(files[base]).hexdigest()) + "  f\n").encode()
        return files[base]
    monkeypatch.setattr(dl, "_get", get)
    return calls


def test_fetch_days_verifies_stops_at_unpublished_day_and_is_idempotent(tmp_path, monkeypatch):
    days = [date(2026, 10, d) for d in (1, 2, 3)]
    files = {}
    for d in days[:2]:
        p = _kline_zip(tmp_path / "src" / f"{d}.zip", [(epoch_15m(d), 1.0)])
        files[F.kline_key("perp", "AAAUSDT", d)] = p.read_bytes()
    calls = _fake_net(monkeypatch, files)
    r = F.fetch_days("perp", "AAAUSDT", date(2026, 10, 1), date(2026, 10, 9), out=tmp_path / "raw")
    assert r == {"market": "perp", "symbol": "AAAUSDT", "new": 2, "last_day": "2026-10-02"}
    assert (tmp_path / "raw/perp_15m/AAAUSDT/AAAUSDT-15m-2026-10-02.zip.ok").exists()
    n = len(calls)
    r2 = F.fetch_days("perp", "AAAUSDT", date(2026, 10, 1), date(2026, 10, 9), out=tmp_path / "raw")
    assert r2["new"] == 0 and len(calls) == n + 1                       # only the 404 probe of day 3
    key = F.kline_key("perp", "BBBUSDT", days[0])
    _fake_net(monkeypatch, {key: files[F.kline_key("perp", "AAAUSDT", days[0])]}, bad_checksum={key})
    with pytest.raises(ValueError, match="checksum"):
        F.fetch_days("perp", "BBBUSDT", days[0], days[0], out=tmp_path / "raw")
    assert not list((tmp_path / "raw/perp_15m/BBBUSDT").glob("*.zip")) if (tmp_path / "raw/perp_15m/BBBUSDT").exists() else True
    assert F.kline_key("spot", "BTCUSDT", days[0]) == "data/spot/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2026-10-01.zip"


def test_fetch_funding_appends_only_new_records(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FUNDING_PAGE", 4)
    t0 = int(np.datetime64("2026-10-01T00:00:00", "ms").astype(np.int64))
    recs = [{"fundingTime": t0 + i * 8 * 3600_000 + 5, "fundingRate": str(1e-4 * (i + 1))} for i in range(6)]
    seen = []

    def get(url):
        seen.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        return json.dumps([r for r in recs if r["fundingTime"] >= start][:F.FUNDING_PAGE]).encode()
    r = F.fetch_funding("AAAUSDT", out=tmp_path, get=get)
    assert r["new_funding"] == 6
    data = json.loads((tmp_path / "funding_rest/AAAUSDT.json").read_text())
    assert [x["rate"] for x in data] == pytest.approx([1e-4 * (i + 1) for i in range(6)])
    recs.append({"fundingTime": t0 + 6 * 8 * 3600_000 + 5, "fundingRate": "0.5"})
    r2 = F.fetch_funding("AAAUSDT", out=tmp_path, get=get)
    assert r2["new_funding"] == 1 and len(json.loads((tmp_path / "funding_rest/AAAUSDT.json").read_text())) == 7
    assert int(seen[0].split("startTime=")[1].split("&")[0]) == t0                 # first fetch starts at the forward start


def test_membership_helpers():
    ms = F.forward_months()
    assert ms[0] == date(2026, 10, 1) and ms[-1] == date(2027, 3, 1) and len(ms) == 6
    assert F.last_complete_month(date(2026, 10, 14)) == "2026-09" and F.last_complete_month(date(2027, 1, 2)) == "2026-12"
    F.check_stable({"2026-10": ["A", "B"], "2026-11": []}, {"2026-10": ["A", "B"], "2026-11": ["C"]})
    with pytest.raises(ValueError, match="changed"):
        F.check_stable({"2026-10": ["A", "B"]}, {"2026-10": ["A", "C"]})


def test_volumes_merge_base_and_forward_months(tmp_path, monkeypatch):
    def day_zip(path, first_day, qv):
        rows = [f"{(first_day + i) * 86_400_000},1,1,1,1,1,0,{v},0,0,0,0" for i, v in enumerate(qv)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("x.csv", "open_time,o,h,l,c,v,ct,qv,n,tb,tq,i\n" + "\n".join(rows))
    d0 = (date(2026, 9, 1) - date(1970, 1, 1)).days
    day_zip(tmp_path / "base/perp_1d/AAA/AAA-1d-2026-09.zip", d0, [5.0] * 30)
    day_zip(tmp_path / "fwd/perp_1d/AAA/AAA-1d-2026-10.zip", d0 + 30, [7.0] * 31)
    monkeypatch.setattr(F, "BASE_RAW", tmp_path / "base")
    monkeypatch.setattr(F, "RAW", tmp_path / "fwd")
    d, q = F.volumes_of("AAA")
    assert len(d) == 61 and q[0] == 5.0 and q[-1] == 7.0


def test_lock_hash_changes_with_content(tmp_path):
    f = tmp_path / "r.py"
    f.write_text("a")
    h = F.rule_hash([str(f)])
    f.write_text("b")
    assert F.rule_hash([str(f)]) != h and len(h) == 64
    assert all(Path(p).exists() for p in F.RULE_FILES)


# ---------------------------------------------------------------- evaluation on synthetic markets

def _market(T_=14000, seed=0, lead=None, vol=0.0035, gap=0.01):
    rng = np.random.default_rng(seed)
    mu = np.repeat(rng.choice([-1, 0, 1], T_ // 1500 + 1) * 5e-4, 1500)[:T_]
    lr = mu + vol * rng.standard_t(4, T_) / np.sqrt(2)
    lr[rng.random(T_) < 0.004] *= 12
    if lead is not None:
        lr[4:] += 0.5 * lead[:T_ - 4]
    c = 100 * np.exp(np.cumsum(lr))
    o = np.concatenate([[100.0], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0008, T_)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0008, T_)))
    real = rng.random(T_) > gap
    for i in range(1, T_):
        if not real[i]:
            o[i] = h[i] = l[i] = c[i] = c[i - 1]
    t = Q0 + np.arange(T_, dtype=np.int64)
    f = np.full(T_, np.nan)
    s = np.flatnonzero(t % 32 == 0)
    f[s] = rng.normal(1e-4, 3e-4, len(s))
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "real": real, "funding": f}, lr


BTC, LR = _market(14000, 0)
PANELS = {"BTCUSDT@spot": BTC, "BTCUSDT@perp": BTC, "ETHUSDT@perp": _market(14000, 1, lead=LR)[0],
          "AAAUSDT": _market(14000, 2, lead=LR)[0], "BBBUSDT": _market(14000, 3, lead=LR)[0]}
MEMBERS = {"2026-10": ["AAAUSDT", "BBBUSDT"], "2026-11": ["AAAUSDT"], "2026-12": [], "2027-01": []}
ASOF = date(2026, 12, 20)


def test_forward_window_patches_and_restores_the_grid():
    d0, n0 = T.day0(), T.n_days()
    with F.forward_window(date(2026, 10, 10)):
        assert T.day0() == epoch_15m(date(2026, 10, 1)) // 96 and T.n_days() == 10
    assert (T.day0(), T.n_days()) == (d0, n0)
    with pytest.raises(RuntimeError):
        with F.forward_window(date(2026, 10, 10)):
            raise RuntimeError
    assert T.n_days() == n0


def test_evaluate_matches_the_engine_and_reports_pending_months():
    out = F.evaluate(PANELS, MEMBERS, ASOF, n_rnd=30, n_boot=200)
    assert out["asof"] == str(ASOF) and out["days"] == (ASOF - date(2026, 10, 1)).days + 1
    assert out["followers"] == ["AAAUSDT", "BBBUSDT"] and out["pending_months"] == ["2026-12"]
    assert out["n_events"] > 0 and out["n_trades"] >= out["n_events"] and 1 / 31 <= out["null_p"] <= 1
    pc = out["primary_criterion"]
    assert pc["meets"] == (pc["sharpe_gt_0"] and pc["mean_trade_gt_0"]) and pc["min_events"] == 30
    assert set(out["scenarios"]) == {"base", "cost75", "x2", "delay1", "nofund", "nolag", "btc_perp_leader"}
    # direct computation with the engine (what the research 17 test did), member months only
    from qlab.research16 import engine as G
    from qlab.research17 import engine as R
    from qlab.research15.evaluate import stats
    with F.forward_window(ASOF):
        ok = {m: v for m, v in MEMBERS.items() if v}
        P = {k: F.truncate(v, ASOF) for k, v in PANELS.items()}
        leaders = {"spot": R.leader(P["BTCUSDT@spot"]), "perp": R.leader(P["BTCUSDT@perp"])}
        daily = 0
        for s in ("AAAUSDT", "BBBUSDT"):
            a = G.Asset.build(P[s])
            s0 = int(epoch_15m(date(2026, 10, 1)) - a.t[0])
            daily = daily + R.run_follower(a, leaders, [R.Scenario("base")], s0, R.member_mask(a, s, ok))["base"].daily
        assert out["P_stats"]["sharpe"] == pytest.approx(stats(daily)["sharpe"])
        # the first month of the forward test holds both members, the next one only AAAUSDT
        assert set(out["members_used"]) == {"2026-10", "2026-11"}
    assert set(out["per_symbol"]) <= {"AAAUSDT", "BBBUSDT"}
    assert len(out["daily"]["dates"]) == out["days"] and len(out["trades"]) == out["n_trades"]
    assert out["daily"]["dates"][0] == "2026-10-01" and out["trades"] == sorted(out["trades"], key=lambda x: x["entry"])
    assert all(x["entry"] >= "2026-10-01" for x in out["trades"])
    assert sum(x["ret"] for x in out["trades"]) == pytest.approx(sum(v["n"] * v["mean"] for v in out["per_symbol"].values()))


def test_evaluate_ignores_data_after_asof_and_history_start_is_not_traded():
    base = F.evaluate(PANELS, MEMBERS, ASOF, n_rnd=20, n_boot=100)
    rng = np.random.default_rng(1)
    pert = {}
    cut = int((PANELS["BTCUSDT@spot"]["t"] < epoch_15m(ASOF) + 96).sum())
    for k, d in PANELS.items():
        d = {a: v.copy() for a, v in d.items()}
        for a in ("o", "h", "l", "c"):
            d[a][cut:] *= np.exp(rng.normal(0, 0.1, len(d[a]) - cut))
        d["real"][cut:] = rng.random(len(d["real"]) - cut) > 0.5
        pert[k] = d
    again = F.evaluate(pert, MEMBERS, ASOF, n_rnd=20, n_boot=100)
    assert json.dumps(base, default=float) == json.dumps(again, default=float)
    # nothing traded before 2026-10-01: the panels start on 2026-09-01 and the grid starts at the forward start
    with F.forward_window(ASOF):
        assert T.day0() * 96 == epoch_15m(date(2026, 10, 1))


def test_evaluate_without_followers_or_events_is_well_formed():
    out = F.evaluate(PANELS, {"2026-10": [], "2026-11": []}, date(2026, 10, 20), n_rnd=5, n_boot=50)
    assert out["followers"] == [] and out["n_events"] == 0 and out["null_p"] is None
    assert out["primary_criterion"] is None and out["P_stats"]["sharpe"] == 0.0
    assert out["pending_months"] == ["2026-10"]


def test_fetch_days_raises_on_server_errors_instead_of_ending_quietly(tmp_path, monkeypatch):
    def get(url, tries=5):
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)
    monkeypatch.setattr(dl, "_get", get)
    with pytest.raises(urllib.error.HTTPError):
        F.fetch_days("perp", "AAAUSDT", date(2026, 10, 1), date(2026, 10, 3), out=tmp_path)


# ---------------------------------------------------------------- operations: health, alerts, dashboard

from datetime import datetime, timezone  # noqa: E402

from qlab.research17 import dashboard, ops  # noqa: E402

TODAY = date(2026, 10, 20)
FWD_OK = {"asof": "2026-10-19", "stale_symbols": {}, "pending_months": []}


def test_parse_download_picks_error_records_and_exit_code():
    out = "{'market': 'perp', 'symbol': 'A', 'new': 3}\n{'symbol': 'B', 'error': 'URLError: dns'}\nerrors: 1\n"
    assert ops.parse_download(out, 1) == ["{'symbol': 'B', 'error': 'URLError: dns'}"]
    assert ops.parse_download("errors: 0\n", 0) == []
    assert "kódem 2" in ops.parse_download("traceback...", 2)[0]


def test_problems_cover_each_failure_mode():
    ok = {"download": {"errors": [], "exit": 0}, "status": {"exit": 0}}
    assert ops.problems(ok, FWD_OK, TODAY) == []
    assert "stahování: 1 chyb" in ops.problems({**ok, "download": {"errors": ["x"], "exit": 1}}, FWD_OK, TODAY)[0]
    p = ops.problems({**ok, "status": {"exit": 1, "error": "boom"}}, None, TODAY)
    assert len(p) == 1 and "status selhal" in p[0]
    assert "chybí výsledek" in ops.problems(ok, None, TODAY)[0]
    assert "data zaostávají" in ops.problems(ok, {**FWD_OK, "asof": "2026-10-15"}, TODAY)[0]
    assert ops.problems(ok, {**FWD_OK, "asof": "2026-10-17"}, TODAY) == []                  # 3 days is still fine
    assert "zastaralá data u: AAAUSDT" in ops.problems(ok, {**FWD_OK, "stale_symbols": {"AAAUSDT": "2026-10-10"}}, TODAY)[0]
    pend = {**FWD_OK, "asof": "2026-11-02", "pending_months": ["2026-11"]}
    assert ops.problems(ok, pend, date(2026, 11, 3)) == []                                    # ZIPs not published yet
    assert "2026-11" in ops.problems(ok, {**pend, "asof": "2026-11-08"}, date(2026, 11, 9))[0]


def test_notify_never_raises_and_needs_a_topic():
    assert ops.notify(None, "t", "b") is False
    seen = []
    assert ops.notify("topic", "Chyba: stažení", "těžká zpráva", post=lambda r: seen.append(r)) is True
    r = seen[0]
    assert r.full_url == "https://ntfy.sh/topic" and r.data.decode() == "těžká zpráva"
    assert r.get_header("Title") == "Chyba: sta?en?" and r.get_header("Priority") == "high"

    def boom(_):
        raise OSError("no network")
    assert ops.notify("topic", "t", "b", post=boom) is False


def test_run_age():
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    assert ops.run_age_days({"last_run": "2026-10-13T12:00:00+00:00"}, now) == pytest.approx(7)
    assert ops.run_age_days({}, now) is None


def _forward_out():
    return F.evaluate(PANELS, MEMBERS, ASOF, n_rnd=20, n_boot=100) | {"commit": "abc", "rule_hash": "r" * 64,
                                                                    "data_hash": "d" * 64}


def test_dashboard_renders_numbers_alerts_and_escapes_html():
    fwd = _forward_out()
    fwd["trades"][0] = {**fwd["trades"][0], "sym": "<script>alert(1)</script>"}
    state = {"last_run": "2026-10-19T06:30:00+00:00", "last_ok": "2026-10-12T06:30:00+00:00",
             "download": {"exit": 1, "errors": ["x"]}, "status": {"exit": 0}, "alerts": ["stahování: 1 chyb <b>"]}
    page = dashboard.render(fwd, state, datetime(2026, 10, 20, tzinfo=timezone.utc), date(2026, 10, 1),
                            date(2027, 3, 31), 30)
    assert "PROBLÉM" in page and "stahování: 1 chyb &lt;b&gt;" in page and "<script>alert" not in page
    assert f"{fwd['n_events']}" in page and "<svg" in page and "Neprůkazné" in page and "Citlivost" in page
    assert "2026-12-20" in page or "2026-12" in page
    assert "chybí" in page or fwd["pending_months"] == []
    assert page.count("<script") == 0                                  # static page: CSP-friendly, nothing to run


def test_dashboard_handles_no_forward_result_and_ok_state():
    page = dashboard.render(None, {"download": {}, "status": {}, "alerts": []}, datetime(2026, 10, 20, tzinfo=timezone.utc),
                            date(2026, 10, 1), date(2027, 3, 31), 30)
    assert "OK" in page and "Zatím není žádný výsledek" in page
    stale = dashboard.render(None, {"last_run": "2026-10-01T06:30:00+00:00", "alerts": []},
                             datetime(2026, 10, 20, tzinfo=timezone.utc), date(2026, 10, 1), date(2027, 3, 31), 30)
    assert "cron neběží" in stale


@pytest.mark.skipif(not list(Path("docs/research17/results").glob("forward_2*.json")), reason="no forward status yet")
def test_dashboard_renders_the_committed_status_file():
    f = json.loads(sorted(Path("docs/research17/results").glob("forward_2*.json"))[-1].read_text())
    if "daily" not in f:
        pytest.skip("status file predates the daily/trades fields")
    page = dashboard.render(f, {"alerts": [], "download": {}, "status": {}}, datetime.now(timezone.utc),
                            date(2026, 10, 1), date(2027, 3, 31), 30)
    assert f["asof"] in page and str(f["n_events"]) in page
