"""Backtest: převod aggTrades zip → npz, okna ticků, shoda s `arena replay` na syntetických datech, report."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from arena.backtest.data import Store, TickStore, agg_zip_to_npz, dates, months, read_bars, read_funding
from arena.backtest.report import build_report
from arena.config import load_config

HDR = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"


def _zip(path: Path, name: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, text)


def test_agg_zip_to_npz_sorts_dedups_and_parses_bool(tmp_path):
    rows = ["5,100.5,1,5,5,1700000000300,true", "4,100.0,1,4,4,1700000000100,false",
            "5,100.5,1,5,5,1700000000300,true", "6,101.0,2,6,6,1700000000300,false"]
    z = tmp_path / "x.zip"
    _zip(z, "x.csv", HDR + "\n".join(rows) + "\n")
    out = tmp_path / "x.npz"
    assert agg_zip_to_npz(z, out) == 3
    d = np.load(out)
    assert d["a"].tolist() == [4, 5, 6] and d["T"].tolist() == [1700000000100, 1700000000300, 1700000000300]
    assert d["p"].tolist() == [100.0, 100.5, 101.0]


def test_agg_zip_without_header(tmp_path):
    z = tmp_path / "y.zip"
    _zip(z, "y.csv", "1,10.0,1,1,1,1700000000000,false\n2,11.0,1,2,2,1700000000001,false\n")
    assert agg_zip_to_npz(z, tmp_path / "y.npz") == 2


def test_tickstore_window_spans_days_and_skips_missing(tmp_path):
    st = Store(tmp_path)
    day = 86_400_000
    t0 = 1_700_000_000_000 - 1_700_000_000_000 % day          # půlnoc UTC
    for i, d in enumerate(["2023-11-14", "2023-11-15"]):
        base = t0 + i * day
        p = st.ticks("BTCUSDT", d)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, T=np.array([base + 10, base + day - 10]), a=np.array([2 * i, 2 * i + 1]), p=np.array([1.0 + i, 2.0 + i]))
    ts = TickStore(st, "BTCUSDT")
    got = list(ts.window(t0 + 5, t0 + 2 * day - 5))
    assert [g[1] for g in got] == [0, 1, 2, 3]                   # [lo, hi) přes půlnoc
    assert list(ts.window(t0 + 2 * day, t0 + 3 * day)) == []     # den bez souboru


def test_dates_months_and_readers(tmp_path):
    assert dates("2026-01-30", "2026-02-02") == ["2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02"]
    assert months("2025-11-15", "2026-02-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    st = Store(tmp_path)
    _zip(st.kline("BTCUSDT", "2026-01-01"), "k.csv",
         "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
         "1767225600000,1,2,0.5,1.5,10,1767225899999,15,3,4,6,0\n")
    bars = read_bars(st, "BTCUSDT", "2026-01-01", "2026-01-01")
    assert len(bars) == 1 and bars[0].c == 1.5 and bars[0].tbqv == 6.0
    _zip(st.funding("BTCUSDT", "2026-01"), "f.csv",
         "calc_time,funding_interval_hours,last_funding_rate\n1767225600008,8,0.0001\n1767254400002,8,-0.0002\n")
    assert read_funding(st, "BTCUSDT", "2026-01-01", "2026-01-01") == [(1767225600008, 0.0001), (1767254400002, -0.0002)]
    assert read_funding(st, "BTCUSDT", "2026-01-02", "2026-01-02") == []


def _row(slot, i, net, tainted=0, side="LONG", exit_time=1_767_300_000_000):
    return {"trade_id": f"p-{slot}-{i}", "slot_id": slot, "notional": 1000.0, "entry_fee": 0.2, "exit_fee": 0.2,
            "side": side, "entry_time": exit_time - 600_000, "exit_time": exit_time + i * 3_600_000, "exit_reason": "trailing",
            "net_pnl": net / 10, "gross_bps": net + 8, "net_bps": net, "entry_slip_bps": 0.5, "exit_slip_bps": 0.5,
            "funding": 0.0, "mfe_bps": 5.0, "mae_bps": -5.0, "tainted": tainted}


def test_build_report_writes_summary_variants_and_index(tmp_path):
    cfg = load_config(Path(__file__).parent.parent / "config")
    rows = [_row("brk.v00.BTCUSDT", i, 3.0 * (i % 5 - 2)) for i in range(40)] + \
           [_row("ctl.brk.BTCUSDT", i, -1.0) for i in range(40)] + [_row("brk.v00.ETHUSDT", 0, 9.0, tainted=1)]
    meta = {"id": "t", "label": "t", "from": "2026-01-01", "to": "2026-01-03", "start_ms": 1_767_225_600_000,
            "end_ms": 1_767_225_600_000 + 3 * 86_400_000 - 1, "sample": "x", "symbols": ["BTCUSDT"], "runtime_s": 1}
    rep = build_report(cfg, rows, meta, tmp_path, tmp_path / "run")
    assert rep["n_trades"] == 80 and rep["n_tainted_excluded"] == 1
    sm = json.loads((tmp_path / "backtest" / "t" / "summary.json").read_text())
    assert sm["leaderboard"]["brk.v00"]["n"] == 40 and sm["leaderboard"]["brk.v01"]["n"] == 0
    v = json.loads((tmp_path / "backtest" / "t" / "variants" / "brk.v00.json").read_text())
    assert len(v["curve"]["t"]) == 3 and v["by_symbol"]["BTCUSDT"]["n"] == 40
    assert json.loads((tmp_path / "backtest" / "index.json").read_text())["runs"][0]["id"] == "t"
    assert (tmp_path / "run" / "trades.jsonl.gz").exists()
    build_report(cfg, rows, meta, tmp_path)                       # opakovaný běh přepíše, nezdvojí
    assert len(json.loads((tmp_path / "backtest" / "index.json").read_text())["runs"]) == 1


def test_array_feature_source_matches_db_source():
    from arena.backtest.engine import ArrayFeatureSource
    from arena.data.bars import BAR_MS
    from arena.db import Bar, Database
    from arena.slots import FeatureSource
    rng = np.random.default_rng(7)
    db = Database(":memory:")
    t0, n = 1_767_225_600_000, 1500
    for sym in ("BTCUSDT", "ETHUSDT"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        bars = [Bar(sym, t0 + i * BAR_MS, c[i - 1] if i else c[0], c[i] * 1.001, c[i] * 0.999, c[i], 10.0, 1000 + rng.random() * 500,
                    5, 4.0, 400 + rng.random() * 200) for i in range(n) if i != 700]    # díra uprostřed
        db.upsert_bars(bars)
        db.upsert_bar_book([(sym, b.open_time, p, float(rng.normal()), 10, int(rng.random() > 0.05), "bookDepth")
                            for b in bars for p in (0.2, 1.0, 2.0)])
    ref, fast = FeatureSource(db, 864), ArrayFeatureSource(db, 864).load(["BTCUSDT", "ETHUSDT"])
    for i in (300, 699, 701, 900, 1499):
        t = t0 + i * BAR_MS
        for sym in ("BTCUSDT", "ETHUSDT"):
            a, b = ref.arrays("BTCUSDT", t), fast.arrays("BTCUSDT", t)
            fa, fb = ref.features(sym, t, a), fast.features(sym, t, b)
            assert (fa is None) == (fb is None)
            if fa is not None:
                assert fa.keys() == fb.keys()
                for k in fa:
                    assert fa[k] == fb[k] or (np.isnan(fa[k]) and np.isnan(fb[k])), (i, sym, k)
