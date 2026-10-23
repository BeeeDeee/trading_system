"""`arena replay` dvakrát po sobě dá identické obchody (SPEC §17 P3)."""

import gzip
import json

import pytest

from arena.config import load_config
from arena.data.recorder import AGG_HEADER, utc_date
from arena.db import Bar, Database
from arena.replay import iter_agg_trades, run_replay
from tests.conftest import load_klines


def _exchange_info():
    def sym(name, tick, step, mn):
        return {"symbol": name, "status": "TRADING", "underlyingType": "COIN", "triggerProtect": "0.03",
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": tick},
                            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": "1000000"},
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": "1000000"},
                            {"filterType": "MIN_NOTIONAL", "notional": mn}]}
    return {"symbols": [sym("BTCUSDT", "0.10", "0.001", "50"), sym("ETHUSDT", "0.01", "0.001", "20")]}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("replay")
    db = Database(d / "src.db")
    aid = 0
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = load_klines(sym)
        bars = [Bar(sym, int(b.open_time[i]), float(b.o[i]), float(b.h[i]), float(b.l[i]), float(b.c[i]), 1.0,
                    float(b.qv[i]), 10, 0.5, float(b.tbqv[i])) for i in range(len(b))]
        db.upsert_bars(bars)
        # syntetické ticky: o → h → l → c (nebo o → l → h → c) s jemnými mezikroky
        by_day: dict[str, list[str]] = {}
        for bar in bars:
            path = [bar.o, bar.h, bar.l, bar.c] if bar.c >= bar.o else [bar.o, bar.l, bar.h, bar.c]
            pts = []
            for x, y in zip(path, path[1:]):
                pts += [x + (y - x) * k / 6 for k in range(6)]
            pts.append(bar.c)
            for k, px in enumerate(pts):
                aid += 1
                T = bar.open_time + 1000 + k * (290_000 // len(pts))
                day = utc_date(T)
                by_day.setdefault(day, []).append(f"{aid},{px:.2f},0.01,{aid},{aid},{T},{'true' if k % 2 else 'false'}\n")
        for day, rows in by_day.items():
            p = d / "ticks" / "aggTrades" / sym / f"{day}.csv.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "wt") as f:
                f.write(AGG_HEADER + "".join(rows))
    db.meta_set("exchange_info", json.dumps(_exchange_info()))
    yield db, d / "ticks"
    db.close()


def test_iter_agg_trades_sorted_and_dedup(src, tmp_path):
    _, ticks = src
    ts = [(t["T"], t["a"]) for t in iter_agg_trades(ticks, "BTCUSDT", ["2026-10-14"])]
    assert ts == sorted(ts) and len(ts) == len(set(ts)) > 1000


def test_replay_twice_identical(src):
    db, ticks = src
    cfg = load_config()
    r1 = run_replay(cfg, db, "2026-10-14", "2026-10-15", ticks)
    r2 = run_replay(cfg, db, "2026-10-14", "2026-10-15", ticks)
    assert len(r1.trades) > 20, len(r1.trades)
    assert r1.fingerprint == r2.fingerprint
    assert [t.row() for t in r1.trades] == [t.row() for t in r2.trades]
    reasons = {t.exit_reason for t in r1.trades}
    assert "trailing" in reasons
    fams = {t.slot_id.split(".")[0] for t in r1.trades}
    assert {"brk", "ctl"} <= fams
    for t in r1.trades:
        assert t.exit_time >= t.entry_time > 0
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.entry_fee - t.exit_fee + t.funding)
