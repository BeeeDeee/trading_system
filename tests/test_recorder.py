import gzip
import zlib

import pytest

from arena.data.book import LocalBook
from arena.data.recorder import AGG_HEADER, AggTradeRecorder, BookSampler, aggregate_bar_book
from arena.db import Database

DAY1 = 1_792_713_600_000   # 2026-10-23 00:00 UTC
DAY2 = DAY1 + 86_400_000


def agg(a, T, sym="BTCUSDT", m=True):
    return {"e": "aggTrade", "E": T, "s": sym, "a": a, "p": "65000.10", "q": "0.005", "f": a * 10, "l": a * 10 + 2,
            "T": T, "m": m}


def read_partial(path):
    """Přečte gzip i bez koncového bloku (stav po pádu procesu)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return d.decompress(path.read_bytes()).decode()


def test_rows_format_rotation_and_part_segments(tmp_path):
    events = []
    r = AggTradeRecorder(tmp_path, on_event=lambda *a: events.append(a))
    r.on_agg_trade(agg(1, DAY1 + 1000))
    r.on_agg_trade(agg(2, DAY1 + 2000, m=False))
    r.on_agg_trade(agg(2, DAY1 + 2000))            # duplicita po reconnectu
    r.on_agg_trade(agg(5, DAY1 + 3000))            # mezera 3..4
    r.flush()
    f1 = tmp_path / "aggTrades" / "BTCUSDT" / "2026-10-23.csv.gz"
    text = read_partial(f1)                         # čitelné po flush bez close
    assert text == AGG_HEADER + "1,65000.10,0.005,10,12,1792713601000,true\n" \
        "2,65000.10,0.005,20,22,1792713602000,false\n5,65000.10,0.005,50,52,1792713603000,true\n"
    assert events and events[0][1] == "mezera v aggTrades" and events[0][2]["missing"] == 2
    r.on_agg_trade(agg(6, DAY2 + 10))               # rotace o půlnoci
    r.close()
    assert gzip.open(tmp_path / "aggTrades" / "BTCUSDT" / "2026-10-24.csv.gz", "rt").read().endswith(
        "6,65000.10,0.005,60,62,1792800000010,true\n")
    # restart téhož dne → nový segment part2
    r2 = AggTradeRecorder(tmp_path)
    r2.on_agg_trade(agg(7, DAY2 + 20))
    r2.close()
    assert (tmp_path / "aggTrades" / "BTCUSDT" / "2026-10-24.part2.csv.gz").exists()


def test_aggregate_bar_book_mean_and_book_ok():
    rows = []
    for i in range(10):
        ts = DAY1 + i * 30_000
        rows.append({"ts": ts, "pct": 1.0, "bid_notional": 300.0, "ask_notional": 100.0})
    rows.append({"ts": DAY1 + 300_000, "pct": 1.0, "bid_notional": 0.0, "ask_notional": 100.0})  # další bar
    out = aggregate_bar_book("BTCUSDT", DAY1, rows, (1.0, 2.0))
    assert out[0] == ("BTCUSDT", DAY1, 1.0, pytest.approx(0.5), 10, 1, "live")
    assert out[1] == ("BTCUSDT", DAY1, 2.0, None, 0, 0, "live")
    out = aggregate_bar_book("BTCUSDT", DAY1, rows[:7], (1.0,))
    assert out[0][4:6] == (7, 0)          # 7 z 10 → book_ok = 0


def test_sampler_skips_unsynced_or_stale_book():
    db = Database(":memory:")
    fresh = LocalBook("BTCUSDT", bids={99.0: 1.0}, asks={101.0: 1.0}, synced=True, last_apply_local_ms=10_000)
    stale = LocalBook("ETHUSDT", bids={99.0: 1.0}, asks={101.0: 1.0}, synced=True, last_apply_local_ms=5_000)
    unsynced = LocalBook("SOLUSDT", bids={99.0: 1.0}, asks={101.0: 1.0}, synced=False, last_apply_local_ms=10_000)
    s = BookSampler(db, {"BTCUSDT": fresh, "ETHUSDT": stale, "SOLUSDT": unsynced}, (1.0, 2.0),
                    lambda: DAY1, lambda: 10_500, max_diff_age_s=2.0)
    rows = s.sample(DAY1)
    assert {r[0] for r in rows} == {"BTCUSDT"} and s.samples_bad == 2
    assert len(db.book_bands("BTCUSDT", DAY1, DAY1 + 1)) == 2
