import pytest

from arena.data.book_validate import compare, parse_bookdepth_csv

CSV = """timestamp,percentage,depth,notional
2026-10-22 00:00:08,-1.00,10.0,1000.0
2026-10-22 00:00:08,1.00,5.0,500.0
2026-10-22 00:00:08,-2.00,20.0,2000.0
2026-10-22 00:00:38,-1.00,10.0,1100.0
2026-10-22 00:00:38,1.00,5.0,400.0
"""


def test_parse_pairs_bid_and_ask():
    d = parse_bookdepth_csv(CSV)
    t0 = 1_792_627_208_000
    assert d[t0] == {1.0: (1000.0, 500.0)}           # 2 % bez asku se zahodí
    assert d[t0 + 30_000] == {1.0: (1100.0, 400.0)}


def test_compare_identical_series_passes():
    ref = {i * 30_000 + 8_000: {1.0: (1000.0 + 37 * (i % 7), 800.0 + 11 * (i % 5))} for i in range(50)}
    live = {i * 30_000: {1.0: v[1.0]} for i, v in enumerate(ref.values())}     # posun 8 s → mimo ±5 s
    assert compare("X", live, ref, (1.0,))[0].n_pairs == 0
    live = {i * 30_000 + 3_000: {1.0: v[1.0]} for i, v in enumerate(ref.values())}
    r = compare("X", live, ref, (1.0,))[0]
    assert r.n_pairs == 50 and r.obi_corr == pytest.approx(1.0) and r.median_abs_dnotional_pct == 0 and r.passed


async def test_backfill_book_from_fixture(monkeypatch):
    import datetime as dt
    import gzip

    from arena.data import book_validate as bv
    from arena.db import Database
    from tests.conftest import DATA

    async def fake_fetch(sym, date, client=None):
        p = DATA / f"{sym}-bookDepth-{date}.csv.gz"
        return parse_bookdepth_csv(gzip.open(p, "rt").read()) if p.exists() else None

    monkeypatch.setattr(bv, "fetch_bookdepth", fake_fetch)
    db = Database(":memory:")
    db.upsert_bar_book([("BTCUSDT", 1_792_454_400_000, 1.0, 0.123, 10, 1, "live")])   # živá data mají přednost
    out = await bv.backfill_book(db, ("BTCUSDT",), 2, (0.2, 1.0, 2.0, 5.0), today=dt.date(2026, 10, 22))
    assert out == {"BTCUSDT": 576}
    rows = db.execute("SELECT * FROM bar_book WHERE pct=1.0 ORDER BY open_time").fetchall()
    assert len(rows) == 576 and rows[0]["source"] == "live" and rows[0]["obi"] == 0.123
    assert rows[1]["source"] == "bookDepth" and rows[1]["book_ok"] == 1
