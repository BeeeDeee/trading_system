"""Lokální kniha: postup Binance (snapshot + diff), resync při mezeře v `pu`, pásma, simulace fillu."""

import asyncio

import pytest

from arena.data.book import BookSync, LocalBook


def ev(U, u, pu, b=(), a=(), E=0):
    return {"e": "depthUpdate", "E": E, "s": "BTCUSDT", "U": U, "u": u, "pu": pu,
            "b": [[str(p), str(q)] for p, q in b], "a": [[str(p), str(q)] for p, q in a]}


SNAP = {"lastUpdateId": 100, "bids": [["99", "1"], ["98", "2"]], "asks": [["101", "1"], ["102", "3"]]}


def synced_book():
    bk = LocalBook("BTCUSDT")
    bk.on_diff(ev(90, 95, 89, b=[(99, 5)]))          # starší než snapshot → zahodit
    bk.on_diff(ev(96, 105, 95, b=[(99, 7)]))         # pokrývá lastUpdateId=100
    bk.on_diff(ev(106, 110, 105, a=[(101, 0)]))      # smazání úrovně
    assert bk.apply_snapshot(SNAP) == "synced"
    return bk


def test_snapshot_procedure_and_zero_qty_removal():
    bk = synced_book()
    assert bk.synced and bk.last_u == 110
    assert bk.bids == {99.0: 7.0, 98.0: 2.0}
    assert bk.asks == {102.0: 3.0}
    assert bk.on_diff(ev(111, 112, 110, b=[(97, 1)])) == "applied"
    assert bk.on_diff(ev(113, 113, 112, b=[(50, 0)])) == "applied"   # mazání neexistující úrovně je normální


def test_gap_in_pu_forces_resync():
    bk = synced_book()
    assert bk.on_diff(ev(115, 118, 114)) == "gap"     # čekal pu=110
    assert not bk.synced and bk.resyncs == 1 and len(bk.buffer) == 1


def test_stale_snapshot_when_first_event_after_snapshot():
    bk = LocalBook("BTCUSDT")
    bk.on_diff(ev(150, 160, 149))
    assert bk.apply_snapshot(SNAP) == "stale_snapshot"
    assert not bk.synced


def test_snapshot_waits_for_covering_event():
    bk = LocalBook("BTCUSDT")
    bk.on_diff(ev(90, 95, 89))
    assert bk.apply_snapshot(SNAP) == "need_events"
    assert bk.pending is not None
    bk.on_diff(ev(96, 101, 95, b=[(99, 4)]))
    assert bk.try_pending() == "synced"
    assert bk.bids[99.0] == 4.0


def test_broken_pu_chain_inside_buffer():
    bk = LocalBook("BTCUSDT")
    bk.on_diff(ev(96, 105, 95))
    bk.on_diff(ev(107, 110, 106))                    # pu ≠ 105
    assert bk.apply_snapshot(SNAP) == "stale_snapshot" and not bk.synced


def test_crossed_book_resyncs():
    bk = synced_book()
    assert bk.on_diff(ev(111, 111, 110, b=[(103, 1)])) == "gap"


def test_keep_outside_levels_on_preventive_resnapshot():
    old = synced_book()
    old.bids[50.0] = 9.0           # vzdálená úroveň doplněná diffem, ve snapshotu není
    old.asks[150.0] = 4.0
    old.bids[98.5] = 1.0           # uvnitř rozsahu snapshotu → snapshot má přednost (úroveň zmizí)
    new = LocalBook("BTCUSDT")
    new.on_diff(ev(196, 205, 195))
    snap = {"lastUpdateId": 200, "bids": [["99", "1"], ["98", "2"]], "asks": [["101", "1"], ["102", "3"]]}
    assert new.apply_snapshot(snap, keep_outside=old) == "synced"
    assert new.bids == {99.0: 1.0, 98.0: 2.0, 50.0: 9.0}
    assert new.asks == {101.0: 1.0, 102.0: 3.0, 150.0: 4.0}


def test_bands_cumulative_notional_around_mid():
    bk = LocalBook("X", bids={99.9: 1.0, 99.5: 2.0, 98.0: 10.0}, asks={100.1: 1.0, 101.5: 3.0}, synced=True)
    bands = bk.bands((0.2, 1.0, 2.0))
    assert bands[0].mid == pytest.approx(100.0)
    # 0,2 %: bid ≥ 99.8, ask ≤ 100.2
    assert bands[0].bid_notional == pytest.approx(99.9) and bands[0].ask_notional == pytest.approx(100.1)
    # 1 %: bid ≥ 99, ask ≤ 101
    assert bands[1].bid_notional == pytest.approx(99.9 + 199.0) and bands[1].ask_notional == pytest.approx(100.1)
    # 2 %: bid ≥ 98 (včetně), ask ≤ 102
    assert bands[2].bid_notional == pytest.approx(99.9 + 199.0 + 980.0)
    assert bands[2].ask_notional == pytest.approx(100.1 + 304.5)
    assert bands[1].obi == pytest.approx((298.9 - 100.1) / (298.9 + 100.1))


def test_simulate_market_walks_book():
    bk = LocalBook("X", bids={99.0: 1.0, 98.0: 1.0}, asks={101.0: 1.0, 102.0: 2.0}, synced=True)
    f = bk.simulate_market("BUY", 2.0)
    assert f.vwap == pytest.approx(101.5) and f.complete and f.levels == 2
    f = bk.simulate_market("SELL", 3.0)
    assert not f.complete and f.filled_qty == pytest.approx(2.0) and f.vwap == pytest.approx(98.5)


async def test_book_sync_resyncs_after_gap():
    calls = []

    async def fetch(sym):
        calls.append(sym)
        return dict(SNAP, lastUpdateId=100 if len(calls) == 1 else 200)

    bs = BookSync(LocalBook("BTCUSDT"), fetch)
    bs.on_diff(ev(96, 105, 95))
    await asyncio.sleep(0.3)
    assert bs.book.synced and len(calls) == 1
    bs.on_diff(ev(106, 107, 105))
    bs.on_diff(ev(190, 199, 180))       # mezera
    assert not bs.book.synced
    bs.on_diff(ev(200, 205, 199))
    await asyncio.sleep(0.3)
    assert bs.book.synced and len(calls) == 2 and bs.book.last_u == 205
