"""Každá varianta: signál ano / ne na syntetických featurách (SPEC §17 P2)."""

import math

import pytest

from arena.config import load_config
from arena.strategies.base import BarContext, common_callback, round1
from arena.strategies.ctl import slot_rng
from arena.strategies.registry import build_slots, build_strategies, global_warmup

S = build_strategies(load_config())
BASE = {"c": 100.0, "atr14_pct": 0.15, "lvz288": 1.5, "cvd12_z": 0.5, "book_ok": 1.0,
        "don_hi_12": 99.0, "don_lo_12": 90.0, "don_hi_24": 99.5, "don_lo_24": 90.0, "don_hi_48": 101.0, "don_lo_48": 89.0,
        "obi0.2_z": 0.0, "obi1_z": 0.0, "obi2_z": 0.0,
        "ret1": 0.0, "ret1_prev": 0.0, "ret1_btc": 0.0, "ret1_btc_prev": 0.0, "sigma288_btc": 0.001,
        "beta288": 1.0, "c_btc": 65000.0, "ema_576_btc": 64000.0}


def ctx(sym="ETHUSDT", vid="x", **kw):
    return BarContext(sym, 1_792_000_000_000, {**BASE, **kw}, f"{vid}.{sym}")


def sig(vid, sym="ETHUSDT", **kw):
    return S[vid].on_bar(ctx(sym, vid, **kw))


def test_common_exit_rule():
    assert common_callback(0.15, 4, 0.2, 3.0) == 0.6
    assert common_callback(0.01, 4, 0.2, 3.0) == 0.2       # clip dole
    assert common_callback(1.0, 7, 0.2, 3.0) == 3.0        # clip nahoře
    assert round1(0.25) == 0.3 and common_callback(math.nan, 4, 0.2, 3) is None
    it = sig("brk.v00")
    assert it.callback_pct == 0.6 and it.hard_stop_pct == 1.2 and it.max_hold_bars == 36


# --- brk ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("vid,yes,no", [
    ("brk.v00", {}, {"c": 99.4}),                                   # c > don_hi_24 = 99.5
    ("brk.v01", {"c": 99.2}, {"c": 98.9}),                           # N=12
    ("brk.v02", {"c": 101.5}, {}),                                   # N=48: 100 < 101
    ("brk.v03", {"lvz288": -2.0}, {"c": 99.0}),                      # bez objemového filtru
    ("brk.v04", {"lvz288": 2.1}, {}),                                # vz=2: 1.5 nestačí
    ("brk.v05", {}, {"c": 99.0}),
    ("brk.v06", {}, {"c": 99.0}),
    ("brk.v07", {}, {"c": 89.0}),                                    # jen LONG: průraz dolů nic
    ("brk.v08", {}, {"lvz288": 0.5}),
    ("brk.v09", {}, {"c": 99.0}),
])
def test_brk_variants(vid, yes, no):
    assert sig(vid, **yes) is not None
    assert sig(vid, **no) is None


def test_brk_short_and_params():
    it = sig("brk.v00", c=89.0)
    assert it.side == "SHORT"
    assert sig("brk.v00", lvz288=0.9) is None                        # vz=1.0
    assert sig("brk.v05").callback_pct == round1(2.5 * 0.15) == 0.4
    assert sig("brk.v06").callback_pct == 1.1
    assert sig("brk.v08").max_hold_bars == 96
    v9 = sig("brk.v09")
    assert v9.callback_pct == 0.4 and v9.activate_offset_pct == pytest.approx(0.6) and v9.hard_stop_pct == 1.0
    assert sig("brk.v00", c=math.nan) is None


# --- obi ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("vid,key,yes,no", [
    ("obi.v00", "obi1_z", 2.5, 2.4),
    ("obi.v01", "obi1_z", 2.0, 1.9),
    ("obi.v02", "obi1_z", 3.0, 2.9),
    ("obi.v03", "obi2_z", 2.5, 2.4),
    ("obi.v04", "obi0.2_z", 2.5, 2.4),
    ("obi.v05", "obi1_z", 2.6, -2.6),                                # jen LONG
    ("obi.v06", "obi1_z", 2.6, 0.0),
    ("obi.v07", "obi1_z", 2.6, 0.0),
    ("obi.v08", "obi1_z", 2.6, 0.0),
    ("obi.v09", "obi1_z", 2.6, 0.0),
])
def test_obi_variants(vid, key, yes, no):
    assert sig(vid, **{key: yes}) is not None
    assert sig(vid, **{key: no}) is None


def test_obi_details():
    assert sig("obi.v00", **{"obi1_z": -2.5}).side == "SHORT"
    assert sig("obi.v00", **{"obi1_z": 3.0, "obi2_z": 0}).reason.startswith("obi1_z")
    assert sig("obi.v03", **{"obi1_z": 3.0}) is None                 # v03 čte jen 2 %
    # book_ok = 0 → stale, žádný signál
    c = ctx(vid="obi.v00", **{"obi1_z": 4.0, "book_ok": 0.0})
    assert S["obi.v00"].is_stale(c) and S["obi.v00"].on_bar(c) is None
    # v08: taker flow nesmí jít proti
    assert sig("obi.v08", **{"obi1_z": 2.6, "cvd12_z": -0.1}) is None
    assert sig("obi.v08", **{"obi1_z": -2.6, "cvd12_z": -0.1}).side == "SHORT"
    assert sig("obi.v08", **{"obi1_z": -2.6, "cvd12_z": 0.1}) is None
    # v06, v07, v09
    assert sig("obi.v06", **{"obi1_z": 2.6}).max_hold_bars == 96
    assert sig("obi.v07", **{"obi1_z": 2.6}).callback_pct == 1.2       # 8 × 0.15
    v9 = sig("obi.v09", **{"obi1_z": 2.6})
    assert v9.callback_pct == 0.8 and v9.take_profit_px == pytest.approx(100 * (1 + 2 * 0.8 / 100))
    assert sig("obi.v09", **{"obi1_z": -2.6}).take_profit_px == pytest.approx(100 * (1 - 0.016))
    assert sig("obi.v00", **{"obi1_z": 2.6}).take_profit_px is None


# --- ll ----------------------------------------------------------------------------------------

JUMP = {"ret1_btc": 0.0031, "ret1": 0.0005}       # rb = 3.1, alt 5 bps ≤ 0.3 × 1 × 31 bps


@pytest.mark.parametrize("vid,yes,no", [
    ("ll.v00", JUMP, {"ret1_btc": 0.0029, "ret1": 0.0}),
    ("ll.v01", {"ret1_btc": 0.0026, "ret1": 0.0}, {"ret1_btc": 0.0024, "ret1": 0.0}),
    ("ll.v02", {"ret1_btc": 0.0041, "ret1": 0.0}, JUMP),
    ("ll.v03", {"ret1_btc": 0.0031, "ret1": -0.0001}, JUMP),        # λ = 0: alt musí být ≤ 0
    ("ll.v04", JUMP, {"ret1": 0.0010}),                               # alt už dohnal (10 > 9.3 bps)
    ("ll.v05", JUMP, {"ret1_btc": -0.0031}),
    ("ll.v06", {"ret1_btc": -0.0031, "ret1": -0.0005}, {"ret1_btc": -0.0031, "ret1": -0.0020}),
    ("ll.v07", {"ret1_btc": 0.0025, "ret1_btc_prev": 0.0020, "ret1": 0.0}, JUMP),   # 2 bary: 45/(10·√2)=3.18
    ("ll.v08", JUMP, {**JUMP, "c_btc": 63000.0}),                     # BTC pod EMA 576
    ("ll.v09", JUMP, {"ret1_btc": 0.0}),
])
def test_ll_variants(vid, yes, no):
    assert sig(vid, **yes) is not None
    assert sig(vid, **no) is None


def test_ll_details():
    assert sig("ll.v00", ret1_btc=-0.0031, ret1=-0.0005) is None      # v00 jen LONG
    assert sig("ll.v06", ret1_btc=-0.0031, ret1=-0.0005).side == "SHORT"
    assert sig("ll.v04", **JUMP).max_hold_bars == 12 and sig("ll.v05", **JUMP).max_hold_bars == 96
    v9 = sig("ll.v09", **JUMP)
    assert v9.callback_pct is None and v9.hard_stop_pct == 2.0 and v9.max_hold_bars == 48
    assert sig("ll.v00", **JUMP, beta288=math.nan) is None
    assert S["ll.v08"].warmup_bars == 1152
    assert S["ll.v00"].symbols == ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")


# --- ctl a registry ------------------------------------------------------------------------------

def test_ctl_deterministic_and_frequency():
    s = S["ctl.obi"]
    hits = [s.on_bar(BarContext("BTCUSDT", 1_792_000_000_000 + i * 300_000, BASE, "ctl.obi.BTCUSDT",
                                extra={"ctl_entry_prob": 0.05, "ctl_long_share": 0.7})) for i in range(20000)]
    entries = [h for h in hits if h is not None]
    assert 0.04 < len(entries) / 20000 < 0.06
    assert 0.65 < sum(e.side == "LONG" for e in entries) / len(entries) < 0.75
    # výstupy identické s V00 cílové rodiny
    v00 = S["obi.v00"].exits(ctx(vid="obi.v00"), "LONG", "x")
    assert (entries[0].callback_pct, entries[0].hard_stop_pct, entries[0].max_hold_bars) == \
        (v00.callback_pct, v00.hard_stop_pct, v00.max_hold_bars)
    assert slot_rng("a", 1).random() == slot_rng("a", 1).random() != slot_rng("a", 2).random()
    # ctl obi mlčí na baru bez knihy
    assert s.on_bar(BarContext("BTCUSDT", 1, {**BASE, "book_ok": 0.0}, "ctl.obi.BTCUSDT",
                               extra={"ctl_entry_prob": 1.0})) is None


def test_registry_slot_counts():
    slots = build_slots(load_config())
    assert sum(not s.is_control and s.strategy.family != "xp" for s in slots) == 140
    assert sum(s.strategy.family == "xp" for s in slots) == 5
    assert sorted(s.slot_id for s in slots if s.is_control)[:2] == ["ctl.brk.BTCUSDT", "ctl.brk.DOGEUSDT"]
    assert sum(s.is_control for s in slots) == 14
    assert global_warmup(S) == 1440
