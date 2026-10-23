"""TrailingSim na syntetických tickových cestách (SPEC §17 P3)."""

import pytest

from arena.execution.trailing_sim import TrailingSim


def run(sim, path, t0=1000):
    """path = ceny, každá v jiném čase; vrací potvrzený výstup."""
    for i, px in enumerate(path):
        out = sim.on_trade(px, t0 + i * 10, i + 1)
        if out is not None:
            return out
    return sim.flush()


def test_long_trailing_from_entry():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=98.0, callback_pct=1.0)
    out = run(sim, [100.5, 102.0, 101.5, 101.0, 100.97, 100.0])
    assert out.kind == "TS" and out.trigger_px == pytest.approx(102.0 * 0.99)
    assert out.ts == 1040                       # obchod 100.97 ≤ 100.98
    assert sim.mfe_px == 102.0


def test_short_trailing_mirror():
    sim = TrailingSim("SHORT", 100.0, hard_stop_px=102.0, callback_pct=1.0)
    out = run(sim, [99.5, 98.0, 98.5, 98.97, 99.0])
    assert out.kind == "TS" and out.trigger_px == pytest.approx(98.0 * 1.01) and out.ts == 1040


def test_trailing_without_move_triggers_from_entry_hwm():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=95.0, callback_pct=0.5)
    assert run(sim, [99.6, 99.5]).trigger_px == pytest.approx(99.5)


def test_activation_price_long():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=98.0, callback_pct=1.0, activate_px=101.0)
    # před aktivací by pokles o 1 % od maxima trailing nespustil
    out = run(sim, [100.8, 99.7, 100.9, 101.0, 101.6, 100.6, 100.55])
    assert sim.activated
    assert out.kind == "TS" and out.trigger_px == pytest.approx(101.6 * 0.99) and out.ts == 1060


def test_activation_never_reached_hard_stop_protects():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=98.0, callback_pct=1.0, activate_px=101.0)
    out = run(sim, [100.5, 99.0, 98.0])
    assert not sim.activated and out.kind == "HS" and out.trigger_px == 98.0


def test_activation_price_short():
    sim = TrailingSim("SHORT", 100.0, hard_stop_px=102.0, callback_pct=1.0, activate_px=99.0)
    out = run(sim, [99.5, 100.4, 99.0, 98.0, 98.99])
    assert out.kind == "TS" and out.trigger_px == pytest.approx(98.98)


def test_tp_requires_trade_strictly_beyond_limit():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=97.0, callback_pct=3.0, tp_px=102.0)
    assert run(sim, [101.0, 102.0, 102.0, 101.9]) is None      # dotek nestačí
    sim = TrailingSim("LONG", 100.0, hard_stop_px=97.0, callback_pct=3.0, tp_px=102.0)
    out = run(sim, [101.0, 102.0, 102.01])
    assert out.kind == "TP" and out.trigger_px == 102.0 and not out.is_market
    sim = TrailingSim("SHORT", 100.0, hard_stop_px=103.0, callback_pct=3.0, tp_px=98.0)
    assert run(sim, [98.0]) is None
    sim = TrailingSim("SHORT", 100.0, hard_stop_px=103.0, callback_pct=3.0, tp_px=98.0)
    assert run(sim, [97.99]).kind == "TP"


def test_simultaneous_exits_same_time_worse_wins():
    # dva obchody se stejným T: první by vyplnil TP, druhý spustí tvrdý stop → horší (HS) vyhrává
    sim = TrailingSim("LONG", 100.0, hard_stop_px=99.0, callback_pct=None, tp_px=101.0)
    assert sim.on_trade(101.5, 5000, 1) is None
    assert sim.on_trade(98.9, 5000, 2) is None
    out = sim.on_trade(100.0, 5001, 3)                         # vyšší čas potvrdí dávku
    assert out.kind == "HS"


def test_trailing_and_stop_on_same_trade_takes_lower_level():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=99.0, callback_pct=0.5)
    sim.on_trade(101.0, 1000, 1)                               # HWM 101 → úroveň 100.495
    sim.on_trade(98.5, 1010, 2)                                # gap pod obě úrovně
    out = sim.flush()
    assert out.kind == "HS" and out.trigger_px == 99.0          # nižší úroveň = horší pro LONG


def test_tick_order_decides_when_times_differ():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=99.0, callback_pct=None, tp_px=101.0)
    sim.on_trade(101.5, 5000, 1)
    out = sim.on_trade(98.9, 5001, 2)
    assert out.kind == "TP"


def test_ignores_trades_before_entry_and_duplicates():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=99.0, callback_pct=1.0, start_ts=1000)
    assert sim.on_trade(50.0, 999, 1) is None and sim.pending is None
    sim.on_trade(101.0, 1001, 5)
    sim.on_trade(10.0, 1002, 5)                                 # duplicitní aggId
    assert sim.pending is None and sim.extreme == 101.0


def test_json_roundtrip_keeps_state():
    sim = TrailingSim("SHORT", 100.0, hard_stop_px=102.0, callback_pct=1.0, activate_px=99.0)
    sim.on_trade(98.5, 1000, 1)
    sim2 = TrailingSim.from_json(sim.to_json())
    assert sim2 == sim
    assert sim2.on_trade(99.6, 1010, 2) is None and sim2.flush().kind == "TS"


def test_no_trailing_only_hard_stop():
    sim = TrailingSim("LONG", 100.0, hard_stop_px=98.0)
    assert run(sim, [105.0, 99.0, 98.01]) is None
    sim = TrailingSim("LONG", 100.0, hard_stop_px=98.0)
    assert run(sim, [105.0, 98.0]).kind == "HS"
