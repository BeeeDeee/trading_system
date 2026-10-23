"""Paper executor: vstup s latencí, výstupy, poplatky, funding, časový výstup, výpadek."""

from decimal import Decimal

import pytest

from arena.data.bars import BAR_MS
from arena.execution.paper import PaperExecutor, TradeFillModel
from arena.strategies.base import Intent
from tests.test_filters import _sym

BAR = 1_792_000_000_000 - 1_792_000_000_000 % BAR_MS
DEC = BAR + BAR_MS + 500


def make(**kw):
    closed = []
    ex = PaperExecutor({"BTCUSDT": _sym(tick="0.10", step="0.001")}, TradeFillModel(), on_closed=closed.append,
                       **kw)
    return ex, closed


def tr(px, T, a):
    return {"p": str(px), "T": T, "a": a, "s": "BTCUSDT"}


def intent(side="LONG", cb=1.0, hs=2.0, act=None, tp=None, hold=36):
    return Intent(side, cb, act, hs, tp, hold, "test")


def test_entry_latency_fill_fees_and_trailing_exit():
    ex, closed = make()
    ex.open("brk.v00.BTCUSDT", "BTCUSDT", 1, intent(), 7, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 100, 1))          # před latencí → fill ještě ne
    assert ex.positions["brk.v00.BTCUSDT"].status == "entering"
    ex.on_trade("BTCUSDT", tr(50010, DEC + 260, 2))          # první obchod ≥ t_d + 250 ms
    pos = ex.positions["brk.v00.BTCUSDT"]
    assert pos.status == "open" and pos.entry_px == 50010 and pos.qty == 0.02
    assert pos.plan["hard_stop_px"] == pytest.approx(49009.8)   # 50010 × 0.98 = 49009.8 (na tick)
    ex.on_trade("BTCUSDT", tr(51000, DEC + 1000, 3))
    ex.on_trade("BTCUSDT", tr(50480, DEC + 2000, 4))         # ≤ 51000 × 0.99 = 50490 → trigger
    assert ex.positions["brk.v00.BTCUSDT"].status == "open"   # čeká na potvrzení dávky
    ex.on_trade("BTCUSDT", tr(50470, DEC + 2050, 5))         # potvrdí, fill v t+100 ms
    ex.on_trade("BTCUSDT", tr(50450, DEC + 2100, 6))
    assert len(closed) == 1
    t = closed[0]
    assert t.exit_reason == "trailing" and t.exit_px == 50450 and t.exit_trigger_px == pytest.approx(50490)
    assert t.entry_fee == pytest.approx(0.02 * 50010 * 0.0005)
    assert t.exit_fee == pytest.approx(0.02 * 50450 * 0.0005)
    assert t.gross_pnl == pytest.approx(0.02 * 440)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.entry_fee - t.exit_fee)
    assert t.entry_slip_bps == pytest.approx(2.0) and t.exit_slip_bps == pytest.approx(40 / 50490 * 1e4)
    assert t.mfe_bps == pytest.approx((51000 - 50010) / 50010 * 1e4)
    assert t.signal_id == 7 and t.mode == "paper" and t.trade_id == "p-brk.v00.BTCUSDT-1"
    assert not ex.positions


def test_short_take_profit_is_maker_at_limit():
    ex, closed = make()
    ex.open("s", "BTCUSDT", 1, intent("SHORT", cb=3.0, hs=5.0, tp=49000.0), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    ex.on_trade("BTCUSDT", tr(49000, DEC + 400, 2))          # dotek
    assert not closed
    ex.on_trade("BTCUSDT", tr(48999.9, DEC + 500, 3))
    ex.on_trade("BTCUSDT", tr(48999.9, DEC + 600, 4))
    t = closed[0]
    assert t.exit_reason == "take_profit" and t.exit_px == 49000.0
    assert t.exit_fee == pytest.approx(t.qty * 49000 * 0.0002) and t.exit_slip_bps == 0


def test_marketable_tp_is_skipped():
    ex, closed = make()
    ex.open("s", "BTCUSDT", 1, intent(tp=49900.0), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    pos = ex.positions["s"]
    assert pos.plan["tp_px"] is None and "tp_skipped_marketable" in pos.notes


def test_time_exit_after_max_hold_bars_and_bars_held():
    ex, closed = make()
    ex.open("s", "BTCUSDT", 1, intent(hold=3), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    for k in (1, 2):
        ex.on_bar_close(BAR + k * BAR_MS, BAR + (k + 1) * BAR_MS + 500, {"BTCUSDT": 50000.0})
    assert ex.positions["s"].status == "open"
    d = BAR + 4 * BAR_MS + 500
    ex.on_bar_close(BAR + 3 * BAR_MS, d, {"BTCUSDT": 50100.0})
    assert ex.positions["s"].status == "exiting"
    ex.on_trade("BTCUSDT", tr(50090, d + 260, 2))
    t = closed[0]
    assert t.exit_reason == "time" and t.exit_trigger_px == 50100.0 and t.bars_held == 3


def test_funding_long_pays_positive_rate():
    ex, closed = make()
    ex.open("s", "BTCUSDT", 1, intent(), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    ex.on_funding("BTCUSDT", DEC + 10_000, 0.0001, 50000.0)
    ex.on_funding("BTCUSDT", DEC - 10_000, 0.0001, 50000.0)   # před vstupem → nic
    assert ex.positions["s"].funding == pytest.approx(-0.02 * 50000 * 0.0001)


def test_outage_replay_taints_and_penalizes():
    ex, closed = make(outage_penalty_bps=2.0)
    ex.open("s", "BTCUSDT", 1, intent(cb=1.0, hs=2.0), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    ex.replay_outage("BTCUSDT", [tr(50500, DEC + 5000, 2), tr(49990, DEC + 6000, 3)], now=DEC + 60_000)
    t = closed[0]
    assert t.tainted == 1 and t.exit_reason == "trailing"
    assert t.exit_px == pytest.approx(49990 * (1 - 2e-4))


def test_state_roundtrip_restore():
    states = {}
    ex, _ = make()
    ex.on_state = lambda sid, st: states.__setitem__(sid, st) if st else states.pop(sid, None)
    ex.open("s", "BTCUSDT", 1, intent(act=0.5), None, BAR, DEC, 50000.0, "cfg")
    ex.on_trade("BTCUSDT", tr(50000, DEC + 300, 1))
    ex.on_trade("BTCUSDT", tr(50300, DEC + 400, 2))
    ex2, closed = make()
    ex2.restore(states)                     # stav uložený při fillu vstupu (HWM se ukládá na baru)
    assert not ex2.positions["s"].sim.activated
    for t in (tr(50000, DEC + 300, 1), tr(50300, DEC + 400, 2)):   # přehrání obchodů od fillu obnoví HWM
        ex2.on_trade("BTCUSDT", t)
    p1, p2 = ex.positions["s"], ex2.positions["s"]
    assert p2.sim == p1.sim and p2.plan == p1.plan and p2.sim.activated


async def test_fetch_agg_trades_pages_past_sparse_first_window():
    from arena.engine import fetch_agg_trades

    trades = [{"a": i, "T": 1000 + i * 600_000, "p": "1"} for i in range(20)]   # 1 obchod / 10 min

    class FakeRest:
        async def agg_trades(self, symbol, start_ms=None, end_ms=None, from_id=None, limit=1000):
            if from_id is not None:
                return [t for t in trades if t["a"] >= from_id][:3]
            return [t for t in trades if start_ms <= t["T"] <= end_ms][:limit]

    out = await fetch_agg_trades(FakeRest(), "BTCUSDT", 0, trades[15]["T"])
    assert [t["a"] for t in out] == list(range(16))
