"""Live executor a rekonciliace proti falešné burze (akceptace P5 v testech; skutečné ověření na testnetu)."""

import asyncio
import random
from decimal import Decimal

import pytest

from arena.binance import errors
from arena.config import load_config
from arena.db import Database
from arena.execution.live import LiveExecutor
from arena.reconcile import Reconciler
from arena.risk import RiskGuard
from arena.strategies.base import Intent
from tests.fake_exchange import FakeExchange
from tests.test_filters import _sym

SYM = "XRPUSDT"
SLOT = "brk.v00.XRPUSDT"
BAR = 1_792_000_000_000 - 1_792_000_000_000 % 300_000


class Clock:
    def __init__(self):
        self.t = BAR + 300_500

    def __call__(self):
        self.t += 7
        return self.t


def make(fx=None, db=None, deadline=2.0):
    fx = fx or FakeExchange()
    db = db or Database(":memory:")
    alerts, closed = [], []
    ex = LiveExecutor(fx, {SYM: _sym(tick="0.0001", step="0.1", min_notional="5", symbol=SYM, mmax="2000000")}, db,
                      Clock(), lambda s: float(fx.price[s]), lambda t, m: alerts.append((t, m)),
                      lambda rec, pos: closed.append(rec), protect_deadline_s=deadline)
    return ex, fx, db, alerts, closed


def intent(cb=1.0, act=None, tp=None, hs=2.0):
    return Intent("LONG", cb, act, hs, tp, 36, "test")


async def open_(ex, seq=1, it=None):
    return await ex.open(SLOT, SYM, seq, it or intent(), 7, f"p-{SLOT}-{seq}", BAR, BAR + 300_500, 0.5, 150.0, "cfg")


async def deliver(ex, events):
    for e in events:
        await ex.on_user_event(e)


def algo_types(fx):
    return sorted(a["type"] for a in fx.algos.values())


def conditional_on_order_endpoint(fx):
    return [c for c in fx.calls if c[0] == "POST /fapi/v1/order" and "STOP" in str(c[1].get("type"))]


async def test_happy_path_trailing_exit_cancels_siblings_and_records_trade():
    ex, fx, db, alerts, closed = make()
    pos = await open_(ex)
    assert pos.state == "PROTECTED" and Decimal(pos.qty) == 300
    assert algo_types(fx) == ["STOP_MARKET", "TRAILING_STOP_MARKET"]
    hs = fx.algos[pos.orders["HS"]]
    assert Decimal(hs["triggerPrice"]) == Decimal("0.4900") and Decimal(hs["quantity"]) == 300
    ts_req = [c[1] for c in fx.calls if c[1].get("type") == "TRAILING_STOP_MARKET"][0]
    assert ts_req["callbackRate"] == Decimal("1.0") and ts_req["reduceOnly"] == "true"
    assert ts_req["workingType"] == "CONTRACT_PRICE"
    # stínový sim: cena vyroste a trailing se spustí
    ex.on_trade_tick(SYM, {"p": "0.5100", "T": pos.entry_time + 1000, "a": 1})
    fx.price[SYM] = Decimal("0.5049")
    await deliver(ex, fx.trigger_algo(pos.orders["TS"]))
    assert not ex.positions and len(closed) == 1
    t = closed[0]
    assert t.exit_reason == "trailing" and t.mode == "live" and t.twin_trade_id == f"p-{SLOT}-1"
    assert t.exit_trigger_px == pytest.approx(0.51 * 0.99)
    assert t.exit_fee == pytest.approx(300 * 0.5049 * 0.0005)
    assert fx.algos == {}                               # HS zrušen
    assert fx.positions[SYM] == 0
    assert not conditional_on_order_endpoint(fx)
    assert db.execute("SELECT COUNT(*) FROM latencies").fetchone()[0] >= 3


async def test_hard_stop_failure_triggers_emergency_close():
    fx = FakeExchange(algo_fail={"STOP_MARKET": [errors.BinanceError(400, -1000, "x")] * 3})
    ex, fx, db, alerts, closed = make(fx)
    await open_(ex)
    assert any(a[0] == "EMERGENCY_CLOSE" for a in alerts)
    em = [c for c in fx.calls if c[0] == "POST /fapi/v1/order" and c[1].get("newClientOrderId", "").endswith("-EM")]
    assert len(em) == 1 and em[0][1]["reduceOnly"] == "true"
    assert fx.positions[SYM] == 0


async def test_protection_timeout_over_2s_emergency_close(monkeypatch):
    fx = FakeExchange()
    orig = fx.new_algo_order

    async def slow(p):
        await asyncio.sleep(0.3)
        return await orig(p)

    fx.new_algo_order = slow
    ex, fx, db, alerts, closed = make(fx, deadline=0.2)
    await open_(ex)
    assert any(a[0] == "EMERGENCY_CLOSE" for a in alerts) and fx.positions[SYM] == 0


async def test_trailing_minus_2021_retries_without_activate_price():
    ex, fx, db, alerts, closed = make()
    orig = fx.new_algo_order

    async def move_then_place(p):
        if p["type"] == "STOP_MARKET":
            fx.price[SYM] = Decimal("0.5200")  # mezi vstupem a trailingem trh přeskočil aktivační cenu 0.505
        return await orig(p)

    fx.new_algo_order = move_then_place
    pos = await open_(ex, it=intent(act=1.0))
    ts = [c[1] for c in fx.calls if c[1].get("type") == "TRAILING_STOP_MARKET"]
    assert "activatePrice" in ts[0] and "activatePrice" not in ts[1]
    assert pos.state == "PROTECTED" and "trailing bez activatePrice (-2021)" in pos.notes


async def test_entry_timeout_queries_instead_of_resending():
    fx = FakeExchange(entry_timeout_but_filled=True)
    ex, fx, db, alerts, closed = make(fx)
    pos = await open_(ex)
    entries = [c for c in fx.calls if c[1].get("newClientOrderId", "").endswith("-E")]
    assert len(entries) == 1 and pos.state == "PROTECTED"


async def test_entry_timeout_not_found_goes_flat():
    fx = FakeExchange(order_fail=[errors.UnknownOrderStatus("POST", "/fapi/v1/order", "x", "timeout")])
    ex, fx, db, alerts, closed = make(fx)
    assert await open_(ex) is None
    assert not ex.positions and fx.positions.get(SYM, 0) == 0
    assert len([c for c in fx.calls if c[0] == "POST /fapi/v1/order"]) == 1


async def test_kill9_between_entry_and_protection_restart_protects():
    db = Database(":memory:")
    ex, fx, db, alerts, closed = make(db=db)

    async def crash(pos):
        raise asyncio.CancelledError   # proces zemřel po fillu vstupu, před ochranou

    ex._protect = crash
    with pytest.raises(asyncio.CancelledError):
        await open_(ex)
    assert fx.positions[SYM] == Decimal("300.0") and fx.algos == {}
    # restart: nový executor, stav z DB, rekonciliace
    ex2, _, _, alerts2, _ = make(fx, db)
    ex2.restore()
    assert ex2.positions[SYM].state == "PROTECTING"
    risk = RiskGuard(load_config(), db, live_flag=False)
    diffs = await Reconciler(fx, ex2, risk, (SYM,)).run(fix=True)
    assert [d.kind for d in diffs] == ["UNPROTECTED"]
    assert algo_types(fx) == ["STOP_MARKET", "TRAILING_STOP_MARKET"]
    assert ex2.positions[SYM].state == "PROTECTED"


async def test_kill9_after_entry_sent_restart_resolves_entry():
    db = Database(":memory:")
    ex, fx, db, alerts, closed = make(db=db)
    orig = fx.new_order

    async def die_after_send(p):
        await orig(p)
        raise asyncio.CancelledError   # odpověď se už nezpracovala

    fx.new_order = die_after_send
    with pytest.raises(asyncio.CancelledError):
        await open_(ex)
    fx.new_order = orig
    ex2, _, _, _, _ = make(fx, db)
    ex2.restore()
    assert ex2.positions[SYM].state == "ENTRY_SENT"
    diffs = await Reconciler(fx, ex2, RiskGuard(load_config(), db, False), (SYM,)).run(fix=True)
    assert diffs[0].kind == "ENTRY_FILLED_UNPROTECTED"
    assert algo_types(fx) == ["STOP_MARKET", "TRAILING_STOP_MARKET"]


async def test_orphan_a5_algo_cancelled_foreign_untouched():
    ex, fx, db, alerts, closed = make()
    fx.algos["a5-ABCDEF-9-HS"] = {"algoId": 1, "clientAlgoId": "a5-ABCDEF-9-HS", "symbol": SYM, "type": "STOP_MARKET"}
    fx.algos["owner-manual-1"] = {"algoId": 2, "clientAlgoId": "owner-manual-1", "symbol": SYM, "type": "STOP_MARKET"}
    fx.orders["a5-ABCDEF-9-TP"] = {"clientOrderId": "a5-ABCDEF-9-TP", "symbol": SYM, "open": True}
    fx.orders["web_123"] = {"clientOrderId": "web_123", "symbol": SYM, "open": True}
    diffs = await Reconciler(fx, ex, RiskGuard(load_config(), db, False), (SYM,)).run(fix=True)
    assert sorted(d.kind for d in diffs) == ["ORPHAN_ALGO", "ORPHAN_ORDER"]
    assert list(fx.algos) == ["owner-manual-1"] and fx.orders["web_123"]["open"]


async def test_foreign_position_freezes_without_action():
    ex, fx, db, alerts, closed = make()
    fx.positions["XRPUSDT"] = Decimal("100")
    risk = RiskGuard(load_config(), db, False)
    diffs = await Reconciler(fx, ex, risk, (SYM,)).run(fix=True)
    assert diffs[0].kind == "FOREIGN_POSITION" and SYM in risk.frozen_symbols
    assert fx.positions[SYM] == Decimal("100") and not [c for c in fx.calls if c[0].startswith("POST")]


async def test_qty_mismatch_freezes():
    ex, fx, db, alerts, closed = make()
    await open_(ex)
    fx.positions[SYM] += Decimal("10")
    risk = RiskGuard(load_config(), db, False)
    diffs = await Reconciler(fx, ex, risk, (SYM,)).run(fix=True)
    assert diffs[0].kind == "QTY_MISMATCH" and SYM in risk.frozen_symbols


async def test_time_exit_cancels_algos_before_market():
    ex, fx, db, alerts, closed = make()
    pos = await open_(ex)
    n0 = len(fx.calls)
    await ex.on_bar_close(BAR + 36 * 300_000, {SYM: 0.5})
    seq = [c[0] for c in fx.calls[n0:]]
    assert seq.index("DELETE /fapi/v1/algoOrder") < seq.index("POST /fapi/v1/order")
    mkt = [c[1] for c in fx.calls[n0:] if c[0] == "POST /fapi/v1/order"][0]
    assert mkt["reduceOnly"] == "true" and mkt["newClientOrderId"].endswith("-TX")
    t = fx.user_trades_log[-1]
    await deliver(ex, [fx.otu({"clientOrderId": mkt["newClientOrderId"]}, t, True)])
    assert closed[0].exit_reason == "time" and not ex.positions


async def test_minus_2022_race_reconciled_from_user_trades():
    ex, fx, db, alerts, closed = make()
    pos = await open_(ex)
    fx.trigger_algo(pos.orders["HS"])          # stop proběhl, ale události nedorazily (WS výpadek)
    await ex.on_bar_close(BAR + 36 * 300_000, {SYM: 0.5})    # časový výstup → -2022
    assert "-2022 při výstupu TX" in ex.positions[SYM].notes
    diffs = await Reconciler(fx, ex, RiskGuard(load_config(), db, False), (SYM,)).run(fix=True)
    assert diffs[0].kind == "POSITION_GONE"
    assert closed and closed[0].exit_reason == "reconciled" and not ex.positions


async def test_200_round_trips_never_unprotected():
    """Invariant: po návratu z open() má každá živá pozice tvrdý stop, nebo je zavřená."""
    rng = random.Random(5)
    fx = FakeExchange()
    ex, fx, db, alerts, closed = make(fx)
    for i in range(1, 201):
        if rng.random() < 0.1:
            fx.algo_fail["STOP_MARKET"] = [errors.UnknownOrderStatus("POST", "/fapi/v1/algoOrder", None, "t")]
        if rng.random() < 0.05:
            fx.algo_fail["TRAILING_STOP_MARKET"] = [errors.BinanceError(400, -1000, "x")] * 3
        pos = await open_(ex, seq=i)
        await deliver(ex, fx.events)            # filly market příkazů z WS (vstup, případně emergency close)
        fx.events.clear()
        live = fx.positions.get(SYM, 0)
        if live != 0:
            assert any(a["type"] == "STOP_MARKET" for a in fx.algos.values()), i
            await deliver(ex, fx.trigger_algo(pos.orders["HS"]))
        assert fx.positions.get(SYM, 0) == 0 and not ex.positions, i
        fx.algos.clear()
        fx.algo_fail.clear()
    assert len(closed) == 200
    assert not conditional_on_order_endpoint(fx)


async def test_time_exit_rejected_restores_protection():
    ex, fx, db, alerts, closed = make()
    pos = await open_(ex)
    fx.order_fail = [errors.BinanceError(400, -1001, "Internal error")] * 3
    await ex.on_bar_close(BAR + 36 * 300_000, {SYM: 0.5})
    assert ex.positions[SYM].state == "PROTECTED"
    assert algo_types(fx) == ["STOP_MARKET", "TRAILING_STOP_MARKET"]      # stopy zpět na burze
    assert any(a[0] == "výstup selhal" for a in alerts)


async def test_reconcile_retries_pending_emergency_close():
    fx = FakeExchange(algo_fail={"STOP_MARKET": [errors.BinanceError(400, -1000, "x")] * 3})
    ex, fx, db, alerts, closed = make(fx)
    fx.order_fail = []
    orig = fx.new_order
    calls = {"n": 0}

    async def flaky(p):
        if p.get("newClientOrderId", "").endswith("-EM"):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise errors.BinanceError(400, -1001, "Internal error")
        return await orig(p)

    fx.new_order = flaky
    await open_(ex)
    assert ex.positions[SYM].state == "EMERGENCY_CLOSE" and fx.positions[SYM] != 0
    diffs = await Reconciler(fx, ex, RiskGuard(load_config(), db, False), (SYM,)).run(fix=True)
    assert diffs[0].kind == "EXIT_PENDING" and fx.positions[SYM] == 0


async def test_positions_view_has_hard_stop_and_entry_time():
    from types import SimpleNamespace
    from arena.live_runtime import LiveRuntime
    ex, fx, db, alerts, closed = make()
    pos = await open_(ex)
    me = SimpleNamespace(executor=ex, last_price=lambda sym: 0.5)
    (v,) = LiveRuntime.positions_view(me)
    assert v["hard_stop"] == pytest.approx(0.49) and v["entry_time"] == pos.entry_time > 0
