"""BarStore: jen uzavřené bary, mezera → backfill, reconnect doplní mezeru, stale při selhání; koordinátor."""

import asyncio

from arena.data.bars import BAR_MS, BarCloseCoordinator, BarStore
from arena.db import Bar, Database

T0 = 1_792_000_000_000 - 1_792_000_000_000 % BAR_MS


def kline_rest(t, c=100.0):
    return [t, "100", "101", "99", str(c), "10", t + BAR_MS - 1, "1000", 5, "6", "600", "0"]


def kline_ws(sym, t, x=True, c=100.0):
    return {"t": t, "T": t + BAR_MS - 1, "s": sym, "i": "5m", "o": "100", "c": str(c), "h": "101", "l": "99",
            "v": "10", "n": 5, "x": x, "q": "1000", "V": "6", "Q": "600", "B": "0"}


class FakeExchange:
    def __init__(self, upto):
        self.upto = upto
        self.calls = []
        self.fail = False

    async def klines(self, sym, start, end):
        self.calls.append((sym, start, end))
        if self.fail:
            raise ConnectionError("down")
        return [kline_rest(t) for t in range(start, min(end, self.upto) + 1, BAR_MS)][:1500]


def make_store(now, upto, warmup=10, closed=None):
    db = Database(":memory:")
    ex = FakeExchange(upto)
    clock = {"now": now}
    st = BarStore(db, ("BTCUSDT",), ex.klines, lambda: clock["now"], warmup,
                  on_closed=(lambda b, live: closed.append((b.open_time, live))) if closed is not None else None)
    return st, db, ex, clock


def test_ws_and_rest_bars_identical():
    k = kline_rest(T0)
    w = kline_ws("BTCUSDT", T0)
    assert Bar.from_rest("BTCUSDT", k) == Bar.from_ws(w)


async def test_warmup_backfill_and_open_kline_ignored():
    now = T0 + 20 * BAR_MS + 1000
    st, db, ex, _ = make_store(now, upto=T0 + 19 * BAR_MS)
    await st.backfill("BTCUSDT")
    bars = db.bars("BTCUSDT")
    assert len(bars) == 10 and bars[-1].open_time == T0 + 19 * BAR_MS
    await st.on_kline(kline_ws("BTCUSDT", T0 + 20 * BAR_MS, x=False))   # rozpracovaný → ignorovat
    assert db.last_bar_time("BTCUSDT") == T0 + 19 * BAR_MS


async def test_gap_triggers_backfill_before_store():
    now = T0 + 20 * BAR_MS + 1000
    st, db, ex, clock = make_store(now, upto=T0 + 19 * BAR_MS)
    await st.backfill("BTCUSDT")
    # výpadek: bary 20..22 chybí, přijde až 23
    ex.upto = T0 + 23 * BAR_MS
    clock["now"] = T0 + 24 * BAR_MS + 500
    await st.on_kline(kline_ws("BTCUSDT", T0 + 23 * BAR_MS))
    times = [b.open_time for b in db.bars("BTCUSDT")]
    assert times[-4:] == [T0 + i * BAR_MS for i in (20, 21, 22, 23)]
    assert "BTCUSDT" not in st.stale


async def test_reconnect_backfill_fills_gap():
    """Umělé odpojení: po reconnectu backfill_all doplní bary, které WS nedoručil."""
    now = T0 + 20 * BAR_MS + 1000
    st, db, ex, clock = make_store(now, upto=T0 + 19 * BAR_MS)
    await st.backfill_all(from_warmup=True)
    ex.upto = T0 + 25 * BAR_MS
    clock["now"] = T0 + 26 * BAR_MS + 100
    n = await st.backfill_all()
    times = [b.open_time for b in db.bars("BTCUSDT")]
    assert times[-1] == T0 + 25 * BAR_MS
    assert all(b - a == BAR_MS for a, b in zip(times, times[1:]))


async def test_backfill_failure_marks_stale():
    now = T0 + 20 * BAR_MS + 1000
    st, db, ex, clock = make_store(now, upto=T0 + 19 * BAR_MS)
    await st.backfill("BTCUSDT")
    ex.fail = True
    clock["now"] = T0 + 23 * BAR_MS + 100
    await st.on_kline(kline_ws("BTCUSDT", T0 + 22 * BAR_MS))
    assert "BTCUSDT" in st.stale
    clock["now"] = T0 + 24 * BAR_MS + 100
    await st.on_kline(kline_ws("BTCUSDT", T0 + 23 * BAR_MS))   # díra pořád nedoplněná → stále stale
    assert "BTCUSDT" in st.stale
    ex.fail = False
    ex.upto = T0 + 24 * BAR_MS
    clock["now"] = T0 + 25 * BAR_MS + 100
    await st.on_kline(kline_ws("BTCUSDT", T0 + 24 * BAR_MS))   # retry doplní 20..24
    assert "BTCUSDT" not in st.stale
    times = [b.open_time for b in db.bars("BTCUSDT")]
    assert all(b - a == BAR_MS for a, b in zip(times, times[1:])) and times[-1] == T0 + 24 * BAR_MS


async def test_coordinator_waits_for_all_symbols_or_timeout():
    emitted = []
    clock = {"now": T0 + BAR_MS + 50}
    co = BarCloseCoordinator(("BTCUSDT", "ETHUSDT"), lambda t, s, d: emitted.append((t, s)),
                             lambda: clock["now"], wait_s=0.05)
    b = lambda sym, t: Bar(sym, t, 1, 1, 1, 1, 1, 1, 1, 1, 1)  # noqa: E731
    co.on_closed(b("BTCUSDT", T0), True)
    assert emitted == []
    co.on_closed(b("ETHUSDT", T0), True)
    assert emitted == [(T0, frozenset({"BTCUSDT", "ETHUSDT"}))]
    # další bar: ETH nedorazí → po timeoutu jen BTC
    clock["now"] = T0 + 2 * BAR_MS + 50
    co.on_closed(b("BTCUSDT", T0 + BAR_MS), True)
    await asyncio.sleep(0.15)
    assert emitted[-1] == (T0 + BAR_MS, frozenset({"BTCUSDT"}))
    # pozdě dorazivší ETH už nic nevydá, stejně jako historie z backfillu
    co.on_closed(b("ETHUSDT", T0 + BAR_MS), True)
    co.on_closed(b("ETHUSDT", T0 - 10 * BAR_MS), False)
    assert len(emitted) == 2


async def test_coordinator_ignores_late_backfill_bars():
    emitted = []
    clock = {"now": T0 + BAR_MS + 10_000}          # 10 s po hranici
    co = BarCloseCoordinator(("BTCUSDT",), lambda t, s, d: emitted.append(t), lambda: clock["now"], wait_s=3)
    co.on_closed(Bar("BTCUSDT", T0, 1, 1, 1, 1, 1, 1, 1, 1, 1), False)
    assert emitted == []
    clock["now"] = T0 + 2 * BAR_MS + 1_000         # backfill 1 s po hranici → v okně
    co.on_closed(Bar("BTCUSDT", T0 + BAR_MS, 1, 1, 1, 1, 1, 1, 1, 1, 1), False)
    assert emitted == [T0 + BAR_MS]
