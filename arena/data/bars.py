"""BarStore: 5m bary z kline streamu, backfill přes REST, detekce mezer, koordinace uzavření (SPEC §4.2)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from arena.db import Bar, Database

log = logging.getLogger(__name__)

BAR_MS = 300_000
KLINES_LIMIT = 1500


def bar_floor(ms: int) -> int:
    return ms - ms % BAR_MS


KlinesFetcher = Callable[[str, int, int], Awaitable[list[list[Any]]]]   # (symbol, start, end) → REST klines


@dataclass
class BarStore:
    """Uloží bar jen při `k.x == true` (nástraha 3). Před uložením nového baru doplní případnou mezeru.

    Symbol, u kterého backfill selže, je `stale` a nesmí generovat vstupy, dokud další backfill neprojde."""

    db: Database
    symbols: tuple[str, ...]
    fetch_klines: KlinesFetcher
    now_ms: Callable[[], int]
    warmup_bars: int = 864
    on_closed: Callable[[Bar, bool], None] | None = None      # (bar, live) → koordinátor
    on_event: Callable[[str, str, dict[str, Any]], None] | None = None

    last_open: dict[str, int] = field(default_factory=dict, init=False)
    stale: set[str] = field(default_factory=set, init=False)
    gap_from: dict[str, int] = field(default_factory=dict, init=False)   # nedoplněná díra od open_time
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for s in self.symbols:
            lt = self.db.last_bar_time(s)
            if lt is not None:
                self.last_open[s] = lt
            self._locks[s] = asyncio.Lock()

    async def backfill_all(self, from_warmup: bool = False) -> None:
        await asyncio.gather(*(self.backfill(s, from_warmup) for s in self.symbols))

    async def backfill(self, symbol: str, from_warmup: bool = False) -> int:
        """Stáhne uzavřené bary od posledního uloženého (nebo od nedoplněné díry) po poslední uzavřený.

        `from_warmup` (start procesu): stáhne celé warmup okno znovu, upsert je idempotentní a doplní i díry.
        Vrací počet stažených barů."""
        async with self._locks[symbol]:
            now = self.now_ms()
            last_closed_open = bar_floor(now) - BAR_MS
            warmup_start = last_closed_open - (self.warmup_bars - 1) * BAR_MS
            start = self.last_open.get(symbol)
            start = start + BAR_MS if (start is not None and not from_warmup) else warmup_start
            if symbol in self.gap_from:
                start = min(start, self.gap_from[symbol])
            if start > last_closed_open:
                self.stale.discard(symbol)
                return 0
            got: list[Bar] = []
            try:
                cur = start
                while cur <= last_closed_open:
                    rows = await self.fetch_klines(symbol, cur, last_closed_open)
                    bars = [Bar.from_rest(symbol, k) for k in rows if int(k[0]) <= last_closed_open]
                    if not bars:
                        break
                    got.extend(bars)
                    cur = bars[-1].open_time + BAR_MS
                    if len(rows) < KLINES_LIMIT:
                        break
            except Exception as e:  # noqa: BLE001
                self.stale.add(symbol)
                self.gap_from.setdefault(symbol, start)
                self._emit("ERROR", "backfill selhal, symbol stale", {"symbol": symbol, "err": repr(e)})
                return 0
            self.gap_from.pop(symbol, None)
            if got:
                self.db.upsert_bars(got)
                self.last_open[symbol] = max(self.last_open.get(symbol, 0), got[-1].open_time)
            missing = self._missing(symbol, start, last_closed_open, got)
            if missing:
                # burza bar nemá (výpadek burzy); zalogovat, ale stale jen když chybí poslední bar
                self._emit("WARN", "backfill: chybějící bary na burze", {"symbol": symbol, "n": len(missing),
                                                                        "first": missing[0]})
            if self.last_open.get(symbol) == last_closed_open:
                self.stale.discard(symbol)
            else:
                self.stale.add(symbol)
            if got:
                log.info("backfill %s: %d barů (%d → %d)", symbol, len(got), got[0].open_time, got[-1].open_time)
                if self.on_closed is not None:
                    for b in got:
                        self.on_closed(b, False)
            return len(got)

    @staticmethod
    def _missing(symbol: str, start: int, end: int, got: Iterable[Bar]) -> list[int]:
        have = {b.open_time for b in got}
        return [t for t in range(start, end + 1, BAR_MS) if t not in have]

    async def on_kline(self, k: dict[str, Any]) -> None:
        """Kline event (`data.k`). Rozpracované bary (`x == false`) se ignorují."""
        if not k.get("x"):
            return
        bar = Bar.from_ws(k)
        sym = bar.symbol
        prev = self.last_open.get(sym)
        if prev is not None and bar.open_time <= prev:
            return   # už máme (např. z backfillu po reconnectu)
        if (prev is not None and bar.open_time != prev + BAR_MS) or sym in self.gap_from:
            if sym not in self.gap_from:
                self._emit("WARN", "mezera v barech → backfill", {"symbol": sym, "prev": prev, "new": bar.open_time})
            await self.backfill(sym)
            if self.last_open.get(sym, 0) >= bar.open_time:
                return
        self.db.upsert_bars([bar])
        self.last_open[sym] = bar.open_time
        if sym not in self.gap_from:
            self.stale.discard(sym)
        if self.on_closed is not None:
            self.on_closed(bar, True)

    def _emit(self, level: str, msg: str, data: dict[str, Any]) -> None:
        log.log(logging.WARNING if level in ("WARN", "ERROR") else logging.INFO, "%s %s", msg, data)
        if self.on_event is not None:
            self.on_event(level, msg, data)


@dataclass
class BarCloseCoordinator:
    """Vydá „bar uzavřen“ pro čas T, až jsou uzavřené bary všech symbolů, nebo po `wait_s` od hranice.

    Rodina C potřebuje BTC i alt ze stejného baru. Symbol, který do `wait_s` nedorazil, se vyhodnotí zvlášť, až jeho
    bar přijde (s vlastním decided_at), nejpozději `late_max_s` po hranici; později se přeskočí a zaloguje.
    Bary starší než poslední vydaný čas (backfill historie) se nevydávají."""

    symbols: tuple[str, ...]
    on_bar_close: Callable[[int, frozenset[str], int], None]   # (open_time, symboly, decided_at_ms)
    now_ms: Callable[[], int]
    wait_s: float = 3.0
    on_event: Callable[[str, str, dict[str, Any]], None] | None = None
    late_max_s: float = 20.0      # = hranice stale dat (SPEC §10): později by vstupy stejně blokoval risk

    last_emitted: int | None = field(default=None, init=False)
    _pending: dict[int, set[str]] = field(default_factory=dict, init=False)
    _timers: dict[int, asyncio.TimerHandle] = field(default_factory=dict, init=False)
    _late: dict[int, set[str]] = field(default_factory=dict, init=False)   # vydaný bar → symboly, které chyběly

    def on_closed(self, bar: Bar, live: bool) -> None:
        t = bar.open_time
        if self.last_emitted is not None and t <= self.last_emitted:
            late = self._late.get(t)
            if late is not None and bar.symbol in late:
                late.discard(bar.symbol)
                lag_ms = self.now_ms() - (t + BAR_MS)
                if lag_ms <= self.late_max_s * 1000:
                    if self.on_event is not None:
                        self.on_event("INFO", "opožděný bar vyhodnocen zvlášť",
                                      {"open_time": t, "symbol": bar.symbol, "lag_ms": lag_ms})
                    self.on_bar_close(t, frozenset({bar.symbol}), self.now_ms())
                elif self.on_event is not None:
                    self.on_event("WARN", "opožděný bar po limitu, přeskočen",
                                  {"open_time": t, "symbol": bar.symbol, "lag_ms": lag_ms})
            return
        # bar z backfillu (reconnect, start) spustí rozhodování jen v okně čekání po hranici;
        # starší bar by znamenal vstup se zpožděním, které live nikdy nemá
        if not live and self.now_ms() > t + BAR_MS + self.wait_s * 1000:
            return
        got = self._pending.setdefault(t, set())
        got.add(bar.symbol)
        if got >= set(self.symbols):
            self._emit_bar(t)
        elif t not in self._timers:
            boundary = t + BAR_MS
            delay = max(0.0, (boundary - self.now_ms()) / 1000 + self.wait_s)
            self._timers[t] = asyncio.get_running_loop().call_later(delay, self._emit_bar, t)

    def _emit_bar(self, t: int) -> None:
        timer = self._timers.pop(t, None)
        if timer is not None:
            timer.cancel()
        got = self._pending.pop(t, set())
        if self.last_emitted is not None and t <= self.last_emitted:
            return
        missing = set(self.symbols) - got
        if missing and self.on_event is not None:
            self.on_event("WARN", "bar bez všech symbolů, chybějící se vyhodnotí po příchodu",
                          {"open_time": t, "missing": sorted(missing)})
        self.last_emitted = t
        self._late = {t: missing} if missing else {}
        for old in [k for k in self._pending if k < t]:
            self._pending.pop(old, None)
            h = self._timers.pop(old, None)
            if h is not None:
                h.cancel()
        self.on_bar_close(t, frozenset(got), self.now_ms())
