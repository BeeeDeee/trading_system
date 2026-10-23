"""Lokální plná kniha: REST snapshot + diff stream `depth@100ms` (SPEC §4.3, nástraha 15).

Postup Binance (docs/VERIFIED.md): buffer eventů, snapshot `limit=1000`, zahodit `u < lastUpdateId`,
první event `U ≤ lastUpdateId ≤ u`, dál `pu == předchozí u`, jinak resync. Množství je absolutní, 0 = smazat.

Kniha slouží jen pro featury a simulaci fillů, ne pro ceny příkazů, proto je `float`."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from arena.util import wall_ms

log = logging.getLogger(__name__)

MAX_BUFFER = 5000   # eventů; při 100 ms je to ~8 min


@dataclass(frozen=True)
class Band:
    pct: float
    bid_notional: float
    ask_notional: float
    mid: float

    @property
    def obi(self) -> float | None:
        tot = self.bid_notional + self.ask_notional
        return (self.bid_notional - self.ask_notional) / tot if tot > 0 else None


@dataclass(frozen=True)
class MarketFill:
    vwap: float
    filled_qty: float
    levels: int
    complete: bool   # False = kniha neměla dost hloubky, VWAP je jen z toho, co bylo


@dataclass
class LocalBook:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    synced: bool = False
    last_u: int | None = None
    last_event_ms: int | None = None      # E posledního aplikovaného eventu (čas burzy)
    last_apply_local_ms: int | None = None
    snapshot_lo: float | None = None      # cenový rozsah posledního snapshotu (min bid)
    snapshot_hi: float | None = None      # (max ask)
    resyncs: int = 0
    buffer: list[dict[str, Any]] = field(default_factory=list)
    # snapshot, který čeká na event s u ≥ lastUpdateId: (lid, bids, asks, keep_outside)
    pending: tuple[int, dict[float, float], dict[float, float], LocalBook | None] | None = field(default=None, repr=False)

    # --- synchronizace ------------------------------------------------------

    def on_diff(self, ev: dict[str, Any], recv_ms: int | None = None) -> Literal["applied", "buffered", "gap"]:
        """Zpracuj diff event. Vrací `gap`, když se kniha rozjela a je potřeba nový snapshot."""
        if not self.synced:
            self.buffer.append(ev)
            if len(self.buffer) > MAX_BUFFER:
                del self.buffer[: len(self.buffer) - MAX_BUFFER]
            return "buffered"
        if ev["pu"] != self.last_u:
            log.warning("%s: mezera v diff streamu (pu=%s, čekal %s) → resync", self.symbol, ev["pu"], self.last_u)
            self.desync()
            self.buffer.append(ev)
            return "gap"
        self._apply(ev, recv_ms)
        if not self._sane():
            log.warning("%s: překřížená kniha → resync", self.symbol)
            self.desync()
            return "gap"
        return "applied"

    def desync(self) -> None:
        self.synced = False
        self.resyncs += 1
        self.buffer.clear()
        self.pending = None

    def apply_snapshot(self, snap: dict[str, Any], keep_outside: LocalBook | None = None,
                       now_ms: int | None = None) -> Literal["synced", "need_events", "stale_snapshot"]:
        """Nastaví knihu ze snapshotu a přehraje buffer.

        `keep_outside`: preventivní resnapshot synchronizované knihy. Úrovně mimo cenový rozsah snapshotu se
        převezmou z ní (diff stream je udržuje, snapshot je nemá). Po mezeře se nic nepřebírá.
        `need_events`: buffer ještě nemá event pokrývající snapshot, čeká se na další eventy.
        `stale_snapshot`: první event v bufferu je až za snapshotem, je potřeba nový snapshot."""
        lid = int(snap["lastUpdateId"])
        bids = {float(p): float(q) for p, q in snap["bids"] if float(q) > 0}
        asks = {float(p): float(q) for p, q in snap["asks"] if float(q) > 0}
        events = [e for e in self.buffer if e["u"] >= lid]
        if not events:
            self.buffer = []
            self.pending = (lid, bids, asks, keep_outside)
            return "need_events"
        self.pending = None
        if events[0]["U"] > lid:
            self.buffer = events
            return "stale_snapshot"
        lo = min(bids) if bids else None
        hi = max(asks) if asks else None
        if keep_outside is not None and lo is not None and hi is not None:
            for p, q in keep_outside.bids.items():
                if p < lo:
                    bids.setdefault(p, q)
            for p, q in keep_outside.asks.items():
                if p > hi:
                    asks.setdefault(p, q)
        self.bids, self.asks = bids, asks
        self.snapshot_lo, self.snapshot_hi = lo, hi
        self.last_u = None
        prev_u: int | None = None
        for e in events:
            if prev_u is not None and e["pu"] != prev_u:
                self.buffer = []
                self.synced = False
                return "stale_snapshot"
            self._apply(e, now_ms)
            prev_u = e["u"]
        self.buffer = []
        self.synced = True
        return "synced"

    def try_pending(self, now_ms: int | None = None) -> Literal["synced", "need_events", "stale_snapshot", "none"]:
        """Když snapshot čekal na eventy, zkus ho dokončit s tím, co mezitím přišlo do bufferu."""
        if self.pending is None:
            return "none"
        lid, bids, asks, keep = self.pending
        snap = {"lastUpdateId": lid, "bids": list(bids.items()), "asks": list(asks.items())}
        return self.apply_snapshot(snap, keep, now_ms)

    def _apply(self, ev: dict[str, Any], recv_ms: int | None) -> None:
        for p, q in ev["b"]:
            fp, fq = float(p), float(q)
            if fq == 0.0:
                self.bids.pop(fp, None)
            else:
                self.bids[fp] = fq
        for p, q in ev["a"]:
            fp, fq = float(p), float(q)
            if fq == 0.0:
                self.asks.pop(fp, None)
            else:
                self.asks[fp] = fq
        self.last_u = ev["u"]
        self.last_event_ms = ev.get("E")
        self.last_apply_local_ms = recv_ms if recv_ms is not None else wall_ms()

    def _sane(self) -> bool:
        if not self.bids or not self.asks:
            return True
        return max(self.bids) < min(self.asks)

    # --- čtení --------------------------------------------------------------

    def best(self) -> tuple[float, float] | None:
        if not self.bids or not self.asks:
            return None
        return max(self.bids), min(self.asks)

    def is_fresh(self, now_local_ms: int, max_age_ms: float) -> bool:
        return (self.synced and self.last_apply_local_ms is not None
                and now_local_ms - self.last_apply_local_ms <= max_age_ms)

    def bands(self, pcts: Sequence[float]) -> list[Band] | None:
        """Kumulativní notional bid (cena ≥ mid·(1−p/100)) a ask (cena ≤ mid·(1+p/100)) pro každé p."""
        b = self.best()
        if b is None:
            return None
        mid = (b[0] + b[1]) / 2
        lo = [mid * (1 - p / 100) for p in pcts]
        hi = [mid * (1 + p / 100) for p in pcts]
        bid_n = [0.0] * len(pcts)
        ask_n = [0.0] * len(pcts)
        lo_min, hi_max = min(lo), max(hi)
        for p, q in self.bids.items():
            if p >= lo_min:
                n = p * q
                for i, th in enumerate(lo):
                    if p >= th:
                        bid_n[i] += n
        for p, q in self.asks.items():
            if p <= hi_max:
                n = p * q
                for i, th in enumerate(hi):
                    if p <= th:
                        ask_n[i] += n
        return [Band(pcts[i], bid_n[i], ask_n[i], mid) for i in range(len(pcts))]

    def simulate_market(self, side: Literal["BUY", "SELL"], qty: float) -> MarketFill | None:
        """Projde knihu (BUY bere asky vzestupně, SELL bidy sestupně) a vrátí VWAP."""
        if qty <= 0:
            raise ValueError("qty musí být > 0")
        levels = sorted(self.asks.items()) if side == "BUY" else sorted(self.bids.items(), reverse=True)
        if not levels:
            return None
        left, cost, used = qty, 0.0, 0
        for p, q in levels:
            take = min(left, q)
            cost += take * p
            left -= take
            used += 1
            if left <= 1e-15:
                break
        filled = qty - max(left, 0.0)
        return MarketFill(cost / filled, filled, used, left <= 1e-15)


SnapshotFetcher = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass
class BookSync:
    """Řídí synchronizaci jedné knihy: snapshot při startu a mezeře, preventivní resnapshot (SPEC §4.3)."""

    book: LocalBook
    fetch_snapshot: SnapshotFetcher
    resnapshot_every_ms: int = 30 * 60 * 1000
    on_event: Callable[[str, str, dict[str, Any]], None] | None = None   # (level, msg, data) → events

    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _shadow: LocalBook | None = field(default=None, init=False)
    last_snapshot_ms: int | None = field(default=None, init=False)

    def on_diff(self, ev: dict[str, Any]) -> None:
        now = wall_ms()
        res = self.book.on_diff(ev, now)
        if res == "buffered" and self.book.pending is not None:
            done = self.book.try_pending(now)
            if done == "synced":
                self._emit("INFO", "book synchronizována", {"symbol": self.book.symbol})
            elif done == "stale_snapshot":
                self.request_snapshot()
        if self._shadow is not None:
            self._shadow.on_diff(ev, now)
            if self._shadow.pending is not None:
                done = self._shadow.try_pending(now)
                if done == "synced":
                    self._swap_shadow()
                elif done == "stale_snapshot":
                    self._shadow = None      # preventivní resnapshot se zkusí příště
        if res == "gap":
            self._emit("WARN", "book resync (mezera v pu)", {"symbol": self.book.symbol, "u": ev["u"]})
            self._shadow = None
            self.request_snapshot()
        elif not self.book.synced and self._task is None and self.book.pending is None:
            self.request_snapshot()

    def maybe_resnapshot(self, now_ms: int) -> None:
        if (self.book.synced and self._task is None and self._shadow is None and self.last_snapshot_ms is not None
                and now_ms - self.last_snapshot_ms >= self.resnapshot_every_ms):
            self._shadow = LocalBook(self.book.symbol)
            self._task = asyncio.ensure_future(self._snapshot(shadow=True))

    def request_snapshot(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._snapshot(shadow=False))

    async def _snapshot(self, shadow: bool) -> None:
        try:
            # krátké čekání, ať je v bufferu event, který snapshot pokryje
            await asyncio.sleep(0.2)
            for attempt in range(5):
                snap = await self.fetch_snapshot(self.book.symbol)
                self.last_snapshot_ms = wall_ms()
                target = self._shadow if shadow else self.book
                if target is None:
                    return
                res = target.apply_snapshot(snap, keep_outside=self.book if shadow else None)
                if res == "synced":
                    if shadow:
                        self._swap_shadow()
                    else:
                        self._emit("INFO", "book synchronizována", {"symbol": self.book.symbol,
                                                                     "levels": len(target.bids) + len(target.asks)})
                    return
                if res == "need_events":
                    return   # dokončí se v on_diff přes try_pending
                await asyncio.sleep(min(2 ** attempt, 10))
            self._emit("ERROR", "book: snapshot se nepodařilo napojit", {"symbol": self.book.symbol})
        except Exception as e:  # noqa: BLE001
            self._emit("ERROR", "book: snapshot selhal", {"symbol": self.book.symbol, "err": repr(e)})
            if shadow:
                self._shadow = None
        finally:
            self._task = None

    def _swap_shadow(self) -> None:
        assert self._shadow is not None
        # shadow dostal snapshot s převzetím úrovní mimo rozsah; od té doby dostával stejné diffy jako book
        self._shadow.resyncs = self.book.resyncs
        self.book.bids, self.book.asks = self._shadow.bids, self._shadow.asks
        self.book.last_u, self.book.last_event_ms = self._shadow.last_u, self._shadow.last_event_ms
        self.book.last_apply_local_ms = self._shadow.last_apply_local_ms
        self.book.snapshot_lo, self.book.snapshot_hi = self._shadow.snapshot_lo, self._shadow.snapshot_hi
        self._shadow = None

    def _emit(self, level: str, msg: str, data: dict[str, Any]) -> None:
        log.log(logging.WARNING if level in ("WARN", "ERROR") else logging.INFO, "%s %s", msg, data)
        if self.on_event is not None:
            self.on_event(level, msg, data)
