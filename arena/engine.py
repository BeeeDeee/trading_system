"""Orchestrace jednoho asyncio procesu `arena run` (D8). Ve fázi P1 jen datová vrstva:
WS /market + /public, BarStore s backfillem, lokální knihy, recorder, 30s pásma knihy, bar_book."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.binance.rest import RestClient
from arena.binance.ws import WsConnection, combined_url
from arena.config import Config
from arena.data.bars import BAR_MS, BarCloseCoordinator, BarStore
from arena.data.book import BookSync, LocalBook
from arena.data.recorder import AggTradeRecorder, BookSampler, aggregate_bar_book
from arena.data.ticks import TickDispatcher
from arena.db import Database
from arena.util import wall_ms

log = logging.getLogger(__name__)


@dataclass
class Engine:
    cfg: Config
    db: Database
    rest: RestClient
    warmup_bars: int = 864

    stop: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    ticks: TickDispatcher = field(default_factory=TickDispatcher, init=False)

    def __post_init__(self) -> None:
        cfg = self.cfg
        syms = cfg.universe
        self.books: dict[str, BookSync] = {
            s: BookSync(LocalBook(s), self._fetch_depth, cfg.book.resnapshot_every_min * 60_000,
                        on_event=self._event_cb("book"))
            for s in syms
        }
        self.coordinator = BarCloseCoordinator(syms, self.on_bar_close, self.rest.clock.now_ms,
                                               wait_s=cfg.engine.bar_close_wait_s, on_event=self._event_cb("bars"))
        self.bar_store = BarStore(self.db, syms, self._fetch_klines, self.rest.clock.now_ms, self.warmup_bars,
                                  on_closed=self.coordinator.on_closed, on_event=self._event_cb("bars"))
        self.recorder = (AggTradeRecorder(Path(cfg.recorder.raw_dir), on_event=self._event_cb("recorder"))
                         if cfg.recorder.raw_aggtrades else None)
        self.sampler = BookSampler(self.db, {s: b.book for s, b in self.books.items()}, cfg.book.bands_pct,
                                   self.rest.clock.now_ms, wall_ms, cfg.book.sample_every_s, cfg.book.max_diff_age_s,
                                   on_tick=self._on_sample_tick)
        ep = cfg.data_endpoints
        low = [s.lower() for s in syms]
        self.ws_market = WsConnection(
            "market", combined_url(ep.ws_market, [f"{s}@{k}" for s in low for k in ("kline_5m", "aggTrade",
                                                                                     "markPrice@1s")]),
            self.on_market, self.on_market_connect)
        # depth a bookTicker na oddělených spojeních: bookTicker (~2000 zpráv/s) Binance často shazuje
        # (ověřeno 2026-10-23, CHANGELOG P1) a každé shození depth spojení = resync knihy a ztracené vzorky.
        self.ws_public = WsConnection(
            "public", combined_url(ep.ws_public, [f"{s}@depth@100ms" for s in low]),
            self.on_public, self.on_public_connect)
        self.ws_ticker = WsConnection(
            "ticker", combined_url(ep.ws_public, [f"{s}@bookTicker" for s in low]), self.on_public)
        self.bar_close_listeners: list[Any] = []   # P2+: strategie

    # --- REST adaptéry ------------------------------------------------------

    async def _fetch_klines(self, symbol: str, start: int, end: int) -> list[list[Any]]:
        return await self.rest.klines(symbol, "5m", start, end + BAR_MS - 1, 1500)

    async def _fetch_depth(self, symbol: str) -> dict[str, Any]:
        return await self.rest.depth(symbol, 1000)

    # --- události -----------------------------------------------------------

    def _event_cb(self, component: str) -> Any:
        def cb(level: str, msg: str, data: dict[str, Any]) -> None:
            self.db.event(level, component, msg, data)
        return cb

    def on_market(self, stream: str, data: dict[str, Any]) -> None:
        if stream.endswith("@aggTrade"):
            if self.recorder is not None:
                self.recorder.on_agg_trade(data)
            self.ticks.on_agg_trade(data)
        elif "@kline_" in stream:
            if data["k"].get("x"):
                asyncio.ensure_future(self.bar_store.on_kline(data["k"]))
        elif "@markPrice" in stream:
            self.ticks.on_mark_price(data)

    def on_public(self, stream: str, data: dict[str, Any]) -> None:
        if "@depth" in stream:
            self.books[data["s"]].on_diff(data)
        elif stream.endswith("@bookTicker"):
            self.ticks.on_book_ticker(data)

    async def on_market_connect(self, reconnected: bool) -> None:
        if reconnected:
            self.db.event("WARN", "ws", "market reconnect → backfill barů")
            await self.bar_store.backfill_all()

    async def on_public_connect(self, reconnected: bool) -> None:
        if reconnected:
            self.db.event("WARN", "ws", "public reconnect → resync knih")
            for b in self.books.values():
                b.book.desync()

    def _on_sample_tick(self, ts: int) -> None:
        for b in self.books.values():
            b.maybe_resnapshot(wall_ms())

    def on_bar_close(self, open_time: int, symbols: frozenset[str], decided_at: int) -> None:
        for sym in self.cfg.universe:
            rows = self.db.book_bands(sym, open_time, open_time + BAR_MS)
            self.db.upsert_bar_book(aggregate_bar_book(sym, open_time, rows, self.cfg.book.bands_pct))
        log.info("bar %d uzavřen (%s), decided_at +%d ms", open_time, ",".join(sorted(symbols)),
                 decided_at - open_time - BAR_MS)
        for listener in self.bar_close_listeners:
            listener(open_time, symbols, decided_at)

    # --- běh ----------------------------------------------------------------

    async def _time_loop(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=600)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.rest.sync_time()
            except Exception as e:  # noqa: BLE001
                log.warning("resync času selhal: %r", e)

    async def _status_loop(self, every_s: float = 300) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=every_s)
                return
            except asyncio.TimeoutError:
                pass
            log.info("status: %s", self.status())

    def status(self) -> dict[str, Any]:
        now = wall_ms()
        return {
            "ws": {w.name: {"connected": w.connected, "connects": w.connects,
                            "last_msg_age_ms": (now - w.last_msg_ms) if w.last_msg_ms else None}
                   for w in (self.ws_market, self.ws_public, self.ws_ticker)},
            "books": {s: {"synced": b.book.synced, "resyncs": b.book.resyncs,
                          "levels": len(b.book.bids) + len(b.book.asks)} for s, b in self.books.items()},
            "bars_last": dict(self.bar_store.last_open), "stale": sorted(self.bar_store.stale),
            "samples": {"ok": self.sampler.samples_ok, "bad": self.sampler.samples_bad},
            "aggtrades_rows": self.recorder.rows_written if self.recorder else None,
            "weight_1m": self.rest.used_weight_1m,
        }

    async def run(self) -> None:
        await self.rest.sync_time()
        self.db.record_config(self.cfg.config_hash, self.cfg.source_text)
        self.db.event("INFO", "engine", "start", {"env": self.cfg.env, "config_hash": self.cfg.config_hash})
        await self.bar_store.backfill_all(from_warmup=True)
        tasks = [asyncio.create_task(c, name=n) for n, c in (
            ("ws_market", self.ws_market.run(self.stop)), ("ws_public", self.ws_public.run(self.stop)),
            ("ws_ticker", self.ws_ticker.run(self.stop)),
            ("sampler", self.sampler.run(self.stop)), ("time", self._time_loop()), ("status", self._status_loop()),
        )]
        if self.recorder is not None:
            tasks.append(asyncio.create_task(self.recorder.run(self.stop), name="recorder"))
        try:
            await self.stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.recorder is not None:
                self.recorder.close()
            self.db.event("INFO", "engine", "stop")
