"""Orchestrace jednoho asyncio procesu `arena run` (D8): datová vrstva + paper sloty.

WS /market (kline, aggTrade, markPrice), /public (depth) a /public (bookTicker zvlášť), BarStore s backfillem,
lokální knihy, recorder, 30s pásma knihy, bar_book, featury → strategie → paper executor, funding."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.binance.filters import SymbolFilters, parse_exchange_info
from arena.binance.rest import RestClient
from arena.binance.ws import WsConnection, combined_url
from arena.config import Config, load_signer
from arena.data.bars import BAR_MS, BarCloseCoordinator, BarStore
from arena.data.book import BookSync, LocalBook
from arena.data.recorder import AggTradeRecorder, BookSampler, aggregate_bar_book
from arena.data.ticks import TickDispatcher
from arena.alerts import Alerter
from arena.db import Database
from arena.execution.paper import BookFillModel, PaperExecutor
from arena.live_runtime import LiveRuntime
from arena.risk import RiskGuard
from arena.slots import FeatureSource, FundingTracker, SlotRunner
from arena.snapshot import SnapshotWriter
from arena.strategies.registry import build_slots, build_strategies, global_warmup
from arena.util import wall_ms

log = logging.getLogger(__name__)

AGG_WINDOW_MS = 3_600_000   # REST aggTrades: okno startTime–endTime max. 1 h


async def fetch_agg_trades(rest: RestClient, symbol: str, start: int, end: int) -> list[dict[str, Any]]:
    """Chybějící aggTrades přes REST (SPEC §7): časové okno ≤ 1 h najde první obchod, dál stránkování fromId."""
    out: list[dict[str, Any]] = []
    win = start
    batch: list[dict[str, Any]] = []
    while win <= end and not batch:      # první okno, ve kterém je aspoň jeden obchod
        batch = await rest.agg_trades(symbol, start_ms=win, end_ms=min(win + AGG_WINDOW_MS - 1, end))
        win += AGG_WINDOW_MS
    while batch:   # fromId nemá časové okno: prázdná dávka = žádné další obchody
        out.extend(b for b in batch if int(b["T"]) <= end)
        if int(batch[-1]["T"]) >= end:
            break
        batch = await rest.agg_trades(symbol, from_id=int(batch[-1]["a"]) + 1)
    for t in out:
        t.setdefault("s", symbol)
    return out


@dataclass
class Engine:
    cfg: Config
    db: Database
    rest: RestClient
    live_flag: bool = False

    stop: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    ticks: TickDispatcher = field(default_factory=TickDispatcher, init=False)
    runner: SlotRunner | None = field(default=None, init=False)
    executor: PaperExecutor | None = field(default=None, init=False)
    live: LiveRuntime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        cfg = self.cfg
        syms = cfg.universe
        self.strategies = build_strategies(cfg)
        self.slots = build_slots(cfg)
        self.warmup_bars = global_warmup(self.strategies)
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
        self.funding = FundingTracker(self._on_funding)
        self.alerter = Alerter.from_env(cfg.alerts.ntfy_topic_env)
        self.risk = RiskGuard(cfg, self.db, self.live_flag, on_alert=self.alerter.send)
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
            self.funding.on_mark(data)

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

    def _on_funding(self, symbol: str, t: int, rate: float, mark: float) -> None:
        if self.executor is not None:
            self.executor.on_funding(symbol, t, rate, mark)

    def on_bar_close(self, open_time: int, symbols: frozenset[str], decided_at: int) -> None:
        for sym in self.cfg.universe:
            rows = self.db.book_bands(sym, open_time, open_time + BAR_MS)
            self.db.upsert_bar_book(aggregate_bar_book(sym, open_time, rows, self.cfg.book.bands_pct))
        lag = decided_at - open_time - BAR_MS
        log.info("bar %d uzavřen (%s), decided_at +%d ms", open_time, ",".join(sorted(symbols)), lag)
        if self.runner is not None:
            # stale data: bar > 20 s po hranici → žádné nové vstupy (SPEC §10)
            stale = frozenset(self.bar_store.stale) | (frozenset(symbols) if lag > 20_000 else frozenset())
            self.risk.stale_symbols = set(stale) | {s for s, b in self.books.items()
                                                    if not b.book.is_fresh(wall_ms(), 5_000)}
            if self.live is not None:
                closes = {bar.symbol: bar.c for s in symbols for bar in self.db.bars(s, open_time, open_time)}
                asyncio.ensure_future(self.live.on_bar_close(open_time, closes))
            try:
                self.runner.on_bar_close(open_time, symbols, decided_at, stale)
            except Exception:  # noqa: BLE001 – chyba jednoho baru nesmí shodit proces
                log.exception("chyba při zpracování baru %d", open_time)
                self.db.event("ERROR", "engine", "chyba při zpracování baru", {"bar": open_time})

    # --- start: filtry, poplatky, obnova paper pozic ---------------------------

    async def _load_filters(self) -> dict[str, SymbolFilters]:
        info = await self.rest.exchange_info()
        self.db.meta_set("exchange_info", json.dumps(
            {"symbols": [s for s in info["symbols"] if s["symbol"] in self.cfg.universe]}))
        return parse_exchange_info(info, self.cfg.universe)

    async def _load_fees(self) -> tuple[dict[str, float], dict[str, float]]:
        """Skutečné sazby účtu z `commissionRate` (paper = taker sazba účtu, SPEC §7); bez klíče VIP0."""
        taker: dict[str, float] = {}
        maker: dict[str, float] = {}
        signer = load_signer(self.cfg.env)
        if signer is None:
            self.db.event("WARN", "engine", "bez klíče: paper používá VIP0 poplatky (taker 0,05 %, maker 0,02 %)")
            return taker, maker
        try:
            async with RestClient(self.cfg.endpoints.rest, signer=signer) as acct:
                for s in self.cfg.universe:
                    cr = await acct.commission_rate(s)
                    taker[s], maker[s] = float(cr["takerCommissionRate"]), float(cr["makerCommissionRate"])
        except Exception as e:  # noqa: BLE001
            self.db.event("ERROR", "engine", "commissionRate selhal, VIP0 poplatky", {"err": repr(e)})
        return taker, maker

    async def _recover_outage(self) -> None:
        """Otevřené paper pozice po restartu: přehrát chybějící aggTrades, výstupy tainted (SPEC §7)."""
        assert self.executor is not None
        now = self.rest.clock.now_ms()
        by_sym: dict[str, int] = {}
        for p in self.executor.positions.values():
            since = (p.sim.last_ts + 1) if (p.sim and p.sim.last_ts) else (p.entry_time or p.decision_ts)
            by_sym[p.symbol] = min(by_sym.get(p.symbol, since), since)
        for sym, since in by_sym.items():
            try:
                trades = await fetch_agg_trades(self.rest, sym, since, now)
            except Exception as e:  # noqa: BLE001
                self.db.event("ERROR", "paper", "výpadek: aggTrades nejdou stáhnout", {"symbol": sym, "err": repr(e)})
                trades = []
            closes = {b.open_time: b.c for b in self.db.bars(sym, start=since - BAR_MS)}
            self.executor.replay_outage(sym, trades, now, closes)
            self.db.event("WARN", "paper", "výpadek: přehráno aggTrades", {"symbol": sym, "n": len(trades),
                                                                          "since": since})

    async def _advance_loop(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=0.05)
                return
            except asyncio.TimeoutError:
                pass
            if self.executor is not None:
                self.executor.advance(self.rest.clock.now_ms())

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

    def _last_price(self, symbol: str) -> float | None:
        t = self.ticks.last_trade.get(symbol)
        return float(t["p"]) if t else None

    def paper_positions(self) -> list[dict[str, Any]]:
        out = []
        for p in (self.executor.positions.values() if self.executor else []):
            last = self.ticks.last_trade.get(p.symbol)
            px = float(last["p"]) if last else None
            lvl = p.sim.trail_level if p.sim else None
            sg = 1 if p.side == "LONG" else -1
            out.append({"slot": p.slot_id, "side": p.side, "status": p.status, "entry_px": p.entry_px,
                        "entry_time": p.entry_time, "px": px, "trail": lvl, "hard_stop": p.plan.get("hard_stop_px"),
                        "trail_dist_bps": (sg * (px - lvl) / px * 1e4) if (px and lvl) else None,
                        "upnl_bps": (sg * (px - p.entry_px) / p.entry_px * 1e4) if (px and p.entry_px) else None})
        return out

    async def _snapshot_loop(self, snap: SnapshotWriter) -> None:
        """snapshot.json každých 15 s v hlavním vlákně (rychlé), statistiky každých 5 min ve vlákně
        s vlastním DB spojením, aby bootstrap neblokoval čtení WS."""
        heavy_db = Database(self.cfg.paths.db)
        heavy = SnapshotWriter(self.cfg, heavy_db, snap.site_dir, snap.status, snap.positions, snap.live_variants,
                               snap.calibrated)
        heavy_task: asyncio.Future[None] | None = None
        try:
            while not self.stop.is_set():
                if heavy_task is None and wall_ms() - heavy.last_heavy_ms >= 300_000:
                    heavy_task = asyncio.ensure_future(asyncio.to_thread(heavy.write_heavy))
                if heavy_task is not None and heavy_task.done():
                    if heavy_task.exception() is not None:
                        log.error("snapshot: statistiky selhaly: %r", heavy_task.exception())
                        heavy.last_heavy_ms = wall_ms()
                    snap.leaderboard_cache = heavy.leaderboard_cache
                    heavy_task = None
                try:
                    snap.write_snapshot()
                except Exception:  # noqa: BLE001
                    log.exception("snapshot selhal")
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=self.cfg.engine.snapshot_every_s)
                    return
                except asyncio.TimeoutError:
                    pass
        finally:
            if heavy_task is not None:
                await asyncio.gather(heavy_task, return_exceptions=True)
            heavy_db.close()

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
            "paper_open": len(self.executor.positions) if self.executor else None,
            "weight_1m": self.rest.used_weight_1m,
        }

    async def run(self) -> None:
        cfg = self.cfg
        await self.rest.sync_time()
        self.db.record_config(cfg.config_hash, cfg.source_text)
        self.db.event("INFO", "engine", "start", {"env": cfg.env, "data_env": cfg.data_env,
                                                  "config_hash": cfg.config_hash, "slots": len(self.slots)})
        filters = await self._load_filters()
        taker, maker = await self._load_fees()
        fill = BookFillModel({s: b.book for s, b in self.books.items()}, self.ticks, wall_ms,
                             cfg.paper.stale_book_penalty_bps, dict(cfg.paper.extra_slippage_bps))
        self.executor = PaperExecutor(filters, fill, cfg.paper.notional_usdt, cfg.paper.entry_latency_ms,
                                      cfg.paper.trigger_latency_ms, cfg.paper.outage_penalty_bps, taker, maker,
                                      cfg.paper.cost_model_version, on_event=self._event_cb("paper"))
        self.runner = SlotRunner(cfg, self.db, self.slots, self.executor, FeatureSource(self.db, self.warmup_bars))
        for s in cfg.universe:
            self.ticks.subscribe(s, lambda e, s=s: self.executor.on_trade(s, e))  # type: ignore[union-attr]
        live_tasks: list[asyncio.Task[Any]] = []
        if self.risk.live_allowed:
            signer = load_signer(cfg.env)
            if signer is None:
                raise RuntimeError(f"live zapnuto, ale chybí klíč pro {cfg.env}")
            trade_rest = RestClient(cfg.endpoints.rest, signer=signer)
            self.live = LiveRuntime(cfg, self.db, trade_rest, self.risk, self._last_price, self.alerter.send, self.stop)
            await self.live.start()
            self.runner.on_signal = self.live.on_signal
            for s in cfg.live.slots:
                self.ticks.subscribe(s, lambda e, s=s: self.live.on_trade_tick(s, e))  # type: ignore[union-attr]
            live_tasks = self.live.tasks()
            self.db.event("WARN", "engine", f"LIVE zapnuto ({cfg.env}), sloty {dict((k, v.variant) for k, v in cfg.live.slots.items())}")
        elif cfg.live.enabled:
            self.db.event("WARN", "engine", "live.enabled: true, ale bez --live → live executor se nevytvoří")
        await self.bar_store.backfill_all(from_warmup=True)
        if self.executor.positions:
            await self._recover_outage()
        tasks = [asyncio.create_task(c, name=n) for n, c in (
            ("ws_market", self.ws_market.run(self.stop)), ("ws_public", self.ws_public.run(self.stop)),
            ("ws_ticker", self.ws_ticker.run(self.stop)), ("advance", self._advance_loop()),
            ("sampler", self.sampler.run(self.stop)), ("time", self._time_loop()), ("status", self._status_loop()),
        )]
        if self.recorder is not None:
            tasks.append(asyncio.create_task(self.recorder.run(self.stop), name="recorder"))
        tasks.extend(live_tasks)
        snap = SnapshotWriter(cfg, self.db, Path(cfg.paths.site), self.status, self.paper_positions,
                              live_variants=lambda: [s.variant for s in cfg.live.slots.values()] if self.live
                              else [],
                              live_positions=lambda: self.live.positions_view() if self.live else [])
        tasks.append(asyncio.create_task(self._snapshot_loop(snap), name="snapshot"))
        try:
            await self.stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.recorder is not None:
                self.recorder.close()
            self.db.event("INFO", "engine", "stop")
