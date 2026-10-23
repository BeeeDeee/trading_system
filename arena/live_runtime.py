"""Live část procesu (P5): user data stream, startovní kontroly, rekonciliace, equity, income, kill switch.

Vytvoří se jen s `live.enabled: true` a flagem `--live` (SPEC §10)."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from arena.binance import errors
from arena.binance.filters import SymbolFilters, parse_exchange_info
from arena.binance.rest import RestClient
from arena.binance.ws import WsConnection
from arena.config import Config
from arena.db import Database
from arena.execution.base import TradeRecord
from arena.execution.live import LiveExecutor, LivePosition
from arena.reconcile import Reconciler
from arena.risk import RiskGuard
from arena.strategies.base import Intent
from arena.strategies.registry import Slot

log = logging.getLogger(__name__)

KEEPALIVE_S = 30 * 60
INCOME_EVERY_S = 10 * 60
EQUITY_EVERY_S = 60
LEDGER_TOL_USDT = 0.01


class LiveStartError(RuntimeError):
    pass


@dataclass
class LiveRuntime:
    cfg: Config
    db: Database
    rest: RestClient                 # podepsaný klient na cfg.env
    risk: RiskGuard
    last_price: Any
    alert: Any
    stop: asyncio.Event

    executor: LiveExecutor | None = field(default=None, init=False)
    reconciler: Reconciler | None = field(default=None, init=False)
    ws: WsConnection | None = field(default=None, init=False)
    listen_key: str | None = field(default=None, init=False)
    cooldown_until: dict[str, int] = field(default_factory=dict, init=False)
    seq: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.cfg.live.slots)

    # --- start ---------------------------------------------------------------

    async def start(self) -> None:
        await self.rest.sync_time()
        filters = parse_exchange_info(await self.rest.exchange_info(), self.symbols)
        await self._check_account(filters)
        self.executor = LiveExecutor(self.rest, filters, self.db, self.rest.clock.now_ms, self.last_price,
                                     self.alert, self._on_closed, self.cfg.paper.cost_model_version)
        self.executor.restore()
        for r in self.db.execute("SELECT slot_id, trade_id, exit_time FROM trades WHERE mode='live'").fetchall():
            n = int(str(r["trade_id"]).rsplit("-", 1)[-1])
            self.seq[r["slot_id"]] = max(self.seq.get(r["slot_id"], 0), n)
        for p in self.executor.positions.values():
            self.seq[p.slot_id] = max(self.seq.get(p.slot_id, 0), p.seq)
        self.reconciler = Reconciler(self.rest, self.executor, self.risk, self.symbols, self.alert)
        diffs = await self.reconciler.run(fix=True)
        for d in diffs:
            log.warning("rekonciliace při startu: %s", d)
        self.ws = WsConnection("private", self._private_url, self._on_private, self._on_private_connect,
                               idle_timeout_s=300)

    async def _check_account(self, filters: Mapping[str, SymbolFilters]) -> None:
        """One-way, ISOLATED a páka na každém live symbolu; nastavit jen bez otevřené pozice (SPEC §9)."""
        if await self.rest.position_side_dual():
            raise LiveStartError("účet je v hedge módu (dualSidePosition=true); reduceOnly vyžaduje one-way")
        cfgs = {c["symbol"]: c for c in await self.rest.symbol_config()}
        open_pos = {p["symbol"] for p in await self.rest.position_risk() if Decimal(str(p["positionAmt"])) != 0}
        for sym in self.symbols:
            if sym not in filters:
                raise LiveStartError(f"{sym} chybí v exchangeInfo")
            c = cfgs.get(sym, {})
            if c.get("marginType") == "ISOLATED" and int(c.get("leverage", 0)) == self.cfg.live.leverage:
                continue
            if sym in open_pos:
                raise LiveStartError(f"{sym}: margin/páka neodpovídá a je otevřená pozice; nestartuji live")
            if c.get("marginType") != "ISOLATED":
                try:
                    await self.rest.set_margin_type(sym, "ISOLATED")
                except errors.BinanceError as e:
                    if e.code != -4046:   # „No need to change margin type“
                        raise
            await self.rest.set_leverage(sym, self.cfg.live.leverage)
            self.db.event("INFO", "live", f"{sym}: nastaveno ISOLATED {self.cfg.live.leverage}x")

    # --- user data stream ---------------------------------------------------

    async def _private_url(self) -> str:
        self.listen_key = await self.rest.listen_key()
        return f"{self.cfg.endpoints.ws_private.rstrip('/')}/ws/{self.listen_key}"

    async def _on_private_connect(self, reconnected: bool) -> None:
        self.risk.private_down_since = None
        if reconnected and self.reconciler is not None:
            await self.reconciler.run(fix=True)

    def _on_private(self, stream: str, data: dict[str, Any]) -> None:
        et = data.get("e")
        if et == "listenKeyExpired":
            self.db.event("WARN", "live", "listenKeyExpired → nový klíč + rekonciliace")
            assert self.ws is not None
            self.ws.force_reconnect()
            return
        if self.executor is not None and et in ("ORDER_TRADE_UPDATE", "ALGO_UPDATE", "ACCOUNT_UPDATE"):
            asyncio.ensure_future(self.executor.on_user_event(data))

    async def _keepalive_loop(self) -> None:
        while not await _sleep(self.stop, KEEPALIVE_S):
            try:
                await self.rest.keepalive_listen_key()
            except errors.ListenKeyMissing:
                assert self.ws is not None
                self.ws.force_reconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("keepalive listenKey selhal: %r", e)

    # --- smyčky -------------------------------------------------------------

    async def _reconcile_loop(self) -> None:
        while not await _sleep(self.stop, self.cfg.engine.reconcile_every_s):
            try:
                assert self.reconciler is not None
                await self.reconciler.run(fix=True)
            except Exception as e:  # noqa: BLE001
                log.warning("rekonciliace selhala: %r", e)

    async def _equity_loop(self) -> None:
        while True:
            try:
                acct = await self.rest.account()
                wallet = float(acct.get("totalWalletBalance", 0))
                upnl = float(acct.get("totalUnrealizedProfit", 0))
                now = self.rest.clock.now_ms()
                self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)",
                                (now, wallet, upnl, float(acct.get("totalInitialMargin", 0))))
                if self.risk.on_equity(wallet + upnl, now) and self.executor is not None:
                    await self.executor.close_all("live_halt")
            except Exception as e:  # noqa: BLE001
                log.warning("equity selhalo: %r", e)
            if await _sleep(self.stop, EQUITY_EVERY_S):
                return

    async def _income_loop(self) -> None:
        while True:
            try:
                await self.poll_income()
                self.ledger_check()
            except Exception as e:  # noqa: BLE001
                log.warning("income selhal: %r", e)
            if await _sleep(self.stop, INCOME_EVERY_S):
                return

    async def poll_income(self) -> int:
        since = int(self.db.meta_get("income_since") or (self.rest.clock.now_ms() - 7 * 86_400_000))
        n = 0
        while True:
            rows = await self.rest.income(start_ms=since, limit=1000)
            for r in rows:
                if r.get("incomeType") not in ("COMMISSION", "FUNDING_FEE", "REALIZED_PNL"):
                    continue
                n += self.db.execute("INSERT OR IGNORE INTO income VALUES (?,?,?,?,?,?,?)",
                                     (str(r["tranId"]), r.get("symbol"), r["incomeType"], float(r["income"]),
                                      r.get("asset"), int(r["time"]), r.get("tradeId"))).rowcount
            if not rows:
                break
            since = max(int(r["time"]) for r in rows) + 1
            if len(rows) < 1000:
                break
        self.db.meta_set("income_since", str(since))
        return n

    def ledger_check(self) -> list[dict[str, Any]]:
        """Income je ground truth: rozdíl proti lokálnímu ledgeru > 0,01 USDT → varování (SPEC §8)."""
        out = []
        for t in self.db.execute("SELECT * FROM trades WHERE mode='live' ORDER BY exit_time DESC LIMIT 200").fetchall():
            sym = t["slot_id"].rsplit(".", 1)[1]
            r = self.db.execute("SELECT COALESCE(SUM(amount),0) FROM income WHERE symbol=? AND ts>=? AND ts<=? "
                                "AND income_type IN ('COMMISSION','REALIZED_PNL','FUNDING_FEE')",
                                (sym, int(t["entry_time"]) - 1000, int(t["exit_time"]) + 60_000)).fetchone()
            diff = float(r[0]) - float(t["net_pnl"])
            if abs(diff) > LEDGER_TOL_USDT:
                out.append({"trade_id": t["trade_id"], "income": float(r[0]), "ledger": float(t["net_pnl"]),
                            "diff": diff})
        self.db.meta_set("ledger_diffs", json.dumps(out))
        if out:
            self.db.event("WARN", "ledger", f"{len(out)} obchodů nesedí s income > {LEDGER_TOL_USDT} USDT",
                          {"diffs": out[:20]})
        return out

    async def _kill_loop(self) -> None:
        while not await _sleep(self.stop, 1.0):
            if self.risk.kill_active() and self.executor is not None and self.executor.positions:
                await self.executor.close_all("kill switch")
            if self.ws is not None and not self.ws.connected and self.risk.private_down_since is None:
                self.risk.private_down_since = self.rest.clock.now_ms()

    def tasks(self) -> list[asyncio.Task[Any]]:
        assert self.ws is not None
        return [asyncio.create_task(c, name=n) for n, c in (
            ("live_ws", self.ws.run(self.stop)), ("live_keepalive", self._keepalive_loop()),
            ("live_reconcile", self._reconcile_loop()), ("live_equity", self._equity_loop()),
            ("live_income", self._income_loop()), ("live_kill", self._kill_loop()))]

    # --- napojení na sloty --------------------------------------------------

    def on_signal(self, slot: Slot, intent: Intent, signal_id: int, action: str, paper_trade_id: str | None,
                  bar_time: int, decided_at: int, decision_px: float) -> None:
        """Signál paper slotu; když je to live slot daného symbolu, zkusí se živý vstup."""
        ls = self.cfg.live.slots.get(slot.symbol)
        if ls is None or ls.variant != slot.strategy.variant_id or self.executor is None:
            return
        if slot.symbol in self.executor.positions:
            return
        if bar_time < self.cooldown_until.get(slot.slot_id, 0):
            return
        ok, why = self.risk.can_enter(slot.slot_id, slot.symbol, decided_at, len(self.executor.positions))
        if not ok:
            self.db.event("INFO", "risk", f"live vstup blokován: {why}", {"slot": slot.slot_id, "signal": signal_id})
            return
        self.seq[slot.slot_id] = self.seq.get(slot.slot_id, 0) + 1
        asyncio.ensure_future(self.executor.open(
            slot.slot_id, slot.symbol, self.seq[slot.slot_id], intent, signal_id, paper_trade_id, bar_time,
            decided_at, decision_px, ls.notional_usdt, self.cfg.config_hash))

    async def on_bar_close(self, open_time: int, closes: Mapping[str, float]) -> None:
        if self.executor is not None:
            await self.executor.on_bar_close(open_time, closes)

    def on_trade_tick(self, symbol: str, e: Mapping[str, Any]) -> None:
        if self.executor is not None:
            self.executor.on_trade_tick(symbol, e)

    def _on_closed(self, rec: TradeRecord, pos: LivePosition) -> None:
        row = rec.row()
        self.db.execute(f"INSERT OR REPLACE INTO trades({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                        tuple(row.values()))
        cd = self.cfg.engine.cooldown_bars * 300_000
        self.cooldown_until[rec.slot_id] = (rec.exit_time // 300_000) * 300_000 + cd
        self.risk.on_live_trade_closed(rec.slot_id, pos.notional_usdt, rec.exit_time)
        self.db.event("INFO", "live", f"{rec.slot_id} {rec.exit_reason} {rec.net_pnl:+.4f} USDT ({rec.net_bps:+.1f} bps)",
                      {"trade_id": rec.trade_id, **rec.extra})

    def positions_view(self) -> list[dict[str, Any]]:
        out = []
        for p in (self.executor.positions.values() if self.executor else []):
            px = self.last_price(p.symbol)
            sim = p.sim or {}
            lvl = None
            if sim.get("activated") and sim.get("extreme") and sim.get("callback_pct"):
                k = sim["callback_pct"] / 100
                lvl = sim["extreme"] * (1 - k) if p.side == "LONG" else sim["extreme"] * (1 + k)
            sg = 1 if p.side == "LONG" else -1
            out.append({"slot": p.slot_id, "side": p.side, "state": p.state, "entry_px": p.entry_px, "px": px,
                        "trail": lvl, "trail_dist_bps": (sg * (px - lvl) / px * 1e4) if (px and lvl) else None,
                        "algo_id": p.orders.get("TS") or p.orders.get("HS")})
        return out


async def _sleep(stop: asyncio.Event, s: float) -> bool:
    """Počká `s` sekund; True = přišel stop."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=s)
        return True
    except asyncio.TimeoutError:
        return False


def kill_file(cfg: Config) -> Path:
    return Path(cfg.paths.kill_file)
