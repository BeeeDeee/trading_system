"""Rekonciliace DB ↔ burza (SPEC §9), při startu a každých 60 s.

- pozice na burze, v DB žádná → FOREIGN_POSITION: alert, žádná akce, live na symbolu se zmrazí;
- pozice v DB i na burze bez tvrdého stopu nebo trailingu → okamžitě doplnit ochranu;
- pozice v DB, na burze žádná → dohledat výstup přes userTrades, zapsat obchod exit_reason=reconciled;
- osiřelý algo/limit příkaz s prefixem `a5-` bez pozice → zrušit;
- příkazy bez prefixu `a5-` → nikdy nesahat (patří vlastníkovi);
- nesoulad množství → alert + zmrazit symbol."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.binance import errors
from arena.binance.rest import RestClient
from arena.execution.ids import is_ours
from arena.execution.live import LiveExecutor, LivePosition
from arena.risk import RiskGuard

log = logging.getLogger(__name__)


@dataclass
class Diff:
    kind: str
    symbol: str
    detail: str
    action: str = ""

    def __str__(self) -> str:
        return f"{self.kind:20} {self.symbol:9} {self.detail}" + (f" → {self.action}" if self.action else "")


@dataclass
class Reconciler:
    rest: RestClient
    executor: LiveExecutor
    risk: RiskGuard
    symbols: tuple[str, ...]
    alert: Any = lambda t, m: None
    last: list[Diff] = field(default_factory=list)

    async def run(self, fix: bool = True) -> list[Diff]:
        diffs: list[Diff] = []
        ex_pos = {p["symbol"]: Decimal(str(p["positionAmt"])) for p in await self.rest.position_risk()
                  if Decimal(str(p.get("positionAmt", "0"))) != 0}
        orders = await self.rest.open_orders()
        algos = await self.rest.open_algo_orders()
        open_algo = {str(a.get("clientAlgoId")): a for a in algos}
        open_ord = {str(o.get("clientOrderId")): o for o in orders}

        for sym in self.symbols:
            amt = ex_pos.get(sym, Decimal(0))
            pos = self.executor.positions.get(sym)
            if pos is None:
                if amt != 0:
                    diffs.append(Diff("FOREIGN_POSITION", sym, f"na burze {amt}, v DB nic", "zmrazit live symbol"))
                    if fix:
                        self.risk.freeze(sym, "FOREIGN_POSITION")
                continue
            if pos.state == "ENTRY_SENT":
                diffs.append(await self._resolve_entry(pos, fix))
                continue
            if amt == 0:
                diffs.append(Diff("POSITION_GONE", sym, f"{pos.slot_id} v DB, na burze nic", "dohledat výstup"))
                if fix:
                    await self._close_from_user_trades(pos)
                continue
            remaining = pos.qty_d - Decimal(str(round(pos.exit_qty, 12)))
            sign = 1 if pos.side == "LONG" else -1
            if amt != sign * remaining:
                diffs.append(Diff("QTY_MISMATCH", sym, f"burza {amt}, DB {sign * remaining}", "alert + zmrazit"))
                if fix:
                    self.risk.freeze(sym, f"QTY_MISMATCH burza {amt} DB {sign * remaining}")
                continue
            if pos.state in ("EXITING", "EMERGENCY_CLOSE"):
                diffs.append(Diff("EXIT_PENDING", sym, f"{pos.slot_id} {pos.state}, na burze {amt}", "zkusit výstup znovu"))
                if fix:
                    await self.executor.retry_close(pos)
                continue
            have = {c for c in open_algo if c in pos.orders.values()}
            missing = [k for k in ("HS", "TS") if (k == "HS" or pos.plan.get("callback_rate"))
                       and pos.orders.get(k) not in have]
            if missing and pos.state in ("PROTECTING", "PROTECTED"):
                diffs.append(Diff("UNPROTECTED", sym, f"{pos.slot_id} chybí {missing}", "doplnit ochranu"))
                if fix:
                    await self.executor.ensure_protection(pos, have)

        # osiřelé a5- příkazy (bez pozice v DB); cizí příkazy se nikdy neruší
        live_cids = {c for p in self.executor.positions.values() for c in p.orders.values()}
        for cid, a in open_algo.items():
            if is_ours(cid) and cid not in live_cids:
                diffs.append(Diff("ORPHAN_ALGO", str(a.get("symbol")), cid, "zrušit"))
                if fix:
                    try:
                        await self.rest.cancel_algo_order(cid)
                    except (errors.BinanceError, errors.UnknownOrderStatus) as e:
                        log.warning("zrušení %s selhalo: %r", cid, e)
        for cid, o in open_ord.items():
            if is_ours(cid) and cid not in live_cids:
                diffs.append(Diff("ORPHAN_ORDER", str(o.get("symbol")), cid, "zrušit"))
                if fix:
                    try:
                        await self.rest.cancel_order(str(o["symbol"]), cid)
                    except (errors.BinanceError, errors.UnknownOrderStatus) as e:
                        log.warning("zrušení %s selhalo: %r", cid, e)
        for d in diffs:
            if d.kind in ("FOREIGN_POSITION", "QTY_MISMATCH", "UNPROTECTED"):
                self.alert(d.kind, f"{d.symbol}: {d.detail}")
        self.last = diffs
        self.executor.db.meta_set("last_reconcile", str(self.executor.now_ms()))
        if diffs:
            self.executor.db.event("WARN", "reconcile", f"{len(diffs)} rozdílů", {"diffs": [str(d) for d in diffs]})
        return diffs

    async def _resolve_entry(self, pos: LivePosition, fix: bool) -> Diff:
        """Pád mezi odesláním vstupu a ochranou (kill -9): zjistit stav vstupu a dokončit nebo zahodit."""
        cid = pos.orders["E"]
        try:
            o = await self.rest.query_order(pos.symbol, cid)
        except errors.BinanceError as e:
            if e.code == -2013:
                if fix:
                    self.executor.positions.pop(pos.symbol, None)
                    self.executor.save(None, pos.slot_id)
                return Diff("ENTRY_NOT_FOUND", pos.symbol, cid, "zahodit")
            raise
        if Decimal(str(o.get("executedQty", "0"))) == 0:
            if fix:
                self.executor.positions.pop(pos.symbol, None)
                self.executor.save(None, pos.slot_id)
            return Diff("ENTRY_UNFILLED", pos.symbol, f"{cid} {o.get('status')}", "zahodit")
        if fix:
            self.executor._on_entry_filled(pos, o)   # noqa: SLF001 – stejná cesta jako běžný vstup
            await self.executor._protect(pos)        # noqa: SLF001
        return Diff("ENTRY_FILLED_UNPROTECTED", pos.symbol, f"{cid} qty {o.get('executedQty')}", "doplnit ochranu")

    async def _close_from_user_trades(self, pos: LivePosition) -> None:
        """Pozice skončila mimo náš přehled (např. stop za výpadku WS): výstupní filly z userTrades."""
        trades = await self.rest.user_trades(pos.symbol, start_ms=pos.entry_time or pos.decision_ts)
        for t in trades:
            if t.get("side") != pos.close_side or int(t["time"]) < pos.entry_time:
                continue
            ev = {"e": "ORDER_TRADE_UPDATE", "o": {
                "s": pos.symbol, "x": "TRADE", "c": "", "i": t["orderId"], "t": t["id"], "l": t["qty"], "L": t["price"],
                "n": t.get("commission", 0), "N": t.get("commissionAsset"), "T": t["time"], "rp": t.get("realizedPnl", 0),
                "R": True, "S": t["side"], "m": t.get("maker", False)}}
            if pos.exit_kind is None:
                pos.exit_kind = "RC"
            await self.executor.on_user_event(ev)
            if pos.symbol not in self.executor.positions:
                return
        if pos.symbol in self.executor.positions:
            # bez fillů (např. likvidace mimo userTrades okno) → uzavřít s poslední cenou, alert
            self.alert("výstup nedohledán", f"{pos.slot_id}: pozice zmizela bez fillů v userTrades")
            pos.exit_kind = "RC"
            await self.executor._finalize(pos, "reconciled")  # noqa: SLF001
