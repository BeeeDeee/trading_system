"""Live executor: stavový automat jedné pozice (SPEC §8).

FLAT → ENTRY_SENT → PROTECTING → PROTECTED → EXITING → FLAT; selhání ochrany → EMERGENCY_CLOSE.
Invariant: živá pozice nikdy nezůstane bez tvrdého stopu déle než 2 s.

Pravidla:
- po timeoutu/5xx je stav příkazu neznámý → dotaz podle client ID, nikdy slepý retry (nástraha 7);
- podmíněné příkazy jen přes /fapi/v1/algoOrder (D3), quantity = executedQty, reduceOnly;
- Intent → úrovně příkazů přes `plan_exits`, stejně jako paper (nástraha 12);
- skutečné poplatky z ORDER_TRADE_UPDATE (`n`, `N`), realized PnL (`rp`); funding z /fapi/v1/income."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Callable, Literal, Mapping

from arena.binance import errors
from arena.binance.filters import SymbolFilters
from arena.binance.rest import RestClient
from arena.data.bars import BAR_MS
from arena.db import Database
from arena.execution.base import TradeRecord, bps, side_sign
from arena.execution.ids import client_id
from arena.execution.sizing import ExitPlan, entry_qty, plan_exits
from arena.execution.trailing_sim import TrailingSim
from arena.strategies.base import Intent

log = logging.getLogger(__name__)

State = Literal["ENTRY_SENT", "PROTECTING", "PROTECTED", "EXITING", "EMERGENCY_CLOSE"]
PROTECT_DEADLINE_S = 2.0
PROTECT_ATTEMPTS = 3
EXIT_REASONS = {"TS": "trailing", "HS": "hard_stop", "TP": "take_profit", "TX": "time", "EM": "emergency",
                "KILL": "kill", "RC": "reconciled"}


@dataclass
class LivePosition:
    slot_id: str
    symbol: str
    seq: int
    trade_id: str
    side: Literal["LONG", "SHORT"]
    signal_id: int | None
    twin_trade_id: str | None
    entry_bar: int
    decision_ts: int
    decision_px: float
    arrival_mid: float | None
    intent: dict[str, Any]
    notional_usdt: float
    config_hash: str
    state: State = "ENTRY_SENT"
    qty: str = "0"                         # Decimal jako řetězec (executedQty)
    entry_px: float = 0.0
    entry_time: int = 0
    entry_fee: float = 0.0
    plan: dict[str, Any] = field(default_factory=dict)
    orders: dict[str, str] = field(default_factory=dict)      # kind → client id (E, HS, TS, TP, TX, EM)
    algo_order_ids: dict[str, str] = field(default_factory=dict)   # orderId spuštěného market příkazu → kind
    exit_kind: str | None = None
    exit_trigger_px: float | None = None
    exit_trigger_ts: int | None = None
    exit_qty: float = 0.0
    exit_notional: float = 0.0
    exit_fee: float = 0.0
    exit_time: int = 0
    realized: float = 0.0
    sent_ms: int = 0
    ack_ms: int = 0
    fill_ms: int = 0
    sim: dict[str, Any] | None = None      # stínový TrailingSim: HWM pro teoretickou úroveň trailingu
    notes: list[str] = field(default_factory=list)

    @property
    def qty_d(self) -> Decimal:
        return Decimal(self.qty)

    @property
    def close_side(self) -> str:
        return "SELL" if self.side == "LONG" else "BUY"

    def kind_of(self, cid: str | None) -> str | None:
        for k, v in self.orders.items():
            if v == cid:
                return k
        return None


@dataclass
class LiveExecutor:
    rest: RestClient
    filters: Mapping[str, SymbolFilters]
    db: Database
    now_ms: Callable[[], int]
    last_price: Callable[[str], float | None]
    alert: Callable[[str, str], None] = lambda t, m: None
    on_closed: Callable[[TradeRecord, LivePosition], None] | None = None
    cost_model_version: int = 1
    protect_deadline_s: float = PROTECT_DEADLINE_S

    positions: dict[str, LivePosition] = field(default_factory=dict)          # symbol → pozice (1 slot/symbol)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def lock(self, symbol: str) -> asyncio.Lock:
        return self._locks.setdefault(symbol, asyncio.Lock())

    # --- persistence --------------------------------------------------------

    def save(self, pos: LivePosition | None, slot_id: str | None = None) -> None:
        sid = pos.slot_id if pos else slot_id
        self.db.execute("INSERT OR REPLACE INTO slot_state VALUES (?,?,?,?,?)",
                        (sid, "live", pos.state if pos else "FLAT", json.dumps(asdict(pos)) if pos else None,
                         self.now_ms()))

    def restore(self) -> None:
        for r in self.db.execute("SELECT state_json FROM slot_state WHERE mode='live' AND status!='FLAT' "
                                 "AND state_json IS NOT NULL").fetchall():
            pos = LivePosition(**json.loads(r["state_json"]))
            self.positions[pos.symbol] = pos

    def _order_row(self, cid: str, pos: LivePosition, kind: str, is_algo: bool, request: dict[str, Any],
                   status: str, response: Any = None, exchange_id: Any = None) -> None:
        self.db.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(client_id) DO UPDATE SET status=excluded.status, "
            "last_response_json=excluded.last_response_json, exchange_id=COALESCE(excluded.exchange_id, exchange_id), "
            "ack_at=excluded.ack_at",
            (cid, pos.trade_id, kind, int(is_algo), str(exchange_id) if exchange_id is not None else None, status,
             json.dumps(request, default=str), json.dumps(response, default=str) if response is not None else None,
             self.now_ms(), self.now_ms()))

    def _latency(self, trade_id: str, kind: str, ms: int) -> None:
        self.db.execute("INSERT INTO latencies VALUES (?,?,?)", (trade_id, kind, int(ms)))

    # --- vstup --------------------------------------------------------------

    async def open(self, slot_id: str, symbol: str, seq: int, intent: Intent, signal_id: int | None,
                   twin_trade_id: str | None, bar_time: int, decided_at: int, decision_px: float,
                   notional_usdt: float, config_hash: str) -> LivePosition | None:
        async with self.lock(symbol):
            if symbol in self.positions:
                return None
            f = self.filters[symbol]
            try:
                qty = entry_qty(f, notional_usdt, decision_px)
            except ValueError as e:
                self.alert("live vstup pod filtry", f"{slot_id}: {e}")
                return None
            pos = LivePosition(slot_id, symbol, seq, f"l-{slot_id}-{seq}", intent.side, signal_id, twin_trade_id,
                               bar_time, decided_at, decision_px, None, asdict(intent), notional_usdt, config_hash)
            cid = client_id(slot_id, seq, "E")
            pos.orders["E"] = cid
            self.positions[symbol] = pos
            self.save(pos)   # před odesláním: po pádu rekonciliace ví, že vstup mohl odejít
            req = {"symbol": symbol, "side": "BUY" if intent.side == "LONG" else "SELL", "type": "MARKET",
                   "quantity": qty, "newClientOrderId": cid, "newOrderRespType": "RESULT"}
            pos.sent_ms = self.now_ms()
            self._latency(pos.trade_id, "bar_close_to_send", pos.sent_ms - (bar_time + BAR_MS))
            self._latency(pos.trade_id, "decision_to_send", pos.sent_ms - decided_at)
            self._order_row(cid, pos, "E", False, req, "SENT")
            try:
                resp = await self.rest.new_order(req)
            except errors.UnknownOrderStatus:
                resp = await self._query_until_known(symbol, cid)
            except errors.BinanceError as e:
                self._order_row(cid, pos, "E", False, req, "REJECTED", {"code": e.code, "msg": e.msg})
                self.db.event("WARN", "live", f"vstup odmítnut {slot_id}", {"code": e.code, "msg": e.msg})
                del self.positions[symbol]
                self.save(None, slot_id)
                return None
            if resp is None or Decimal(str(resp.get("executedQty", "0"))) == 0:
                self._order_row(cid, pos, "E", False, req, str((resp or {}).get("status", "UNKNOWN")), resp)
                self.db.event("WARN", "live", f"vstup bez fillu {slot_id}", {"resp": resp})
                del self.positions[symbol]
                self.save(None, slot_id)
                return None
            pos.ack_ms = self.now_ms()
            self._latency(pos.trade_id, "send_to_ack", pos.ack_ms - pos.sent_ms)
            self._order_row(cid, pos, "E", False, req, str(resp.get("status")), resp, resp.get("orderId"))
            try:
                self._on_entry_filled(pos, resp)
                await self._protect(pos)
                if "entry_px_approx" in pos.notes and pos.symbol in self.positions:
                    await self._refine_entry_px(pos)
            except asyncio.CancelledError:
                raise   # pád procesu: po restartu dorovná rekonciliace
            except Exception as e:  # noqa: BLE001 – pozice nesmí zůstat bez ochrany
                log.exception("chyba po fillu vstupu %s", slot_id)
                if not pos.plan:
                    pos.qty = str(Decimal(str(resp["executedQty"])))
                if not pos.entry_px:
                    pos.entry_px = float(resp.get("avgPrice") or 0) or self.last_price(symbol) or decision_px
                await self.emergency_close(pos, f"chyba po fillu vstupu: {e!r}")
            return pos

    async def _query_until_known(self, symbol: str, cid: str, tries: int = 5) -> dict[str, Any] | None:
        """Po timeoutu: dotaz podle origClientOrderId. -2013 (neexistuje) = příkaz nedorazil."""
        for i in range(tries):
            try:
                return await self.rest.query_order(symbol, cid)
            except errors.BinanceError as e:
                if e.code == -2013:
                    return None
            except errors.UnknownOrderStatus:
                pass
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.3 * (i + 1))
        self.alert("neznámý stav vstupu", f"{symbol} {cid}: po {tries} dotazech stále neznámý")
        return None

    async def _refine_entry_px(self, pos: LivePosition) -> None:
        """Skutečná cena vstupu, když ji odpověď na MARKET neobsahovala: dotaz na příkaz, jinak vážený průměr fillů."""
        try:
            q = await self.rest.query_order(pos.symbol, pos.orders["E"])
            px = float(q.get("avgPrice") or 0)
            if px <= 0:
                oid = str(q.get("orderId"))
                fills = [t for t in await self.rest.user_trades(pos.symbol, start_ms=pos.entry_time - 5000)
                         if str(t.get("orderId")) == oid]
                qty = sum(float(t["qty"]) for t in fills)
                px = sum(float(t["price"]) * float(t["qty"]) for t in fills) / qty if qty else 0.0
            if px > 0:
                pos.notes.append(f"entry_px {pos.entry_px} → {px}")
                pos.entry_px = px
                self.save(pos)
                return
        except Exception as e:  # noqa: BLE001 – ochrana už stojí, jen zpřesnění
            log.warning("cena vstupu %s se nepodařilo zpřesnit: %r", pos.slot_id, e)
        self.db.event("WARN", "live", f"entry_px zůstává odhad {pos.slot_id}", {"px": pos.entry_px})

    def _on_entry_filled(self, pos: LivePosition, resp: Mapping[str, Any]) -> None:
        f = self.filters[pos.symbol]
        qty = Decimal(str(resp["executedQty"]))
        px = Decimal(str(resp.get("avgPrice") or 0))
        if px <= 0:
            px = Decimal(str(resp.get("price") or 0))     # u MARKET je price 0
        if px <= 0:
            # demo/testnet vrací u RESULT bez avgPrice (zjištěno 2026-10-25): ochrana se nesmí zdržovat dotazem,
            # úrovně se počítají z poslední ceny a skutečná cena vstupu se dohledá až po ochraně (_refine_entry_px)
            last = self.last_price(pos.symbol)
            if not last:
                raise RuntimeError("cena vstupu neznámá: odpověď bez avgPrice a bez lokální ceny")
            px = Decimal(str(last))
            pos.notes.append("entry_px_approx")
        plan = plan_exits(Intent(**pos.intent), f, px, f.floor_qty(qty))
        pos.qty, pos.entry_px = str(plan.qty), float(px)
        pos.entry_time = int(resp.get("updateTime") or self.now_ms())
        pos.fill_ms = self.now_ms()
        self._latency(pos.trade_id, "ack_to_fill", max(0, pos.entry_time - pos.ack_ms))
        pos.plan = {"callback_rate": str(plan.callback_rate) if plan.callback_rate is not None else None,
                    "activate_px": str(plan.activate_px) if plan.activate_px is not None else None,
                    "hard_stop_px": str(plan.hard_stop_px), "tp_px": str(plan.tp_px) if plan.tp_px is not None else None,
                    "max_hold_bars": plan.max_hold_bars}
        pos.notes.extend(plan.notes)
        pos.sim = asdict(TrailingSim(pos.side, float(px), float(plan.hard_stop_px),
                                     float(plan.callback_rate) if plan.callback_rate is not None else None,
                                     float(plan.activate_px) if plan.activate_px is not None else None,
                                     None, start_ts=pos.entry_time))
        pos.state = "PROTECTING"
        self.save(pos)

    # --- ochrana ------------------------------------------------------------

    def _plan(self, pos: LivePosition) -> ExitPlan:
        p = pos.plan
        return ExitPlan(pos.side, pos.qty_d, Decimal(str(pos.entry_px)),
                        Decimal(p["callback_rate"]) if p["callback_rate"] else None,
                        Decimal(p["activate_px"]) if p["activate_px"] else None, Decimal(p["hard_stop_px"]),
                        Decimal(p["tp_px"]) if p["tp_px"] else None, int(p["max_hold_bars"]))

    async def _place_algo(self, pos: LivePosition, kind: str, params: dict[str, Any]) -> bool:
        """Jeden algo příkaz s ≤ 3 pokusy. Při neznámém stavu dotaz podle clientAlgoId (idempotence)."""
        cid = client_id(pos.slot_id, pos.seq, kind)
        pos.orders[kind] = cid
        req = {"symbol": pos.symbol, "side": pos.close_side, "quantity": pos.qty_d, "reduceOnly": "true",
               "workingType": "CONTRACT_PRICE", "clientAlgoId": cid, "newOrderRespType": "RESULT", **params}
        for attempt in range(PROTECT_ATTEMPTS):
            try:
                resp = await self.rest.new_algo_order(req)
                self._order_row(cid, pos, kind, True, req, str(resp.get("algoStatus", "NEW")), resp, resp.get("algoId"))
                return True
            except errors.UnknownOrderStatus:
                try:
                    resp = await self.rest.query_algo_order(cid)
                    if resp and resp.get("algoStatus") in ("NEW", "TRIGGERING", "TRIGGERED"):
                        self._order_row(cid, pos, kind, True, req, str(resp["algoStatus"]), resp, resp.get("algoId"))
                        return True
                except Exception:  # noqa: BLE001
                    pass
            except errors.WouldImmediatelyTrigger as e:
                if kind == "TS" and "activatePrice" in req:
                    # -2021: aktivační cena už je za trhem → trailing bez aktivace (SPEC §8)
                    req.pop("activatePrice")
                    pos.notes.append("trailing bez activatePrice (-2021)")
                    self.db.event("WARN", "live", "trailing -2021 → bez activatePrice", {"slot": pos.slot_id})
                    continue
                self._order_row(cid, pos, kind, True, req, "REJECTED", {"code": e.code, "msg": e.msg})
                return False
            except errors.BinanceError as e:
                self._order_row(cid, pos, kind, True, req, "REJECTED", {"code": e.code, "msg": e.msg})
                if e.code == -2022:   # reduceOnly odmítnut: pozice už není
                    return False
            await asyncio.sleep(0.2)
        return False

    async def _protect(self, pos: LivePosition) -> None:
        """(a) tvrdý STOP_MARKET, (b) TRAILING_STOP_MARKET, (c) volitelně TP LIMIT. (a)/(b) do 2 s, jinak
        EMERGENCY_CLOSE + alert."""
        plan = self._plan(pos)

        async def hard_and_trail() -> bool:
            # HS a TS se posílají souběžně: na demu trvá jedno volání ~1 s a dvě po sobě by nestihla limit 2 s
            jobs = [self._place_algo(pos, "HS", {"type": "STOP_MARKET", "triggerPrice": plan.hard_stop_px})]
            if plan.callback_rate is not None:
                tp: dict[str, Any] = {"type": "TRAILING_STOP_MARKET", "callbackRate": plan.callback_rate}
                if plan.activate_px is not None:
                    tp["activatePrice"] = plan.activate_px
                jobs.append(self._place_algo(pos, "TS", tp))
            res = await asyncio.gather(*jobs, return_exceptions=True)
            for r in res:
                if isinstance(r, BaseException):
                    raise r
            return all(res)

        try:
            ok = await asyncio.wait_for(hard_and_trail(), timeout=self.protect_deadline_s)
        except asyncio.TimeoutError:
            ok = False
        if not ok:
            await self.emergency_close(pos, "selhání ochrany")
            return
        if plan.tp_px is not None:
            last = self.last_price(pos.symbol) or pos.entry_px
            tp = float(plan.tp_px)
            if (pos.side == "LONG" and tp <= last) or (pos.side == "SHORT" and tp >= last):
                pos.notes.append("tp_skipped_marketable")
                self.db.event("INFO", "live", "tp_skipped_marketable", {"slot": pos.slot_id, "tp": tp, "last": last})
            else:
                cid = client_id(pos.slot_id, pos.seq, "TP")
                pos.orders["TP"] = cid
                req = {"symbol": pos.symbol, "side": pos.close_side, "type": "LIMIT", "timeInForce": "GTC",
                       "quantity": pos.qty_d, "price": plan.tp_px, "reduceOnly": "true", "newClientOrderId": cid}
                try:
                    resp = await self.rest.new_order(req)
                    self._order_row(cid, pos, "TP", False, req, str(resp.get("status")), resp, resp.get("orderId"))
                except errors.UnknownOrderStatus:
                    resp = await self._query_until_known(pos.symbol, cid)
                    self._order_row(cid, pos, "TP", False, req, str((resp or {}).get("status", "UNKNOWN")), resp)
                except errors.BinanceError as e:
                    self._order_row(cid, pos, "TP", False, req, "REJECTED", {"code": e.code, "msg": e.msg})
                    pos.notes.append(f"tp odmítnut {e.code}")
        if pos.state == "PROTECTING":
            pos.state = "PROTECTED"
        self.save(pos)

    async def ensure_protection(self, pos: LivePosition, open_algo_cids: set[str], locked: bool = False) -> None:
        """Rekonciliace: pozice bez tvrdého stopu nebo trailingu → okamžitě doplnit (SPEC §9)."""
        if not locked:
            async with self.lock(pos.symbol):
                await self._ensure_protection(pos, open_algo_cids)
        else:
            await self._ensure_protection(pos, open_algo_cids)

    async def _ensure_protection(self, pos: LivePosition, open_algo_cids: set[str]) -> None:
        plan = self._plan(pos)
        need_hs = pos.orders.get("HS") not in open_algo_cids
        need_ts = plan.callback_rate is not None and pos.orders.get("TS") not in open_algo_cids
        if not (need_hs or need_ts):
            return
        self.alert("doplňuji ochranu", f"{pos.slot_id}: HS={'chybí' if need_hs else 'ok'} "
                                       f"TS={'chybí' if need_ts else 'ok'}")
        ok = True
        if need_hs:
            ok = await self._place_algo(pos, "HS", {"type": "STOP_MARKET", "triggerPrice": plan.hard_stop_px})
        if ok and need_ts:
            tp: dict[str, Any] = {"type": "TRAILING_STOP_MARKET", "callbackRate": plan.callback_rate}
            ok = await self._place_algo(pos, "TS", tp)
        if not ok:
            await self.emergency_close(pos, "ochranu nejde doplnit")
            return
        pos.state = "PROTECTED"
        self.save(pos)

    # --- výstupy ------------------------------------------------------------

    async def _cancel_siblings(self, pos: LivePosition, keep: str | None = None) -> None:
        for kind in ("HS", "TS"):
            cid = pos.orders.get(kind)
            if cid and kind != keep:
                try:
                    await self.rest.cancel_algo_order(cid)
                    self._order_row(cid, pos, kind, True, {}, "CANCELED")
                except errors.BinanceError:
                    pass   # už spuštěný / zrušený
                except errors.UnknownOrderStatus:
                    pass   # rekonciliace zruší osiřelé
        cid = pos.orders.get("TP")
        if cid and keep != "TP":
            try:
                await self.rest.cancel_order(pos.symbol, cid)
                self._order_row(cid, pos, "TP", False, {}, "CANCELED")
            except (errors.BinanceError, errors.UnknownOrderStatus):
                pass

    async def market_exit(self, pos: LivePosition, kind: str, trigger_px: float | None = None) -> None:
        """Časový výstup / kill: nejdřív zrušit algo a TP, pak MARKET reduceOnly. -2022 → rekonciliace."""
        async with self.lock(pos.symbol):
            if pos.state in ("EXITING", "EMERGENCY_CLOSE") or pos.symbol not in self.positions:
                return
            pos.state = "EXITING"
            pos.exit_kind = kind
            pos.exit_trigger_px = trigger_px if trigger_px is not None else self.last_price(pos.symbol)
            pos.exit_trigger_ts = self.now_ms()
            self.save(pos)
            await self._cancel_siblings(pos)
            if not await self._send_close(pos, "TX" if kind == "TX" else "EM"):
                # stopy jsou zrušené a market výstup neprošel → vrátit ochranu, výstup zkusí rekonciliace
                self.alert("výstup selhal", f"{pos.slot_id}: {kind} neprošel, obnovuji ochranu")
                pos.state, pos.exit_kind = "PROTECTED", None
                pos.orders.pop("HS", None)
                pos.orders.pop("TS", None)
                await self.ensure_protection(pos, set(), locked=True)

    async def emergency_close(self, pos: LivePosition, reason: str) -> None:
        pos.state = "EMERGENCY_CLOSE"
        pos.exit_kind = pos.exit_kind or "EM"
        pos.exit_trigger_px = pos.exit_trigger_px or self.last_price(pos.symbol)
        pos.exit_trigger_ts = self.now_ms()
        self.save(pos)
        self.alert("EMERGENCY_CLOSE", f"{pos.slot_id}: {reason}")
        await self._cancel_siblings(pos)
        await self._send_close(pos, "EM")

    async def _send_close(self, pos: LivePosition, kind: str, attempts: int = 3) -> bool:
        """MARKET reduceOnly na zbytek pozice. True = příkaz prošel nebo pozice už není (-2022)."""
        cid = client_id(pos.slot_id, pos.seq, kind)
        pos.orders[kind] = cid
        remaining = pos.qty_d - Decimal(str(pos.exit_qty))
        if remaining <= 0:
            return True
        req = {"symbol": pos.symbol, "side": pos.close_side, "type": "MARKET", "quantity": remaining,
               "reduceOnly": "true", "newClientOrderId": cid, "newOrderRespType": "RESULT"}
        for attempt in range(attempts):
            try:
                resp = await self.rest.new_order(req)
            except errors.UnknownOrderStatus:
                resp = await self._query_until_known(pos.symbol, cid)
                if resp is None:
                    continue
            except errors.ReduceOnlyRejected:
                # -2022: pozice mezitím skončila jinak → rekonciliace dopočítá (SPEC §8)
                self._order_row(cid, pos, kind, False, req, "REJECTED_2022")
                self.db.event("WARN", "live", "-2022 při výstupu → rekonciliace", {"slot": pos.slot_id})
                pos.notes.append(f"-2022 při výstupu {kind}")
                pos.exit_kind = "RC"
                self.save(pos)
                return True
            except errors.BinanceError as e:
                self._order_row(cid, pos, kind, False, req, "REJECTED", {"code": e.code, "msg": e.msg})
                self.alert("výstup odmítnut", f"{pos.slot_id}: {e} (pokus {attempt + 1})")
                await asyncio.sleep(0.3)
                continue
            self._order_row(cid, pos, kind, False, req, str((resp or {}).get("status", "UNKNOWN")), resp,
                            (resp or {}).get("orderId"))
            if resp and Decimal(str(resp.get("executedQty", "0"))) > 0 and pos.exit_qty == 0:
                # fill z RESULT; ORDER_TRADE_UPDATE ho doplní poplatkem (idempotentně podle trade id)
                pos.exit_time = int(resp.get("updateTime") or self.now_ms())
            return True
        return False

    async def retry_close(self, pos: LivePosition) -> None:
        """Rekonciliace: pozice v EXITING/EMERGENCY_CLOSE je pořád na burze → zkusit výstup znovu."""
        async with self.lock(pos.symbol):
            kind = "EM" if pos.state == "EMERGENCY_CLOSE" or pos.exit_kind != "TX" else "TX"
            if not await self._send_close(pos, kind) and pos.state == "EXITING":
                pos.state, pos.exit_kind = "PROTECTED", None
                pos.orders.pop("HS", None)
                pos.orders.pop("TS", None)
                await self.ensure_protection(pos, set(), locked=True)

    # --- user data stream ---------------------------------------------------

    def on_trade_tick(self, symbol: str, e: Mapping[str, Any]) -> None:
        """aggTrade → stínový TrailingSim (HWM pro teoretickou úroveň trailingu = exit_trigger_px)."""
        pos = self.positions.get(symbol)
        if pos is None or pos.sim is None or pos.state not in ("PROTECTING", "PROTECTED"):
            return
        sim = TrailingSim(**{**pos.sim, "pending": None, "done": None})
        sim.on_trade(float(e["p"]), int(e["T"]), int(e["a"]))
        d = asdict(sim)
        d["pending"] = d["done"] = None
        pos.sim = d

    async def on_user_event(self, e: Mapping[str, Any]) -> None:
        et = e.get("e")
        if et == "ORDER_TRADE_UPDATE":
            await self._on_order_update(e)
        elif et == "ALGO_UPDATE":
            await self._on_algo_update(e)
        elif et == "ACCOUNT_UPDATE":
            await self._on_account_update(e)

    async def _on_algo_update(self, e: Mapping[str, Any]) -> None:
        o = e["o"]
        pos = self.positions.get(o.get("s", ""))
        if pos is None:
            return
        kind = pos.kind_of(o.get("caid"))
        if kind is None:
            return
        st = o.get("X")
        self.db.execute("UPDATE orders SET status=?, last_response_json=?, ack_at=? WHERE client_id=?",
                        (st, json.dumps(dict(o)), self.now_ms(), o.get("caid")))
        if st in ("TRIGGERING", "TRIGGERED", "FINISHED") and kind in ("HS", "TS"):
            if o.get("ai"):
                pos.algo_order_ids[str(o["ai"])] = kind
            if pos.exit_kind is None:
                pos.exit_kind = kind
                pos.exit_trigger_ts = int(o.get("tt") or e.get("T") or self.now_ms())
                if kind == "HS":
                    pos.exit_trigger_px = float(pos.plan["hard_stop_px"])
                else:
                    sim = pos.sim or {}
                    lvl = TrailingSim(**{**sim, "pending": None, "done": None}).trail_level if sim else None
                    pos.exit_trigger_px = lvl or (float(o["tp"]) if o.get("tp") else self.last_price(pos.symbol))
                pos.state = "EXITING"
            self.save(pos)
        elif st in ("REJECTED", "EXPIRED") and kind in ("HS", "TS") and pos.state in ("PROTECTING", "PROTECTED"):
            self.alert("ochrana zmizela", f"{pos.slot_id}: {kind} {st} {o.get('rm', '')}")
            pos.orders.pop(kind, None)
            await self.ensure_protection(pos, set())

    async def _on_order_update(self, e: Mapping[str, Any]) -> None:
        o = e["o"]
        sym = o.get("s", "")
        pos = self.positions.get(sym)
        if pos is None or o.get("x") != "TRADE":
            return
        cid, oid = o.get("c"), str(o.get("i"))
        kind = pos.kind_of(cid) or pos.algo_order_ids.get(oid)
        if kind is None and o.get("R") and o.get("S") == pos.close_side:
            kind = pos.exit_kind or "EM"   # spuštěný algo příkaz s neznámým client ID: jediný slot na symbolu
        if kind is None:
            return
        tid = str(o.get("t"))
        qty, px, fee = float(o.get("l", 0)), float(o.get("L", 0)), float(o.get("n", 0) or 0)
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tid, cid, pos.trade_id, px, qty, fee, o.get("N"), int(bool(o.get("m"))), int(o.get("T", 0)),
             float(o.get("rp", 0) or 0))).rowcount
        if not inserted:
            return
        if kind == "E":
            pos.entry_fee += fee
            self.save(pos)
            return
        pos.exit_qty += qty
        pos.exit_notional += qty * px
        pos.exit_fee += fee
        pos.realized += float(o.get("rp", 0) or 0)
        pos.exit_time = int(o.get("T", self.now_ms()))
        if pos.exit_kind is None:
            pos.exit_kind = kind
            pos.exit_trigger_px = float(pos.plan["tp_px"]) if kind == "TP" and pos.plan.get("tp_px") else px
            pos.exit_trigger_ts = pos.exit_time
        if pos.exit_kind in ("HS", "TS") and pos.exit_trigger_ts:
            self._latency(pos.trade_id, "trigger_to_fill", pos.exit_time - pos.exit_trigger_ts)
        self.save(pos)
        if Decimal(str(round(pos.exit_qty, 12))) >= pos.qty_d:
            await self._finalize(pos)

    async def _on_account_update(self, e: Mapping[str, Any]) -> None:
        for p in e.get("a", {}).get("P", []):
            pos = self.positions.get(p.get("s", ""))
            if pos is not None and Decimal(str(p.get("pa", "0"))) == 0 and pos.state in ("PROTECTED", "EXITING",
                                                                                        "EMERGENCY_CLOSE"):
                # pozice je pryč; filly dorazí v ORDER_TRADE_UPDATE, sourozenci se mají zrušit hned
                asyncio.ensure_future(self._cancel_siblings(pos, keep=None))

    async def _finalize(self, pos: LivePosition, reason_override: str | None = None) -> TradeRecord:
        await self._cancel_siblings(pos, keep=pos.exit_kind)
        funding = await self._funding(pos)
        q = float(pos.qty_d)
        exit_px = pos.exit_notional / pos.exit_qty if pos.exit_qty else (pos.exit_trigger_px or pos.entry_px)
        sg = side_sign(pos.side)
        notional = q * pos.entry_px
        gross = sg * (exit_px - pos.entry_px) * q
        net = gross - pos.entry_fee - pos.exit_fee + funding
        trig = pos.exit_trigger_px or exit_px
        sim = pos.sim or {}
        rec = TradeRecord(
            trade_id=pos.trade_id, slot_id=pos.slot_id, mode="live", signal_id=pos.signal_id, side=pos.side, qty=q,
            notional=notional, decision_px=pos.decision_px, arrival_mid=pos.arrival_mid, entry_px=pos.entry_px,
            entry_time=pos.entry_time, entry_fee=pos.entry_fee, exit_trigger_px=trig, exit_px=exit_px,
            exit_time=pos.exit_time or self.now_ms(), exit_fee=pos.exit_fee,
            exit_reason=reason_override or EXIT_REASONS.get(pos.exit_kind or "", pos.exit_kind or "?"),
            funding=funding, gross_pnl=gross, net_pnl=net, gross_bps=bps(gross, notional), net_bps=bps(net, notional),
            entry_slip_bps=sg * (pos.entry_px - pos.decision_px) / pos.decision_px * 1e4,
            exit_slip_bps=sg * (trig - exit_px) / trig * 1e4 if trig else 0.0,
            mfe_bps=(sg * (sim["mfe_px"] - pos.entry_px) / pos.entry_px * 1e4) if sim.get("mfe_px") and pos.entry_px else None,
            mae_bps=(sg * (sim["mae_px"] - pos.entry_px) / pos.entry_px * 1e4) if sim.get("mae_px") and pos.entry_px else None,
            bars_held=max(0, ((pos.exit_time or self.now_ms()) - (pos.entry_bar + BAR_MS)) // BAR_MS),
            tainted=0, cost_model_version=self.cost_model_version, config_hash=pos.config_hash,
            twin_trade_id=pos.twin_trade_id, extra={"notes": pos.notes, "realized_pnl": pos.realized},
        )
        self.positions.pop(pos.symbol, None)
        self.save(None, pos.slot_id)
        if self.on_closed is not None:
            self.on_closed(rec, pos)
        return rec

    async def _funding(self, pos: LivePosition) -> float:
        try:
            rows = await self.rest.signed("GET", "/fapi/v1/income", {"symbol": pos.symbol, "incomeType": "FUNDING_FEE",
                                                                     "startTime": pos.entry_time})
            end = pos.exit_time or self.now_ms()
            return float(sum(float(r["income"]) for r in rows if int(r["time"]) <= end))
        except Exception:  # noqa: BLE001 – income poller to dorovná, ledger se porovnává proti income
            return 0.0

    # --- časový výstup a kill ------------------------------------------------

    async def on_bar_close(self, open_time: int, closes: Mapping[str, float]) -> None:
        for pos in list(self.positions.values()):
            if pos.state == "PROTECTED" and pos.plan and \
                    open_time >= pos.entry_bar + int(pos.plan["max_hold_bars"]) * BAR_MS:
                await self.market_exit(pos, "TX", closes.get(pos.symbol))

    async def close_all(self, reason: str) -> None:
        for pos in list(self.positions.values()):
            if pos.state in ("PROTECTING", "PROTECTED"):
                await self.market_exit(pos, "KILL")
        self.db.event("WARN", "live", f"zavřít vše: {reason}")
