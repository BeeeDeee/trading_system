"""Falešná USDⓈ-M burza pro testy live executoru a rekonciliace (jen metody, které kód používá)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.binance import errors
from arena.binance.rest import CONDITIONAL_TYPES


@dataclass
class FakeExchange:
    price: dict[str, Decimal] = field(default_factory=lambda: {"XRPUSDT": Decimal("0.5000")})
    positions: dict[str, Decimal] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)       # clientOrderId → order
    algos: dict[str, dict[str, Any]] = field(default_factory=dict)        # clientAlgoId → algo
    user_trades_log: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    # injekce chyb
    algo_fail: dict[str, list[Exception]] = field(default_factory=dict)   # typ → fronta výjimek
    order_fail: list[Exception] = field(default_factory=list)
    entry_timeout_but_filled: bool = False
    no_avg_price: bool = False        # demo/testnet: MARKET RESULT bez avgPrice, price "0.00"
    events: list[dict[str, Any]] = field(default_factory=list)            # ORDER_TRADE_UPDATE market fillů (WS)
    _ids: Any = field(default_factory=lambda: itertools.count(1000))

    # --- pomocné ------------------------------------------------------------

    def _fill(self, sym: str, side: str, qty: Decimal, cid: str, oid: int, reduce: bool = False) -> dict[str, Any]:
        px = self.price[sym]
        delta = qty if side == "BUY" else -qty
        cur = self.positions.get(sym, Decimal(0))
        if reduce:
            if cur == 0 or (cur > 0) == (delta > 0):
                raise errors.ReduceOnlyRejected(400, -2022, "ReduceOnly Order is rejected.")
            qty = min(qty, abs(cur))
            delta = -qty if cur > 0 else qty
        self.positions[sym] = cur + delta
        tid = next(self._ids)
        t = {"id": tid, "orderId": oid, "symbol": sym, "side": side, "qty": str(qty), "price": str(px),
             "commission": str(qty * px * Decimal("0.0005")), "commissionAsset": "USDT", "time": 1_792_000_000_000 + tid,
             "realizedPnl": "0", "maker": False}
        self.user_trades_log.append(t)
        return {"orderId": oid, "clientOrderId": cid, "status": "FILLED", "executedQty": str(qty), "avgPrice": str(px),
                "updateTime": t["time"], "_trade": t}

    def otu(self, order: dict[str, Any], trade: dict[str, Any], reduce: bool) -> dict[str, Any]:
        """ORDER_TRADE_UPDATE pro fill."""
        return {"e": "ORDER_TRADE_UPDATE", "T": trade["time"], "o": {
            "s": trade["symbol"], "c": order.get("clientOrderId", ""), "i": trade["orderId"], "x": "TRADE",
            "X": "FILLED", "S": trade["side"], "l": trade["qty"], "L": trade["price"], "n": trade["commission"],
            "N": "USDT", "T": trade["time"], "t": trade["id"], "rp": "0", "R": reduce, "m": False}}

    def trigger_algo(self, cid: str) -> list[dict[str, Any]]:
        """Spustí algo příkaz: vrátí události ALGO_UPDATE (TRIGGERED s `ai`) a ORDER_TRADE_UPDATE fillu."""
        a = self.algos.pop(cid)
        oid = next(self._ids)
        res = self._fill(a["symbol"], a["side"], Decimal(str(a["quantity"])), "", oid, reduce=True)
        algo_ev = {"e": "ALGO_UPDATE", "T": res["updateTime"], "o": {
            "caid": cid, "aid": a["algoId"], "s": a["symbol"], "X": "TRIGGERED", "ai": str(oid), "tt": res["updateTime"],
            "o": a["type"], "tp": str(a.get("triggerPrice", ""))}}
        return [algo_ev, self.otu({"clientOrderId": f"system-{oid}"}, res["_trade"], True)]

    # --- REST rozhraní ------------------------------------------------------

    async def new_order(self, p: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("POST /fapi/v1/order", dict(p)))
        if str(p.get("type")) in CONDITIONAL_TYPES:
            raise errors.WrongEndpoint(400, -4120, "STOP_ORDER_SWITCH_ALGO")
        if self.order_fail:
            raise self.order_fail.pop(0)
        cid, oid = p["newClientOrderId"], next(self._ids)
        reduce = str(p.get("reduceOnly")).lower() == "true"
        if p["type"] == "MARKET":
            res = self._fill(p["symbol"], p["side"], Decimal(str(p["quantity"])), cid, oid, reduce)
            self.orders[cid] = {**res, "symbol": p["symbol"], "type": "MARKET", "open": False}
            self.events.append(self.otu({"clientOrderId": cid}, res["_trade"], reduce))
            if self.entry_timeout_but_filled:
                self.entry_timeout_but_filled = False
                raise errors.UnknownOrderStatus("POST", "/fapi/v1/order", cid, "ReadTimeout")
            out = {k: v for k, v in res.items() if k != "_trade"}
            if self.no_avg_price:
                out.pop("avgPrice"); out["price"] = "0.00"
            return out
        self.orders[cid] = {"orderId": oid, "clientOrderId": cid, "symbol": p["symbol"], "type": p["type"],
                            "status": "NEW", "executedQty": "0", "open": True, "price": str(p.get("price"))}
        return dict(self.orders[cid])

    async def query_order(self, symbol: str, cid: str) -> dict[str, Any]:
        o = self.orders.get(cid)
        if o is None:
            raise errors.BinanceError(400, -2013, "Order does not exist.")
        return {k: v for k, v in o.items() if k != "_trade"}

    async def cancel_order(self, symbol: str, cid: str) -> dict[str, Any]:
        self.calls.append(("DELETE /fapi/v1/order", {"cid": cid}))
        o = self.orders.get(cid)
        if o is None or not o.get("open"):
            raise errors.BinanceError(400, -2011, "Unknown order sent.")
        o["open"], o["status"] = False, "CANCELED"
        return o

    async def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [dict(o) for o in self.orders.values() if o.get("open")]

    async def new_algo_order(self, p: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("POST /fapi/v1/algoOrder", dict(p)))
        q = self.algo_fail.get(p["type"])
        if q:
            raise q.pop(0)
        if p["type"] == "TRAILING_STOP_MARKET" and "activatePrice" in p:
            ap, last = Decimal(str(p["activatePrice"])), self.price[p["symbol"]]
            if (p["side"] == "SELL" and ap <= last) or (p["side"] == "BUY" and ap >= last):
                raise errors.WouldImmediatelyTrigger(400, -2021, "Order would immediately trigger.")
        cid = p["clientAlgoId"]
        if cid in self.algos:
            raise errors.BinanceError(400, -4116, "ClientOrderId is duplicated.")
        a = {"algoId": next(self._ids), "clientAlgoId": cid, "symbol": p["symbol"], "side": p["side"],
             "type": p["type"], "quantity": str(p["quantity"]), "algoStatus": "NEW",
             "triggerPrice": str(p.get("triggerPrice", "")), "reduceOnly": True}
        self.algos[cid] = a
        return dict(a)

    async def query_algo_order(self, cid: str) -> dict[str, Any]:
        a = self.algos.get(cid)
        if a is None:
            raise errors.BinanceError(400, -2013, "Order does not exist.")
        return dict(a)

    async def cancel_algo_order(self, cid: str) -> dict[str, Any]:
        self.calls.append(("DELETE /fapi/v1/algoOrder", {"cid": cid}))
        a = self.algos.pop(cid, None)
        if a is None:
            raise errors.BinanceError(400, -2011, "Unknown order sent.")
        return {"algoId": a["algoId"], "clientAlgoId": cid}

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [dict(a) for a in self.algos.values()]

    async def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [{"symbol": s, "positionAmt": str(a)} for s, a in self.positions.items() if a != 0]

    async def user_trades(self, symbol: str, start_ms: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        return [t for t in self.user_trades_log if t["symbol"] == symbol and (start_ms is None or t["time"] >= start_ms)]

    async def signed(self, method: str, path: str, params: dict[str, Any] | None = None, client_id: str | None = None):
        if path == "/fapi/v1/income":
            return []
        raise NotImplementedError(path)
