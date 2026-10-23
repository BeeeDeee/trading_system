"""Tenký async REST klient pro USDⓈ-M futures (SPEC §6). Žádná trading knihovna (D3)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping
from urllib.parse import quote, urlencode

import httpx

from arena.binance import errors
from arena.binance.signing import Signer
from arena.clock import ServerClock

log = logging.getLogger(__name__)

# Podmíněné typy smí jen na /fapi/v1/algoOrder (D3, nástraha 1).
CONDITIONAL_TYPES = frozenset({"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"})
ORDER_PATHS = frozenset({"/fapi/v1/order", "/fapi/v1/batchOrders"})
# Endpointy, u kterých timeout/5xx znamená neznámý stav příkazu (nikdy slepý retry, nástraha 7).
MUTATING_ORDER_PATHS = frozenset({"/fapi/v1/order", "/fapi/v1/batchOrders", "/fapi/v1/algoOrder",
                                  "/fapi/v1/algoOpenOrders", "/fapi/v1/allOpenOrders"})


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, float):
        # float do burzy nepatří (nástraha 6); povolené jen u ne-cenových parametrů jako callbackRate přes Decimal
        raise TypeError("float parametr: použij Decimal nebo str")
    return str(v)


def encode_params(params: Mapping[str, Any]) -> str:
    """Query string v pořadí vložení, percent-encoded (`%20`, ne `+`). Stejný řetězec se podepisuje."""
    return urlencode([(k, _fmt(v)) for k, v in params.items() if v is not None], quote_via=quote)


@dataclass
class RestClient:
    base_url: str
    signer: Signer | None = None
    clock: ServerClock = field(default_factory=ServerClock)
    recv_window: int = 5000
    timeout_s: float = 10.0
    get_retries: int = 3
    transport: httpx.AsyncBaseTransport | None = None

    used_weight_1m: int | None = field(default=None, init=False)
    order_count: dict[str, int] = field(default_factory=dict, init=False)
    pause_until: float = field(default=0.0, init=False)   # monotonic čas, do kdy čekat (429)
    banned: bool = field(default=False, init=False)        # 418 → live stop

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s, transport=self.transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> RestClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- jádro --------------------------------------------------------------

    async def sync_time(self) -> int:
        t0 = self.clock.local_ms()
        data = await self.public("GET", "/fapi/v1/time")
        t1 = self.clock.local_ms()
        self.clock.update(int(data["serverTime"]), t0, t1)
        log.info("čas burzy: offset %d ms, rtt %d ms", self.clock.offset_ms, self.clock.last_rtt_ms)
        return self.clock.offset_ms

    async def public(self, method: str, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._request(method, path, dict(params or {}), signed=False)

    async def signed(self, method: str, path: str, params: Mapping[str, Any] | None = None,
                     client_id: str | None = None) -> Any:
        """Podepsaný požadavek. `client_id` se uvede v UnknownOrderStatus, aby šlo stav dohledat."""
        if self.signer is None:
            raise errors.ClientGuardError("podepsaný požadavek bez klíče")
        return await self._request(method, path, dict(params or {}), signed=True, client_id=client_id)

    def _guard(self, method: str, path: str, params: Mapping[str, Any]) -> None:
        if method == "POST" and path in ORDER_PATHS:
            orders = params.get("batchOrders")
            if isinstance(orders, str):
                orders = json.loads(orders)
            types = [str(params.get("type", "")).upper()]
            if isinstance(orders, list):
                types += [str(o.get("type", "")).upper() for o in orders]
            bad = CONDITIONAL_TYPES.intersection(types)
            if bad:
                raise errors.ClientGuardError(f"{sorted(bad)} patří na /fapi/v1/algoOrder, ne na {path} (-4120)")

    async def _request(self, method: str, path: str, params: dict[str, Any], *, signed: bool,
                       client_id: str | None = None) -> Any:
        method = method.upper()
        self._guard(method, path, params)
        if self.banned:
            raise errors.IpBanned(418, None, "IP je zabanovaná, požadavky stojí")
        mutating = method != "GET" and path in MUTATING_ORDER_PATHS
        if signed and self.clock.needs_resync():
            await self.sync_time()

        ts_retry_done = False
        attempt = 0
        while True:
            attempt += 1
            wait = self.pause_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            query = encode_params(params)
            if signed:
                assert self.signer is not None
                stamp = f"recvWindow={self.recv_window}&timestamp={self.clock.now_ms()}"
                query = f"{query}&{stamp}" if query else stamp
                query = f"{query}&signature={quote(self.signer.sign(query), safe='')}"
            url = f"{path}?{query}" if query else path
            headers = {"X-MBX-APIKEY": self.signer.api_key} if (signed and self.signer) else {}
            try:
                resp = await self._http.request(method, url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if mutating:
                    raise errors.UnknownOrderStatus(method, path, client_id, type(e).__name__) from e
                if attempt <= self.get_retries:
                    await asyncio.sleep(0.5 * 2 ** (attempt - 1))
                    continue
                raise
            self._track_headers(resp.headers)
            if resp.status_code < 400:
                return resp.json()

            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text
            if resp.status_code >= 500:
                if mutating:
                    raise errors.UnknownOrderStatus(method, path, client_id, f"HTTP {resp.status_code}")
                if attempt <= self.get_retries:
                    await asyncio.sleep(0.5 * 2 ** (attempt - 1))
                    continue
            err = errors.from_response(resp.status_code, body, resp.headers.get("Retry-After"))
            if isinstance(err, errors.IpBanned):
                self.banned = True
                log.error("HTTP 418: IP ban, %s %s", method, path)
            elif isinstance(err, errors.RateLimited):
                backoff = err.retry_after_s if err.retry_after_s is not None else 60.0
                self.pause_until = max(self.pause_until, time.monotonic() + backoff)
                log.warning("rate limit (%s), pauza %.0f s", err.code or err.status, backoff)
            elif isinstance(err, errors.TimestampError) and signed and not ts_retry_done:
                ts_retry_done = True
                await self.sync_time()
                continue
            raise err

    def _track_headers(self, headers: httpx.Headers) -> None:
        for k, v in headers.items():
            kl = k.lower()
            if kl == "x-mbx-used-weight-1m":
                self.used_weight_1m = int(v)
            elif kl.startswith("x-mbx-order-count-"):
                self.order_count[kl.removeprefix("x-mbx-order-count-")] = int(v)

    # --- veřejné endpointy --------------------------------------------------

    async def exchange_info(self) -> dict[str, Any]:
        return await self.public("GET", "/fapi/v1/exchangeInfo")

    async def klines(self, symbol: str, interval: str = "5m", start_ms: int | None = None,
                     end_ms: int | None = None, limit: int = 1500) -> list[list[Any]]:
        return await self.public("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                                           "startTime": start_ms, "endTime": end_ms, "limit": limit})

    async def agg_trades(self, symbol: str, start_ms: int | None = None, end_ms: int | None = None,
                         from_id: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        return await self.public("GET", "/fapi/v1/aggTrades", {"symbol": symbol, "fromId": from_id,
                                                              "startTime": start_ms, "endTime": end_ms, "limit": limit})

    async def depth(self, symbol: str, limit: int = 1000) -> dict[str, Any]:
        return await self.public("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})

    async def funding_info(self) -> list[dict[str, Any]]:
        return await self.public("GET", "/fapi/v1/fundingInfo")

    async def premium_index(self, symbol: str | None = None) -> Any:
        return await self.public("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})

    # --- účet (podepsané, read-only) ---------------------------------------

    async def account(self) -> dict[str, Any]:
        return await self.signed("GET", "/fapi/v3/account")

    async def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return await self.signed("GET", "/fapi/v3/positionRisk", {"symbol": symbol})

    async def commission_rate(self, symbol: str) -> dict[str, Any]:
        return await self.signed("GET", "/fapi/v1/commissionRate", {"symbol": symbol})

    async def position_side_dual(self) -> bool:
        data = await self.signed("GET", "/fapi/v1/positionSide/dual")
        return bool(data["dualSidePosition"])

    async def symbol_config(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return await self.signed("GET", "/fapi/v1/symbolConfig", {"symbol": symbol})

    async def account_config(self) -> dict[str, Any]:
        return await self.signed("GET", "/fapi/v1/accountConfig")

    # --- příkazy (SPEC §6, §8) ---------------------------------------------

    async def new_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """POST /fapi/v1/order. Podmíněné typy odmítne guard (patří na algoOrder). Timeout = UnknownOrderStatus."""
        return await self.signed("POST", "/fapi/v1/order", params, client_id=params.get("newClientOrderId"))

    async def query_order(self, symbol: str, client_id: str) -> dict[str, Any]:
        return await self.signed("GET", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id})

    async def cancel_order(self, symbol: str, client_id: str) -> dict[str, Any]:
        return await self.signed("DELETE", "/fapi/v1/order", {"symbol": symbol, "origClientOrderId": client_id},
                                 client_id=client_id)

    async def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return await self.signed("GET", "/fapi/v1/openOrders", {"symbol": symbol})

    async def new_algo_order(self, params: dict[str, Any]) -> dict[str, Any]:
        p = {"algoType": "CONDITIONAL", **params}
        return await self.signed("POST", "/fapi/v1/algoOrder", p, client_id=p.get("clientAlgoId"))

    async def query_algo_order(self, client_algo_id: str) -> dict[str, Any]:
        return await self.signed("GET", "/fapi/v1/algoOrder", {"clientAlgoId": client_algo_id})

    async def cancel_algo_order(self, client_algo_id: str) -> dict[str, Any]:
        return await self.signed("DELETE", "/fapi/v1/algoOrder", {"clientAlgoId": client_algo_id},
                                 client_id=client_algo_id)

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        data = await self.signed("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol})
        return data.get("orders", data) if isinstance(data, dict) else data

    async def user_trades(self, symbol: str, start_ms: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        return await self.signed("GET", "/fapi/v1/userTrades", {"symbol": symbol, "startTime": start_ms,
                                                               "limit": limit})

    async def income(self, start_ms: int | None = None, income_type: str | None = None,
                     limit: int = 1000) -> list[dict[str, Any]]:
        return await self.signed("GET", "/fapi/v1/income", {"incomeType": income_type, "startTime": start_ms,
                                                           "limit": limit})

    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return await self.signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict[str, Any]:
        return await self.signed("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})

    async def listen_key(self) -> str:
        data = await self.signed("POST", "/fapi/v1/listenKey")
        return str(data["listenKey"])

    async def keepalive_listen_key(self) -> None:
        await self.signed("PUT", "/fapi/v1/listenKey")
