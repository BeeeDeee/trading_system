"""Mapování chyb Binance na výjimky (SPEC §6)."""

from __future__ import annotations

from typing import Any


class BinanceError(Exception):
    """Obecná chyba z API. `code` je Binance kód (záporné číslo), `status` HTTP status."""

    def __init__(self, status: int, code: int | None, msg: str, payload: Any = None):
        super().__init__(f"HTTP {status} code={code}: {msg}")
        self.status = status
        self.code = code
        self.msg = msg
        self.payload = payload


class TimestampError(BinanceError):
    """-1021: timestamp mimo recvWindow. Klient resyncne čas a zkusí to jednou znovu."""


class WouldImmediatelyTrigger(BinanceError):
    """-2021: podmíněný příkaz by se spustil hned (např. activatePrice na špatné straně)."""


class ReduceOnlyRejected(BinanceError):
    """-2022: reduceOnly odmítnut, pozice už je pryč. Volající rekonciliuje."""


class WrongEndpoint(BinanceError):
    """-4120 STOP_ORDER_SWITCH_ALGO: podmíněný příkaz poslán na /fapi/v1/order. Je to bug."""


class RateLimited(BinanceError):
    """HTTP 429 nebo -1003. `retry_after_s` z hlavičky Retry-After, pokud přišla."""

    def __init__(self, status: int, code: int | None, msg: str, retry_after_s: float | None, payload: Any = None):
        super().__init__(status, code, msg, payload)
        self.retry_after_s = retry_after_s


class IpBanned(BinanceError):
    """HTTP 418: IP ban. Live se zastaví a jde alert."""


class ListenKeyMissing(BinanceError):
    """-1125: listenKey neexistuje, je potřeba nový."""


class UnknownOrderStatus(Exception):
    """Timeout, 5xx nebo 503 „Unknown error“ u příkazu: stav je neznámý.

    Nikdy neopakovat naslepo. Dotázat se podle client ID (`origClientOrderId` / `clientAlgoId`)."""

    def __init__(self, method: str, path: str, client_id: str | None, cause: str):
        super().__init__(f"{method} {path} client_id={client_id}: stav neznámý ({cause})")
        self.method = method
        self.path = path
        self.client_id = client_id
        self.cause = cause


class ClientGuardError(Exception):
    """Klient odmítl odeslat požadavek, který by byl chybou (např. podmíněný typ na /fapi/v1/order)."""


_CODE_MAP: dict[int, type[BinanceError]] = {
    -1021: TimestampError,
    -2021: WouldImmediatelyTrigger,
    -2022: ReduceOnlyRejected,
    -4120: WrongEndpoint,
    -1125: ListenKeyMissing,
}


def from_response(status: int, body: Any, retry_after: str | None) -> BinanceError:
    code: int | None = None
    msg = ""
    if isinstance(body, dict):
        raw_code = body.get("code")
        code = int(raw_code) if isinstance(raw_code, (int, str)) and str(raw_code).lstrip("-").isdigit() else None
        msg = str(body.get("msg", ""))
    else:
        msg = str(body)[:500]
    if status == 418:
        return IpBanned(status, code, msg, body)
    if status == 429 or code == -1003:
        ra = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
        return RateLimited(status, code, msg, ra, body)
    cls = _CODE_MAP.get(code or 0, BinanceError)
    return cls(status, code, msg, body)
