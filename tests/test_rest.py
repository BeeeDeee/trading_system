"""REST klient s mockovaným transportem: podpis, guard algo endpointu, mapování chyb, neznámý stav."""

import base64
from decimal import Decimal
from urllib.parse import unquote

import httpx
import pytest

from arena.binance import errors
from arena.binance.rest import RestClient, encode_params
from arena.binance.signing import Ed25519Signer
from arena.clock import ServerClock
from tests.test_signing import DOCS_ED25519_PEM


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_700_000_000_000})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(responses, signer=True):
    rec = Recorder(responses)
    clock = ServerClock(local_ms=lambda: 1_700_000_000_000)
    s = Ed25519Signer.from_pem("APIKEY", DOCS_ED25519_PEM) if signer else None
    return RestClient("https://x.test", signer=s, clock=clock, transport=httpx.MockTransport(rec)), rec


def _non_time(rec):
    return [r for r in rec.requests if r.url.path != "/fapi/v1/time"]


async def test_signed_request_signature_verifies_and_is_last():
    client, rec = make([httpx.Response(200, json={"ok": 1})])
    await client.signed("GET", "/fapi/v3/account", {"symbol": "BTCUSDT"})
    req = _non_time(rec)[0]
    q = req.url.query.decode()
    assert req.headers["X-MBX-APIKEY"] == "APIKEY"
    payload, sig = q.rsplit("&signature=", 1)
    assert payload.startswith("symbol=BTCUSDT&recvWindow=5000&timestamp=")
    key = Ed25519Signer.from_pem("k", DOCS_ED25519_PEM).private_key.public_key()
    key.verify(base64.b64decode(unquote(sig)), payload.encode())   # vyhodí při neplatném podpisu


async def test_trailing_stop_on_order_endpoint_fails_without_sending():
    """Nástraha 1: TRAILING_STOP_MARKET na /fapi/v1/order musí selhat (ne se odeslat)."""
    client, rec = make([])
    for t in ("TRAILING_STOP_MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"):
        with pytest.raises(errors.ClientGuardError):
            await client.signed("POST", "/fapi/v1/order", {"symbol": "BTCUSDT", "type": t})
    with pytest.raises(errors.ClientGuardError):
        await client.signed("POST", "/fapi/v1/batchOrders",
                            {"batchOrders": '[{"type":"MARKET"},{"type":"TRAILING_STOP_MARKET"}]'})
    assert _non_time(rec) == []


async def test_algo_endpoint_accepts_trailing():
    client, rec = make([httpx.Response(200, json={"algoId": 1})])
    await client.signed("POST", "/fapi/v1/algoOrder", {"algoType": "CONDITIONAL", "type": "TRAILING_STOP_MARKET",
                                                       "callbackRate": Decimal("0.5")})
    assert "callbackRate=0.5" in _non_time(rec)[0].url.query.decode()


async def test_minus_4120_maps_to_wrong_endpoint():
    client, _ = make([httpx.Response(400, json={"code": -4120, "msg": "STOP_ORDER_SWITCH_ALGO"})])
    with pytest.raises(errors.WrongEndpoint):
        await client.signed("POST", "/fapi/v1/order", {"type": "MARKET"})


@pytest.mark.parametrize("code,cls", [(-2021, errors.WouldImmediatelyTrigger), (-2022, errors.ReduceOnlyRejected),
                                      (-1125, errors.ListenKeyMissing), (-2019, errors.BinanceError)])
async def test_error_mapping(code, cls):
    client, _ = make([httpx.Response(400, json={"code": code, "msg": "x"})])
    with pytest.raises(cls):
        await client.signed("POST", "/fapi/v1/algoOrder", {"type": "STOP_MARKET"})


async def test_timestamp_error_resyncs_and_retries_once():
    client, rec = make([httpx.Response(400, json={"code": -1021, "msg": "ts"}),
                        httpx.Response(400, json={"code": -1021, "msg": "ts"})])
    with pytest.raises(errors.TimestampError):
        await client.signed("GET", "/fapi/v3/account")
    assert len(_non_time(rec)) == 2
    assert sum(1 for r in rec.requests if r.url.path == "/fapi/v1/time") == 2   # úvodní sync + resync


async def test_timeout_on_order_is_unknown_status_and_never_retried():
    client, rec = make([httpx.ReadTimeout("t")])
    with pytest.raises(errors.UnknownOrderStatus) as ei:
        await client.signed("POST", "/fapi/v1/order", {"type": "MARKET"}, client_id="a5-ABCDEF-1-E")
    assert ei.value.client_id == "a5-ABCDEF-1-E"
    assert len(_non_time(rec)) == 1


async def test_5xx_on_algo_order_is_unknown_status():
    client, rec = make([httpx.Response(503, json={"code": -1000, "msg": "Unknown error"})])
    with pytest.raises(errors.UnknownOrderStatus):
        await client.signed("POST", "/fapi/v1/algoOrder", {"type": "STOP_MARKET"})
    assert len(_non_time(rec)) == 1


async def test_get_retries_on_5xx(monkeypatch):
    async def nosleep(_):
        return None
    monkeypatch.setattr("arena.binance.rest.asyncio.sleep", nosleep)
    client, rec = make([httpx.Response(502, text="bad"), httpx.Response(200, json=[1])], signer=False)
    assert await client.public("GET", "/fapi/v1/klines", {"symbol": "BTCUSDT"}) == [1]
    assert len(rec.requests) == 2


async def test_429_sets_pause_and_418_bans():
    client, _ = make([httpx.Response(429, json={"code": -1003, "msg": "slow"}, headers={"Retry-After": "7"})],
                     signer=False)
    with pytest.raises(errors.RateLimited) as ei:
        await client.public("GET", "/fapi/v1/exchangeInfo")
    assert ei.value.retry_after_s == 7 and client.pause_until > 0

    client, rec = make([httpx.Response(418, json={"code": -1003, "msg": "banned"})], signer=False)
    with pytest.raises(errors.IpBanned):
        await client.public("GET", "/fapi/v1/exchangeInfo")
    assert client.banned
    with pytest.raises(errors.IpBanned):
        await client.public("GET", "/fapi/v1/exchangeInfo")
    assert len(rec.requests) == 1


async def test_weight_headers_tracked():
    client, _ = make([httpx.Response(200, json={}, headers={"X-MBX-USED-WEIGHT-1M": "42",
                                                            "X-MBX-ORDER-COUNT-10S": "3"})], signer=False)
    await client.public("GET", "/fapi/v1/exchangeInfo")
    assert client.used_weight_1m == 42 and client.order_count == {"10s": 3}


def test_encode_params_decimal_bool_and_no_float():
    assert encode_params({"q": Decimal("0.0010"), "r": True, "s": None, "id": "a5-X:Y/1"}) == \
        "q=0.0010&r=true&id=a5-X%3AY%2F1"
    with pytest.raises(TypeError):
        encode_params({"price": 1.5})
