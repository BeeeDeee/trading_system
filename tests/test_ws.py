"""WsConnection proti lokálnímu serveru: doručení combined streamu a vynucený reconnect."""

import asyncio
import json

import websockets

from arena.binance.ws import WsConnection, combined_url


def test_combined_url():
    assert combined_url("wss://x/market/", ["a@kline_5m", "b@aggTrade"]) == \
        "wss://x/market/stream?streams=a@kline_5m/b@aggTrade"


async def test_messages_and_forced_reconnect():
    async def handler(ws):
        i = 0
        try:
            while True:
                await ws.send(json.dumps({"stream": "btcusdt@aggTrade", "data": {"a": i}}))
                i += 1
                await asyncio.sleep(0.01)
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        got, connects = [], []

        async def on_connect(reconnected):
            connects.append(reconnected)

        conn = WsConnection("t", f"ws://127.0.0.1:{port}", lambda s, d: got.append((s, d)), on_connect)
        stop = asyncio.Event()
        task = asyncio.create_task(conn.run(stop))
        await asyncio.sleep(0.2)
        conn.force_reconnect()
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(task, 3)
    assert connects == [False, True]
    assert got and got[0][0] == "btcusdt@aggTrade"


async def test_idle_timeout_reconnects():
    async def handler(ws):
        await ws.send(json.dumps({"stream": "x", "data": {}}))
        await asyncio.sleep(5)       # pak ticho

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connects = []

        async def on_connect(reconnected):
            connects.append(reconnected)

        conn = WsConnection("t", f"ws://127.0.0.1:{port}", lambda s, d: None, on_connect, idle_timeout_s=0.3)
        stop = asyncio.Event()
        task = asyncio.create_task(conn.run(stop))
        await asyncio.sleep(2.5)
        stop.set()
        await asyncio.wait_for(task, 3)
    assert connects[:2] == [False, True]


async def test_idle_timeout_measured_from_current_connection():
    """Inspekce 2026-10-25: po prvním idle timeoutu se `last_msg_ms` z minulého spojení nevynuloval a tichý private
    stream se pak odpojoval každých ~13 s. Nové spojení musí mít celý idle timeout."""
    async def handler(ws):
        if not hasattr(handler, "n"):
            handler.n = 0
        handler.n += 1
        if handler.n == 1:
            await ws.send(json.dumps({"stream": "x", "data": {}}))
        await asyncio.sleep(10)       # pak ticho

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connects = []

        async def on_connect(reconnected):
            connects.append(reconnected)

        conn = WsConnection("t", f"ws://127.0.0.1:{port}", lambda s, d: None, on_connect, idle_timeout_s=1.0,
                            backoff_max_s=0.05, check_every_s=0.1)   # kontrola častěji než timeout, jako v provozu
        stop = asyncio.Event()
        task = asyncio.create_task(conn.run(stop))
        await asyncio.sleep(1.5)      # 1. spojení: zpráva, pak > 1 s ticha → reconnect
        assert len(connects) == 2
        await asyncio.sleep(0.4)      # 2. spojení je tiché < 1 s: nesmí se hned odpojit (dřív za 0,1 s)
        assert len(connects) == 2
        stop.set()
        await asyncio.wait_for(task, 3)
