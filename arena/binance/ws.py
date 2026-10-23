"""WS spojení s reconnectem, backoffem a preventivní rotací (SPEC §4.1, nástraha 8).

Ping/pong: Binance posílá ping každé 3 min, knihovna `websockets` odpovídá pongem sama. Navíc posílá
vlastní ping každých 20 s, takže mrtvé spojení se pozná rychle."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import websockets

from arena.util import wall_ms

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict[str, Any]], None]
ConnectHandler = Callable[[bool], Awaitable[None]]


def combined_url(base: str, streams: list[str]) -> str:
    """`wss://.../market` + [btcusdt@kline_5m, ...] → combined stream URL (`stream` mód, SPEC §4.1)."""
    return f"{base.rstrip('/')}/stream?streams={'/'.join(streams)}"


@dataclass
class WsConnection:
    """Jedno dlouhodobé WS spojení. `url` může být funkce (např. private stream s čerstvým listenKey).

    `on_message(stream, data)` dostane rozbalený combined payload (u raw spojení je stream = typ eventu).
    `on_connect(reconnected)` se zavolá po každém úspěšném připojení; po reconnectu má volající doplnit mezeru."""

    name: str
    url: str | Callable[[], Awaitable[str]]
    on_message: MessageHandler
    on_connect: ConnectHandler | None = None
    rotate_after_s: float = 23 * 3600
    idle_timeout_s: float = 60.0
    backoff_max_s: float = 60.0

    connected: bool = field(default=False, init=False)
    connected_since_ms: int | None = field(default=None, init=False)
    last_msg_ms: int | None = field(default=None, init=False)
    disconnected_since_ms: int | None = field(default=None, init=False)
    connects: int = field(default=0, init=False)
    _kick: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    def force_reconnect(self) -> None:
        """Umělé odpojení (test reconnectu, rotace listenKey)."""
        self._kick.set()

    async def _resolve_url(self) -> str:
        return self.url if isinstance(self.url, str) else await self.url()

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                url = await self._resolve_url()
                async with websockets.connect(url, open_timeout=15, ping_interval=20, ping_timeout=20,
                                              max_size=8 * 2**20, close_timeout=5) as ws:
                    reconnected = self.connects > 0
                    self.connects += 1
                    self.connected = True
                    self.connected_since_ms = wall_ms()
                    self.disconnected_since_ms = None
                    backoff = 1.0
                    log.info("WS %s připojeno (reconnect=%s)", self.name, reconnected)
                    if self.on_connect is not None:
                        await self.on_connect(reconnected)
                    await self._recv_loop(ws, stop)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 – každá chyba = reconnect
                log.warning("WS %s chyba: %r", self.name, e)
            finally:
                if self.connected:
                    self.disconnected_since_ms = wall_ms()
                self.connected = False
            if stop.is_set():
                break
            delay = min(backoff, self.backoff_max_s) * (0.8 + 0.4 * random.random())
            backoff = min(backoff * 2, self.backoff_max_s)
            if self._kick.is_set():          # vynucený reconnect bez čekání
                self._kick.clear()
                delay = 0.0
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _watch(self, ws: Any, stop: asyncio.Event) -> None:
        """Hlídač spojení: stop, vynucený reconnect, preventivní rotace a idle timeout zavřou socket,
        `async for` ve čtecí smyčce pak skončí. Čtení samo nevytváří future na každou zprávu."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.rotate_after_s
        stop_task = asyncio.ensure_future(stop.wait())
        kick_task = asyncio.ensure_future(self._kick.wait())
        try:
            while True:
                done, _ = await asyncio.wait({stop_task, kick_task}, timeout=min(5.0, self.idle_timeout_s),
                                             return_when=asyncio.FIRST_COMPLETED)
                if done:
                    break
                if loop.time() >= deadline:
                    log.info("WS %s: preventivní rotace", self.name)
                    self._kick.set()
                    break
                last = self.last_msg_ms or self.connected_since_ms or wall_ms()
                if wall_ms() - last > self.idle_timeout_s * 1000:
                    log.warning("WS %s: žádná zpráva %.0f s → reconnect", self.name, self.idle_timeout_s)
                    break
        finally:
            stop_task.cancel()
            kick_task.cancel()
            await ws.close()

    async def _recv_loop(self, ws: Any, stop: asyncio.Event) -> None:
        watcher = asyncio.ensure_future(self._watch(ws, stop))
        try:
            async for raw in ws:
                self.last_msg_ms = wall_ms()
                msg = json.loads(raw)
                if "stream" in msg and "data" in msg:
                    stream, data = msg["stream"], msg["data"]
                else:
                    stream, data = str(msg.get("e", "")), msg
                try:
                    self.on_message(stream, data)
                except Exception:  # noqa: BLE001 – chyba handleru nesmí shodit spojení
                    log.exception("WS %s: chyba handleru pro %s", self.name, stream)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
