"""Čas burzy: lokální hodiny + offset z `GET /fapi/v1/time` (SPEC §6, nástraha 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from arena.util import wall_ms


@dataclass
class ServerClock:
    """Offset = serverTime − střed lokálního okna požadavku. Přepočítává se každých `resync_every_ms`."""

    resync_every_ms: int = 10 * 60 * 1000
    local_ms: Callable[[], int] = wall_ms
    offset_ms: int = 0
    last_sync_ms: int | None = field(default=None)
    last_rtt_ms: int | None = field(default=None)

    def now_ms(self) -> int:
        return self.local_ms() + self.offset_ms

    def needs_resync(self) -> bool:
        return self.last_sync_ms is None or self.local_ms() - self.last_sync_ms >= self.resync_every_ms

    def update(self, server_ms: int, sent_local_ms: int, recv_local_ms: int) -> None:
        mid = (sent_local_ms + recv_local_ms) // 2
        self.offset_ms = server_ms - mid
        self.last_rtt_ms = recv_local_ms - sent_local_ms
        self.last_sync_ms = recv_local_ms
