"""Alerty přes ntfy (D8). Téma z proměnné prostředí (`alerts.ntfy_topic_env`, default NTFY_TOPIC)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)


@dataclass
class Alerter:
    topic: str | None
    server: str = "https://ntfy.sh"
    min_interval_s: float = 60.0          # stejná zpráva max. jednou za minutu
    _last: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_name: str = "NTFY_TOPIC") -> Alerter:
        return cls(os.environ.get(env_name) or None)

    def send(self, title: str, msg: str, priority: str = "high") -> None:
        """Neblokující odeslání. Bez tématu se jen zaloguje."""
        log.warning("ALERT %s: %s", title, msg)
        if not self.topic:
            return
        key = f"{title}|{msg}"
        now = time.monotonic()
        if now - self._last.get(key, -1e9) < self.min_interval_s:
            return
        self._last[key] = now
        try:
            asyncio.get_running_loop().create_task(self._post(title, msg, priority))
        except RuntimeError:
            asyncio.run(self._post(title, msg, priority))

    async def _post(self, title: str, msg: str, priority: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{self.server}/{self.topic}", content=msg.encode(),
                             headers={"Title": f"arena5m: {title}", "Priority": priority, "Tags": "warning"})
        except Exception as e:  # noqa: BLE001
            log.warning("ntfy selhal: %r", e)
