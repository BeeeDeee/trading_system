from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from scout.utils.errors import ScoutError


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class BarClock:
    """Backtest clock. `now()` returns the current bar's close time."""

    def __init__(self, ts: datetime | None = None) -> None:
        self._ts: datetime | None = None
        if ts is not None:
            self.set(ts)

    def set(self, ts: datetime) -> None:
        if ts.tzinfo is None or ts.utcoffset() != timedelta(0):
            raise ScoutError("BarClock requires a timezone-aware UTC datetime")
        self._ts = ts

    def now(self) -> datetime:
        if self._ts is None:
            raise ScoutError("BarClock.now() called before set()")
        return self._ts


class WallClock:
    """Live clock. The only place datetime.now(UTC) appears."""

    def now(self) -> datetime:
        return datetime.now(UTC)
