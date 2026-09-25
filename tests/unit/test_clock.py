from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from scout.utils.clock import BarClock, Clock, WallClock
from scout.utils.errors import ScoutError


def test_bar_clock_returns_set_time() -> None:
    ts = datetime(2020, 6, 15, 20, 0, tzinfo=UTC)
    clock = BarClock()
    clock.set(ts)
    assert clock.now() is ts


def test_bar_clock_accepts_initial_ts() -> None:
    ts = datetime(2019, 1, 2, 21, 0, tzinfo=UTC)
    clock = BarClock(ts)
    assert clock.now() == ts


def test_bar_clock_now_before_set_raises() -> None:
    clock = BarClock()
    with pytest.raises(ScoutError, match="before set"):
        clock.now()


def test_bar_clock_rejects_naive_datetime() -> None:
    clock = BarClock()
    with pytest.raises(ScoutError, match="timezone-aware UTC"):
        clock.set(datetime(2020, 1, 1, 12, 0))  # noqa: DTZ001


def test_bar_clock_rejects_non_utc() -> None:
    clock = BarClock()
    eastern = timezone(timedelta(hours=-5))
    with pytest.raises(ScoutError, match="timezone-aware UTC"):
        clock.set(datetime(2020, 1, 1, 12, 0, tzinfo=eastern))


def test_bar_clock_satisfies_protocol() -> None:
    clock = BarClock(datetime(2020, 1, 1, tzinfo=UTC))
    assert isinstance(clock, Clock)


def test_wall_clock_is_aware_utc() -> None:
    now = WallClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_wall_clock_satisfies_protocol() -> None:
    assert isinstance(WallClock(), Clock)
