from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scout.data.calendar import build_calendar
from scout.data.schemas import CALENDAR_COLUMNS
from scout.utils.errors import ScoutDataError

# Independently known NYSE full-day closures (not derived from exchange_calendars).
NYSE_HOLIDAYS_2008 = (
    date(2008, 1, 1),  # New Year's Day
    date(2008, 1, 21),  # Martin Luther King Jr. Day
    date(2008, 2, 18),  # Washington's Birthday
    date(2008, 3, 21),  # Good Friday
    date(2008, 5, 26),  # Memorial Day
    date(2008, 7, 4),  # Independence Day
    date(2008, 9, 1),  # Labor Day
    date(2008, 11, 27),  # Thanksgiving
    date(2008, 12, 25),  # Christmas
)

SANDY_CLOSED = (date(2012, 10, 29), date(2012, 10, 30))
SANDY_FRIDAY = date(2012, 10, 26)
SANDY_REOPEN = date(2012, 10, 31)

# Day before Independence Day 2008: early close 1:00 pm ET = 17:00 UTC.
HALF_DAY_2008 = date(2008, 7, 3)
FULL_DAY_BEFORE_HALF = date(2008, 7, 2)

THANKSGIVING_2012 = date(2012, 11, 22)
WED_BEFORE_THANKSGIVING_2012 = date(2012, 11, 21)
FRIDAY_AFTER_THANKSGIVING_2012 = date(2012, 11, 23)
MONDAY_AFTER_THANKSGIVING_2012 = date(2012, 11, 26)


def _sessions(frame: pd.DataFrame) -> set[date]:
    return {pd.Timestamp(s).date() for s in frame["session"]}


def _row(frame: pd.DataFrame, session: date) -> pd.Series:
    matched = frame.loc[pd.to_datetime(frame["session"]).dt.date == session]
    assert len(matched) == 1, f"expected one row for {session}"
    return matched.iloc[0]


def test_calendar_columns_match_schema() -> None:
    frame = build_calendar(date(2008, 7, 1), date(2008, 7, 7))
    assert list(frame.columns) == list(CALENDAR_COLUMNS)


def test_2008_nyse_holidays_are_absent() -> None:
    frame = build_calendar(date(2008, 1, 1), date(2008, 12, 31))
    present = _sessions(frame)
    for holiday in NYSE_HOLIDAYS_2008:
        assert holiday not in present, f"{holiday} should be a NYSE holiday"
    # Nearby weekdays that must still be sessions.
    assert date(2008, 1, 2) in present
    assert date(2008, 7, 3) in present
    assert date(2008, 11, 26) in present
    assert date(2008, 12, 24) in present


def test_2012_hurricane_sandy_closure() -> None:
    frame = build_calendar(date(2012, 10, 1), date(2012, 11, 2))
    present = _sessions(frame)
    for closed in SANDY_CLOSED:
        assert closed not in present
    assert SANDY_FRIDAY in present
    assert SANDY_REOPEN in present
    friday = _row(frame, SANDY_FRIDAY)
    reopen = _row(frame, SANDY_REOPEN)
    # Friday 26th → Wednesday 31st is one session step, not three calendar days.
    assert int(reopen["session_index"]) - int(friday["session_index"]) == 1


def test_half_day_is_a_session() -> None:
    frame = build_calendar(date(2008, 7, 1), date(2008, 7, 7))
    half = _row(frame, HALF_DAY_2008)
    full = _row(frame, FULL_DAY_BEFORE_HALF)
    assert bool(half["is_half_day"]) is True
    assert bool(full["is_half_day"]) is False
    assert int(half["session_index"]) - int(full["session_index"]) == 1
    close = pd.Timestamp(half["close_utc"])
    assert close.hour == 17
    assert close.minute == 0
    assert str(close.tz) == "UTC"


def test_session_index_skips_holidays() -> None:
    frame = build_calendar(date(2012, 11, 19), date(2012, 11, 26))
    present = _sessions(frame)
    assert THANKSGIVING_2012 not in present
    assert WED_BEFORE_THANKSGIVING_2012 in present
    assert FRIDAY_AFTER_THANKSGIVING_2012 in present
    assert MONDAY_AFTER_THANKSGIVING_2012 in present
    wed = _row(frame, WED_BEFORE_THANKSGIVING_2012)
    fri = _row(frame, FRIDAY_AFTER_THANKSGIVING_2012)
    monday = _row(frame, MONDAY_AFTER_THANKSGIVING_2012)
    # Wednesday → Friday skips Thanksgiving; Friday → Monday skips the weekend.
    assert int(fri["session_index"]) - int(wed["session_index"]) == 1
    assert int(monday["session_index"]) - int(fri["session_index"]) == 1
    assert bool(fri["is_half_day"]) is True


def test_close_utc_is_dst_correct() -> None:
    frame = build_calendar(date(2008, 1, 2), date(2008, 7, 2))
    winter = _row(frame, date(2008, 1, 2))
    summer = _row(frame, date(2008, 7, 2))
    winter_close = pd.Timestamp(winter["close_utc"])
    summer_close = pd.Timestamp(summer["close_utc"])
    # 16:00 America/New_York: 21:00 UTC in EST, 20:00 UTC in EDT.
    assert winter_close.hour == 21
    assert summer_close.hour == 20
    assert str(winter_close.tz) == "UTC"
    assert str(summer_close.tz) == "UTC"


def test_unknown_calendar_raises() -> None:
    with pytest.raises(ScoutDataError, match="unknown exchange calendar"):
        build_calendar(date(2008, 1, 1), date(2008, 1, 3), calendar_code="NOT_A_CAL")


def test_end_before_start_raises() -> None:
    with pytest.raises(ScoutDataError, match="end must be on or after start"):
        build_calendar(date(2008, 1, 3), date(2008, 1, 1))
