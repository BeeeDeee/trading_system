"""XNYS session grid via exchange_calendars. Bar arithmetic uses session_index."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from exchange_calendars.errors import InvalidCalendarName  # type: ignore[import-untyped]

from scout.data.schemas import CALENDAR_COLUMNS
from scout.data.store import write_parquet_atomic
from scout.utils.errors import ScoutDataError

DEFAULT_CALENDAR = "XNYS"


def reference_calendar_path(processed_dir: Path, calendar_code: str = DEFAULT_CALENDAR) -> Path:
    return processed_dir.parent / "reference" / f"calendar_{calendar_code.lower()}.parquet"


def build_calendar(
    start: date,
    end: date,
    *,
    calendar_code: str = DEFAULT_CALENDAR,
) -> pd.DataFrame:
    """One row per trading session. `close_utc` is the panel `ts`."""
    if end < start:
        raise ScoutDataError("calendar end must be on or after start")
    try:
        cal = xcals.get_calendar(calendar_code)
    except InvalidCalendarName as exc:
        raise ScoutDataError(
            f"unknown exchange calendar {calendar_code!r}; "
            f"known names include 'XNYS'"
        ) from exc
    index = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(index) == 0:
        return pd.DataFrame(columns=list(CALENDAR_COLUMNS))
    opens = cal.opens.loc[index]
    closes = cal.closes.loc[index]
    early = pd.DatetimeIndex(cal.early_closes)
    frame = pd.DataFrame(
        {
            "session": pd.Series(index.date, dtype="object"),
            "open_utc": opens.to_numpy(),
            "close_utc": closes.to_numpy(),
            "is_half_day": index.isin(early),
            "session_index": np.arange(len(index), dtype=np.int32),
        }
    )
    frame = frame.assign(
        open_utc=pd.to_datetime(frame["open_utc"], utc=True),
        close_utc=pd.to_datetime(frame["close_utc"], utc=True),
        is_half_day=frame["is_half_day"].astype(bool),
        session_index=frame["session_index"].astype("int32"),
    )
    return frame.loc[:, list(CALENDAR_COLUMNS)].reset_index(drop=True)


def write_calendar(path: Path, frame: pd.DataFrame) -> None:
    missing = [c for c in CALENDAR_COLUMNS if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"calendar missing columns: {missing}")
    write_parquet_atomic(path, frame.loc[:, list(CALENDAR_COLUMNS)])
