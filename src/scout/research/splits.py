"""Canonical data-split constants. The holdout range is not a YAML knob."""

from __future__ import annotations

from datetime import UTC, datetime

WARMUP_START = datetime(1998, 1, 1, tzinfo=UTC)
WARMUP_END = datetime(2004, 1, 1, tzinfo=UTC)  # exclusive of trading
DEVELOPMENT_START = datetime(2004, 1, 1, tzinfo=UTC)
DEVELOPMENT_END = datetime(2017, 12, 31, 23, 59, 59, tzinfo=UTC)
HOLDOUT_START = datetime(2018, 1, 1, tzinfo=UTC)


def period_overlaps_holdout(start: datetime, end: datetime) -> bool:
    """True when [start, end] intersects [HOLDOUT_START, +inf)."""
    return end >= HOLDOUT_START and start <= end
