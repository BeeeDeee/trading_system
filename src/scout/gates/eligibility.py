"""Ordered decision-time gates. First failure is the recorded reason.

Owns 14-CONFIG.md §5 rows 1-8. Rows 9-10 (regime) are the engine's.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import GatesConfig
from scout.domain.enums import Direction, RejectionReason
from scout.domain.features import FeatureRow
from scout.domain.market import EarningsEvent
from scout.domain.universe import UniverseEntry

# ADR-020: when available_ts is missing, treat the print as a ±N session band
# on the XNYS grid. Not a YAML knob — widening it is a research decision.
EARNINGS_UNCERTAINTY_SESSIONS = 4

DEFAULT_HARD_TO_BORROW_MAX_BPS = 300.0


def evaluate_gates(
    feature_row: FeatureRow | None,
    universe_entry: UniverseEntry | None,
    cfg: GatesConfig,
    *,
    is_etf: bool,
    earnings: Sequence[EarningsEvent],
    max_hold_bars: int,
    calendar: pd.DataFrame,
    min_bars_since_gap: int,
    bar_age_bars: int = 0,
    direction: Direction | None = None,
    borrow_bps_per_year: float | None = None,
    hard_to_borrow_max_bps_per_year: float = DEFAULT_HARD_TO_BORROW_MAX_BPS,
) -> RejectionReason | None:
    """Return the first failing gate's reason, or None if rows 1-8 all pass."""
    if cfg.require_universe_eligible:
        if universe_entry is None:
            return RejectionReason.NOT_IN_UNIVERSE
        if not universe_entry.eligible:
            assert universe_entry.reason is not None
            return universe_entry.reason
    if feature_row is None:
        return RejectionReason.INSUFFICIENT_HISTORY
    if cfg.require_warm and not feature_row.is_warm:
        return RejectionReason.INSUFFICIENT_HISTORY
    if feature_row.bars_since_gap < min_bars_since_gap:
        return RejectionReason.DATA_GAP
    if bar_age_bars > cfg.max_bar_staleness_bars:
        return RejectionReason.STALE_DATA
    if feature_row.xs_population < cfg.min_xs_population:
        return RejectionReason.THIN_CROSS_SECTION
    if earnings_in_window(
        ts=feature_row.ts,
        max_hold_bars=max_hold_bars,
        events=earnings,
        is_etf=is_etf,
        calendar=calendar,
    ):
        return RejectionReason.EARNINGS_IN_WINDOW
    if (
        not cfg.skip_hard_to_borrow
        and direction is Direction.SHORT
        and borrow_bps_per_year is not None
        and borrow_bps_per_year > hard_to_borrow_max_bps_per_year
    ):
        return RejectionReason.HARD_TO_BORROW
    return None


def earnings_in_window(
    *,
    ts: datetime,
    max_hold_bars: int,
    events: Sequence[EarningsEvent],
    is_etf: bool,
    calendar: pd.DataFrame,
) -> bool:
    """True if the hold window intersects an earnings print. ETFs always False.

    Holding period is `(t, t+max_hold_bars]` in session index. Called only from
    `evaluate_gates` — not a second public gate API.
    """
    if is_etf:
        return False
    t_idx = _session_index_at_ts(calendar, ts)
    if t_idx is None:
        return True
    known = tuple(
        event
        for event in events
        if event.available_ts is None or event.available_ts <= ts
    )
    if not known:
        return True
    hold_lo = t_idx + 1
    hold_hi = t_idx + max_hold_bars
    for event in known:
        e_idx = _session_index_on_or_after(calendar, event.earnings_date)
        if e_idx is None:
            return True
        if event.available_ts is None:
            lo = e_idx - EARNINGS_UNCERTAINTY_SESSIONS
            hi = e_idx + EARNINGS_UNCERTAINTY_SESSIONS
            if _closed_intersect(lo, hi, hold_lo, hold_hi):
                return True
        elif _closed_intersect(e_idx, e_idx, hold_lo, hold_hi):
            return True
    return False


def _closed_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= a1 and b0 <= b1 and a0 <= b1 and b0 <= a1


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed: date = pd.Timestamp(value).date()
    return parsed


def _session_index_at_ts(calendar: pd.DataFrame, ts: datetime) -> int | None:
    if calendar.empty:
        return None
    ts_utc = pd.Timestamp(ts).tz_convert("UTC")
    index_col = calendar["session_index"]
    if "close_utc" in calendar.columns:
        closes = pd.DatetimeIndex(pd.to_datetime(calendar["close_utc"], utc=True)).floor("s")
        target = ts_utc.floor("s")
        hits = closes.get_indexer([target])
        pos = int(hits[0])
        if pos != -1:
            return int(index_col.iloc[pos])
    day = ts_utc.date()
    sessions = calendar["session"]
    for i in range(len(calendar)):
        if _as_date(sessions.iloc[i]) == day:
            return int(index_col.iloc[i])
    return None


def _session_index_on_or_after(calendar: pd.DataFrame, d: date) -> int | None:
    if calendar.empty:
        return None
    sessions = calendar["session"]
    index_col = calendar["session_index"]
    best_idx: int | None = None
    best_date: date | None = None
    for i in range(len(calendar)):
        session_d = _as_date(sessions.iloc[i])
        if session_d >= d and (best_date is None or session_d < best_date):
            best_date = session_d
            best_idx = int(index_col.iloc[i])
    return best_idx
