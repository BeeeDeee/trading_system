"""Decision-time gates: first failure wins; earnings window is max_hold_bars."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from scout.config.schema import GatesConfig
from scout.domain.enums import Direction, MarketRegime, Regime, RejectionReason, VolBucket
from scout.domain.features import FeatureRow
from scout.domain.market import EarningsEvent
from scout.domain.universe import UniverseEntry
from scout.gates.eligibility import (
    EARNINGS_UNCERTAINTY_SESSIONS,
    evaluate_gates,
)

T_INDEX = 15
N_SESSIONS = 50
MAX_HOLD = 21


def _calendar() -> pd.DataFrame:
    sessions = pd.bdate_range(date(2015, 1, 5), periods=N_SESSIONS, freq="B")
    close = pd.DatetimeIndex(sessions.tz_localize("UTC")) + pd.Timedelta(hours=21)
    return pd.DataFrame(
        {
            "session": [ts.date() for ts in sessions],
            "close_utc": close,
            "session_index": list(range(N_SESSIONS)),
        }
    )


def _ts_at(calendar: pd.DataFrame, index: int) -> datetime:
    value = calendar["close_utc"].iloc[index]
    ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def _session_date(calendar: pd.DataFrame, index: int) -> date:
    value = calendar["session"].iloc[index]
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _feature_row(calendar: pd.DataFrame, **overrides: Any) -> FeatureRow:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": _ts_at(calendar, T_INDEX),
        "close": 100.0,
        "atr_14": 2.0,
        "atr_pct": 0.02,
        "vol_20": 0.2,
        "vol_60": 0.25,
        "ema_fast": 101.0,
        "ema_slow": 99.0,
        "ema_spread_atr": 1.0,
        "slope_atr_20": 0.5,
        "efficiency_ratio_20": 0.8,
        "efficiency_ratio_60": 0.7,
        "atr_percentile_1y": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "donchian_high_20": 102.0,
        "donchian_low_20": 90.0,
        "donchian_high_55": 105.0,
        "donchian_low_55": 85.0,
        "keltner_upper": 105.0,
        "keltner_lower": 93.0,
        "dist_to_high_atr": 2.5,
        "dist_to_low_atr": 7.5,
        "mom_252_skip21": 0.15,
        "mom_126_skip21": 0.08,
        "mom_21": 0.02,
        "mom_252_xs_pct": 0.92,
        "vol_xs_pct": 0.4,
        "xs_population": 400,
        "gap_atr": 0.1,
        "gap_abs_mean_20": 0.2,
        "overnight_var_share_60": 0.3,
        "market_regime": MarketRegime.RISK_ON,
        "spy_dd_252": 0.05,
        "spy_above_ma": True,
        "vix_close": 18.0,
        "beta_bench_90": 1.1,
        "corr_bench_90": 0.6,
        "bars_available": 400,
        "bars_since_gap": 50,
        "is_warm": True,
    }
    kwargs.update(overrides)
    return FeatureRow(**kwargs)


def _entry(**overrides: Any) -> UniverseEntry:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "eligible": True,
        "reason": None,
        "adv_usd_30": 1e8,
        "spread_bps_est": 1.0,
        "bars_available": 400,
        "listed_days": 1000.0,
    }
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


def _event(
    calendar: pd.DataFrame,
    index: int,
    *,
    available_index: int | None = 0,
    is_confirmed: bool = True,
) -> EarningsEvent:
    available = None if available_index is None else _ts_at(calendar, available_index)
    return EarningsEvent(
        asset_id="AAPL",
        earnings_date=_session_date(calendar, index),
        is_confirmed=is_confirmed,
        available_ts=available,
        timing="AMC",
    )


def _call(
    calendar: pd.DataFrame,
    *,
    feature_row: FeatureRow | None = ...,  # type: ignore[assignment]
    universe_entry: UniverseEntry | None = ...,  # type: ignore[assignment]
    cfg: GatesConfig | None = None,
    is_etf: bool = False,
    earnings: list[EarningsEvent] | None = None,
    max_hold_bars: int = MAX_HOLD,
    min_bars_since_gap: int = 10,
    bar_age_bars: int = 0,
    direction: Direction | None = Direction.LONG,
    borrow_bps_per_year: float | None = 30.0,
    hard_to_borrow_max_bps_per_year: float = 300.0,
    **feature_overrides: Any,
) -> RejectionReason | None:
    if feature_row is ...:
        feature_row = _feature_row(calendar, **feature_overrides)
    if universe_entry is ...:
        universe_entry = _entry()
    if earnings is None:
        earnings = [_event(calendar, 0)]
    return evaluate_gates(
        feature_row,
        universe_entry,
        cfg if cfg is not None else GatesConfig(),
        is_etf=is_etf,
        earnings=earnings,
        max_hold_bars=max_hold_bars,
        calendar=calendar,
        min_bars_since_gap=min_bars_since_gap,
        bar_age_bars=bar_age_bars,
        direction=direction,
        borrow_bps_per_year=borrow_bps_per_year,
        hard_to_borrow_max_bps_per_year=hard_to_borrow_max_bps_per_year,
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    return _calendar()


def test_all_pass(calendar: pd.DataFrame) -> None:
    assert _call(calendar) is None


def test_not_in_universe_when_entry_missing(calendar: pd.DataFrame) -> None:
    assert _call(calendar, universe_entry=None) is RejectionReason.NOT_IN_UNIVERSE


def test_ineligible_returns_snapshot_reason(calendar: pd.DataFrame) -> None:
    entry = _entry(eligible=False, reason=RejectionReason.LOW_PRICE)
    got = _call(
        calendar,
        universe_entry=entry,
        is_warm=False,
        earnings=[_event(calendar, T_INDEX + 2)],
    )
    assert got is RejectionReason.LOW_PRICE


def test_insufficient_history(calendar: pd.DataFrame) -> None:
    assert _call(calendar, is_warm=False) is RejectionReason.INSUFFICIENT_HISTORY


def test_missing_feature_row_is_insufficient_history(calendar: pd.DataFrame) -> None:
    assert _call(calendar, feature_row=None) is RejectionReason.INSUFFICIENT_HISTORY


def test_data_gap(calendar: pd.DataFrame) -> None:
    assert (
        _call(calendar, bars_since_gap=3, min_bars_since_gap=10) is RejectionReason.DATA_GAP
    )


def test_stale_data(calendar: pd.DataFrame) -> None:
    assert _call(calendar, bar_age_bars=3) is RejectionReason.STALE_DATA


def test_bar_age_at_max_is_not_stale(calendar: pd.DataFrame) -> None:
    assert _call(calendar, bar_age_bars=2) is None


def test_thin_cross_section(calendar: pd.DataFrame) -> None:
    assert _call(calendar, xs_population=99) is RejectionReason.THIN_CROSS_SECTION


def test_earnings_in_window_blocks(calendar: pd.DataFrame) -> None:
    in_window = [_event(calendar, T_INDEX + MAX_HOLD)]
    assert _call(calendar, earnings=in_window) is RejectionReason.EARNINGS_IN_WINDOW
    just_after = [_event(calendar, T_INDEX + MAX_HOLD + 1)]
    assert _call(calendar, earnings=just_after) is None
    on_t = [_event(calendar, T_INDEX)]
    assert _call(calendar, earnings=on_t) is None
    assert _call(calendar, is_etf=True, earnings=[]) is None


def test_earnings_two_sessions_ahead_blocks(calendar: pd.DataFrame) -> None:
    earnings = [_event(calendar, T_INDEX + 2)]
    assert _call(calendar, earnings=earnings) is RejectionReason.EARNINGS_IN_WINDOW


def test_earnings_five_sessions_after_does_not_block(calendar: pd.DataFrame) -> None:
    earnings = [_event(calendar, T_INDEX - 5)]
    assert _call(calendar, earnings=earnings) is None


def test_earnings_window_is_max_hold_not_two(calendar: pd.DataFrame) -> None:
    earnings = [_event(calendar, T_INDEX + 10)]
    assert _call(calendar, earnings=earnings, max_hold_bars=5) is None
    assert (
        _call(calendar, earnings=earnings, max_hold_bars=21) is RejectionReason.EARNINGS_IN_WINDOW
    )


def test_missing_earnings_blocks_stock(calendar: pd.DataFrame) -> None:
    assert _call(calendar, earnings=[]) is RejectionReason.EARNINGS_IN_WINDOW
    assert _call(calendar, is_etf=True, earnings=[]) is None


def test_earnings_ignores_unavailable_as_of(calendar: pd.DataFrame) -> None:
    future_only = [_event(calendar, T_INDEX + 2, available_index=T_INDEX + 1)]
    assert _call(calendar, earnings=future_only) is RejectionReason.EARNINGS_IN_WINDOW
    known_past_and_future = [
        _event(calendar, T_INDEX - 5, available_index=0),
        _event(calendar, T_INDEX + 2, available_index=T_INDEX + 1),
    ]
    assert _call(calendar, earnings=known_past_and_future) is None


def test_earnings_uncertainty_intersects_hold(calendar: pd.DataFrame) -> None:
    assert EARNINGS_UNCERTAINTY_SESSIONS == 4
    near_past = [_event(calendar, T_INDEX - 2, available_index=None)]
    assert _call(calendar, earnings=near_past) is RejectionReason.EARNINGS_IN_WINDOW
    far_past = [_event(calendar, T_INDEX - 5, available_index=None)]
    assert _call(calendar, earnings=far_past) is None


def test_hard_to_borrow_shorts_only(calendar: pd.DataFrame) -> None:
    cfg = GatesConfig(skip_hard_to_borrow=False)
    assert (
        _call(
            calendar,
            cfg=cfg,
            direction=Direction.SHORT,
            borrow_bps_per_year=500.0,
        )
        is RejectionReason.HARD_TO_BORROW
    )
    assert (
        _call(
            calendar,
            cfg=cfg,
            direction=Direction.LONG,
            borrow_bps_per_year=500.0,
        )
        is None
    )
    skipped = GatesConfig(skip_hard_to_borrow=True)
    assert (
        _call(
            calendar,
            cfg=skipped,
            direction=Direction.SHORT,
            borrow_bps_per_year=500.0,
        )
        is None
    )
    assert (
        _call(
            calendar,
            cfg=cfg,
            direction=Direction.SHORT,
            borrow_bps_per_year=None,
        )
        is None
    )


def test_gate_order_stable(calendar: pd.DataFrame) -> None:
    stale_and_earnings = _call(
        calendar,
        bar_age_bars=5,
        earnings=[_event(calendar, T_INDEX + 2)],
        cfg=GatesConfig(skip_hard_to_borrow=False),
        direction=Direction.SHORT,
        borrow_bps_per_year=500.0,
    )
    assert stale_and_earnings is RejectionReason.STALE_DATA
    thin_and_earnings = _call(
        calendar,
        xs_population=10,
        earnings=[_event(calendar, T_INDEX + 2)],
    )
    assert thin_and_earnings is RejectionReason.THIN_CROSS_SECTION
    earnings_and_htb = _call(
        calendar,
        earnings=[_event(calendar, T_INDEX + 2)],
        cfg=GatesConfig(skip_hard_to_borrow=False),
        direction=Direction.SHORT,
        borrow_bps_per_year=500.0,
    )
    assert earnings_and_htb is RejectionReason.EARNINGS_IN_WINDOW
    gap_and_stale = _call(calendar, bars_since_gap=0, bar_age_bars=9)
    assert gap_and_stale is RejectionReason.DATA_GAP
