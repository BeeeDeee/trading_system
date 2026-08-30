"""Ordered universe eligibility rules. First failure is the recorded reason.

Evaluated at each snapshot ts using only sessions with ts_session <= ts.
See docs/04-DATA_AND_UNIVERSE.md §7.2.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import UniverseConfig
from scout.domain.enums import RejectionReason

# Trailing ADV window. Not configurable: 04-DATA and ADR-018 fix it at 60.
ADV_WINDOW_BARS = 60

RULE_ORDER: tuple[RejectionReason, ...] = (
    RejectionReason.INSUFFICIENT_HISTORY,
    RejectionReason.STALE_DATA,
    RejectionReason.DATA_SUSPECT,
    RejectionReason.DATA_GAP,
    RejectionReason.LOW_PRICE,
    RejectionReason.LOW_LIQUIDITY,
    RejectionReason.NOT_IN_UNIVERSE,
    RejectionReason.WIDE_SPREAD,
)


def first_failing_reason(
    *,
    bars_available: int,
    has_session_at_ts: bool,
    is_delisted: bool,
    suspect_in_lookback: bool,
    bars_since_gap: int,
    close_raw: float,
    adv_usd_60: float,
    adv_rank: float,
    spread_bps_est: float,
    cfg: UniverseConfig,
) -> RejectionReason | None:
    """Return the first failing rule, or None if all eight pass."""
    if bars_available < cfg.min_history_bars:
        return RejectionReason.INSUFFICIENT_HISTORY
    if (not has_session_at_ts) or is_delisted:
        return RejectionReason.STALE_DATA
    if suspect_in_lookback:
        return RejectionReason.DATA_SUSPECT
    if bars_since_gap < cfg.min_bars_since_gap:
        return RejectionReason.DATA_GAP
    if not (close_raw >= cfg.min_price_usd):
        return RejectionReason.LOW_PRICE
    if not (adv_usd_60 >= cfg.min_adv_usd):
        return RejectionReason.LOW_LIQUIDITY
    if not (adv_rank <= cfg.universe_size):
        return RejectionReason.NOT_IN_UNIVERSE
    if not (spread_bps_est <= cfg.max_spread_bps):
        return RejectionReason.WIDE_SPREAD
    return None


def apply_eligibility_rules(frame: pd.DataFrame, cfg: UniverseConfig) -> pd.DataFrame:
    """Vectorised first-failure assignment. `reason` is empty iff eligible."""
    n = len(frame)
    if n == 0:
        return frame.assign(
            eligible=pd.Series(dtype=bool),
            reason=pd.Series(dtype="object"),
        )
    reason = pd.Series("", index=frame.index, dtype="object")
    fail_history = frame["bars_available"].to_numpy() < cfg.min_history_bars
    reason = _fill_first(reason, fail_history, RejectionReason.INSUFFICIENT_HISTORY)
    fail_stale = (~frame["has_session_at_ts"].to_numpy()) | frame["is_delisted"].to_numpy()
    reason = _fill_first(reason, fail_stale, RejectionReason.STALE_DATA)
    reason = _fill_first(
        reason, frame["suspect_in_lookback"].to_numpy(), RejectionReason.DATA_SUSPECT
    )
    fail_gap = frame["bars_since_gap"].to_numpy() < cfg.min_bars_since_gap
    reason = _fill_first(reason, fail_gap, RejectionReason.DATA_GAP)
    close_raw = frame["close_raw"].astype("float64")
    fail_price = ~(close_raw >= cfg.min_price_usd)
    reason = _fill_first(reason, fail_price.to_numpy(), RejectionReason.LOW_PRICE)
    adv = frame["adv_usd_60"].astype("float64")
    fail_liq = ~(adv >= cfg.min_adv_usd)
    reason = _fill_first(reason, fail_liq.to_numpy(), RejectionReason.LOW_LIQUIDITY)
    rank = pd.to_numeric(frame["adv_rank"], errors="coerce")
    fail_rank = ~rank.le(cfg.universe_size)
    reason = _fill_first(reason, fail_rank.fillna(True).to_numpy(), RejectionReason.NOT_IN_UNIVERSE)
    spread = frame["spread_bps_est"].astype("float64")
    fail_spread = ~(spread <= cfg.max_spread_bps)
    reason = _fill_first(reason, fail_spread.to_numpy(), RejectionReason.WIDE_SPREAD)
    eligible = reason.eq("")
    return frame.assign(eligible=eligible, reason=reason)


def _fill_first(
    reason: pd.Series, fail: object, code: RejectionReason
) -> pd.Series:
    mask = reason.eq("") & pd.Series(fail, index=reason.index, dtype=bool)
    return reason.mask(mask, code.value)
