"""Corwin-Schultz high-low spread estimator, then a liquidity-tier floor.

The estimator is scale-free and uses ADJUSTED high/low. Negative estimates
clamp to 0; the per-tier floor then makes the result never optimistic.
See docs/04-DATA_AND_UNIVERSE.md §7.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import SpreadFloorConfig, UniverseConfig

_K = 3.0 - 2.0 * np.sqrt(2.0)
_TIER_500M = 500_000_000.0
_TIER_100M = 100_000_000.0
_TIER_20M = 20_000_000.0


def corwin_schultz_bps(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Trailing-window Corwin-Schultz effective spread, in basis points.

    At bar t the last pair is (t-1, t), so the window never reads a future bar.
    Warm-up and invalid high/low yield NaN, never 0.
    """
    if window < 1:
        raise ValueError(f"spread window must be >= 1; got {window}")
    h = high.astype("float64")
    low_ = low.astype("float64")
    valid = (h > 0.0) & (low_ > 0.0) & (h >= low_)
    ln_hl = np.log(h / low_).where(valid)
    prev_h = h.shift(1)
    prev_l = low_.shift(1)
    prev_valid = (prev_h > 0.0) & (prev_l > 0.0) & (prev_h >= prev_l)
    pair_ok = valid & prev_valid
    beta_i = (ln_hl.pow(2) + ln_hl.shift(1).pow(2)).where(pair_ok)
    two_h = np.maximum(h, prev_h)
    two_l = np.minimum(low_, prev_l)
    gamma_i = np.log(two_h / two_l).pow(2).where(pair_ok)
    beta = beta_i.rolling(window, min_periods=window).mean()
    gamma = gamma_i.rolling(window, min_periods=window).mean()
    sqrt_beta = np.sqrt(beta)
    alpha = (np.sqrt(2.0 * beta) - sqrt_beta) / _K - np.sqrt(gamma / _K)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    bps = spread.clip(lower=0.0) * 1e4
    return pd.Series(bps.to_numpy(), index=high.index, dtype="float64")


def spread_floor_bps(adv_usd_60: pd.Series, floors: SpreadFloorConfig) -> pd.Series:
    """Pessimistic floor by trailing ADV. NaN ADV -> NaN (not a fake tight spread)."""
    adv = adv_usd_60.astype("float64")
    out = pd.Series(np.float64(floors.tier_below), index=adv.index, dtype="float64")
    out = out.mask(adv > _TIER_20M, np.float64(floors.tier_20m))
    out = out.mask(adv > _TIER_100M, np.float64(floors.tier_100m))
    out = out.mask(adv > _TIER_500M, np.float64(floors.tier_500m))
    return out.mask(adv.isna(), np.nan)


def spread_bps_est(
    high: pd.Series,
    low: pd.Series,
    adv_usd_60: pd.Series,
    cfg: UniverseConfig,
) -> pd.Series:
    """`max(corwin_schultz_bps, tier_floor)`. NaN CS stays NaN."""
    cs = corwin_schultz_bps(high, low, cfg.spread_window_bars)
    floor = spread_floor_bps(adv_usd_60, cfg.spread_floor_bps_by_tier)
    stacked = np.maximum(cs.to_numpy(dtype="float64"), floor.to_numpy(dtype="float64"))
    return pd.Series(stacked, index=high.index, dtype="float64")
