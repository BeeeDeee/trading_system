"""Per-symbol regime label. Pure, stateless, five thresholds. No history."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import RegimeConfig
from scout.domain.enums import Regime


def classify_regime(
    er_20: float,
    er_60: float,
    slope_atr: float,
    vol_pct: float,
    cfg: RegimeConfig,
) -> Regime:
    """Map four floats to a Regime. NaN on any input → UNKNOWN.

    CHOP is the default: anything not clearly trending or clearly ranging.
    """
    if any(math.isnan(float(x)) for x in (er_20, er_60, slope_atr, vol_pct)):
        return Regime.UNKNOWN

    trending = (er_20 >= cfg.er_trend_min) and (er_60 >= cfg.er_long_trend_min)

    if trending and slope_atr >= cfg.slope_min:
        return Regime.TREND_UP
    if trending and slope_atr <= -cfg.slope_min:
        return Regime.TREND_DOWN
    if (er_20 <= cfg.er_range_max) and (vol_pct <= cfg.vol_range_max):
        return Regime.RANGE
    return Regime.CHOP


def classify_regime_vectorized(
    er_20: pd.Series,
    er_60: pd.Series,
    slope_atr: pd.Series,
    vol_pct: pd.Series,
    cfg: RegimeConfig,
) -> pd.Series:
    """Vectorised classify_regime. Same branches, np.select, no row-wise Python."""
    e20 = er_20.to_numpy(dtype="float64", copy=False)
    e60 = er_60.to_numpy(dtype="float64", copy=False)
    slope = slope_atr.to_numpy(dtype="float64", copy=False)
    vol = vol_pct.to_numpy(dtype="float64", copy=False)
    unknown = np.isnan(e20) | np.isnan(e60) | np.isnan(slope) | np.isnan(vol)
    trending = (e20 >= cfg.er_trend_min) & (e60 >= cfg.er_long_trend_min)
    labels = np.select(
        [
            unknown,
            trending & (slope >= cfg.slope_min),
            trending & (slope <= -cfg.slope_min),
            (e20 <= cfg.er_range_max) & (vol <= cfg.vol_range_max),
        ],
        [
            Regime.UNKNOWN.value,
            Regime.TREND_UP.value,
            Regime.TREND_DOWN.value,
            Regime.RANGE.value,
        ],
        default=Regime.CHOP.value,
    )
    return pd.Series(labels, index=er_20.index, dtype=object).map(Regime)
