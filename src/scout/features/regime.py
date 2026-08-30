"""Per-symbol regime label. Pure, stateless, five thresholds. No history."""

from __future__ import annotations

import math

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
