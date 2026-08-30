"""Market-regime columns from the benchmark (SPY) close. Computed once per ts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import MarketRegimeConfig
from scout.domain.enums import MarketRegime

SESSIONS_PER_YEAR: int = 252
DD_LOOKBACK: int = 252
VOL_LOOKBACK: int = 20
VOL_PCT_WINDOW: int = 756
VOL_PCT_MIN_PERIODS: int = 252


def spy_sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def spy_drawdown(close: pd.Series, n: int = DD_LOOKBACK) -> pd.Series:
    return 1.0 - close / close.rolling(n, min_periods=n).max()


def spy_realised_vol(
    close: pd.Series,
    n: int = VOL_LOOKBACK,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n, min_periods=n).std() * np.sqrt(sessions_per_year)


def spy_vol_percentile(
    vol: pd.Series,
    window: int = VOL_PCT_WINDOW,
    min_periods: int = VOL_PCT_MIN_PERIODS,
) -> pd.Series:
    return vol.rolling(window, min_periods=min_periods).rank(pct=True)


def classify_market_regime(
    above_ma: bool,
    dd: float,
    vol_pct: float,
    cfg: MarketRegimeConfig,
) -> MarketRegime:
    """Pure, stateless. UNKNOWN during benchmark warm-up; RISK_OFF is a gate."""
    if any(x is None or math.isnan(float(x)) for x in (dd, vol_pct)):
        return MarketRegime.UNKNOWN

    stressed = (dd >= cfg.dd_stress) or (vol_pct >= cfg.vol_stress_pct)

    if above_ma and dd <= cfg.dd_warn and not stressed:
        return MarketRegime.RISK_ON
    if (not above_ma) and stressed:
        return MarketRegime.RISK_OFF
    return MarketRegime.NEUTRAL


def compute_market_columns(close: pd.Series, cfg: MarketRegimeConfig) -> pd.DataFrame:
    """One row per timestamp: spy_above_ma, spy_dd_252, market_regime.

    Vectorised with np.select so the full history is not classified row-by-row.
    """
    sma = spy_sma(close, cfg.sma_n)
    above = close > sma
    dd = spy_drawdown(close)
    vol_pct = spy_vol_percentile(spy_realised_vol(close))

    unknown = dd.isna() | vol_pct.isna()
    stressed = ((dd >= cfg.dd_stress) | (vol_pct >= cfg.vol_stress_pct)).eq(True)
    risk_on = above.eq(True) & (dd <= cfg.dd_warn).eq(True) & ~stressed
    risk_off = ~above.eq(True) & stressed

    labels = np.select(
        [unknown.to_numpy(), risk_on.to_numpy(), risk_off.to_numpy()],
        [
            MarketRegime.UNKNOWN.value,
            MarketRegime.RISK_ON.value,
            MarketRegime.RISK_OFF.value,
        ],
        default=MarketRegime.NEUTRAL.value,
    )
    regime = pd.Series(labels, index=close.index, dtype=object).map(MarketRegime)
    return pd.DataFrame(
        {
            "spy_above_ma": above.to_numpy(),
            "spy_dd_252": dd.to_numpy(),
            "market_regime": regime.to_numpy(),
        },
        index=close.index,
    )
