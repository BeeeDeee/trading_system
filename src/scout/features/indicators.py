"""Per-symbol indicator formulas. Causal: a value at t uses close_time <= t.

No TA-Lib, no pandas_ta. Each function is a thin pandas/numpy expression
tested against a hand-computed fixture in tests/unit/test_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import FeaturesConfig
from scout.domain.enums import VolBucket

SESSIONS_PER_YEAR: int = 252
DONCHIAN_N_SHORT: int = 20
VOL_N_SHORT: int = 20
VOL_N_LONG: int = 60
GAP_MEAN_N: int = 20
OVERNIGHT_VAR_N: int = 60
MOM_MID_LOOKBACK: int = 126
MOM_SHORT: int = 21
VOL_LOW_PCT: float = 0.33
VOL_HIGH_PCT: float = 0.67


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder smoothing == EMA with alpha = 1/n. adjust=False is REQUIRED:
    # adjust=True produces a different (and non-recursive) series.
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr_pct(atr: pd.Series, close: pd.Series) -> pd.Series:
    return atr / close


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def ema_spread_atr(fast: pd.Series, slow: pd.Series, atr: pd.Series) -> pd.Series:
    return (fast - slow) / atr


def slope_atr(close: pd.Series, atr: pd.Series, n: int) -> pd.Series:
    return (close - close.shift(n)) / (atr * np.sqrt(n))


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    # .shift(1) is mandatory: the channel must exclude the current bar or the
    # breakout condition becomes self-referential.
    return high.rolling(n, min_periods=n).max().shift(1)


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    # .shift(1) is mandatory: the channel must exclude the current bar or the
    # breakout condition becomes self-referential.
    return low.rolling(n, min_periods=n).min().shift(1)


def dist_to_high_atr(high: pd.Series, close: pd.Series, atr: pd.Series) -> pd.Series:
    return (high - close) / atr


def dist_to_low_atr(low: pd.Series, close: pd.Series, atr: pd.Series) -> pd.Series:
    return (close - low) / atr


def keltner_upper(slow: pd.Series, atr: pd.Series, k: float) -> pd.Series:
    return slow + k * atr


def keltner_lower(slow: pd.Series, atr: pd.Series, k: float) -> pd.Series:
    return slow - k * atr


def realised_vol(close: pd.Series, n: int, sessions_per_year: int = SESSIONS_PER_YEAR) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n, min_periods=n).std() * np.sqrt(sessions_per_year)


def gap_atr(open_: pd.Series, close: pd.Series, atr: pd.Series) -> pd.Series:
    return (open_ - close.shift(1)) / atr.shift(1)


def gap_abs_mean(gap: pd.Series, n: int = GAP_MEAN_N) -> pd.Series:
    return gap.abs().rolling(n, min_periods=n).mean()


def overnight_var_share(
    open_: pd.Series, close: pd.Series, n: int = OVERNIGHT_VAR_N
) -> pd.Series:
    r_overnight = np.log(open_ / close.shift(1))
    r_intraday = np.log(close / open_)
    var_on = r_overnight.rolling(n, min_periods=n).var()
    var_id = r_intraday.rolling(n, min_periods=n).var()
    denom = var_on + var_id
    return (var_on / denom).where(denom > 0)


def momentum_skip(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    return (close.shift(skip) / close.shift(lookback)) - 1.0


def momentum(close: pd.Series, n: int) -> pd.Series:
    return (close / close.shift(n)) - 1.0


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman's Efficiency Ratio: net directional progress / path length.

    Range [0, 1]. 1.0 is a perfect trend; 0.0 is a path that returned to start.
    """
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return (net / path).where(path > 0)


def atr_percentile(atr_pct_series: pd.Series, window_bars: int, min_periods: int) -> pd.Series:
    """Rank of the current atr_pct within its own trailing window. Result in [0, 1]."""
    return atr_pct_series.rolling(window_bars, min_periods=min_periods).rank(pct=True)


def classify_vol_bucket(
    percentile: pd.Series,
    *,
    low_pct: float = VOL_LOW_PCT,
    high_pct: float = VOL_HIGH_PCT,
) -> pd.Series:
    """Per-symbol vol bucket from a trailing ATR percentile. NaN → UNKNOWN."""
    valid = percentile.notna()
    labels = np.select(
        [~valid, percentile < low_pct, percentile >= high_pct],
        [VolBucket.UNKNOWN.value, VolBucket.LOW.value, VolBucket.HIGH.value],
        default=VolBucket.MID.value,
    )
    return pd.Series(labels, index=percentile.index, dtype=object).map(VolBucket)


def rolling_beta(close: pd.Series, close_bench: pd.Series, window: int) -> pd.Series:
    """Rolling beta of `close` vs the benchmark. Inner join on the index (ts)."""
    r_sym, r_bench = _aligned_log_returns(close, close_bench)
    min_periods = window // 2
    cov = r_sym.rolling(window, min_periods=min_periods).cov(r_bench)
    var = r_bench.rolling(window, min_periods=min_periods).var()
    return cov / var


def rolling_corr(close: pd.Series, close_bench: pd.Series, window: int) -> pd.Series:
    r_sym, r_bench = _aligned_log_returns(close, close_bench)
    min_periods = window // 2
    return r_sym.rolling(window, min_periods=min_periods).corr(r_bench)


def required_warmup_bars(cfg: FeaturesConfig) -> int:
    """Sessions required before is_warm. Derived from config, not a constant."""
    return max(
        cfg.ema_slow,
        cfg.atr_n,
        cfg.donchian_n + 1,
        cfg.slope_lookback,
        cfg.er_long_n,
        VOL_N_LONG,
        cfg.beta_window // 2,
        cfg.vol_min_periods,
        cfg.mom_lookback_bars + cfg.mom_skip_bars,
    )


def _aligned_log_returns(close: pd.Series, close_bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat({"sym": close, "bench": close_bench}, axis=1, join="inner")
    r_sym = np.log(joined["sym"] / joined["sym"].shift(1))
    r_bench = np.log(joined["bench"] / joined["bench"].shift(1))
    return r_sym, r_bench
