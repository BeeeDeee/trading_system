"""Family signals (spec §6.2): score, entry and exit matrices, all point in time.

A signal says, for each day t and security, how attractive it is (`score`, higher is better, NaN =
not rankable), whether it may be newly bought (`entry`) and whether a holding must be sold
(`exit`). Portfolio construction (qlab.strategies.portfolio) turns it into target weights.
"""

from dataclasses import dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.features.basic import (rolling_max, rolling_mean, rolling_min, sma_ratio, tr_index,
                                 trailing_return, trailing_volatility)

FAMILIES = ("xs_momentum", "ts_trend", "breakout", "st_reversal", "low_vol")


@dataclass(frozen=True)
class Signal:
    score: np.ndarray
    entry: np.ndarray   # bool; may be bought
    exit: np.ndarray    # bool; must be sold


def _listed_at_least(panel: Panel, days: int) -> np.ndarray:
    return (np.cumsum(panel.listed, axis=0, dtype=np.int32) >= days) & panel.listed


def build_signal(panel: Panel, family: str, params: dict) -> Signal:
    finite = lambda s: np.isfinite(s)  # noqa: E731
    no_exit = np.zeros(panel.shape, bool)

    if family == "xs_momentum":
        score = trailing_return(panel, params["lookback"], params["skip"])
        return Signal(score, finite(score), no_exit)

    if family == "low_vol":
        score = -trailing_volatility(panel, params["lookback"])
        return Signal(score, finite(score), no_exit)

    if family == "ts_trend":
        score = sma_ratio(panel, params["sma"])
        return Signal(score, finite(score) & (score > 0), finite(score) & (score < 0))

    if family == "st_reversal":
        score = -trailing_return(panel, params["lookback"])
        above = sma_ratio(panel, 200) > 0  # NaN compares False
        return Signal(score, finite(score) & above, ~above & panel.listed)

    if family == "breakout":
        lookback, recent = params["lookback"], params["recent_days"]
        idx = tr_index(panel)
        prev_high = np.vstack([np.full((1, panel.shape[1]), np.nan),
                               rolling_max(idx, lookback)[:-1]])
        new_high = (idx >= prev_high) & _listed_at_least(panel, lookback + 1)
        recent_high = rolling_max(new_high.astype(float), recent) > 0
        exit_window = max(lookback // 2, 2)
        prev_low = np.vstack([np.full((1, panel.shape[1]), np.nan),
                              rolling_min(idx, exit_window)[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            score = idx / rolling_max(idx, lookback) - 1.0  # 0 at the high, negative below
        score[~_listed_at_least(panel, lookback + 1)] = np.nan
        return Signal(score, recent_high & np.isfinite(score), (idx < prev_low) & panel.listed)

    raise ValueError(f"unknown family {family!r}")


def market_trend(panel: Panel, asset: int, window: int) -> np.ndarray:
    """1.0 on days the asset's TR index is above its `window`-day SMA, else 0.0 (point in time)."""
    idx = tr_index(panel)[:, asset]
    return (idx > rolling_mean(idx[:, None], window)[:, 0]).astype(float)


def market_vol_scale(panel: Panel, asset: int, target: float = 0.15, window: int = 63) -> np.ndarray:
    """min(1, target / realized annualized vol of the asset); 1.0 before enough history."""
    vol = trailing_volatility(panel, window)[:, asset] * np.sqrt(252)
    return np.where(np.isfinite(vol) & (vol > 0), np.minimum(1.0, target / vol), 1.0)
