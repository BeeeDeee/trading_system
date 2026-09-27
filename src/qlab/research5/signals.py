"""STR-TF signals (research 5 pre-registration §4).

Every function maps a Panel (plus unadjusted high/low where needed) to a (T, N) array whose row t
uses data up to and including the close of day t. Indicators use the total-return index (adjusted
prices); only IBS and ATR use unadjusted same-day high/low/close.
"""

import numpy as np

from qlab.data.panel import Panel
from qlab.features.basic import (_listed_days, rolling_mean, sma_ratio, tr_index, trailing_return,
                                 trailing_volatility)

VOL_WINDOW = 20


def z_reversal(panel: Panel, lookback: int, vol_window: int = VOL_WINDOW) -> np.ndarray:
    """ret_L / (vol_20 * sqrt(L)): the L-day return in units of its expected daily-vol scale."""
    ret = trailing_return(panel, lookback)
    vol = trailing_volatility(panel, vol_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ret / (vol * np.sqrt(lookback))
    z[~np.isfinite(z)] = np.nan
    return z


def sma_gap(panel: Panel, window: int) -> np.ndarray:
    """TR index / SMA(window) - 1 (> 0: above the average); NaN without `window` listed days."""
    return sma_ratio(panel, window)


def daily_return(panel: Panel) -> np.ndarray:
    r = (1.0 + np.asarray(panel.ret_co)) * (1.0 + np.asarray(panel.ret_oc)) - 1.0
    return np.where(panel.listed, r, np.nan)


def rsi(panel: Panel, n: int = 2) -> np.ndarray:
    """Wilder RSI(n) of the TR index; NaN until n changes are known."""
    idx = tr_index(panel)
    diff = np.vstack([np.zeros((1, idx.shape[1])), np.diff(idx, axis=0)])
    gain, loss = np.maximum(diff, 0.0), np.maximum(-diff, 0.0)
    listed = np.asarray(panel.listed)
    days = _listed_days(panel)
    out = np.full(idx.shape, np.nan)
    avg_g = np.zeros(idx.shape[1])
    avg_l = np.zeros(idx.shape[1])
    for t in range(idx.shape[0]):
        d = days[t]
        first = d <= n + 1  # plain average over the first n changes (day 1 has no change)
        w = np.where(first, 1.0 / np.maximum(d - 1, 1), 1.0 / n)
        upd = listed[t] & (d >= 2)
        avg_g = np.where(upd, avg_g + w * (gain[t] - avg_g), avg_g)
        avg_l = np.where(upd, avg_l + w * (loss[t] - avg_l), avg_l)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(avg_l > 0, 100.0 - 100.0 / (1.0 + avg_g / avg_l),
                           np.where(avg_g > 0, 100.0, 50.0))
        out[t] = np.where(listed[t] & (d >= n + 1), val, np.nan)
    return out


def ibs(panel: Panel, high_u: np.ndarray, low_u: np.ndarray) -> np.ndarray:
    """Internal bar strength (close - low) / (high - low) of day t; NaN for a zero range."""
    close = np.asarray(panel.close_u, dtype=float)
    hi, lo = np.asarray(high_u, dtype=float), np.asarray(low_u, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (close - lo) / (hi - lo)
    out[~((hi > lo) & np.asarray(panel.listed))] = np.nan
    return out


def atr_fraction(panel: Panel, high_u: np.ndarray, low_u: np.ndarray, n: int = 14) -> np.ndarray:
    """SMA(n) of the true range as a fraction of the close.

    The previous close is brought onto today's share basis via the total return
    (close_u / (1 + r)), so splits do not create fake ranges.
    """
    close = np.asarray(panel.close_u, dtype=float)
    hi, lo = np.asarray(high_u, dtype=float), np.asarray(low_u, dtype=float)
    prev = close / (1.0 + daily_return(panel))
    tr = np.fmax(hi - lo, np.fmax(np.abs(hi - prev), np.abs(lo - prev))) / close
    tr = np.where(np.isfinite(tr), tr, 0.0)
    out = rolling_mean(tr, n)
    out[(_listed_days(panel) <= n) | ~np.asarray(panel.listed)] = np.nan
    return out
