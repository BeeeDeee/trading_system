"""Swing pivots and Smart Zones ranges (prereg §4).

Point in time: a pivot of day j with lookback k exists only from the close of day j+k on; every
output row t uses rows <= t only. NaN high/low (no candle) inside a pivot window means no pivot.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAX_AGE = 90       # days since the pivot day; older ranges expire
MIN_WIDTH = 0.05   # (top - bottom) / bottom


def pivots(x: np.ndarray, k: int, kind: str) -> np.ndarray:
    """(T, N) bool: a pivot of day t-k is confirmed at the close of t.

    high: x_j > max(x_{j-k..j-1}) and x_j >= max(x_{j+1..j+k}); low mirrored (<, <=).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    if len(x) < 2 * k + 1:
        return out
    w = sliding_window_view(x, 2 * k + 1, axis=0)          # (T-2k, N, 2k+1), centre = day j
    c, left, right = w[..., k], w[..., :k], w[..., k + 1:]
    with np.errstate(invalid="ignore"):
        if kind == "high":
            piv = (c > left.max(-1)) & (c >= right.max(-1))
        elif kind == "low":
            piv = (c < left.min(-1)) & (c <= right.min(-1))
        else:
            raise ValueError(kind)
    out[2 * k:] = piv                                        # centre j = i + k, confirmed at i + 2k
    return out


def ranges(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(top, bottom, valid), each (T, N), as of the close of each day.

    top = max(last confirmed pivot high, highs since its pivot day), bottom mirrored. valid: both
    pivots exist and are at most MAX_AGE days old, and the range is at least MIN_WIDTH of the bottom.
    """
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    T, N = high.shape
    ph, pl_ = pivots(high, k, "high"), pivots(low, k, "low")
    top, bot = np.full((T, N), np.nan), np.full((T, N), np.nan)
    valid = np.zeros((T, N), dtype=bool)
    run_top, run_bot = np.full(N, np.nan), np.full(N, np.nan)
    day_h, day_l = np.full(N, -10**9), np.full(N, -10**9)
    for t in range(T):
        run_top = np.where(np.isnan(run_top), np.nan, np.fmax(run_top, high[t]))
        run_bot = np.where(np.isnan(run_bot), np.nan, np.fmin(run_bot, low[t]))
        if ph[t].any():
            run_top[ph[t]], day_h[ph[t]] = high[t - k, ph[t]], t - k
        if pl_[t].any():
            run_bot[pl_[t]], day_l[pl_[t]] = low[t - k, pl_[t]], t - k
        top[t], bot[t] = run_top, run_bot
        with np.errstate(invalid="ignore", divide="ignore"):
            valid[t] = ((t - day_h <= MAX_AGE) & (t - day_l <= MAX_AGE) & (run_top > run_bot)
                        & ((run_top - run_bot) / run_bot >= MIN_WIDTH))
    return top, bot, valid
