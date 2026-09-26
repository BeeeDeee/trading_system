"""Basic point-in-time features on a Panel. Row t uses data up to the close of day t."""

import numpy as np

from qlab.data.panel import Panel


def tr_index(panel: Panel) -> np.ndarray:
    """Total-return index at each close (1 before the first day)."""
    return np.cumprod((1.0 + panel.ret_co) * (1.0 + panel.ret_oc), axis=0)


def trailing_return(panel: Panel, lookback: int, skip: int = 0) -> np.ndarray:
    """Total return from close t-lookback to close t-skip; NaN without enough listed history."""
    idx = tr_index(panel)
    listed_days = np.cumsum(panel.listed, axis=0)
    out = np.full(idx.shape, np.nan)
    if lookback < len(idx):
        out[lookback:] = idx[lookback - skip: len(idx) - skip] / idx[: len(idx) - lookback] - 1.0
    out[listed_days <= lookback] = np.nan
    out[~panel.listed] = np.nan
    return out


def trailing_volatility(panel: Panel, lookback: int) -> np.ndarray:
    """Standard deviation of daily close-to-close returns over the last `lookback` days."""
    r = (1.0 + panel.ret_co) * (1.0 + panel.ret_oc) - 1.0
    c1 = np.cumsum(np.vstack([np.zeros((1, r.shape[1])), r]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros((1, r.shape[1])), r * r]), axis=0)
    n = lookback
    out = np.full(r.shape, np.nan)
    if n < len(r):
        s1 = c1[n + 1:] - c1[1:-n]
        s2 = c2[n + 1:] - c2[1:-n]
        out[n:] = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0) * n / (n - 1))
    out[np.cumsum(panel.listed, axis=0) <= lookback] = np.nan
    out[~panel.listed] = np.nan
    return out
