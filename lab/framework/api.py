"""Helpers a strategy may import (`from lab.framework.api import ...`). All are point-in-time:
row t of the output uses rows 0..t of the input only."""

import numpy as np

from qlab.schedule import period_starts  # noqa: F401  first trading day of each week ('W') / month ('M')


def total_return_index(data) -> np.ndarray:
    """(T, N) total-return index at each close, 1 before the first row."""
    return np.cumprod((1.0 + data.ret_co) * (1.0 + data.ret_oc), axis=0)


def trailing_return(index: np.ndarray, lookback: int, skip: int = 0) -> np.ndarray:
    """index[t - skip] / index[t - lookback] - 1; NaN for the first `lookback` rows."""
    out = np.full(index.shape, np.nan)
    if lookback < len(index):
        out[lookback:] = index[lookback - skip: len(index) - skip] / index[: len(index) - lookback] - 1.0
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` rows (NaN until the window is full); NaNs count as 0."""
    c = np.cumsum(np.nan_to_num(x), axis=0)
    out = np.full(np.shape(x), np.nan)
    out[window - 1:] = c[window - 1:]
    out[window:] -= c[:-window]
    out[window - 1:] /= window
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    m = rolling_mean(x, window)
    m2 = rolling_mean(np.square(np.nan_to_num(x)), window)
    return np.sqrt(np.maximum(m2 - m * m, 0.0) * window / max(window - 1, 1))


def only_on(days: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Keep decisions only on `days` (bool, T); other rows become NaN = hold."""
    out = np.asarray(weights, dtype=float).copy()
    out[~np.asarray(days, dtype=bool)] = np.nan
    return out
