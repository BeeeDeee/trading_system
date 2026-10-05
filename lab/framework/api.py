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


# ---------------------------------------------------------------------------- multiple timeframes
# A strategy decides once a day, but its signals may live on any horizon: weekly or monthly bars built
# from the daily rows, slow filters confirming fast signals. These helpers build higher-timeframe bars
# point-in-time: a week or month counts only once it is complete, i.e. from the first row of the next one.

def _period_key(dates, unit: str) -> np.ndarray:
    d = np.asarray(dates, dtype="datetime64[D]")
    if unit == "W":
        return (d.astype(np.int64) + 3) // 7          # Monday-based weeks (1970-01-01 was a Thursday)
    if unit == "M":
        return d.astype("datetime64[M]").astype(np.int64)
    if unit == "Q":
        return d.astype("datetime64[M]").astype(np.int64) // 3
    raise ValueError(f"unknown unit {unit!r} (W, M, Q)")


def last_completed_row(dates, unit: str) -> np.ndarray:
    """(T,) index of the last row of the most recent *completed* period before row t's period; -1 if none.
    Row t's own period is still open, so its last row is not known yet."""
    key = _period_key(dates, unit)
    start = np.r_[True, key[1:] != key[:-1]]
    idx = np.arange(len(key))
    cur_start = np.maximum.accumulate(np.where(start, idx, 0))
    return cur_start - 1


def completed_period_value(x: np.ndarray, dates, unit: str) -> np.ndarray:
    """x (T,) or (T, N) as of the close of the last completed week/month/quarter; NaN before the first."""
    x = np.asarray(x, dtype=float)
    rows = last_completed_row(dates, unit)
    out = x[np.maximum(rows, 0)].copy()
    out[rows < 0] = np.nan
    return out


def period_return(index: np.ndarray, dates, unit: str, periods: int) -> np.ndarray:
    """Return over the last `periods` completed weeks/months/quarters, from a (T, N) total-return index."""
    key = _period_key(dates, unit)
    ends = np.flatnonzero(np.r_[key[1:] != key[:-1], False])     # last row of every completed period
    rows = last_completed_row(dates, unit)
    pos = np.searchsorted(ends, rows)                               # position of that row among the period ends
    back = pos - periods
    ok = (rows >= 0) & (back >= 0)
    index = np.asarray(index, dtype=float)
    out = np.full(index.shape, np.nan)
    out[ok] = index[rows[ok]] / index[ends[back[ok]]] - 1.0
    return out


def ema(x: np.ndarray, span: float) -> np.ndarray:
    """Exponential moving average along rows (alpha = 2 / (span + 1)); NaNs carry the previous value."""
    x = np.asarray(x, dtype=float)
    a = 2.0 / (span + 1.0)
    out = np.full(x.shape, np.nan)
    prev = np.full(x.shape[1:], np.nan)
    for t in range(len(x)):
        cur = np.where(np.isnan(prev), x[t], a * x[t] + (1 - a) * prev)
        prev = np.where(np.isnan(x[t]), prev, cur)
        out[t] = prev
    return out
