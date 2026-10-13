"""Helpers a scalping strategy may import (`from lab.framework.intraday_api import ...`). Everything here is
point-in-time: row t of a result uses rows <= t only, and higher timeframes count only once a block has closed.

All functions work column-wise on (T, N) float arrays (axis 0 = time). NaN in = NaN out where a window is not full.
"""

import numpy as np


def shift(x: np.ndarray, k: int) -> np.ndarray:
    """x[t - k] (k >= 0; NaN for the first k rows). Negative k would look ahead and is refused."""
    if k < 0:
        raise ValueError("shift(x, k) needs k >= 0: a negative shift looks into the future")
    out = np.full(x.shape, np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k] if k else x
    return out


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of the last n rows including the current one; NaN until n rows exist."""
    c = np.cumsum(np.nan_to_num(np.asarray(x, dtype=float)), axis=0)
    out = np.full(x.shape, np.nan)
    out[n - 1:] = c[n - 1:] - np.vstack([np.zeros((1,) + x.shape[1:]), c[: len(x) - n]])
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    return rolling_sum(x, n) / n


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = rolling_mean(x, n)
    v = rolling_mean(x * x, n) - m * m
    return np.sqrt(np.maximum(v, 0.0) * n / max(n - 1, 1))


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    out = np.full(x.shape, np.nan)
    if len(x) >= n:
        out[n - 1:] = swv(np.asarray(x, dtype=float), n, axis=0).max(axis=-1)
    return out


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    return -rolling_max(-np.asarray(x, dtype=float), n)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average with span n (alpha = 2 / (n + 1)), started at the first finite value."""
    a = 2.0 / (n + 1.0)
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    cur = np.full(x.shape[1:], np.nan)
    for t in range(len(x)):
        ok = np.isfinite(x[t])
        cur = np.where(ok, np.where(np.isfinite(cur), cur + a * (x[t] - cur), x[t]), cur)
        out[t] = cur
    return out


def log_return(c: np.ndarray, k: int = 1) -> np.ndarray:
    """log(c[t] / c[t - k])."""
    return np.log(c / shift(c, k))


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = shift(c, 1)
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    return rolling_mean(true_range(h, l, c), n)


def zscore(x: np.ndarray, n: int) -> np.ndarray:
    """(x - rolling mean) / rolling std over the last n rows (never the full sample)."""
    sd = rolling_std(x, n)
    return np.where(sd > 0, (x - rolling_mean(x, n)) / np.where(sd > 0, sd, 1.0), np.nan)


def block_value(x: np.ndarray, k: int, how: str = "last") -> np.ndarray:
    """Higher-timeframe bar of k rows (k divides the day, blocks aligned to 00:00 UTC), as seen at each row:
    the aggregate of the LAST COMPLETED block ("last" = its final value, "first", "max", "min", "sum").
    Rows inside a block see the previous block; the current, unfinished block is never used."""
    T = len(x)
    nb = T // k
    blocks = np.asarray(x[: nb * k], dtype=float).reshape((nb, k) + x.shape[1:])
    agg = {"last": lambda b: b[:, -1], "first": lambda b: b[:, 0], "max": lambda b: b.max(axis=1),
           "min": lambda b: b.min(axis=1), "sum": lambda b: b.sum(axis=1)}[how](blocks)
    out = np.full(x.shape, np.nan)
    rows = np.arange(T)
    done = rows // k - 1                      # the block before the one containing row t ...
    done = np.where(rows % k == k - 1, rows // k, done)   # ... or the block that closes exactly at row t
    ok = (done >= 0) & (done < nb)
    out[ok] = agg[done[ok]]
    return out


def hold_for(signal: np.ndarray, n: int) -> np.ndarray:
    """Turn entry signals (+1 / -1 / 0 / NaN) into positions held for n rows after the entry row (a new
    non-zero signal replaces the old one). Row t decides after the close of t, as everywhere."""
    s = np.nan_to_num(np.asarray(signal, dtype=float))
    T = len(s)
    idx = np.where(s != 0, np.arange(T)[:, None], -10**9)
    np.maximum.accumulate(idx, axis=0, out=idx)
    held = np.take_along_axis(s, np.clip(idx, 0, T - 1), axis=0)
    return np.where((np.arange(T)[:, None] - idx) < n, held, 0.0)


def normalize_gross(w: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Scale each row so that sum |w| <= cap."""
    g = np.abs(np.nan_to_num(w)).sum(axis=1, keepdims=True)
    return np.where(g > cap, w * cap / np.where(g > 0, g, 1.0), w)


def cross_rank(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Cross-sectional rank in [-0.5, 0.5] per row over the finite (and masked) entries; NaN elsewhere."""
    v = np.where(mask, x, np.nan) if mask is not None else np.asarray(x, dtype=float)
    ok = np.isfinite(v)
    r = np.argsort(np.argsort(np.where(ok, v, np.inf), axis=1), axis=1).astype(float)
    n = ok.sum(axis=1, keepdims=True)
    return np.where(ok, (r + 0.5) / np.maximum(n, 1) - 0.5, np.nan)
