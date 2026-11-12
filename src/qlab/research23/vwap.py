"""Session VWAP, its volume-weighted standard deviation and ATR (prereg §4).

Row t uses candles of its own UTC session up to and including t (the close of t), ATR the 14 candles up to t.
"""

import numpy as np

from qlab.research23.data import PER_DAY

ATR_N = 14


def bar_price(h, l, c, v, qv) -> np.ndarray:
    """Exact VWAP of each candle (quote / base volume); (h+l+c)/3 where there is no volume."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, qv / v, (h + l + c) / 3.0)


def session_vwap(t, p, v) -> tuple[np.ndarray, np.ndarray]:
    """(V, sigma) per candle, cumulative within the UTC session; NaN until the session has volume.

    Prices are shifted by the session's first price before squaring (the variance is shift-invariant)."""
    sess = np.asarray(t, np.int64) // PER_DAY
    first = np.r_[True, sess[1:] != sess[:-1]]
    ref = p[np.maximum.accumulate(np.where(first, np.arange(len(p)), 0))]
    x = p - ref

    def scum(y):
        cs = np.cumsum(y)
        start = np.maximum.accumulate(np.where(first, np.arange(len(y)), 0))
        return cs - (cs[start] - y[start])

    sv, sx, sxx = scum(v), scum(v * x), scum(v * x * x)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(sv > 0, sx / sv, np.nan)
        var = np.where(sv > 0, sxx / sv - mx * mx, np.nan)
    return mx + ref, np.sqrt(np.maximum(var, 0.0))


def atr(h, l, c, n: int = ATR_N) -> np.ndarray:
    """Mean true range of the n candles up to and including t (NaN for the first n-1)."""
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    cs = np.cumsum(tr)
    out = np.full(len(tr), np.nan)
    out[n - 1:] = (cs[n - 1:] - np.r_[0.0, cs[:-n]]) / n
    return out
