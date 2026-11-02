"""H4: sleeves mixed monthly, crypto calendar aligned to US trading days (prereg §4.5)."""

from collections.abc import Callable

import numpy as np

from qlab.research19 import setup as S
from qlab.schedule import period_starts


def align_to_calendar(crypto_dates: np.ndarray, crypto_ret: np.ndarray, us_dates: np.ndarray) -> np.ndarray:
    """Return of a crypto day-series assigned to each US trading day d: the compounded return of crypto days in
    (d_prev, d]. A weekend rolls into Monday. Crypto days after the last US day are dropped; for the first US day
    everything up to it is compounded."""
    cd = np.asarray(crypto_dates, dtype="datetime64[D]")
    ud = np.asarray(us_dates, dtype="datetime64[D]")
    r = np.nan_to_num(np.asarray(crypto_ret, dtype=float))
    hi = np.searchsorted(cd, ud, side="right")
    lo = np.r_[0, hi[:-1]]
    return np.array([np.prod(1.0 + r[a:b]) - 1.0 for a, b in zip(lo, hi)])


def mix(returns: np.ndarray, dates: np.ndarray, weight_fn: Callable[[np.ndarray], np.ndarray] | np.ndarray,
        cost_bps: float = S.ETF_COST_BPS) -> np.ndarray:
    """Sleeves (columns) reset to target weights after the close of the first trading day of each month (same timing
    as `research10.sim.monthly_mix`), drifting in between. The target comes from `weight_fn(returns[:t])` (past data
    only) or is fixed. A reset costs cost_bps * sum|target - drifted weights| on the day it takes effect (not for
    the first allocation)."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    T, k = r.shape
    reset = np.r_[False, period_starts(dates, "M")[:-1]]
    reset[0] = True
    out = np.empty(T)
    w_end = None
    starts = list(np.flatnonzero(reset)) + [T]
    for a, b in zip(starts[:-1], starts[1:]):
        w = np.asarray(weight_fn(r[:a]) if callable(weight_fn) else weight_fn, dtype=float)
        g = np.cumprod(1.0 + r[a:b], axis=0)
        pv = g @ w
        prev = np.r_[1.0, pv[:-1]]
        seg = pv / prev - 1.0
        if w_end is not None:
            seg[0] = (1.0 + seg[0]) * (1.0 - cost_bps / 1e4 * np.abs(w - w_end).sum()) - 1.0
        out[a:b] = seg
        w_end = w * g[-1] / pv[-1]
    return out


def equal_weights(k: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda past: np.full(k, 1.0 / k)


def inverse_vol_weights(window: int = 252) -> Callable[[np.ndarray], np.ndarray]:
    """Inverse volatility over the last `window` days (equal weights until enough history)."""
    def fn(past: np.ndarray) -> np.ndarray:
        k = past.shape[1]
        if len(past) < window:
            return np.full(k, 1.0 / k)
        sd = past[-window:].std(axis=0, ddof=1)
        if not np.all(sd > 0):
            return np.full(k, 1.0 / k)
        w = 1.0 / sd
        return w / w.sum()
    return fn
