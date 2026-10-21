"""H-0069 mechanism-test events: |r - median| >= k_sigma * own trailing sigma, either direction, side -1.

Same event definition as strategy.py (code duplicated so this module needs only numpy):
member and tradable on t, valid close-to-close return on t, >= 2/3 of the previous lookback_days
sessions with a valid return, z = (r - cross-sectional median of members) / sigma(t-L .. t-1).
All events are marked (the strategy's cap of 20 shorts per day is a book rule, not part of the event).
"""
import numpy as np

MIN_HISTORY_FRACTION = 2.0 / 3.0


def _row_median(x, valid):
    T, N = x.shape
    vals = np.where(valid, x, np.inf)
    s = np.sort(vals, axis=1)
    n = valid.sum(axis=1)
    lo = np.clip((n - 1) // 2, 0, max(N - 1, 0))
    hi = np.clip(n // 2, 0, max(N - 1, 0))
    rows = np.arange(T)
    med = np.full(T, np.nan)
    has = n > 0
    if N > 0:
        a = s[rows, lo]
        b = s[rows, hi]
        med[has] = 0.5 * (a[has] + b[has])
    return med


def events(data, params):
    k_sigma = float(params["k_sigma"])
    lookback = int(params["lookback_days"])
    if data.universe is None:
        raise ValueError("H-0069 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    T, N = ret_co.shape

    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    has_px = np.isfinite(close) & listed
    prev_px = np.zeros((T, N), dtype=bool)
    if T > 1:
        prev_px[1:] = has_px[:-1]
    valid = has_px & prev_px & tradable & np.isfinite(r)

    rv = np.where(valid, r, 0.0)
    P1 = np.zeros((T + 1, N))
    P2 = np.zeros((T + 1, N))
    PC = np.zeros((T + 1, N))
    np.cumsum(rv, axis=0, out=P1[1:])
    np.cumsum(rv * rv, axis=0, out=P2[1:])
    np.cumsum(valid.astype(np.float64), axis=0, out=PC[1:])
    t_idx = np.arange(T)
    start = np.maximum(t_idx - lookback, 0)
    s1 = P1[t_idx] - P1[start]
    s2 = P2[t_idx] - P2[start]
    cnt = np.rint(PC[t_idx] - PC[start])
    enough = cnt >= MIN_HISTORY_FRACTION * lookback - 1e-9
    enough &= cnt >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.where(cnt > 0, cnt, 1.0)) / np.where(cnt > 1, cnt - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    sigma = np.sqrt(var)

    m = _row_median(r, valid & universe)

    base = universe & tradable & valid & enough & (sigma > 0) & np.isfinite(m)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(base, (r - m[:, None]) / np.where(sigma > 0, sigma, 1.0), np.nan)
    eligible = base & np.isfinite(z)
    mask = eligible & (np.abs(np.where(eligible, z, 0.0)) >= k_sigma)
    side = np.full(mask.shape, -1.0)
    return mask, side
