"""H-0191 mechanism event: forward-split ex-date of a LIQ-500 member (same detection as strategy.py)."""
import numpy as np

SPLIT_RATIOS = (1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
TOLERANCE = 0.03
GAP_SESSIONS = 21


def _split_factor(data):
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    T, N = close.shape
    prev = np.full((T, N), np.nan)
    if T > 1:
        prev[1:] = close[:-1]
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = (np.isfinite(prev) & (prev > 0) & np.isfinite(close) & (close > 0)
             & np.isfinite(ret_co) & np.isfinite(ret_oc))
    f = np.full((T, N), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        f[valid] = prev[valid] * (1.0 + r[valid]) / close[valid]
    return f


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0191 needs the LIQ-500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    f = _split_factor(data)
    hit = np.zeros(f.shape, dtype=bool)
    for R in SPLIT_RATIOS:
        if R < float(params["min_ratio"]) - 1e-12:
            continue
        with np.errstate(invalid="ignore"):
            hit |= np.isfinite(f) & (np.abs(f / R - 1.0) <= TOLERANCE)
    cand = uni & tradable & hit
    mask = np.zeros(cand.shape, dtype=bool)
    rows, cols = np.nonzero(cand)
    order = np.lexsort((rows, cols))
    last = {}
    for k in order:
        t, j = int(rows[k]), int(cols[k])
        p = last.get(j)
        if p is not None and t - p < GAP_SESSIONS:
            continue
        mask[t, j] = True
        last[j] = t
    return mask, None
