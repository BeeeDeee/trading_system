"""H-0059 mechanism-test events: the same R&D-increase event the strategy uses (side +1).

The event code is a verbatim copy of strategy.compute_events (only numpy may be imported here).
"""
import numpy as np

LAG = 252
REFIRE = 252
MIN_INTENSITY = 0.05
MIN_INT_GROWTH = 0.05


def _changed(x):
    fin = np.isfinite(x)
    out = np.zeros(x.shape, dtype=bool)
    if x.shape[0] > 1:
        prev_fin = fin[:-1]
        cur_fin = fin[1:]
        with np.errstate(invalid="ignore"):
            diff = x[1:] != x[:-1]
        out[1:] = cur_fin & (~prev_fin | (prev_fin & diff))
    return out


def _lag(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def _compute_events(rnd, assets, eligible, growth_min):
    T, N = rnd.shape
    new_filing = _changed(rnd) | _changed(assets)
    R0, A0 = rnd, assets
    R1, A1 = _lag(rnd, LAG), _lag(assets, LAG)
    with np.errstate(invalid="ignore", divide="ignore"):
        fin = np.isfinite(R0) & np.isfinite(A0) & np.isfinite(R1) & np.isfinite(A1)
        pos = fin & (R1 > 0) & (A0 > 0) & (A1 > 0)
        safe_A0 = np.where(pos, A0, 1.0)
        safe_A1 = np.where(pos, A1, 1.0)
        safe_R1 = np.where(pos, R1, 1.0)
        int0 = np.where(pos, R0, 0.0) / safe_A0
        int1 = safe_R1 / safe_A1
        cond_b = int0 >= MIN_INTENSITY
        cond_c = (np.where(pos, R0, 0.0) / safe_R1 - 1.0) >= growth_min
        cond_d = (int0 / int1 - 1.0) >= MIN_INT_GROWTH
    cand = new_filing & pos & cond_b & cond_c & cond_d & eligible
    ev = np.zeros((T, N), dtype=bool)
    last = np.full(N, -(10 ** 9), dtype=np.int64)
    for t in np.flatnonzero(cand.any(axis=1)):
        fire = cand[t] & ((t - last) > REFIRE)
        ev[t] = fire
        last[fire] = t
    return ev


def events(data, params):
    growth_min = float(params["growth_min"])
    rnd = np.asarray(data.extras["sf1_art_rnd"], dtype=np.float64)
    assets = np.asarray(data.extras["sf1_art_assets"], dtype=np.float64)
    if data.universe is None:
        raise ValueError("H-0059 needs the point-in-time LIQ-500 universe")
    eligible = np.asarray(data.universe, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    mask = _compute_events(rnd, assets, eligible, growth_min)
    return mask, None
