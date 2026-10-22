"""H-0073 mechanism event: after the close of t the latest usable VIX/VIX3M ratio (same timing and
staleness rule as the strategy, code duplicated from strategy.py so this file is self-contained) is at or
below k - d. VIXY is marked with side -1; the peers SHY, IEF, TLT, AGG, GLD are never marked."""
import numpy as np

MAX_AGE = 5
TRADED = "VIXY"
VIX_KEY = "fred_macro.VIXCLS"
VXV_KEY = "fred_vxv_1d.VXVCLS"


def _ffill_and_last_obs(x):
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    idx = np.arange(T)
    fin = np.isfinite(x)
    prev = np.empty(T)
    prev[0] = np.nan
    prev[1:] = x[:-1]
    new = fin & ~(np.isfinite(prev) & (x == prev))
    last_new = np.maximum.accumulate(np.where(new, idx, -1))
    last_fin = np.maximum.accumulate(np.where(fin, idx, -1))
    vals = np.where(last_fin >= 0, x[np.maximum(last_fin, 0)], np.nan)
    return vals, last_new


def _ratio_and_age(vix, vxv):
    v1, s1 = _ffill_and_last_obs(vix)
    v2, s2 = _ffill_and_last_obs(vxv)
    T = v1.shape[0]
    idx = np.arange(T)
    s = np.minimum(s1, s2)
    age = np.where(s >= 0, idx - s + 1, np.iinfo(np.int64).max).astype(np.int64)
    ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
    r = np.full(T, np.nan)
    r[ok] = v1[ok] / v2[ok]
    return r, age


def events(data, params):
    k = float(params["k"])
    d = float(params["d"])
    T = len(data.dates)
    N = len(data.instruments)
    j = data.col(TRADED)
    vix = np.asarray(data.series[VIX_KEY], dtype=float).reshape(T)
    vxv = np.asarray(data.series[VXV_KEY], dtype=float).reshape(T)
    r, age = _ratio_and_age(vix, vxv)
    usable = np.isfinite(r) & (age <= MAX_AGE)
    steep = usable & (np.where(usable, r, np.inf) <= k - d + 1e-12)

    ok = np.asarray(data.tradable, dtype=bool)[:, j] & np.asarray(data.listed, dtype=bool)[:, j]
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)[:, j]

    mask = np.zeros((T, N), dtype=bool)
    mask[:, j] = steep & ok
    side = np.ones((T, N), dtype=float)
    side[:, j] = -1.0
    return mask, side
