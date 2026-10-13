"""H-0028 mechanism event: Item 2.02 reaction session with |z| >= z_threshold (same rule as the strategy)."""
import math

import numpy as np

SIGMA_WINDOW = 63
SIGMA_MIN_OBS = 40


def _abnormal_returns(ret_co, ret_oc, valid, universe):
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    ok = np.asarray(valid, dtype=bool) & np.isfinite(r)
    r = np.where(ok, r, np.nan)
    mem = np.asarray(universe, dtype=bool) & ok
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return r - mkt[:, None]


def _pre_event_sigma(a, window=SIGMA_WINDOW, min_obs=SIGMA_MIN_OBS):
    T, N = a.shape
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    zero = np.zeros((1, N))
    C = np.vstack([zero, np.cumsum(ok, axis=0, dtype=np.float64)])
    S1 = np.vstack([zero, np.cumsum(x, axis=0)])
    S2 = np.vstack([zero, np.cumsum(x * x, axis=0)])
    t = np.arange(T)
    end = np.clip(t - 1, 0, T)
    start = np.clip(t - 1 - window, 0, T)
    n = C[end] - C[start]
    s1 = S1[end] - S1[start]
    s2 = S2[end] - S2[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    sigma = np.sqrt(np.maximum(var, 0.0))
    return np.where(n >= min_obs, sigma, np.nan)


def events(data, params):
    z_threshold = float(params["z_threshold"])
    if data.universe is None:
        raise ValueError("H-0028 needs the LIQ-500 point-in-time universe")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    a = _abnormal_returns(data.ret_co, data.ret_oc, listed & tradable, universe)
    sigma = _pre_event_sigma(a)
    prev = np.full_like(a, np.nan)
    prev[1:] = a[:-1]
    ar = (1.0 + prev) * (1.0 + a) - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ar / (sigma * math.sqrt(2.0))
    good = np.isfinite(z) & (sigma > 0)
    zz = np.where(good, z, 0.0)
    mask = (ea == 1.0) & universe & tradable & good & (np.abs(zz) >= z_threshold)
    side = np.where(zz > 0, 1.0, -1.0)
    return mask, side
