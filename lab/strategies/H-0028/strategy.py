"""H-0028 Earnings-reaction drift on real 8-K release dates (LIQ-500, dollar-neutral).

Decision after the close of row t (rows 0..t only), executed at the open of t+1.

(1) a[s, i] = r[s, i] - equal-weight mean r of the LIQ-500 members on s,
    r = (1 + ret_co)(1 + ret_oc) - 1; r is valid only where the stock is listed and traded that session.
(2) Event at t: ea_reaction_day[t, i] == 1, member at t, tradable at t.
    AR = (1 + a[t-1])(1 + a[t]) - 1, sigma = std(a[t-64 .. t-2]) (63 rows ending t-2, >= 40 valid),
    z = AR / (sigma * sqrt(2)); z >= z_threshold -> long, z <= -z_threshold -> short.
(3) Target held on rows e .. e+hold_days-1 (in the market for hold_days sessions), zero from row e+hold_days.
    A new event replaces the old one; leaving LIQ-500 or not trading at t drops the stock.
(4) +0.5 / max(nL, 10) per long, -0.5 / max(nS, 10) per short, hedge -net equally over the other tradable members.
"""
import math

import numpy as np

PARAMS = {"hold_days": 5, "z_threshold": 2.0}

SIGMA_WINDOW = 63      # sessions ending at t-2
SIGMA_MIN_OBS = 40     # valid abnormal returns required in that window
DIVISOR_FLOOR = 10     # max(nL, 10), max(nS, 10)
LEG_GROSS = 0.5


def abnormal_returns(ret_co, ret_oc, valid, universe):
    """(T, N) abnormal close-to-close return; NaN where the stock's return is not valid."""
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    ok = np.asarray(valid, dtype=bool) & np.isfinite(r)
    r = np.where(ok, r, np.nan)
    mem = np.asarray(universe, dtype=bool) & ok
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return r - mkt[:, None]


def pre_event_sigma(a, window=SIGMA_WINDOW, min_obs=SIGMA_MIN_OBS):
    """sigma[t] = sample std (ddof=1) of a over rows t-window-1 .. t-2; NaN with fewer than min_obs values."""
    T, N = a.shape
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    zero = np.zeros((1, N))
    C = np.vstack([zero, np.cumsum(ok, axis=0, dtype=np.float64)])   # C[k] = sum over rows < k
    S1 = np.vstack([zero, np.cumsum(x, axis=0)])
    S2 = np.vstack([zero, np.cumsum(x * x, axis=0)])
    t = np.arange(T)
    end = np.clip(t - 1, 0, T)                 # exclusive: rows <= t-2
    start = np.clip(t - 1 - window, 0, T)
    n = C[end] - C[start]
    s1 = S1[end] - S1[start]
    s2 = S2[end] - S2[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    var = np.maximum(var, 0.0)
    sigma = np.sqrt(var)
    return np.where(n >= min_obs, sigma, np.nan)


def reaction_z(a, sigma):
    """z[t] = ((1 + a[t-1])(1 + a[t]) - 1) / (sigma[t] * sqrt(2)); NaN where undefined."""
    prev = np.full_like(a, np.nan)
    prev[1:] = a[:-1]
    ar = (1.0 + prev) * (1.0 + a) - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ar / (sigma * math.sqrt(2.0))
    return np.where(np.isfinite(z) & (sigma > 0), z, np.nan)


def event_sides(ea, universe, tradable, z, z_threshold):
    """(long_ev, short_ev) bool (T, N): events after the close of t."""
    ea = np.asarray(ea, dtype=np.float64)
    base = (ea == 1.0) & np.asarray(universe, dtype=bool) & np.asarray(tradable, dtype=bool) & np.isfinite(z)
    zz = np.where(np.isfinite(z), z, 0.0)
    long_ev = base & (zz >= z_threshold)
    short_ev = base & (zz <= -z_threshold)
    return long_ev, short_ev


def book_weights(long_ev, short_ev, universe, tradable, hold_days):
    """Event state machine and dollar-neutral weights, one decision per row."""
    T, N = long_ev.shape
    universe = np.asarray(universe, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    W = np.zeros((T, N), dtype=np.float64)
    side = np.zeros(N, dtype=np.float64)
    age = np.zeros(N, dtype=np.int64)
    active = np.zeros(N, dtype=bool)
    for t in range(T):
        age[active] += 1
        active &= age < hold_days
        new_l = long_ev[t]
        new_s = short_ev[t]
        new = new_l | new_s
        side[new_l] = 1.0
        side[new_s] = -1.0
        age[new] = 0
        active |= new
        elig = universe[t] & tradable[t]
        active &= elig
        side[~active] = 0.0
        longs = active & (side > 0)
        shorts = active & (side < 0)
        nl = int(longs.sum())
        ns = int(shorts.sum())
        if nl == 0 and ns == 0:
            continue
        row = np.zeros(N, dtype=np.float64)
        row[longs] = LEG_GROSS / max(nl, DIVISOR_FLOOR)
        row[shorts] = -LEG_GROSS / max(ns, DIVISOR_FLOOR)
        net = row.sum()
        hedge = elig & ~active
        nh = int(hedge.sum())
        if nh > 0 and net != 0.0:
            row[hedge] = -net / nh
        W[t] = row
    return W


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    z_threshold = float(params["z_threshold"])
    if data.universe is None:
        raise ValueError("H-0028 needs the LIQ-500 point-in-time universe")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ea = data.extras["ea_reaction_day"]
    a = abnormal_returns(data.ret_co, data.ret_oc, listed & tradable, universe)
    sigma = pre_event_sigma(a)
    z = reaction_z(a, sigma)
    long_ev, short_ev = event_sides(ea, universe, tradable, z, z_threshold)
    return book_weights(long_ev, short_ev, universe, tradable, hold_days)
