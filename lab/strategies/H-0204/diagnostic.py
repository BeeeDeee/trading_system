"""H-0204 mechanism_test events: the same event rule as strategy.py (copied, separate file).

Event on (t, j): LIQ-500 member and tradable on t, predicted next release (last year's
Item 2.02 reaction session + 364 days, first weekday on or after) exactly lead_sessions
weekdays after t, last reaction 40..110 sessions before t, |z(R_last)| >= z_min.
Side -1 if z >= z_min (short leg), +1 if z <= -z_min (long leg). All events are marked,
also those the book skips (slot already open, gross cap).
"""
import math

import numpy as np

SD_WINDOW = 63
SD_MIN_VALID = 40
GAP_MIN = 40
GAP_MAX = 110
OLD_MIN = 230
OLD_MAX = 290
YEAR_DAYS = 364


def _daily_returns(data):
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(close) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = np.where(valid, r, 0.0)
    return r, valid


def _reaction_z(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0204 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = flag.shape
    r, valid = _daily_returns(data)

    g2 = np.full((T, N), np.nan)
    v2 = np.zeros((T, N), dtype=bool)
    if T >= 2:
        v2[1:] = valid[1:] & valid[:-1]
        g2[1:] = (1.0 + r[:-1]) * (1.0 + r[1:]) - 1.0
    m2 = np.full(T, np.nan)
    if T >= 3:
        mem = uni[:-2] & v2[2:]
        cnt = mem.sum(axis=1)
        s = np.where(mem, g2[2:], 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m2[2:] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    z = np.full((T, N), np.nan, dtype=np.float64)
    R, J = np.nonzero(flag == 1.0)
    keep = (R + 1 <= T - 1) & (R - 2 >= 0)
    R, J = R[keep], J[keep]
    if R.size == 0:
        return z
    k = R + 1
    abn = g2[k, J] - m2[k]
    ok2 = v2[k, J] & np.isfinite(abn)
    off = np.arange(-(SD_WINDOW + 1), -1)
    idx = R[:, None] + off[None, :]
    inside = idx >= 0
    idxc = np.maximum(idx, 0)
    vv = valid[idxc, J[:, None]] & inside
    xx = np.where(vv, r[idxc, J[:, None]], 0.0)
    n = vv.sum(axis=1)
    nn = np.maximum(n, 2)
    mean = xx.sum(axis=1) / nn
    dev = np.where(vv, xx - mean[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / (nn - 1)
    sd = np.sqrt(var)
    good = ok2 & (n >= SD_MIN_VALID) & (sd > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = abn / (math.sqrt(2.0) * sd)
    z[R[good], J[good]] = zz[good]
    return z


def events(data, params):
    lead = int(params["lead_sessions"])
    z_min = float(params["z_min"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0204 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = flag.shape
    rows = np.arange(T)
    is_ev = flag == 1.0

    z = _reaction_z(data)

    last_incl = np.where(is_ev, rows[:, None], -1).astype(np.int64)
    np.maximum.accumulate(last_incl, axis=0, out=last_incl)
    r_last = np.full((T, N), -1, dtype=np.int64)
    if T >= 2:
        r_last[1:] = last_incl[:-1]
    gap = rows[:, None] - r_last
    elig = (r_last >= 0) & (gap >= GAP_MIN) & (gap <= GAP_MAX)
    z_last = z[np.maximum(r_last, 0), np.arange(N)[None, :]]
    z_last = np.where(r_last >= 0, z_last, np.nan)

    nan_c = np.zeros((T + 1, N), dtype=np.int64)
    nan_c[1:] = np.cumsum(~np.isfinite(flag), axis=0)
    lo = rows - OLD_MAX
    has_hist = lo >= 0
    nan_in = nan_c[rows] - nan_c[np.maximum(lo, 0)]
    flag_ok = has_hist[:, None] & (nan_in == 0)

    d = np.asarray(data.dates).astype("datetime64[D]")
    d_year = (d.astype(np.int64) + YEAR_DAYS).astype("datetime64[D]")   # + 364 calendar days
    pw = np.busday_offset(d_year, 0, roll="forward")
    tgt = np.busday_offset(d, lead, roll="forward")
    kk = np.searchsorted(pw, tgt, side="left")
    kc = np.minimum(kk, T - 1)
    hit = (kk < T) & (pw[kc] == tgt) & (rows - kc >= OLD_MIN) & (rows - kc <= OLD_MAX)
    pred = hit[:, None] & is_ev[kc, :]

    zl_ok = np.isfinite(z_last)
    zf = np.where(zl_ok, z_last, 0.0)
    big = zl_ok & (np.abs(zf) >= z_min)
    mask = pred & elig & flag_ok & uni & tradable & big
    side = np.where(zf >= z_min, -1.0, 1.0)
    side = np.where(mask, side, 0.0)
    return mask, side
