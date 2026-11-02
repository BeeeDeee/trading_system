"""H-0091 mechanism events: an 8-K Item 2.02 reaction session on t with |z| >= z_min (market-adjusted
close-to-close return over its own 63-session sd of t-63 .. t-1), in the top (+1) or bottom (-1) tercile of
the LIQ-500 capital-gains overhang at the close of t-1. Same logic as strategy.event_sides (copied so that
this file is self-contained)."""
import numpy as np

SD_WINDOW = 63
SD_MIN_VALID = 40
REF_MIN_SHARE_NUM = 4
REF_MIN_SHARE_DEN = 5


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _cum0(x, dtype):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=dtype)
    np.cumsum(x, axis=0, dtype=dtype, out=out[1:])
    return out


def _daily_returns(data):
    r = (1.0 + _f64(data.ret_co)) * (1.0 + _f64(data.ret_oc)) - 1.0
    close = _f64(data.close)
    listed = np.asarray(data.listed, dtype=bool)
    has_px = np.isfinite(close)
    prev_px = np.zeros_like(has_px)
    prev_px[1:] = has_px[:-1]
    valid = listed & has_px & prev_px & np.isfinite(r)
    r[~valid] = np.nan
    return r, valid


def _market_adjusted(r, valid, uni):
    mem = valid & uni
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    m = np.full(r.shape[0], np.nan)
    pos = cnt > 0
    m[pos] = tot[pos] / cnt[pos]
    return r - m[:, None]


def _prior_sd(a):
    T = a.shape[0]
    ok = np.isfinite(a)
    a0 = np.where(ok, a, 0.0)
    c = _cum0(ok, np.int32)
    s1 = _cum0(a0, np.float64)
    s2 = _cum0(a0 * a0, np.float64)
    del a0
    idx = np.arange(T)
    lo = np.maximum(idx - SD_WINDOW, 0)
    n = (c[idx] - c[lo]).astype(np.float64)
    m1 = s1[idx] - s1[lo]
    m2 = s2[idx] - s2[lo]
    del s1, s2, c
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (m2 - m1 * m1 / n) / (n - 1.0)
    var[n < SD_MIN_VALID] = np.nan
    var[var <= 0] = np.nan
    return np.sqrt(var)


def _overhang(r, valid, data, ref_window):
    W = int(ref_window)
    if W < 1:
        raise ValueError("ref_window must be >= 1")
    T = r.shape[0]
    P = np.cumprod(1.0 + np.where(valid, r, 0.0), axis=0)
    dv = _f64(data.dollar_volume)
    has_px = np.isfinite(_f64(data.close)) & np.asarray(data.listed, dtype=bool)
    ok = has_px & np.isfinite(dv) & (dv > 0)
    dv = np.where(ok, dv, 0.0)
    cx = _cum0(dv * P, np.float64)
    cw = _cum0(dv, np.float64)
    del dv
    cn = _cum0(ok, np.int32)
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - W, 0)
    n = cn[hi] - cn[lo]
    sx = cx[hi] - cx[lo]
    del cx
    sw = cw[hi] - cw[lo]
    del cw, cn
    good = (n * REF_MIN_SHARE_DEN >= REF_MIN_SHARE_NUM * W) & (sw > 0) & has_px
    with np.errstate(invalid="ignore", divide="ignore"):
        cgo = P * sw / sx - 1.0
    cgo[~good] = np.nan
    cgo[~np.isfinite(cgo)] = np.nan
    return cgo


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0091 needs the liq_n point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = _f64(data.extras["ea_reaction_day"])
    z_min = float(params["z_min"])
    ref_window = int(params["ref_window"])

    r, valid = _daily_returns(data)
    a = _market_adjusted(r, valid, uni)
    sd = _prior_sd(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = a / sd
    del a, sd
    large = (ea == 1.0) & uni & tradable & np.isfinite(z) & (np.abs(z) >= z_min)
    del z, ea
    large[0] = False

    cgo = _overhang(r, valid, data, ref_window)
    del r, valid

    T, N = uni.shape
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(large.any(axis=1)):
        prev = cgo[t - 1]
        pool = uni[t - 1] & np.isfinite(prev)
        if not pool.any():
            continue
        q1, q2 = np.quantile(prev[pool], [1.0 / 3.0, 2.0 / 3.0])
        cand = large[t] & np.isfinite(prev)
        hi = cand & (prev >= q2)
        lo = cand & (prev <= q1)
        both = hi & lo
        side[t, hi & ~both] = 1.0
        side[t, lo & ~both] = -1.0
    mask = side != 0
    return mask, side
