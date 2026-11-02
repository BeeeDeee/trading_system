"""H-0091 Disposition-gated earnings-news drift (LIQ-500, dollar-neutral, 5-session hold).

Row t is decided after the close of t from rows 0..t only; the engine fills at the open of t+1.
"""
import numpy as np

PARAMS = {"ref_window": 252, "z_min": 1.0}

SD_WINDOW = 63        # fixed in the card: own volatility over sessions t-63 .. t-1
SD_MIN_VALID = 40     # at least 40 valid values, else no event
REF_MIN_SHARE_NUM = 4  # at least 80 % (= 4/5) of the reference window valid
REF_MIN_SHARE_DEN = 5
HOLD = 5              # events of t-4 .. t are active after the close of t
MIN_PER_SIDE = 3
LEG = 0.5


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _cum0(x, dtype):
    """Cumulative sum along rows with a leading zero row: C[t] = sum of rows < t."""
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=dtype)
    np.cumsum(x, axis=0, dtype=dtype, out=out[1:])
    return out


def daily_returns(data):
    """r[t,j] = (1 + ret_co) * (1 + ret_oc) - 1, NaN where the stock has no close-to-close return on t
    (not listed, no close on t, or no close on t-1)."""
    r = (1.0 + _f64(data.ret_co)) * (1.0 + _f64(data.ret_oc)) - 1.0
    close = _f64(data.close)
    listed = np.asarray(data.listed, dtype=bool)
    has_px = np.isfinite(close)
    prev_px = np.zeros_like(has_px)
    prev_px[1:] = has_px[:-1]
    valid = listed & has_px & prev_px & np.isfinite(r)
    r[~valid] = np.nan
    return r, valid


def market_adjusted(r, valid, uni):
    """a[t,j] = r[t,j] - equal-weighted mean of r[t,.] over universe members with a return on t."""
    mem = valid & uni
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    m = np.full(r.shape[0], np.nan)
    pos = cnt > 0
    m[pos] = tot[pos] / cnt[pos]
    return r - m[:, None]


def prior_sd(a):
    """Sample standard deviation (ddof=1) of a[.,j] over rows t-63 .. t-1 (clipped at row 0),
    NaN with fewer than 40 valid values."""
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


def overhang(r, valid, data, ref_window):
    """CGO[s,j] = P[s,j] / R[s,j] - 1 at the close of s, where P is the total-return index (cumulative
    product of 1 + r) and R the dollar-volume-weighted mean of P over sessions s-W+1 .. s (the card's
    t-W .. t-1 for s = t-1). NaN unless at least 80 % of the W sessions have a positive dollar volume
    and a price, or without a price on s."""
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


def event_sides(data, params):
    """(T, N) int8: +1 / -1 for a large-reaction release on t in the top / bottom CGO tercile of t-1, else 0."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0091 needs the liq_n point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = _f64(data.extras["ea_reaction_day"])
    z_min = float(params["z_min"])
    ref_window = int(params["ref_window"])

    r, valid = daily_returns(data)
    a = market_adjusted(r, valid, uni)
    sd = prior_sd(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = a / sd
    del a, sd
    large = (ea == 1.0) & uni & tradable & np.isfinite(z) & (np.abs(z) >= z_min)
    del z, ea
    large[0] = False  # no t-1 overhang on the first row

    cgo = overhang(r, valid, data, ref_window)
    del r, valid

    T, N = uni.shape
    side = np.zeros((T, N), dtype=np.int8)
    for t in np.flatnonzero(large.any(axis=1)):
        prev = cgo[t - 1]
        pool = uni[t - 1] & np.isfinite(prev)
        if not pool.any():
            continue
        q1, q2 = np.quantile(prev[pool], [1.0 / 3.0, 2.0 / 3.0])
        cand = large[t] & np.isfinite(prev)
        hi = cand & (prev >= q2)
        lo = cand & (prev <= q1)
        both = hi & lo  # only possible if q1 == q2 (degenerate pool): ambiguous, no event
        side[t, hi & ~both] = 1
        side[t, lo & ~both] = -1
    return side


def book(side, uni, tradable):
    """Active set after the close of t: latest event of t-4 .. t per stock, still a member and tradable.
    +0.5 equal-weighted over longs, -0.5 over shorts if both have >= 3 names, else flat."""
    T, N = side.shape
    act = np.zeros((T, N), dtype=np.int8)
    for lag in range(HOLD - 1, -1, -1):  # oldest first, newer events overwrite
        sh = np.zeros_like(side)
        sh[lag:] = side[: T - lag]
        act = np.where(sh != 0, sh, act)
    act[~(uni & tradable)] = 0
    nl = (act > 0).sum(axis=1)
    ns = (act < 0).sum(axis=1)
    on = (nl >= MIN_PER_SIDE) & (ns >= MIN_PER_SIDE)
    w = np.zeros((T, N), dtype=np.float64)
    wl = np.where(on, LEG / np.maximum(nl, 1), 0.0)
    ws = np.where(on, LEG / np.maximum(ns, 1), 0.0)
    w += np.where(act > 0, wl[:, None], 0.0)
    w -= np.where(act < 0, ws[:, None], 0.0)
    return w


def target_weights(data, params):
    side = event_sides(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    return book(side, uni, tradable)
