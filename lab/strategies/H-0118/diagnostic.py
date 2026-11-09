"""H-0118 mechanism event: at an in-window weekly decision date a scored LIQ-500 member enters the
model's long leg (+1) or short leg (-1) after not being in that leg at the previous decision date.

Same features, clocks, walk_forward and legs as strategy.py (copied: this file must stand alone).
"""
import numpy as np
from lab.framework.api import period_starts, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 252
CLOCK_CAP = 63
WINDOW = 15
MIN_NAMES = 50
HIST = 252
RET_N = 21
VOL_N = 63
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}


def _mon(k):
    return np.array(k).astype("timedelta64[M]")


def _day(k):
    return np.array(k).astype("timedelta64[D]")


def _weekday(d):
    return (d.astype("datetime64[D]").astype(np.int64) + 3) % 7


def _holidays(dates):
    yrs = np.unique(dates.astype("datetime64[Y]"))
    m0 = yrs.astype("datetime64[M]")
    sep1 = (m0 + _mon(8)).astype("datetime64[D]")
    labor = np.busday_offset(sep1, 0, roll="forward", weekmask="Mon")
    nov1 = (m0 + _mon(10)).astype("datetime64[D]")
    thanks = np.busday_offset(nov1, 3, roll="forward", weekmask="Thu")
    dec25 = (m0 + _mon(11)).astype("datetime64[D]") + _day(24)
    wd = _weekday(dec25)
    shift = np.where(wd == 5, -1, np.where(wd == 6, 1, 0)).astype("timedelta64[D]")
    xmas = dec25 + shift
    return np.sort(np.concatenate([labor, thanks, xmas]))


def _last_strict(flag):
    T = flag.shape[0]
    a = np.where(flag, np.arange(T), -1)
    out = np.full(T, -1, dtype=np.int64)
    if T > 1:
        out[1:] = np.maximum.accumulate(a)[:-1]
    return out


def deadline_clocks(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    hol = _holidays(dates)
    y0 = dates.astype("datetime64[Y]").astype("datetime64[M]")
    oct31 = (y0 + _mon(10)).astype("datetime64[D]") - _day(1)
    dec31 = (y0 + _mon(12)).astype("datetime64[D]") - _day(1)
    d_fund = np.busday_offset(oct31, 0, roll="backward", holidays=hol)
    d_ind = np.busday_offset(dec31, 0, roll="backward", holidays=hol)
    nxt = np.where(dates <= d_fund, d_fund, d_ind)
    cnt = np.busday_count(dates + _day(1), nxt + _day(1), holidays=hol)
    clock_to = np.minimum(np.maximum(cnt, 0), CLOCK_CAP).astype(np.float64)
    is_dl = (dates == d_fund) | (dates == d_ind)
    mon_id = dates.astype("datetime64[M]").astype(np.int64)
    month = mon_id % 12 + 1
    ends = np.zeros(T, dtype=bool)
    if T > 1:
        ends[:-1] = mon_id[1:] != mon_id[:-1]
    last_fund = _last_strict(ends & (month == 10))
    last_ind = _last_strict(ends & (month == 12))
    last_any = np.maximum(last_fund, last_ind)
    idx = np.arange(T)
    since = np.where(last_any >= 0, idx - last_any, np.nan).astype(np.float64)
    since = np.where(is_dl, 0.0, since)
    clock_since = np.minimum(since, CLOCK_CAP)
    return clock_to, clock_since, last_fund, last_ind


def in_tax_window(clock_to, clock_since):
    with np.errstate(invalid="ignore"):
        return (clock_to <= WINDOW) | ((clock_since >= 1) & (clock_since <= WINDOW))


def _csum(x):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def _window_sum(cs, n, T):
    out = np.full((T,) + cs.shape[1:], np.nan)
    if T >= n:
        out[n - 1:] = cs[n:] - cs[:T - n + 1]
    return out


def stock_features(tri, dv, listed, last_fund, last_ind):
    tri = np.asarray(tri, dtype=np.float64)
    T, N = tri.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.asarray(listed, dtype=bool) & np.isfinite(tri) & (tri > 0)
        hist = _window_sum(_csum(ok.astype(np.float64)), HIST, T)
        full = hist >= HIST - 0.5
        trz = np.where(ok, tri, 0.0)
        dv = np.asarray(dv, dtype=np.float64)
        dvz = np.where(ok & np.isfinite(dv) & (dv > 0), dv, 0.0)
        num = _window_sum(_csum(dvz * trz), HIST, T)
        den = _window_sum(_csum(dvz), HIST, T)
        del dvz
        ref = np.where(den > 0, num / den, np.nan)
        del num, den
        overhang = np.log(tri / ref)
        del ref
        ret21 = np.full((T, N), np.nan)
        if T > RET_N:
            ret21[RET_N:] = tri[RET_N:] / tri[:-RET_N] - 1.0
        r = np.zeros((T, N))
        if T > 1:
            r[1:] = np.where(ok[1:] & ok[:-1], trz[1:] / np.where(ok[:-1], trz[:-1], 1.0) - 1.0, 0.0)
        s1 = _window_sum(_csum(r), VOL_N, T)
        s2 = _window_sum(_csum(r * r), VOL_N, T)
        del r
        var = (s2 - s1 * s1 / VOL_N) / (VOL_N - 1)
        vol63 = np.sqrt(np.maximum(var, 0.0))
        del s1, s2, var

        def since_ref(last):
            has = last >= 0
            li = np.where(has, last, 0)
            base = np.where(has[:, None] & ok[li], trz[li], np.nan)
            return tri / base - 1.0

        tax_year = since_ref(last_ind)
        fund_year = since_ref(last_fund)
    feats = [tax_year, fund_year, overhang, ret21, vol63]
    good = full.copy()
    for f in feats:
        good &= np.isfinite(f)
    return [np.where(good, f, np.nan) for f in feats]


def stock_features_blocked(tri, dv, listed, last_fund, last_ind, block=256):
    tri = np.asarray(tri)
    dv = np.asarray(dv)
    listed = np.asarray(listed)
    T, N = tri.shape
    out = [np.full((T, N), np.nan, dtype=np.float32) for _ in range(5)]
    for j0 in range(0, N, block):
        j1 = min(N, j0 + block)
        fb = stock_features(tri[:, j0:j1], dv[:, j0:j1], listed[:, j0:j1], last_fund, last_ind)
        for k in range(5):
            out[k][:, j0:j1] = fb[k]
        del fb
    return out


def model_inputs(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0118 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    clock_to, clock_since, last_fund, last_ind = deadline_clocks(data.dates)
    tri = total_return_index(data)
    feats = stock_features_blocked(tri, data.dollar_volume, data.listed, last_fund, last_ind)
    del tri
    T, N = uni.shape
    vec = uni.copy()
    for f in feats:
        vec &= np.isfinite(f)
    zs = []
    while feats:
        f = feats.pop(0)
        f[~vec] = np.nan
        z = np.asarray(cs_zscore(f, uni), dtype=np.float32)
        del f
        z = np.array(z, dtype=np.float32, copy=True)
        z[~vec] = np.nan
        zs.append(z)
    ct = np.broadcast_to(clock_to.astype(np.float32)[:, None], (T, N))
    cs = np.broadcast_to(clock_since.astype(np.float32)[:, None], (T, N))
    X = stack(*zs, ct, cs)
    del zs
    return X, uni, clock_to, clock_since


def model_scores(data, params):
    X, uni, clock_to, clock_since = model_inputs(data)
    h = HORIZON
    y = cs_rank(forward_return(data, h), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=h, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), uni, clock_to, clock_since


def _mask(days, T):
    d = np.asarray(days)
    if d.dtype == bool and d.shape == (T,):
        return d.copy()
    m = np.zeros(T, dtype=bool)
    m[d.astype(np.int64)] = True
    return m


def events(data, params):
    leg = float(params["leg"])
    score, uni, clock_to, clock_since = model_scores(data, params)
    T, N = score.shape
    ws = _mask(period_starts(data.dates, "W"), T)
    dec = ws & in_tax_window(clock_to, clock_since)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    idx = np.flatnonzero(dec)
    if idx.size == 0:
        return mask, side
    sw = np.asarray(score_weights(score[idx], uni[idx], top=leg, bottom=leg, min_names=MIN_NAMES),
                    dtype=np.float64)
    sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
    sw[~uni[idx]] = 0.0
    tradable = np.asarray(data.tradable, dtype=bool)
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    for k, r in enumerate(idx):
        lo = sw[k] > 0
        sh = sw[k] < 0
        new_lo = lo & ~prev_long
        new_sh = sh & ~prev_short
        ok = uni[r] & tradable[r]
        mask[r] = (new_lo | new_sh) & ok
        side[r] = np.where(new_lo, 1.0, np.where(new_sh, -1.0, 0.0))
        prev_long, prev_short = lo, sh
    side[~mask] = 0.0
    return mask, side
