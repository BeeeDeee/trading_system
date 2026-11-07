"""H-0109 mechanism event: a window entrant enters the LOW set (side -1) or the HIGH set (side +1) at t.

Copy of the strategy's entrant, feature, walk_forward and set code (diagnostic.py is a separate file).
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
N_ESTIMATORS = 100
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 200
ANNIV_DAYS = 364
WINDOW_DAYS = 5
NO_REACTION = 40
TRAIL = 63
MIN_POOL = 50


def _roll_sum_full(x, n):
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T < n:
        return out
    ok = np.isfinite(x)
    k = np.zeros((T + 1,) + x.shape[1:], dtype=np.int32)
    np.cumsum(ok, axis=0, out=k[1:])
    full = (k[n:] - k[:-n]) == n
    del k
    c = np.zeros((T + 1,) + x.shape[1:])
    np.cumsum(np.where(ok, x, 0.0), axis=0, out=c[1:])
    del ok
    s = c[n:] - c[:-n]
    del c
    s[~full] = np.nan
    out[n - 1:] = s
    return out


def _roll_mean_full(x, n):
    return _roll_sum_full(x, n) / n


def _roll_std_full(x, n):
    s1 = _roll_sum_full(x, n)
    s2 = _roll_sum_full(x * x, n)
    var = (s2 - s1 * s1 / n) / (n - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _roll_median_members(x, n, mask, chunk=100000):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    ti, tj = np.nonzero(mask)
    keep = ti >= n - 1
    ti, tj = ti[keep], tj[keep]
    offs = np.arange(n - 1, -1, -1)
    for a in range(0, len(ti), chunk):
        r = ti[a:a + chunk]
        j = tj[a:a + chunk]
        win = x[r[:, None] - offs[None, :], j[:, None]]
        med = np.median(win, axis=1)
        med[~np.all(np.isfinite(win), axis=1)] = np.nan
        out[r, j] = med
    return out


def _member_mean(x, uni):
    ok = uni & np.isfinite(x)
    cnt = ok.sum(axis=1)
    s = np.where(ok, x, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _reaction(ea):
    return np.nan_to_num(np.asarray(ea, dtype=float), nan=0.0) == 1.0


def _window_candidates(days, react):
    T, N = react.shape
    cand = np.zeros((T, N), dtype=bool)
    s, j = np.nonzero(react)
    if len(s) == 0:
        return cand
    anniv = days[s] + ANNIV_DAYS
    r = np.searchsorted(days, anniv - WINDOW_DAYS, side="left")
    ok = r < T
    s, j, anniv, r = s[ok], j[ok], anniv[ok], r[ok]
    ok = (days[r] < anniv) & (r > s)
    cand[r[ok], j[ok]] = True
    return cand


def _recent_reaction(react, n):
    T = react.shape[0]
    c = np.zeros((T + 1,) + react.shape[1:], dtype=np.int64)
    np.cumsum(react, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) + 1 - n, 0)
    return (c[1:] - c[lo]) > 0


def _announcement_risk(react, ex_abs):
    T, N = react.shape
    out = np.full((T, N), np.nan)
    s, j = np.nonzero(react & np.isfinite(ex_abs))
    if len(s) == 0:
        return out
    order = np.lexsort((s, j))
    s, j = s[order], j[order]
    v = ex_abs[s, j]
    tot = v.copy()
    cnt = np.ones(len(v))
    for lag in (1, 2, 3):
        same = np.zeros(len(v), dtype=bool)
        same[lag:] = j[lag:] == j[:-lag]
        add = np.zeros(len(v))
        add[lag:] = v[:-lag]
        tot = tot + np.where(same, add, 0.0)
        cnt = cnt + same
    val = np.where(cnt >= 2, tot / cnt, np.nan)
    has = np.zeros((T, N), dtype=bool)
    grid = np.full((T, N), np.nan)
    has[s, j] = True
    grid[s, j] = val
    last = np.maximum.accumulate(np.where(has, np.arange(T)[:, None], -1), axis=0)
    cols = np.broadcast_to(np.arange(N)[None, :], (T, N))
    filled = grid[np.maximum(last, 0), cols]
    out[last >= 0] = filled[last >= 0]
    return out


def _features(data, uni, react):
    T, N = len(data.dates), len(data.instruments)
    idx = total_return_index(data)
    prev = np.vstack([np.ones((1, N)), idx[:-1]])
    r = idx / prev - 1.0
    r[~(np.asarray(data.listed, dtype=bool) & np.isfinite(data.close))] = np.nan
    del idx, prev
    mkt = _member_mean(r, uni)
    ex = r - mkt[:, None]
    f_ivol = _roll_std_full(ex, 63).astype(np.float32)
    f_arisk = _announcement_risk(react, np.abs(ex)).astype(np.float32)
    del ex
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(np.where(r > -1.0, r, np.nan))
    del r
    run21 = np.expm1(_roll_sum_full(lr, 21))
    f_runup = (run21 - _member_mean(run21, uni)[:, None]).astype(np.float32)
    del run21
    m231 = _roll_sum_full(lr, 231)
    del lr
    f_mom = np.full((T, N), np.nan, dtype=np.float32)
    if T > 21:
        f_mom[21:] = np.expm1(m231[:-21])
    del m231
    dv = np.asarray(data.dollar_volume, dtype=float)
    dv = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
    m21 = _roll_mean_full(dv, 21)
    m252 = _roll_mean_full(dv, 252)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_turn = np.log(m21 / m252).astype(np.float32)
    del m21, m252
    dv = dv.astype(np.float32)
    med = _roll_median_members(dv, 63, uni)
    del dv
    with np.errstate(divide="ignore", invalid="ignore"):
        f_liq = np.log(med).astype(np.float32)
    del med
    return [f_ivol, f_turn, f_runup, f_liq, f_arisk, f_mom]


def _entrants_and_scores(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("liq_n card needs data.universe")
    uni = np.asarray(uni, dtype=bool)
    react = _reaction(data.extras["ea_reaction_day"])
    days = np.asarray(data.dates).astype(np.int64)
    feats = _features(data, uni, react)
    for i in range(len(feats)):
        feats[i] = np.asarray(cs_zscore(feats[i], uni), dtype=np.float32)
    X = stack(*feats)
    del feats
    allfin = np.isfinite(X).all(axis=2)
    ent = _window_candidates(days, react) & ~_recent_reaction(react, NO_REACTION) & uni & allfin
    X[~ent] = np.nan
    y = forward_return(data, HORIZON)
    gbm = {"n_estimators": N_ESTIMATORS, "learning_rate": LEARNING_RATE,
           "min_child_samples": MIN_CHILD_SAMPLES, "num_leaves": int(params["num_leaves"])}
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.where(ent, score, np.nan)
    return uni, ent, score


def _sets(score, tail):
    T, N = score.shape
    side = np.zeros((T, N))
    scored = np.isfinite(score)
    st, sj = np.nonzero(scored)
    sv = score[st, sj]
    off = np.zeros(T + 1, dtype=np.int64)
    np.cumsum(np.bincount(st, minlength=T), out=off[1:])
    for t in np.unique(st):
        pool = sv[off[max(0, t - TRAIL + 1)]:off[t + 1]]
        if len(pool) < MIN_POOL:
            continue
        lo = np.quantile(pool, tail)
        hi = np.quantile(pool, 1.0 - tail)
        js = sj[off[t]:off[t + 1]]
        v = sv[off[t]:off[t + 1]]
        low = v <= lo
        high = v >= hi
        both = low & high
        side[t, js[low & ~both]] = -1.0
        side[t, js[high & ~both]] = 1.0
    return side


def events(data, params):
    uni, ent, score = _entrants_and_scores(data, params)
    side = _sets(score, float(params["tail"]))
    mask = (side != 0) & uni & np.asarray(data.tradable, dtype=bool)
    return mask, np.where(mask, side, 0.0)
