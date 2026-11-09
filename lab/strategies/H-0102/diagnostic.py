"""H-0102 mechanism event: a release row (ea_reaction_day = 1 on t, LIQ-500 member on t) whose
walk-forward gbm score is at or above the (1 - top) quantile of all scored release rows of the trailing
126 sessions including t (at least 100 scored rows). Side +1. Same features, model and entry rule as
strategy.py (copied: diagnostic.py is a separate file); no slot cap (the event is the score event)."""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
WINDOW = 63
RUNUP_WINDOW = 21
QUANTILE_WINDOW = 126
MIN_SCORED = 100
GBM_FIXED = {
    "num_leaves": 8,
    "min_child_samples": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample": 1.0,
}


def _lagged_mean_std(x, n):
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    mean = np.full((T, N), np.nan)
    std = np.full((T, N), np.nan)
    if T <= n:
        return mean, std
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    z = np.zeros((1, N))
    cs = np.concatenate([z, np.cumsum(v, axis=0)])
    cs2 = np.concatenate([z, np.cumsum(v * v, axis=0)])
    cc = np.concatenate([z, np.cumsum(valid, axis=0, dtype=np.float64)])
    s = cs[n:T] - cs[0:T - n]
    s2 = cs2[n:T] - cs2[0:T - n]
    c = cc[n:T] - cc[0:T - n]
    full = c == n
    m = s / n
    var = np.maximum((s2 - s * m) / (n - 1), 0.0)
    mean[n:] = np.where(full, m, np.nan)
    std[n:] = np.where(full, np.sqrt(var), np.nan)
    return mean, std


def _cs_median(x, mask):
    x = np.asarray(x, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    xs = np.sort(np.where(ok, x, np.nan), axis=1)
    cnt = ok.sum(axis=1)
    rows = np.arange(x.shape[0])
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.maximum(cnt // 2, 0)
    hi = np.minimum(hi, x.shape[1] - 1)
    med = 0.5 * (xs[rows, lo] + xs[rows, hi])
    return np.where(cnt > 0, med, np.nan)


def _cs_mean(x, mask):
    x = np.asarray(x, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def _lagged_median_on(x, mask, n):
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for t in range(n, T):
        cols = np.nonzero(mask[t])[0]
        if cols.size == 0:
            continue
        out[t, cols] = np.median(x[t - n:t, cols], axis=0)
    return out


def _lag(x, k):
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def _release_rows(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0102 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    return (ea == 1.0) & uni, uni


def _raw_features(data, uni):
    listed = np.asarray(data.listed, dtype=bool)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = np.where(listed, r, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, sd63 = _lagged_mean_std(r, WINDOW)
        react = (r - _cs_median(r, uni)[:, None]) / sd63
        react = np.where(np.isfinite(react) & (sd63 > 0), react, np.nan)
        io = np.asarray(data.extras["f13_io"], dtype=np.float64)
        dv = np.asarray(data.dollar_volume, dtype=np.float64)
        mdv, _ = _lagged_mean_std(dv, WINDOW)
        liq = np.where(mdv > 0, np.log(np.where(mdv > 0, mdv, 1.0)), np.nan)
        meddv = _lagged_median_on(dv, uni, WINDOW)
        ratio = dv / meddv
        okr = np.isfinite(ratio) & (ratio > 0)
        relvol = np.where(okr, np.log(np.where(okr, ratio, 1.0)), np.nan)
        idx = np.asarray(total_return_index(data), dtype=np.float64)
        run = _lag(idx, 1) / _lag(idx, RUNUP_WINDOW + 1) - 1.0
        run = np.where(np.isfinite(run), run, np.nan)
        runup = run - _cs_median(run, uni)[:, None]
        resid = r - _cs_mean(r, uni)[:, None]
        _, ivol = _lagged_mean_std(resid, WINDOW)
    return [react, io, liq, relvol, runup, ivol]


def _entry_mask(score, top, window=QUANTILE_WINDOW, min_scored=MIN_SCORED):
    score = np.asarray(score, dtype=np.float64)
    T = score.shape[0]
    fin = np.isfinite(score)
    has = fin.any(axis=1)
    out = np.zeros(score.shape, dtype=bool)
    q = 1.0 - float(top)
    for t in range(T):
        if not has[t]:
            continue
        lo = max(0, t - window + 1)
        pool = score[lo:t + 1][fin[lo:t + 1]]
        if pool.size < min_scored:
            continue
        thr = np.quantile(pool, q)
        out[t] = fin[t] & (score[t] >= thr)
    return out


def events(data, params):
    rel, uni = _release_rows(data)
    feats = _raw_features(data, uni)
    X = stack(*[np.where(rel, cs_zscore(f, uni), np.nan) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score = np.where(rel & np.isfinite(score), score, np.nan)
    entry = _entry_mask(score, params["top"])
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = entry & uni & tradable
    return mask, None
