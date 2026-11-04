"""H-0095 mechanism events: a LIQ-500 member's release-session reaction (ea_reaction_day == 1 on t),
scored by the walk-forward model, whose score is at or above the entry_quantile of the event scores
of the previous 252 sessions (the stock enters the tilt at the open of t+1). Side +1.
Same feature / walk_forward / percentile code as strategy.py (copied)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
VOL_WIN = 63
VOLU_WIN = 63
RUNUP_WIN = 21
PREV_WIN = 140
LIQ_WIN = 63
MIN_FIT_ROWS = 504
RETRAIN = 63
PCT_WIN = 252
PCT_MIN = 126
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}
MEDIAN_CHUNK = 128


def member_mean(x, uni):
    m = uni & np.isfinite(x)
    s = np.where(m, x, 0.0).sum(axis=1)
    c = m.sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


def _prefix(x):
    T, N = x.shape
    v = np.isfinite(x)
    cs = np.zeros((T + 1, N))
    cs[1:] = np.cumsum(np.where(v, x, 0.0), axis=0)
    cc = np.zeros((T + 1, N), dtype=np.int64)
    cc[1:] = np.cumsum(v, axis=0)
    return cs, cc


def window_sum_count(x, n, include_current):
    T, N = x.shape
    cs, cc = _prefix(x)
    end = np.arange(T) + (1 if include_current else 0)
    st = np.clip(end - n, 0, None)
    s = cs[end] - cs[st]
    c = cc[end] - cc[st]
    return s, c


def lagged_std(x, n):
    s, c = window_sum_count(x, n, False)
    s2, _ = window_sum_count(x * x, n, False)
    full = c == n
    var = np.full(x.shape, np.nan)
    var[full] = (s2[full] - s[full] * s[full] / n) / (n - 1)
    var = np.where(var < 0, 0.0, var)
    return np.sqrt(var)


def trailing_mean(x, n):
    s, c = window_sum_count(x, n, True)
    out = np.full(x.shape, np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


def lagged_median(x, mask, n):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    xp = np.vstack([np.full((n, N), np.nan), x])
    for a in range(0, T, MEDIAN_CHUNK):
        b = min(a + MEDIAN_CHUNK, T)
        cj = np.flatnonzero(mask[a:b].any(axis=0))
        if cj.size == 0:
            continue
        seg = xp[a:b - 1 + n][:, cj]
        L = b - a
        win = np.stack([seg[k:k + L] for k in range(n)], axis=2)
        cnt = np.isfinite(win).sum(axis=2)
        srt = np.sort(win, axis=2)
        lo = np.clip((cnt - 1) // 2, 0, None)[..., None]
        hi = np.clip(cnt // 2, 0, None)[..., None]
        med = 0.5 * (np.take_along_axis(srt, lo, axis=2)[..., 0] + np.take_along_axis(srt, hi, axis=2)[..., 0])
        med[cnt == 0] = np.nan
        out[a:b, cj] = med
    return out


def shift_down(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:x.shape[0] - k]
    return out


def raw_features(ret_co, ret_oc, listed, dollar_volume, uni, ea):
    T, N = ret_co.shape
    lst = np.asarray(listed, dtype=bool)
    rco = np.where(lst, np.asarray(ret_co, dtype=np.float64), np.nan)
    roc = np.where(lst, np.asarray(ret_oc, dtype=np.float64), np.nan)
    cc = (1.0 + rco) * (1.0 + roc) - 1.0
    abn = cc - member_mean(cc, uni)[:, None]
    sd = lagged_std(abn, VOL_WIN)
    sd = np.where(sd > 0, sd, np.nan)

    f1 = (rco - member_mean(rco, uni)[:, None]) / sd
    f2 = (roc - member_mean(roc, uni)[:, None]) / sd

    dv = np.asarray(dollar_volume, dtype=np.float64)
    dv = np.where(lst & np.isfinite(dv) & (dv > 0), dv, np.nan)
    med = lagged_median(dv, uni, VOLU_WIN)
    med = np.where(med > 0, med, np.nan)
    f3 = np.log(dv / med)

    prod = np.ones((T, N))
    for k in range(1, RUNUP_WIN + 1):
        prod = prod * (1.0 + shift_down(abn, k))
    f4 = (prod - 1.0) / (sd * np.sqrt(RUNUP_WIN))

    eaf = np.asarray(ea, dtype=np.float64)
    ev_all = np.isfinite(eaf) & (eaf == 1.0)
    rz = abn / sd
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(ev_all, rows, -1), axis=0) if T > 0 else np.zeros((0, N), int)
    prev = np.vstack([np.full((1, N), -1), last[:-1]]) if T > 0 else last
    okp = (prev >= 0) & (rows - prev <= PREV_WIN)
    val = rz[np.clip(prev, 0, None), np.arange(N)[None, :]] if T > 0 else rz
    f5 = np.where(okp & np.isfinite(val), val, 0.0)

    mdv = trailing_mean(dv, LIQ_WIN)
    return f1, f2, f3, f4, f5, mdv


def event_mask(ea, uni):
    eaf = np.asarray(ea, dtype=np.float64)
    return np.isfinite(eaf) & (eaf == 1.0) & uni


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def first_fit_row(ea_full):
    any_ea = np.isfinite(np.asarray(ea_full, dtype=np.float64)).any(axis=1)
    idx = np.flatnonzero(any_ea)
    if idx.size == 0:
        return any_ea.shape[0]
    return int(idx[0]) + MIN_FIT_ROWS


def event_scores(data, params):
    if data.universe is None:
        raise ValueError("H-0095 needs the liq_n universe")
    uni_full = np.asarray(data.universe, dtype=bool)
    ea_full = data.extras["ea_reaction_day"]
    T, N = uni_full.shape
    cols = np.flatnonzero(uni_full.any(axis=0))
    score_full = np.full((T, N), np.nan)
    if cols.size == 0:
        return score_full
    uni = uni_full[:, cols]
    ea = np.asarray(ea_full, dtype=np.float64)[:, cols]
    f1, f2, f3, f4, f5, mdv = raw_features(np.asarray(data.ret_co)[:, cols], np.asarray(data.ret_oc)[:, cols],
                                           np.asarray(data.listed)[:, cols],
                                           np.asarray(data.dollar_volume)[:, cols], uni, ea)
    f6 = cs_rank(mdv, uni)
    X = stack(cs_zscore(f1, uni), cs_zscore(f2, uni), cs_zscore(f3, uni), cs_zscore(f4, uni),
              cs_zscore(f5, uni), cs_zscore(f6, uni))
    ev = event_mask(ea, uni)
    X[~ev] = np.nan
    y = np.array(cs_rank(np.asarray(forward_return(data, HORIZON))[:, cols], uni), dtype=np.float64)
    y[~ev] = np.nan
    score = np.array(walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params(params),
                                  retrain_every=RETRAIN, universe=uni), dtype=np.float64)
    score[~ev] = np.nan
    score[:min(first_fit_row(ea_full), T)] = np.nan
    score_full[:, cols] = score
    return score_full


def entry_flags(score, q):
    T, N = score.shape
    entry = np.zeros((T, N), dtype=bool)
    r, c = np.nonzero(np.isfinite(score))
    if r.size == 0:
        return entry
    s = score[r, c]
    first = int(r[0])
    starts = np.searchsorted(r, np.arange(T + 1), side="left")
    for t in np.unique(r):
        if t - first < PCT_MIN:
            continue
        lo = starts[max(t - PCT_WIN, 0)]
        hi = starts[t]
        if hi <= lo:
            continue
        pool = s[lo:hi]
        a, b = starts[t], starts[t + 1]
        cur = s[a:b]
        pct = (pool[None, :] < cur[:, None]).sum(axis=1) / pool.size
        entry[t, c[a:b]] = pct >= q
    return entry


def events(data, params):
    q = float(params["entry_quantile"])
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    score = event_scores(data, params)
    mask = entry_flags(score, q) & uni & tradable
    return mask, None
