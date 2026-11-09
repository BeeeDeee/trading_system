"""H-0099 mechanism test: entries into the top / bottom fifth of the walk-forward GBM score.

Same features and walk_forward call as strategy.py (copied). At each weekly decision (first trading
session of the calendar week) a scored universe member whose score rank is in the top K = round(top * N)
and was not in the top K at the previous weekly decision (or not scored then) is a +1 event; the mirror
entry into the bottom K is a -1 event. No buffer rule. Only members tradable on row t are marked.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 126
BETA_WIN = 126
BETA_MIN_OBS = 100
IVOL_WIN = 63
STATE_WIN = 252
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500,
             "subsample": 0.8, "colsample": 1.0}


def _shift(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rsum(x, n):
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(x, axis=0)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] >= n:
        out[n - 1] = c[n - 1]
        out[n:] = c[n:] - c[:-n]
    return out


def _compound(r, n):
    g = np.ones(r.shape, dtype=np.float64)
    for k in range(n):
        g = g * (1.0 + _shift(r, k))
    return g - 1.0


def _rolling_median_full(v, n):
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if v.shape[0] >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = np.median(win, axis=1)
    return out


def _rolling_std_full(v, n):
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if v.shape[0] >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = np.std(win, axis=1, ddof=1)
    return out


def _daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    return np.where(listed & np.isfinite(r), r, np.nan)


def _market_return(r, uni):
    mem = uni & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _stock_features(r, m):
    T, N = r.shape
    mm = np.broadcast_to(m[:, None], (T, N))
    pv = np.isfinite(r) & np.isfinite(mm)
    x = np.where(pv, mm, 0.0)
    y = np.where(pv, r, 0.0)
    w = pv.astype(np.float64)
    n = _rsum(w, BETA_WIN)
    sx, sy = _rsum(x, BETA_WIN), _rsum(y, BETA_WIN)
    sxy, sxx = _rsum(x * y, BETA_WIN), _rsum(x * x, BETA_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        varx = sxx - sx * sx / n
        beta = np.where((n >= BETA_MIN_OBS) & (varx > 0), cov / varx, np.nan)
    rs5 = _compound(r, 5)
    rm5 = _compound(m, 5)
    res5 = rs5 - beta * rm5[:, None]
    rs21x = _shift(_compound(r, 21), 5)
    rm21x = _shift(_compound(m, 21), 5)
    res21x = rs21x - beta * rm21x[:, None]
    n63 = _rsum(w, IVOL_WIN)
    sx63, sy63 = _rsum(x, IVOL_WIN), _rsum(y, IVOL_WIN)
    sxy63, sxx63, syy63 = _rsum(x * y, IVOL_WIN), _rsum(x * x, IVOL_WIN), _rsum(y * y, IVOL_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        sz = sy63 - beta * sx63
        szz = syy63 - 2.0 * beta * sxy63 + beta * beta * sxx63
        var = (szz - sz * sz / n63) / (n63 - 1.0)
        ivol = np.where(n63 >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    ivol = np.where(np.isfinite(beta), ivol, np.nan)
    return res5, res21x, ivol, beta


def _vix_state(vix):
    v = np.asarray(vix, dtype=np.float64).reshape(-1)
    v = np.where(v > 0, v, np.nan)
    med = _rolling_median_full(v, STATE_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(v / med)


def _mkt21_state(m):
    m21 = _compound(m, 21)
    sd = _rolling_std_full(m21, STATE_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, m21 / sd, np.nan)


def _model_scores(data, params):
    if data.universe is None:
        raise ValueError("H-0099 needs the point-in-time LIQ-200 universe")
    uni = np.asarray(data.universe, dtype=bool)
    T, N = uni.shape
    r = _daily_returns(data)
    m = _market_return(r, uni)
    res5, res21x, ivol, beta = _stock_features(r, m)
    vs = _vix_state(data.series["fred_macro.VIXCLS"])
    if vs.shape[0] != T:
        raise ValueError("VIXCLS series does not match the calendar")
    ms = _mkt21_state(m)
    complete = (uni & np.isfinite(res5) & np.isfinite(res21x) & np.isfinite(ivol) & np.isfinite(beta)
                & np.isfinite(vs)[:, None] & np.isfinite(ms)[:, None])
    feats = [cs_zscore(np.where(complete, f, np.nan), uni) for f in (res5, res21x, ivol, beta)]
    X = stack(*feats, np.where(complete, vs[:, None], np.nan), np.where(complete, ms[:, None], np.nan))
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), uni


def entry_events(score, uni, tradable, decision, top):
    """Entries into the top / bottom K = round(top * N) of the scored members at weekly decisions."""
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev_top = np.zeros(N, dtype=bool)
    prev_bot = np.zeros(N, dtype=bool)
    for t in np.flatnonzero(decision):
        scored = uni[t] & np.isfinite(score[t])
        n_sc = int(scored.sum())
        k = int(np.floor(top * n_sc + 0.5))
        cur_top = np.zeros(N, dtype=bool)
        cur_bot = np.zeros(N, dtype=bool)
        if k > 0:
            cols = np.flatnonzero(scored)
            order = cols[np.argsort(-score[t, cols], kind="stable")]
            cur_top[order[:k]] = True
            cur_bot[order[n_sc - k:]] = True
        ok = uni[t] & tradable[t]
        new_top = cur_top & ~prev_top & ok
        new_bot = cur_bot & ~prev_bot & ok & ~cur_top
        mask[t] = new_top | new_bot
        side[t, new_top] = 1.0
        side[t, new_bot] = -1.0
        prev_top, prev_bot = cur_top, cur_bot
    return mask, side


def events(data, params):
    score, uni = _model_scores(data, params)
    decision = np.asarray(period_starts(data.dates, "W"), dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    return entry_events(score, uni, tradable, decision, float(params["top"]))
