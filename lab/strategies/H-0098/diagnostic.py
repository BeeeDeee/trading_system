"""H-0098 mechanism events: a LIQ-500 member enters the top `top` fraction (top fifth at the primary
value) of the model score at decision date t, having not been in it at the previous decision date
(the previous row; decisions are daily). Side +1. Same feature / walk_forward / score_weights
pipeline as strategy.py (copied, this is a separate file)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

RELEASE_KEY = "macro_release_calendar_1d.next_release_sessions"
HORIZON = 5
RETRAIN_EVERY = 63
BETA_LOOKBACK = 252
BETA_MIN_VALID = 189
IVOL_LOOKBACK = 63
IVOL_MIN_VALID = 2
LIQ_LOOKBACK = 63
REL_CAP = 10.0
MIN_NAMES = 50
GBM_FIXED = {
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 2000,
    "subsample": 0.5,
    "colsample": 1.0,
}
COL_BLOCK = 256
CELL_BLOCK = 200000


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0098 needs the point-in-time LIQ-500 universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def _daily_returns(data):
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    has = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = np.zeros_like(has)
    valid[1:] = has[1:] & has[:-1]
    return np.where(valid & np.isfinite(r), r, np.nan)


def _market_return(r, uni):
    ok = uni & np.isfinite(r)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, r, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def _rolling_sum(a, n):
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > n:
        out[n:] -= c[:-n]
    return out


def _beta_ivol(r, m):
    T, N = r.shape
    beta = np.full((T, N), np.nan, dtype=np.float32)
    ivol = np.full((T, N), np.nan, dtype=np.float32)
    mf = np.isfinite(m)
    m0 = np.where(mf, m, 0.0)[:, None]
    for c0 in range(0, N, COL_BLOCK):
        rb = r[:, c0:c0 + COL_BLOCK]
        ok = np.isfinite(rb) & mf[:, None]
        o = ok.astype(np.float64)
        x = m0 * o
        y = np.where(ok, rb, 0.0)
        xy = x * y
        xx = x * x
        yy = y * y
        n = _rolling_sum(o, BETA_LOOKBACK)
        sx = _rolling_sum(x, BETA_LOOKBACK)
        sy = _rolling_sum(y, BETA_LOOKBACK)
        sxy = _rolling_sum(xy, BETA_LOOKBACK)
        sxx = _rolling_sum(xx, BETA_LOOKBACK)
        nn = np.maximum(n, 1.0)
        varx = sxx - sx * sx / nn
        cov = sxy - sx * sy / nn
        good = (n >= BETA_MIN_VALID) & (varx > 0)
        b = np.full(varx.shape, np.nan)
        np.divide(cov, varx, out=b, where=good)
        a = (sy - b * sx) / nn
        n6 = _rolling_sum(o, IVOL_LOOKBACK)
        sx6 = _rolling_sum(x, IVOL_LOOKBACK)
        sy6 = _rolling_sum(y, IVOL_LOOKBACK)
        sxy6 = _rolling_sum(xy, IVOL_LOOKBACK)
        sxx6 = _rolling_sum(xx, IVOL_LOOKBACK)
        syy6 = _rolling_sum(yy, IVOL_LOOKBACK)
        ssr = (syy6 - 2.0 * a * sy6 - 2.0 * b * sxy6 + a * a * n6
               + 2.0 * a * b * sx6 + b * b * sxx6)
        gi = np.isfinite(b) & (n6 >= IVOL_MIN_VALID)
        iv = np.full(ssr.shape, np.nan)
        np.divide(np.maximum(ssr, 0.0), np.maximum(n6, 1.0), out=iv, where=gi)
        beta[:, c0:c0 + COL_BLOCK] = b
        ivol[:, c0:c0 + COL_BLOCK] = np.sqrt(iv)
    return beta, ivol


def _liq_63(dollar_volume, uni):
    dv = np.asarray(dollar_volume, dtype=np.float64)
    T, N = dv.shape
    L = LIQ_LOOKBACK
    dvp = np.vstack([np.full((L - 1, N), np.nan), dv])
    out = np.full((T, N), np.nan, dtype=np.float32)
    tt, jj = np.nonzero(uni)
    offs = np.arange(L)[None, :]
    for s in range(0, tt.size, CELL_BLOCK):
        t = tt[s:s + CELL_BLOCK]
        j = jj[s:s + CELL_BLOCK]
        win = dvp[t[:, None] + offs, j[:, None]]
        empty = ~np.isfinite(win).any(axis=1)
        win[empty] = 1.0
        med = np.nanmedian(win, axis=1)
        bad = empty | ~(med > 0)
        lv = np.full(med.shape, np.nan)
        np.log(med, out=lv, where=~bad)
        out[t, j] = lv
    return out


def _date_features(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    d_next = (d.astype(np.int64) + 1).astype("datetime64[D]")
    month = d.astype("datetime64[M]")
    first = month.astype("datetime64[D]")
    nxt = (month.astype(np.int64) + 1).astype("datetime64[M]").astype("datetime64[D]")
    wd_left = np.busday_count(d_next, nxt).astype(np.float64)
    wd_in = np.busday_count(first, d_next).astype(np.float64)
    return wd_left, wd_in


def _release_feature(series):
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(s), np.minimum(s, REL_CAP), REL_CAP)


def _features(data, uni):
    T, N = uni.shape
    r = _daily_returns(data)
    m = _market_return(r, uni)
    beta, ivol = _beta_ivol(r, m)
    del r
    liq = _liq_63(data.dollar_volume, uni)
    zb = np.where(uni, cs_zscore(beta, uni), np.nan)
    zi = np.where(uni, cs_zscore(ivol, uni), np.nan)
    zl = np.where(uni, cs_zscore(liq, uni), np.nan)
    del beta, ivol, liq
    wd_left, wd_in = _date_features(data.dates)
    rel = _release_feature(data.series[RELEASE_KEY])
    if rel.shape[0] != T:
        raise ValueError("release series length does not match the view")
    f4 = np.broadcast_to(wd_left[:, None], (T, N))
    f5 = np.broadcast_to(wd_in[:, None], (T, N))
    f6 = np.broadcast_to(rel[:, None], (T, N))
    return stack(zb, zi, zl, f4, f5, f6)


def _gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def top_entries(in_top, uni, tradable):
    """Member in the top set at t and not at t-1 (the previous daily decision date)."""
    prev = np.zeros_like(in_top)
    prev[1:] = in_top[:-1]
    return in_top & ~prev & uni & tradable


def events(data, params):
    uni = _universe(data)
    X = _features(data, uni)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=_gbm_params(params),
                         retrain_every=RETRAIN_EVERY, universe=uni)
    c = np.asarray(score_weights(score, uni, top=float(params["top"]), bottom=0.0,
                                 min_names=MIN_NAMES), dtype=np.float64)
    in_top = np.isfinite(c) & (c > 0)
    mask = top_entries(in_top, uni, np.asarray(data.tradable, dtype=bool))
    return mask, None
