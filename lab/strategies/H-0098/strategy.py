"""H-0098: learned calendar timing of beta in LIQ-500, long-only 5-day cohorts (gbm).

Decision after the close of row t from rows <= t; executed at the open of t+1.

Features (card f1..f6):
  f1 beta_252  OLS slope of the stock's daily total return on the equal-weight mean daily total
               return of the LIQ-500 members, last 252 rows, >= 189 valid days (else NaN); z-scored
               across the members of date t (cs_zscore).
  f2 ivol_63   std of the residuals of that same regression (alpha/beta of the 252 window) over the
               last 63 rows; z-scored.
  f3 liq_63    log of the median daily dollar volume over the last 63 rows; z-scored.
  f4 wd_left   weekdays strictly after t up to the last calendar day of t's month (date alone), RAW.
  f5 wd_in     weekdays from the first calendar day of t's month through t (date alone), RAW.
  f6 rel       macro_release_calendar_1d.next_release_sessions at t, clipped at 10, NaN -> 10, RAW.
Label: total return open t+1 -> open t+6 (forward_return, horizon 5), ranked across members of t.
Model: walk_forward gbm, expanding window, refit every 63 rows, fixed settings from the card.
Book: score_weights(top, min_names 50) per date = one cohort; each cohort gets 1/5 of capital and is
held 5 rows; a stock that stops being a member is dropped from every cohort from that row on.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

RELEASE_KEY = "macro_release_calendar_1d.next_release_sessions"
HORIZON = 5              # ml.horizon_days = label horizon = cohort holding period (sessions)
RETRAIN_EVERY = 63       # ml.retrain_every_days
BETA_LOOKBACK = 252
BETA_MIN_VALID = 189
IVOL_LOOKBACK = 63
IVOL_MIN_VALID = 2       # a standard deviation needs at least two residuals (not in the card)
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


def daily_returns(data):
    """Daily total return (1+ret_co)(1+ret_oc)-1; NaN unless the stock has a price at t and t-1."""
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    has = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = np.zeros_like(has)
    valid[1:] = has[1:] & has[:-1]
    return np.where(valid & np.isfinite(r), r, np.nan)


def market_return(r, uni):
    """Equal-weight mean daily total return of the members of each date (NaN if none valid)."""
    ok = uni & np.isfinite(r)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, r, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def _rolling_sum(a, n):
    """Trailing sum over the last n rows including the current one (shorter at the start)."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > n:
        out[n:] -= c[:-n]
    return out


def beta_ivol(r, m):
    """beta_252 (>= 189 valid days) and ivol_63 (residual std of the same regression, last 63 rows)."""
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


def liq_63(dollar_volume, uni):
    """log median dollar volume of the last 63 rows, computed on member cells only (NaN elsewhere)."""
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


def date_features(dates):
    """wd_left, wd_in from the calendar date of each row alone (Mon-Fri, holidays ignored)."""
    d = np.asarray(dates).astype("datetime64[D]")
    d_next = (d.astype(np.int64) + 1).astype("datetime64[D]")          # calendar day after t
    month = d.astype("datetime64[M]")
    first = month.astype("datetime64[D]")                              # 1st calendar day of t's month
    nxt = (month.astype(np.int64) + 1).astype("datetime64[M]").astype("datetime64[D]")  # 1st of next month
    wd_left = np.busday_count(d_next, nxt).astype(np.float64)
    wd_in = np.busday_count(first, d_next).astype(np.float64)
    return wd_left, wd_in


def release_feature(series):
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(s), np.minimum(s, REL_CAP), REL_CAP)


def features(data, uni):
    T, N = uni.shape
    r = daily_returns(data)
    m = market_return(r, uni)
    beta, ivol = beta_ivol(r, m)
    del r
    liq = liq_63(data.dollar_volume, uni)
    zb = np.where(uni, cs_zscore(beta, uni), np.nan)
    zi = np.where(uni, cs_zscore(ivol, uni), np.nan)
    zl = np.where(uni, cs_zscore(liq, uni), np.nan)
    del beta, ivol, liq
    wd_left, wd_in = date_features(data.dates)
    rel = release_feature(data.series[RELEASE_KEY])
    if rel.shape[0] != T:
        raise ValueError("release series length does not match the view")
    f4 = np.broadcast_to(wd_left[:, None], (T, N))
    f5 = np.broadcast_to(wd_in[:, None], (T, N))
    f6 = np.broadcast_to(rel[:, None], (T, N))
    return stack(zb, zi, zl, f4, f5, f6)


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def model_scores(data, params):
    uni = _universe(data)
    X = features(data, uni)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params(params),
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return score, uni


def cohort_weights(score, uni, top):
    """Daily cohort: equal weight on the top fraction of scored members; < 50 scored -> cash."""
    c = np.asarray(score_weights(score, uni, top=float(top), bottom=0.0, min_names=MIN_NAMES),
                   dtype=np.float64)
    return np.where(np.isfinite(c), c, 0.0)


def overlap_book(cohorts, uni, hold=HORIZON):
    """Row t holds cohorts t-hold+1..t at 1/hold each; a cohort's stock is dropped from the first
    row on which it is not a member (and stays dropped for that cohort)."""
    T, N = cohorts.shape
    w = np.zeros((T, N), dtype=np.float64)
    cont = uni.copy()                       # member on every row t-k..t
    for k in range(min(hold, T)):
        if k > 0:
            sh = np.zeros_like(uni)
            sh[k:] = uni[:-k]
            cont &= sh
        ck = np.zeros((T, N), dtype=np.float64)
        ck[k:] = cohorts[:T - k]
        w += np.where(cont, ck, 0.0)
    return w / float(hold)


def target_weights(data, params):
    score, uni = model_scores(data, params)
    cohorts = cohort_weights(score, uni, params["top"])
    return overlap_book(cohorts, uni)
