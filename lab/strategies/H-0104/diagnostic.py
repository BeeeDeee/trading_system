"""H-0104 mechanism event: gbm-only top-fraction entries.

At a weekly decision row t (Friday, same rule as the book) a member is marked when it is in the gbm top
`top` fraction of the scored members on t, was not in the gbm top fraction at the previous weekly decision
row, and is not in the top fraction of the ridge score (walk_forward, model ridge, alpha 1.0) fitted on the
same features, rows, label and refit schedule. Side +1. Decision rows on which either model scores fewer
than 50 members (no model fitted yet), or whose previous decision row had no gbm ranking, have no events.
Feature, label and walk_forward code is copied from strategy.py.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
SCALE_WINDOW = 63
SCALE_MIN_VALID = 40
EVENT_WINDOW = 126
PRE_WINDOW = 21
MIN_SCORED = 50
GBM_FIXED = {"n_estimators": 100, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}
RIDGE_FIXED = {"alpha": 1.0}


def daily_returns(ret_co, ret_oc, listed):
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    r[~np.asarray(listed, dtype=bool)] = np.nan
    return r


def universe_relative(r, uni):
    uni = np.asarray(uni, dtype=bool)
    ok = uni & np.isfinite(r)
    n = ok.sum(axis=1)
    s = np.where(ok, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return r - m[:, None]


def _padded_cumsum(x):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def prior_scale(u, window=SCALE_WINDOW, min_valid=SCALE_MIN_VALID):
    T = u.shape[0]
    valid = np.isfinite(u)
    x = np.where(valid, u, 0.0)
    cn = _padded_cumsum(valid.astype(np.float64))
    cx = _padded_cumsum(x)
    cxx = _padded_cumsum(x * x)
    b = np.arange(T)
    a = np.maximum(b - window, 0)
    n = cn[b] - cn[a]
    sx = cx[b] - cx[a]
    sxx = cxx[b] - cxx[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sxx - sx * sx / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd[n < min_valid] = np.nan
    sd[sd <= 0.0] = np.nan
    return sd


def _last_event_index(ev):
    T = ev.shape[0]
    idx = np.where(ev, np.arange(T)[:, None], -1)
    return np.maximum.accumulate(idx, axis=0)


def event_features(u, ev, window=EVENT_WINDOW, pre_window=PRE_WINDOW):
    T, N = u.shape
    ev = np.asarray(ev, dtype=bool)
    S = prior_scale(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_ev = u / S
    valid = np.isfinite(u)
    x = np.where(valid, u, 0.0)
    cu = _padded_cumsum(x)
    cn = _padded_cumsum(valid.astype(np.float64))
    b = np.arange(T)
    a = np.maximum(b - pre_window, 0)
    pre_sum = cu[b] - cu[a]
    pre_n = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_z = pre_sum / (S * np.sqrt(float(pre_window)))
    pre_z[pre_n < pre_window] = np.nan
    del pre_sum, pre_n, cn

    last = _last_event_index(ev)
    rows = np.arange(T)[:, None]
    r1 = last
    have1 = (r1 >= 0) & ((rows - r1) <= window - 1)
    r1c = np.where(have1, r1, 0)

    react_z = np.take_along_axis(z_ev, r1c, axis=0)
    since = (rows - r1c).astype(np.float64)
    s_r1 = np.take_along_axis(S, r1c, axis=0)
    post_sum = cu[1:] - np.take_along_axis(cu, r1c + 1, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        post_z = post_sum / s_r1
    pre_r1 = np.take_along_axis(pre_z, r1c, axis=0)

    prev_before = np.full((T, N), -1, dtype=last.dtype)
    if T > 1:
        prev_before[1:] = last[:-1]
    r0 = np.take_along_axis(prev_before, r1c, axis=0)
    have0 = have1 & (r0 >= 0) & ((r1c - r0) <= window)
    r0c = np.where(have0, r0, 0)
    react_prev = np.take_along_axis(z_ev, r0c, axis=0)

    for f in (react_z, since, post_z, pre_r1):
        f[~have1] = np.nan
    react_prev[~have0] = np.nan
    return react_z, react_prev, since, post_z, pre_r1


def reaction_flag(data):
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    return np.nan_to_num(flag, nan=0.0) > 0.5


def build_features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0104 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    r = daily_returns(data.ret_co, data.ret_oc, data.listed)
    u = universe_relative(r, uni)
    del r
    raw = event_features(u, reaction_flag(data))
    del u
    complete = uni.copy()
    for f in raw:
        complete &= np.isfinite(f)
    feats = []
    for f in raw:
        f = np.where(complete, f, np.nan)
        feats.append(cs_zscore(f, uni))
    del raw
    return stack(*feats)


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return ((d + 3) % 7) == 4


def top_sets(score, uni, top, min_scored=MIN_SCORED):
    """(T, N) bool in the top fraction of the scored members, and (T,) bool ranking available."""
    uni = np.asarray(uni, dtype=bool)
    score = np.where(uni, np.asarray(score, dtype=np.float64), np.nan)
    ok = np.isfinite(score).sum(axis=1) >= min_scored
    w = np.asarray(score_weights(score, uni, top=float(top), bottom=0.0, min_names=min_scored),
                   dtype=np.float64)
    in_top = (np.nan_to_num(w, nan=0.0) > 0) & ok[:, None] & uni
    return in_top, ok


def gbm_only_entries(gbm, ridge, uni, tradable, dec, top, min_scored=MIN_SCORED):
    uni = np.asarray(uni, dtype=bool)
    g_top, g_ok = top_sets(gbm, uni, top, min_scored)
    r_top, r_ok = top_sets(ridge, uni, top, min_scored)
    dec_idx = np.flatnonzero(np.asarray(dec, dtype=bool))
    mask = np.zeros(uni.shape, dtype=bool)
    if dec_idx.size < 2:
        return mask
    cur = dec_idx[1:]
    prev = dec_idx[:-1]
    ok = (g_ok[cur] & r_ok[cur] & g_ok[prev])[:, None]
    ev = ok & g_top[cur] & ~g_top[prev] & ~r_top[cur]
    mask[cur] = ev
    mask &= uni & np.asarray(tradable, dtype=bool)
    return mask


def events(data, params):
    X = build_features(data)
    y = cs_rank(forward_return(data, HORIZON), data.universe)
    gp = dict(GBM_FIXED)
    gp["num_leaves"] = int(params["num_leaves"])
    gbm = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp, retrain_every=RETRAIN_EVERY,
                       universe=data.universe)
    ridge = walk_forward(X, y, horizon=HORIZON, model="ridge", params=dict(RIDGE_FIXED),
                         retrain_every=RETRAIN_EVERY, universe=data.universe)
    del X, y
    mask = gbm_only_entries(gbm, ridge, data.universe, data.tradable, decision_rows(data.dates), params["top"])
    return mask, None
