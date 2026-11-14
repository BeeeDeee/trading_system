"""H-0144 mechanism event: a coin enters the top `top` fraction of the model score at a weekly decision row.

Same features, label and walk_forward as strategy.py (copied: diagnostic.py is a separate file). An event at
(t, j): j is a scored member of the PIT top-50 on weekly decision row t, is in the top set (floor(top x n),
at least 3, only when >= 15 members are scored) on t, and was not in the top set of the previous weekly
decision row; j is tradable on t. Side +1.
"""
import math

import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 56
RETRAIN_EVERY = 56
MIN_SCORED = 15
MIN_PICKS = 3
MIN_VALID_BARS = 150
FORMATION = 180
SKIP = 30
VOL_WIN = 90
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 400, "subsample": 0.7,
             "colsample": 1.0}


def _lag(x, k):
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if k < T:
        out[k:] = x[:T - k] if k > 0 else x
    return out


def _window_sum(x, n):
    x0 = np.where(np.isfinite(x), x, 0.0).astype(np.float64)
    T = x0.shape[0]
    c = np.vstack([np.zeros((1,) + x0.shape[1:]), np.cumsum(x0, axis=0)])
    out = np.full(x0.shape, np.nan)
    if n <= T:
        out[n - 1:] = c[n:] - c[:T - n + 1]
    return out


def _window_mean_valid(x, valid, n):
    xs = np.where(valid, x, 0.0)
    s = _window_sum(xs, n)
    c = _window_sum(valid.astype(np.float64), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan), c


def _window_max(x, n):
    T = x.shape[0]
    m = np.full(x.shape, np.nan)
    for k in range(min(n, T)):
        m = np.fmax(m, _lag(x, k))
    if n - 1 > 0:
        m[:min(n - 1, T)] = np.nan
    return m


def raw_features(ret_co, ret_oc, close, qv, listed):
    close = np.asarray(close, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    valid = np.isfinite(close) & np.asarray(listed, dtype=bool)
    rco = np.where(np.isfinite(ret_co), ret_co, 0.0).astype(np.float64)
    roc = np.where(np.isfinite(ret_oc), ret_oc, 0.0).astype(np.float64)
    lr = np.log1p(np.maximum(rco, -0.9999)) + np.log1p(np.maximum(roc, -0.9999))
    cl = np.cumsum(lr, axis=0)
    trend = _lag(cl, SKIP) - _lag(cl, FORMATION)
    recent = cl - _lag(cl, SKIP)
    cmax = _window_max(np.where(valid, close, np.nan), FORMATION)
    with np.errstate(invalid="ignore", divide="ignore"):
        stage = np.minimum(np.log(np.where(valid, close, np.nan) / cmax), 0.0)
    qv_ok = valid & np.isfinite(qv)
    m30, _ = _window_mean_valid(qv, qv_ok, SKIP)
    m180, _ = _window_mean_valid(qv, qv_ok, FORMATION)
    with np.errstate(invalid="ignore", divide="ignore"):
        vtrend = np.where((m30 > 0) & (m180 > 0), np.log(m30 / m180), np.nan)
    lrv = np.where(valid, lr, np.nan)
    mu, c90 = _window_mean_valid(lrv, valid, VOL_WIN)
    m2, _ = _window_mean_valid(lrv * lrv, valid, VOL_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (m2 - mu * mu) * c90 / (c90 - 1.0)
    vol90 = np.where(c90 >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    nvalid = _window_sum(valid.astype(np.float64), FORMATION)
    nvalid = np.where(np.isfinite(nvalid), nvalid, 0.0)
    return [trend, recent, stage, vtrend, vol90], nvalid


def scored_features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0144 needs the crypto_top_n point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    feats, nvalid = raw_features(data.ret_co, data.ret_oc, data.close, data.dollar_volume, data.listed)
    scored = uni & (nvalid >= MIN_VALID_BARS)
    for f in feats:
        scored &= np.isfinite(f)
    zs = [cs_zscore(np.where(scored, f, np.nan), scored) for f in feats]
    return zs, scored, uni


def model_score(data, params):
    zs, scored, uni = scored_features(data)
    X = stack(*zs)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY,
                         universe=uni, train_days=None, seed=0)
    score = np.where(scored & np.isfinite(score), score, np.nan)
    return score, uni


def decision_rows(dates):
    ps = np.asarray(period_starts(dates, "W"))
    if ps.dtype == bool:
        return np.flatnonzero(ps)
    mask = np.zeros(len(dates), dtype=bool)
    mask[ps.astype(int)] = True
    return np.flatnonzero(mask)


def top_set(score_row, top):
    ok = np.isfinite(score_row)
    n = int(ok.sum())
    if n < MIN_SCORED:
        return None
    m = max(MIN_PICKS, int(math.floor(float(top) * n + 1e-9)))
    cols = np.flatnonzero(ok)
    order = np.argsort(-score_row[cols], kind="stable")
    return cols[order[:m]]


def entry_events(score, member, tradable, dec_rows, top):
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    prev = np.zeros(N, dtype=bool)
    for t in dec_rows:
        cur = np.zeros(N, dtype=bool)
        picks = top_set(score[t], top)
        if picks is not None:
            cur[picks] = True
        mask[t] = cur & ~prev & member[t] & tradable[t]
        prev = cur
    return mask


def events(data, params):
    score, uni = model_score(data, params)
    drows = decision_rows(data.dates)
    mask = entry_events(score, uni, np.asarray(data.tradable, dtype=bool), drows, params["top"])
    return mask, None
