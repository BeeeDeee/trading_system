"""H-0096 mechanism event: a LIQ-500 member that is a predicted-release-window entrant (win = 1) and whose
gbm score percentile among all scored members on t puts it in the long leg (+1, >= 1 - leg_fraction) or
the short leg (-1, <= leg_fraction), on a day when both legs are non-empty. Same feature, walk_forward
and leg code as strategy.py (copied: diagnostic.py is a separate file). Marked only where the stock is a
member and tradable on t.

Calendar approximation (as in strategy.py): the predicted release and the 3-session offset use the
weekday calendar (np.busday_offset), exchange holidays ignored.
"""
import numpy as np

from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
TRAIN_DAYS = 1260
N_ESTIMATORS = 100
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 200
PRED_LAG_DAYS = 364
SESSIONS_BEFORE = 3
RECENT_SESSIONS = 20
SUE_LAG = 252
SUE4_LAG = 189
PRE_DAYS = 10
VOL_DAYS = 63
EPS = 1e-9


def lag(x, k):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if k <= 0:
        return x.copy()
    if k < T:
        out[k:] = x[: T - k]
    return out


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r[~np.asarray(data.listed, dtype=bool)] = np.nan
    return r


def rolling_std_full(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T < n:
        return out
    f = np.isfinite(x)
    x0 = np.where(f, x, 0.0)
    shape = (T + 1,) + x.shape[1:]
    c1 = np.zeros(shape)
    c2 = np.zeros(shape)
    cf = np.zeros(shape)
    c1[1:] = np.cumsum(x0, axis=0)
    c2[1:] = np.cumsum(x0 * x0, axis=0)
    cf[1:] = np.cumsum(f, axis=0)
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    k = cf[n:] - cf[:-n]
    var = (s2 - s1 * s1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    out[n - 1:] = np.where(k >= n - 0.5, np.sqrt(var), np.nan)
    return out


def ear1_feature(r, ea, vol, uni):
    T, N = r.shape
    mem = uni & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    um = np.full(T, np.nan)
    ok = cnt > 0
    um[ok] = tot[ok] / cnt[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (r - um[:, None]) / vol
    val[~(np.isfinite(vol) & (vol > 0))] = np.nan
    E = ea == 1
    last = np.where(E, np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    cols = np.arange(N)[None, :]
    out = np.where(last >= 0, val[np.clip(last, 0, None), cols], np.nan)
    return out


def window_flag(ea, dates):
    ea = np.asarray(ea, dtype=np.float64)
    T, N = ea.shape
    E = (ea == 1).astype(np.int32)
    ccum = np.zeros((T + 1, N), dtype=np.int32)
    ccum[1:] = np.cumsum(E, axis=0)
    shifted = (dates.astype(np.int64) + PRED_LAG_DAYS).astype(dates.dtype)
    pred_dates = np.busday_offset(shifted, 0, roll="forward")
    target = np.busday_offset(dates, SESSIONS_BEFORE, roll="forward")
    lo = np.searchsorted(pred_dates, target, side="left")
    hi = np.searchsorted(pred_dates, target, side="right")
    predicted = (ccum[hi] - ccum[lo]) > 0
    rows = np.arange(T)
    start = np.maximum(rows + 1 - RECENT_SESSIONS, 0)
    recent = ccum[rows + 1] - ccum[start]
    return predicted & (recent == 0) & np.isfinite(ea)


def features(data, uni):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    netinc = np.asarray(data.extras["sf1_arq_netinc"], dtype=np.float64)
    assets = np.asarray(data.extras["sf1_arq_assets"], dtype=np.float64)
    assets = np.where(assets > 0, assets, np.nan)
    sue1 = (netinc - lag(netinc, SUE_LAG)) / assets
    sue4 = lag(sue1, SUE4_LAG)
    r = daily_returns(data)
    vol63 = rolling_std_full(r, VOL_DAYS)
    ear1 = ear1_feature(r, ea, vol63, uni)
    pre10 = np.asarray(trailing_return(total_return_index(data), PRE_DAYS), dtype=np.float64)
    win = window_flag(ea, data.dates)
    cont = [cs_zscore(cs_rank(x, uni), uni) for x in (sue1, sue4, ear1, pre10, vol63)]
    X = stack(win.astype(np.float32), *cont)
    return X, win


def model_scores(data, params, X, uni):
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm_params = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "num_leaves": int(params["num_leaves"]),
    }
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                        retrain_every=RETRAIN_EVERY, universe=uni, train_days=TRAIN_DAYS)


def cohort_legs(score, win, uni, leg_fraction):
    score = np.asarray(score, dtype=np.float64)
    T, N = score.shape
    legs = np.zeros((T, N), dtype=np.int8)
    scored = uni & np.isfinite(score)
    cand_all = scored & win
    lf = float(leg_fraction)
    for t in np.nonzero(cand_all.any(axis=1))[0]:
        mem = scored[t]
        vals = np.sort(score[t, mem])
        n = vals.size
        if n < 2:
            continue
        cand = np.nonzero(cand_all[t])[0]
        v = score[t, cand]
        less = np.searchsorted(vals, v, side="left")
        leq = np.searchsorted(vals, v, side="right")
        pct = ((less + leq - 1) / 2.0) / (n - 1)
        lo = pct <= lf + EPS
        hi = pct >= 1.0 - lf - EPS
        is_long = hi & ~lo
        is_short = lo & ~hi
        if is_long.any() and is_short.any():
            legs[t, cand[is_long]] = 1
            legs[t, cand[is_short]] = -1
    return legs


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0096 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(data.universe, dtype=bool)
    X, win = features(data, uni)
    score = model_scores(data, params, X, uni)
    legs = cohort_legs(score, win, uni, params["leg_fraction"])
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (legs != 0) & uni & tradable
    side = np.where(legs < 0, -1.0, 1.0)
    return mask, side
