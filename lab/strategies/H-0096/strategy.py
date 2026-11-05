"""H-0096 Release-window concentration of earnings information, learned.

LIQ-500 dollar-neutral. One LightGBM model (framework walk_forward, model gbm) is trained on all member
rows with six features (win flag, sue1, sue4, ear1, pre10, vol63). Only predicted-release-window entrants
(win = 1) are traded: long those whose score percentile among all scored members is >= 1 - leg_fraction,
short those <= leg_fraction; one cohort per day, held 5 sessions, gross min(0.20, 0.02 x marked names),
+g/2 over the longs and -g/2 over the shorts.

Calendar approximation: XNYS sessions after day t are not knowable from data.dates, so the predicted
release P (first session on or after r + 364 calendar days) and "t is 3 sessions before P" are computed
on the weekday calendar (np.busday_offset, exchange holidays ignored).
"""
import numpy as np

from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"leg_fraction": 0.33, "num_leaves": 8}

# fixed settings of the card (not grid parameters)
HORIZON = 5              # ml.horizon_days, label horizon and purge
HOLD = 5                 # cohort holding period in sessions
RETRAIN_EVERY = 63       # ml.retrain_every_days
TRAIN_DAYS = 1260        # ml.train_days, rolling training window
N_ESTIMATORS = 100
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 200
COHORT_CAP = 0.20        # max gross of one cohort
PER_NAME = 0.02          # cohort gross per marked name
PRED_LAG_DAYS = 364      # predicted release = reaction session + 364 calendar days
SESSIONS_BEFORE = 3      # entry row t is 3 sessions before the predicted release
RECENT_SESSIONS = 20     # no release in the 20 sessions up to and including t
SUE_LAG = 252            # seasonal change: 252 sessions
SUE4_LAG = 189           # sue4 = sue1 as known 189 sessions earlier
PRE_DAYS = 10
VOL_DAYS = 63
EPS = 1e-9


# ----------------------------------------------------------------------------------------- helpers
def lag(x, k):
    """x shifted down by k rows (row t holds row t-k), NaN for the first k rows."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if k <= 0:
        return x.copy()
    if k < T:
        out[k:] = x[: T - k]
    return out


def daily_returns(data):
    """Close-to-close total return of each row, NaN where the instrument is not listed."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r[~np.asarray(data.listed, dtype=bool)] = np.nan
    return r


def rolling_std_full(x, n):
    """Trailing sample std over n rows ending at t; NaN unless all n values are finite."""
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
    """(r - universe mean r) / vol63 on the most recent ea_reaction_day = 1 session, carried forward."""
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
    """win[t, j]: t is exactly 3 (weekday) sessions before a predicted release of j, no release in the
    last 20 sessions up to and including t, and ea_reaction_day finite at t."""
    ea = np.asarray(ea, dtype=np.float64)
    T, N = ea.shape
    E = (ea == 1).astype(np.int32)
    ccum = np.zeros((T + 1, N), dtype=np.int32)
    ccum[1:] = np.cumsum(E, axis=0)
    shifted = (dates.astype(np.int64) + PRED_LAG_DAYS).astype(dates.dtype)      # r + 364 calendar days
    pred_dates = np.busday_offset(shifted, 0, roll="forward")                    # P of a reaction on row s
    target = np.busday_offset(dates, SESSIONS_BEFORE, roll="forward")            # 3 sessions after row t
    lo = np.searchsorted(pred_dates, target, side="left")
    hi = np.searchsorted(pred_dates, target, side="right")
    predicted = (ccum[hi] - ccum[lo]) > 0
    rows = np.arange(T)
    start = np.maximum(rows + 1 - RECENT_SESSIONS, 0)
    recent = ccum[rows + 1] - ccum[start]
    return predicted & (recent == 0) & np.isfinite(ea)


def features(data, uni):
    """The card's six features, (T, N, 6): win unscaled, the five continuous ones cs-ranked + z-scored."""
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
    """(T, N) int8: +1 long leg, -1 short leg of the cohort opened after the close of t, 0 otherwise.
    Percentile = average rank (0-based) of the candidate's score among all scored members / (n - 1)."""
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


def book_from_legs(legs, uni, hold=HOLD):
    """Overlapping 5-session cohorts; a stock that leaves the universe is dropped from its cohorts."""
    T, N = legs.shape
    nl = (legs == 1).sum(axis=1).astype(np.float64)
    ns = (legs == -1).sum(axis=1).astype(np.float64)
    g = np.minimum(COHORT_CAP, PER_NAME * (nl + ns))
    with np.errstate(divide="ignore", invalid="ignore"):
        wl = np.where(nl > 0, g / 2.0 / nl, 0.0)
        ws = np.where(ns > 0, -g / 2.0 / ns, 0.0)
    C = np.where(legs == 1, wl[:, None], np.where(legs == -1, ws[:, None], 0.0))
    W = np.zeros((T, N))
    alive = uni.copy()                      # member on every row from the cohort's open to t
    for k in range(hold):
        if k >= T:
            break
        if k > 0:
            alive[k:] &= uni[: T - k]
            alive[:k] = False
        W[k:] += np.where(alive[k:], C[: T - k], 0.0)
    return W


# ----------------------------------------------------------------------------------------- strategy
def target_weights(data, params):
    if data.universe is None:
        raise ValueError("H-0096 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(data.universe, dtype=bool)
    X, win = features(data, uni)
    score = model_scores(data, params, X, uni)
    legs = cohort_legs(score, win, uni, params["leg_fraction"])
    return book_from_legs(legs, uni)
