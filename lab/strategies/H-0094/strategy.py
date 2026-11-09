"""H-0094: news-conditioned drift vs reversal, learned (LightGBM), LIQ-500 dollar-neutral, daily 5-day cohorts.

Point-in-time: row t uses rows 0..t only (trailing windows built from backward shifts), decided after the
close of t and executed at the open of t+1. The label (forward_return) is passed to walk_forward only.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"leg_fraction": 0.1, "n_estimators": 100}

# fixed by the card (not in the grid)
HORIZON = 5            # label horizon_days and cohort holding period (sessions)
N_COHORTS = 5          # live cohorts, each 1/5 of the book
VOL_WIN = 63           # vol63
DV_SHORT = 5           # mean dollar volume over t-4..t
DV_LONG = 63           # mean dollar volume over t-67..t-5
EA_WIN = 5             # earnings-release window t-4..t
RETRAIN_EVERY = 63
MIN_NAMES = 50
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 1000, "subsample": 0.5,
             "colsample": 1.0}


def _shift(x, k):
    """x[t-k] at row t (NaN / False for t < k). k >= 0 only."""
    out = np.full_like(x, np.nan if x.dtype.kind == "f" else False)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def trailing_sum(x, w):
    """Sum of x over rows t-w+1..t; NaN if any of them is NaN or the window is not full."""
    acc = x.astype(np.float64).copy()
    for k in range(1, w):
        if k >= x.shape[0]:
            break
        acc[k:] += x[:-k]
    acc[: min(w - 1, x.shape[0])] = np.nan
    return acc


def trailing_std(x, w):
    """Sample standard deviation (ddof=1) over rows t-w+1..t; NaN unless all w values are valid."""
    m = trailing_sum(x, w) / w
    ss = np.zeros_like(m)
    for k in range(0, w):
        if k >= x.shape[0]:
            break
        if k == 0:
            ss += (x - m) ** 2
        else:
            ss[k:] += (x[:-k] - m[k:]) ** 2
    return np.sqrt(ss / (w - 1))


def daily_returns(ret_co, ret_oc, close, listed):
    """r = (1+ret_co)(1+ret_oc)-1; NaN where the stock has no price that day (no history)."""
    r = (1.0 + ret_co.astype(np.float64)) * (1.0 + ret_oc.astype(np.float64)) - 1.0
    r[~(listed & np.isfinite(close))] = np.nan
    return r


def residual_returns(r, uni):
    """e = r minus the equal-weight mean r of the universe members (with a return) that day."""
    ok = uni & np.isfinite(r)
    cnt = ok.sum(axis=1)
    s = np.where(ok, r, 0.0).sum(axis=1)
    mean = np.full(r.shape[0], np.nan)
    np.divide(s, cnt, out=mean, where=cnt > 0)
    return r - mean[:, None]


def raw_features(r, uni, flag, dv):
    """The card's six features (unscaled): z1, z5, ea5, ea_z, dv_ab, lvol; each (T, N) float64."""
    e = residual_returns(r, uni)
    vol = trailing_std(r, VOL_WIN)
    vol[~(vol > 0)] = np.nan
    z1 = e / vol
    z5 = trailing_sum(e, 5) / (vol * np.sqrt(5.0))
    # earnings-release window: ea_reaction_day == 1 on any session t-4..t (NaN counts as 0)
    hit = flag == 1.0
    found = np.zeros(r.shape, dtype=bool)
    ea_z = np.zeros(r.shape, dtype=np.float64)
    for k in range(EA_WIN):                     # k = 0 is the most recent session
        f = _shift(hit, k)
        sel = f & ~found
        ea_z[sel] = _shift(z1, k)[sel]          # e / vol63 on that session (NaN there -> unscored)
        found |= f
    ea5 = found.astype(np.float64)
    dvf = dv.astype(np.float64)
    dv_short = trailing_sum(dvf, DV_SHORT) / DV_SHORT
    dv_long = _shift(trailing_sum(dvf, DV_LONG) / DV_LONG, DV_SHORT)
    ratio = dv_short / dv_long
    dv_ab = np.full(r.shape, np.nan)
    np.log(ratio, out=dv_ab, where=ratio > 0)
    lvol = np.log(vol)
    return z1, z5, ea5, ea_z, dv_ab, lvol


def membership_run(uni):
    """Number of consecutive rows up to and including t on which the stock is a member."""
    run = np.zeros(uni.shape, dtype=np.int32)
    prev = np.zeros(uni.shape[1], dtype=np.int32)
    for t in range(uni.shape[0]):
        prev = np.where(uni[t], prev + 1, 0)
        run[t] = prev
    return run


def cohort_book(cohort, uni):
    """Sum of the last N_COHORTS daily cohorts, each scaled 1/N_COHORTS. A cohort formed at t-k keeps a
    stock only while it has been a member on every row t-k..t (closed at the first non-member row)."""
    c = np.nan_to_num(cohort.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    run = membership_run(uni)
    book = np.zeros(c.shape, dtype=np.float64)
    for k in range(N_COHORTS):
        book += np.where(run >= k + 1, _shift(c, k), 0.0)
    book = np.nan_to_num(book, nan=0.0)
    return book / N_COHORTS


def model_scores(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0094 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    flag = data.extras["ea_reaction_day"]
    r = daily_returns(data.ret_co, data.ret_oc, data.close, np.asarray(data.listed, dtype=bool))
    z1, z5, ea5, ea_z, dv_ab, lvol = raw_features(r, uni, flag, data.dollar_volume)
    ea5 = np.where(uni, ea5, np.nan)            # 0/1 flag passed unscaled (members only)
    X = stack(cs_zscore(z1, uni), cs_zscore(z5, uni), ea5, cs_zscore(ea_z, uni),
              cs_zscore(dv_ab, uni), cs_zscore(lvol, uni))
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY,
                         universe=uni)
    return score, uni


def target_weights(data, params):
    score, uni = model_scores(data, params)
    lf = float(params["leg_fraction"])
    cohort = score_weights(score, uni, top=lf, bottom=lf, min_names=MIN_NAMES)   # 0.5 long / 0.5 short
    # every trading day is a decision row (only_on over all days is the identity), flat before the first fit
    return cohort_book(cohort, uni)
