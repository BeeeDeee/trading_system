"""H-0146 Empire-building overinvestment, learned.

LIQ-500 long-only book of six overlapping monthly cohorts; each cohort is the top `top` fraction of a
LightGBM (framework gbm) score on six point-in-time features (asset growth, capex change, cash flow,
leverage, ROA change, 12-1 return), equal weight, 1/6 of the book, held 126 rows.
"""
import numpy as np
from lab.framework.api import period_starts, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"num_leaves": 8, "top": 0.2}

# fixed by the card (not grid parameters)
LAG = 252            # "the value 252 rows earlier"
SKIP = 21            # ret_12_1 ends at the close of t-21
HORIZON = 60         # ml.horizon_days
RETRAIN_EVERY = 63   # ml.retrain_every_days
HOLD_ROWS = 126      # each cohort is held 126 trading rows
N_COHORTS = 6        # each cohort carries 1/6 of the book
MIN_NAMES = 20       # a cohort with fewer than 20 scored members is not formed
GBM_FIXED = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "min_child_samples": 2000,
    "subsample": 0.8,
    "colsample": 1.0,
}


def lag_rows(x, k):
    """x as it was known k rows earlier (NaN for the first k rows). Never a restated value: it is the
    daily point-in-time matrix shifted down by k rows."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    T = x.shape[0]
    if k < T:
        out[k:] = x[: T - k]
    return out


def _finite(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.isfinite(x), x, np.float32(np.nan)).astype(np.float32)


def raw_features(assets, capex, ncfo, debt, roa, tri, listed):
    """The card's six raw features (T, N) each, plus ok = assets_t > 0 and assets_{t-252} > 0."""
    assets = np.asarray(assets, dtype=np.float32)
    a_lag = lag_rows(assets, LAG)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pos_now = assets > 0
        ok = pos_now & (a_lag > 0)
        # (1) asset_growth = log(assets_t / assets_{t-252})
        f1 = np.where(ok, np.log(assets / a_lag), np.nan)
        # (2) capex_change = |capex_t| / assets_t - |capex_{t-252}| / assets_{t-252}
        capex = np.asarray(capex, dtype=np.float32)
        f2 = np.where(ok, np.abs(capex) / assets - np.abs(lag_rows(capex, LAG)) / a_lag, np.nan)
        # (3) cash_flow = ncfo_t / assets_t
        f3 = np.where(pos_now, np.asarray(ncfo, dtype=np.float32) / assets, np.nan)
        # (4) leverage = debt_t / assets_t
        f4 = np.where(pos_now, np.asarray(debt, dtype=np.float32) / assets, np.nan)
        # (5) roa_change = roa_t - roa_{t-252}
        roa = np.asarray(roa, dtype=np.float32)
        f5 = roa - lag_rows(roa, LAG)
        # (6) ret_12_1 = TR index at close t-21 / TR index at close t-252 - 1 (needs a listing at t-252)
        tri = np.asarray(tri, dtype=np.float64)
        T = tri.shape[0]
        num = np.full(tri.shape, np.nan)
        den = np.full(tri.shape, np.nan)
        lst = np.zeros(tri.shape, dtype=bool)
        if SKIP < T:
            num[SKIP:] = tri[: T - SKIP]
        if LAG < T:
            den[LAG:] = tri[: T - LAG]
            lst[LAG:] = np.asarray(listed, dtype=bool)[: T - LAG]
        f6 = np.where(lst & (den > 0), num / den - 1.0, np.nan)
    feats = [_finite(f) for f in (f1, f2, f3, f4, f5, f6)]
    return feats, ok


def model_inputs(data):
    """Point-in-time feature tensor X (T, N, 6) and the universe mask. A stock-date has a complete
    feature vector only if it is a member on t, all six features are finite and both asset values > 0."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0146 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    ex = data.extras
    tri = total_return_index(data)
    feats, ok = raw_features(ex["sf1_art_assets"], ex["sf1_art_capex"], ex["sf1_art_ncfo"],
                             ex["sf1_art_debt"], ex["sf1_art_roa"], tri, data.listed)
    del tri
    valid = uni & ok
    zs = []
    for f in feats:
        z = np.asarray(cs_zscore(f, uni), dtype=np.float32)
        valid &= np.isfinite(z)
        zs.append(z)
    del feats
    X = stack(*zs)
    del zs
    X[~valid] = np.nan
    return X, uni, valid


def model_score(data, params):
    X, uni, valid = model_inputs(data)
    gbm = dict(GBM_FIXED)
    gbm["num_leaves"] = int(params["num_leaves"])
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score[~valid] = np.nan
    return score, uni


def decision_mask(dates):
    """First trading day of each month (point-in-time: needs only the dates of rows <= t)."""
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return ps.copy()
    m = np.zeros(len(dates), dtype=bool)
    m[ps.astype(int)] = True
    return m


def cohort_book(decision, cohort_w, alive, hold=HOLD_ROWS, n_cohorts=N_COHORTS):
    """Sum of the live cohorts.

    decision: (T,) bool, cohort formation rows; cohort_w: (T, N) weights of the cohort formed on each
    decision row (already scaled to the cohort's share); alive: (T, N) bool, member and listed.
    A cohort formed at row d is held on rows d .. d+hold-1, or until the cohort n_cohorts decisions later
    replaces it (so at most n_cohorts are live). A stock is dropped from a cohort for good at the first row
    on which it is not alive (sold at the next open, not redistributed)."""
    alive = np.asarray(alive, dtype=bool)
    T, N = alive.shape
    W = np.zeros((T, N), dtype=np.float64)
    rows = np.arange(T, dtype=np.int32)[:, None]
    last_out = np.maximum.accumulate(np.where(alive, np.int32(-1), rows), axis=0)
    dec_rows = np.flatnonzero(np.asarray(decision, dtype=bool))
    for k, d in enumerate(dec_rows):
        w = np.nan_to_num(np.asarray(cohort_w[d], dtype=np.float64), nan=0.0)
        w = np.where(alive[d], np.maximum(w, 0.0), 0.0)
        if not np.any(w > 0):
            continue                      # cohort not formed: its share stays in cash
        end = d + hold
        if k + n_cohorts < len(dec_rows):
            end = min(end, int(dec_rows[k + n_cohorts]))
        end = min(end, T)
        keep = last_out[d:end] < d        # alive on every row since d
        W[d:end] += np.where(keep, w[None, :], 0.0)
    return W


def book_from_score(score, uni, listed, dates, top):
    """Monthly cohorts of the top `top` fraction of scored members (equal weight, >= 20 scored members)."""
    sw = score_weights(score, uni, top=float(top), min_names=MIN_NAMES)
    sw = np.nan_to_num(np.asarray(sw, dtype=np.float64), nan=0.0)
    decision = decision_mask(dates)
    alive = np.asarray(uni, dtype=bool) & np.asarray(listed, dtype=bool)
    return cohort_book(decision, sw / N_COHORTS, alive)


def target_weights(data, params):
    score, uni = model_score(data, params)
    return book_from_score(score, uni, data.listed, data.dates, params["top"])
