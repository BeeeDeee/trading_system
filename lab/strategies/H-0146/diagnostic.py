"""H-0146 mechanism event: at a monthly decision date a scored LIQ-500 member newly enters the top fifth
(side +1) or the bottom fifth (side -1) of the model score, having not been in that fifth at the previous
monthly decision. Same features, label and walk_forward as strategy.py (copied: separate file)."""
import numpy as np
from lab.framework.api import period_starts, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

LAG = 252
SKIP = 21
HORIZON = 60
RETRAIN_EVERY = 63
MIN_NAMES = 20
GBM_FIXED = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "min_child_samples": 2000,
    "subsample": 0.8,
    "colsample": 1.0,
}


def lag_rows(x, k):
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
    assets = np.asarray(assets, dtype=np.float32)
    a_lag = lag_rows(assets, LAG)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pos_now = assets > 0
        ok = pos_now & (a_lag > 0)
        f1 = np.where(ok, np.log(assets / a_lag), np.nan)
        capex = np.asarray(capex, dtype=np.float32)
        f2 = np.where(ok, np.abs(capex) / assets - np.abs(lag_rows(capex, LAG)) / a_lag, np.nan)
        f3 = np.where(pos_now, np.asarray(ncfo, dtype=np.float32) / assets, np.nan)
        f4 = np.where(pos_now, np.asarray(debt, dtype=np.float32) / assets, np.nan)
        roa = np.asarray(roa, dtype=np.float32)
        f5 = roa - lag_rows(roa, LAG)
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


def model_score(data, params):
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
    gbm = dict(GBM_FIXED)
    gbm["num_leaves"] = int(params["num_leaves"])
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score[~valid] = np.nan
    return score, uni


def decision_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return ps.copy()
    m = np.zeros(len(dates), dtype=bool)
    m[ps.astype(int)] = True
    return m


def entry_events(decision, legs, eligible):
    """legs (T, N): > 0 in the top fraction, < 0 in the bottom fraction on a decision row.
    An event is a new entry into a tail relative to the previous decision row."""
    T, N = legs.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev_top = np.zeros(N, dtype=bool)
    prev_bot = np.zeros(N, dtype=bool)
    for d in np.flatnonzero(np.asarray(decision, dtype=bool)):
        top = legs[d] > 0
        bot = legs[d] < 0
        new_top = top & ~prev_top & eligible[d]
        new_bot = bot & ~prev_bot & eligible[d]
        mask[d] = new_top | new_bot
        side[d] = np.where(new_top, 1.0, np.where(new_bot, -1.0, 0.0))
        prev_top, prev_bot = top, bot
    return mask, side


def events(data, params):
    score, uni = model_score(data, params)
    q = float(params["top"])       # the card's "fifth" = top at its primary value 0.2
    legs = score_weights(score, uni, top=q, bottom=q, min_names=MIN_NAMES)
    legs = np.nan_to_num(np.asarray(legs, dtype=np.float64), nan=0.0)
    decision = decision_mask(data.dates)
    eligible = uni & np.asarray(data.tradable, dtype=bool)
    return entry_events(decision, legs, eligible)
