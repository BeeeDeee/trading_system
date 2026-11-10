"""H-0122: round-number support/resistance, learned (gbm on the unadjusted close's place on the dollar grid).

LIQ-500, daily decision after the close of t, trades at the open of t+1. Six features per stock and date,
framework walk-forward LightGBM (expanding window, refit every 63 rows), dollar-neutral top/bottom `top` book.
"""
import numpy as np
from lab.framework.api import only_on
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.1}

HORIZON = 1            # card ml.horizon_days
RETRAIN_EVERY = 63     # card ml.retrain_every_days
VOL_WINDOW = 21        # card feature (6)
SPLIT_TOL = 0.01       # card feature (4): 1 % mismatch between unadjusted and total return
MIN_NAMES = 100        # card: flat on days with fewer than 100 scored members
GBM_FIXED = {"num_leaves": 15, "min_child_samples": 200}
RIDGE_ALPHA = 1.0      # diagnostic control only


def daily_total_return(ret_co, ret_oc):
    """r_t = (1 + ret_co) * (1 + ret_oc) - 1."""
    return (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0


def grid_features(close_u, r):
    """cents_off, roundness, log price and crossing from the unadjusted close (rows <= t only).

    close_u: (T, N) unadjusted close, NaN without a price. r: (T, N) daily total return.
    Returns four (T, N) float64 arrays; NaN where the close (or, for crossing, the previous close) is missing.
    """
    p = np.asarray(close_u, dtype=np.float64)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    level = np.floor(p + 0.5)                      # nearest whole dollar, ties (x.50) go up -> cents in [-0.5, 0.5)
    cents = p - level
    roundness = np.where(np.mod(level, 50) == 0, 3.0,
                         np.where(np.mod(level, 10) == 0, 2.0,
                                  np.where(np.mod(level, 5) == 0, 1.0, 0.0)))
    roundness = np.where(np.isfinite(level), roundness, np.nan)
    logp = np.log(p)

    prev = np.full_like(p, np.nan)
    prev[1:] = p[:-1]                              # close_u_{t-1} (previous row), backward-looking only
    up = (prev < level) & (level <= p)
    down = (prev >= level) & (level > p)
    crossing = np.where(up, 1.0, np.where(down, -1.0, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mismatch = np.abs((p / prev) / (1.0 + r) - 1.0) > SPLIT_TOL
    crossing = np.where(mismatch, 0.0, crossing)   # split / corporate action moved the unadjusted price
    crossing = np.where(np.isfinite(p) & np.isfinite(prev), crossing, np.nan)
    return cents, roundness, logp, crossing


def realized_vol(r, window):
    """Trailing std (ddof=1) of r over the last `window` rows; NaN unless all `window` values are valid."""
    r = np.asarray(r, dtype=np.float64)
    T, N = r.shape
    valid = np.isfinite(r)
    x = np.where(valid, r, 0.0)
    z = np.zeros((1, N))
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(valid.astype(np.float64), axis=0)])
    out = np.full((T, N), np.nan)
    if T < window:
        return out
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    n = cn[window:] - cn[:-window]
    var = (s2 - s1 * s1 / window) / (window - 1)
    vol = np.sqrt(np.maximum(var, 0.0))
    out[window - 1:] = np.where(n == window, vol, np.nan)
    return out


def build_features(data):
    """(T, N, 6) feature tensor in card order; features 3, 5, 6 z-scored across members per date."""
    close_u = data.close
    r = daily_total_return(data.ret_co, data.ret_oc)
    p = np.asarray(close_u, dtype=np.float64)
    prev = np.full_like(p, np.nan)
    prev[1:] = p[:-1]
    r = np.where(np.isfinite(p) & np.isfinite(prev), r, np.nan)   # a real one-day return needs both closes
    cents, roundness, logp, crossing = grid_features(close_u, r)
    vol = realized_vol(r, VOL_WINDOW)
    uni = data.universe
    X = stack(cents, roundness, cs_zscore(logp, uni), crossing, cs_zscore(r, uni), cs_zscore(vol, uni))
    return X


def label(data):
    """Open t+1 -> open t+2 total return, ranked across members per date."""
    return cs_rank(forward_return(data, HORIZON), data.universe)


def gbm_scores(data, params, X=None, y=None):
    if X is None:
        X = build_features(data)
    if y is None:
        y = label(data)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp,
                        retrain_every=RETRAIN_EVERY, universe=data.universe)


def ridge_scores(data, params, X=None, y=None):
    if X is None:
        X = build_features(data)
    if y is None:
        y = label(data)
    return walk_forward(X, y, horizon=HORIZON, model="ridge", params={"alpha": RIDGE_ALPHA},
                        retrain_every=RETRAIN_EVERY, universe=data.universe)


def book(score, universe, top, min_names=MIN_NAMES):
    top = float(top)
    return score_weights(score, universe, top=top, bottom=top, min_names=min_names)


def target_weights(data, params):
    if data.universe is None:
        raise ValueError("H-0122 needs the LIQ-500 point-in-time universe")
    score = gbm_scores(data, params)
    w = book(score, data.universe, params["top"])
    every_day = np.ones(len(data.dates), dtype=bool)          # daily rebalance
    return only_on(every_day, w)
