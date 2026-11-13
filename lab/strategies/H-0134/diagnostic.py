"""H-0134 mechanism event: gbm vs ridge leg disagreement at weekly decision rows.

Same features, rows, label and refit schedule as strategy.py (code copied: diagnostic.py is a separate file).
side +1: in the gbm long leg but not the ridge long leg, or in the ridge short leg but not the gbm short leg.
side -1: in the gbm short leg but not the ridge short leg, or in the ridge long leg but not the gbm long leg.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 21
REFIT_EVERY = 63
TOP_N = 30
VOL_WINDOW_UNIVERSE = 30
MIN_SCORED = 15
SUNDAY = 6
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 300,
             "subsample": 0.8, "colsample": 1.0}
RIDGE_FIXED = {"alpha": 1.0}


def _lag(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _rolling_sum_count(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cc = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cs[1:] = np.cumsum(v, axis=0)
    cc[1:] = np.cumsum(valid, axis=0)
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return cs[hi] - cs[lo], cc[hi] - cc[lo]


def weekday(dates):
    d = np.asarray(dates).astype(np.int64)
    return (d + 3) % 7


def pit_universe(data):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    s, c = _rolling_sum_count(dv, VOL_WINDOW_UNIVERSE)
    eligible = np.asarray(data.listed, dtype=bool) & (c > 0) & (s > 0)
    key = np.where(eligible, s, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    ranks = np.empty_like(order)
    T, N = key.shape
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N), (T, N)).copy(), axis=1)
    return eligible & (ranks < TOP_N)


def compute_features(close, qv, basis, member):
    close = np.asarray(close, dtype=np.float64)
    c = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    logc = np.log(c)
    mom28 = logc - _lag(logc, 28)
    ret7 = logc - _lag(logc, 7)
    bs, bc = _rolling_sum_count(basis, 7)
    basis7 = np.where(bc >= 5, bs / np.maximum(bc, 1.0), np.nan)
    q7s, q7c = _rolling_sum_count(qv, 7)
    q90s, q90c = _rolling_sum_count(qv, 90)
    m7 = np.where(q7c > 0, q7s / np.maximum(q7c, 1.0), np.nan)
    m90 = np.where(q90c > 0, q90s / np.maximum(q90c, 1.0), np.nan)
    ok = np.isfinite(m7) & np.isfinite(m90) & (m7 > 0) & (m90 > 0)
    volshock = np.full(c.shape, np.nan)
    volshock[ok] = np.log(m7[ok] / m90[ok])
    r = logc - _lag(logc, 1)
    rs, rc = _rolling_sum_count(r, 28)
    rss, _ = _rolling_sum_count(r * r, 28)
    var = (rss - rs * rs / np.maximum(rc, 1.0)) / np.maximum(rc - 1.0, 1.0)
    rvol28 = np.where(rc >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    _, cc90 = _rolling_sum_count(c, 90)
    scored = np.asarray(member, dtype=bool) & (cc90 >= 80) & (bc >= 5)
    feats = [np.where(scored, f, np.nan) for f in (mom28, ret7, basis7, volshock, rvol28)]
    return feats, scored


def model_inputs(data):
    member = pit_universe(data)
    basis = np.asarray(data.extras["basis"], dtype=np.float64)
    qv = np.asarray(data.dollar_volume, dtype=np.float64)
    feats, scored = compute_features(data.close, qv, basis, member)
    X = stack(*[cs_zscore(f, scored) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), member)
    return member, X, y


def legs(score_row, member_row, n_leg, min_scored=MIN_SCORED):
    """Boolean long / short legs of one decision row (same rule as the strategy), or None if too few scored."""
    idx = np.flatnonzero(member_row & np.isfinite(score_row))
    if idx.size < min_scored:
        return None
    s = score_row[idx]
    N = score_row.shape[0]
    lo = np.zeros(N, dtype=bool)
    sh = np.zeros(N, dtype=bool)
    lo[idx[np.argsort(-s, kind="stable")[:n_leg]]] = True
    sh[idx[np.argsort(s, kind="stable")[:n_leg]]] = True
    return lo, sh


def disagreement(g_score, r_score, member, tradable, decision, n_leg):
    T, N = g_score.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(decision):
        g = legs(g_score[t], member[t], n_leg)
        r = legs(r_score[t], member[t], n_leg)
        if g is None or r is None:
            continue
        gl, gs = g
        rl, rs = r
        plus = (gl & ~rl) | (rs & ~gs)
        minus = (gs & ~rs) | (rl & ~gl)
        ev = (plus | minus) & member[t] & tradable[t]
        mask[t] = ev
        side[t] = np.where(plus, 1.0, np.where(minus, -1.0, 0.0)) * ev
    return mask, side


def events(data, params):
    member, X, y = model_inputs(data)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    g_score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                           retrain_every=REFIT_EVERY, universe=member)
    r_score = walk_forward(X, y, horizon=HORIZON, model="ridge", params=dict(RIDGE_FIXED),
                           retrain_every=REFIT_EVERY, universe=member)
    decision = weekday(data.dates) == SUNDAY
    tradable = np.asarray(data.tradable, dtype=bool)
    return disagreement(np.asarray(g_score, dtype=np.float64), np.asarray(r_score, dtype=np.float64),
                        member, tradable, decision, int(params["n_leg"]))
