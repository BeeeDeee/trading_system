"""H-0134: leverage-crowded vs spot-led trends, learned (gbm on momentum x basis), top-30 Binance perps.

Weekly (Sunday bar) dollar-neutral cohorts of the n_leg highest / lowest gbm scores, each at 1/(6 n_leg),
held 21 rows, three cohorts overlapping. Everything is point-in-time: row t uses rows 0..t only.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "n_leg": 6}

# fixed by the card (not in signal.params)
HORIZON = 21            # label horizon = holding period (rows)
REFIT_EVERY = 63        # ml.retrain_every_days
TOP_N = 30              # universe crypto_top_n, n 30 (fallback only, when the gate passes no universe)
VOL_WINDOW_UNIVERSE = 30
MIN_SCORED = 15         # minimum scored members to form a cohort
SUNDAY = 6              # weekday index, Monday = 0
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 300,
             "subsample": 0.8, "colsample": 1.0}


# ----------------------------------------------------------------------------- helpers (point-in-time)

def _lag(x, k):
    """x shifted down by k rows (row t holds x[t-k]); NaN for the first k rows. Accepts views shorter than k."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _rolling_sum_count(x, n):
    """Trailing sum of the finite values over rows t-n+1..t and their count (NaN-aware, shorter views ok)."""
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
    """Monday 0 .. Sunday 6, from the date of each row alone (epoch day 0 was a Thursday, hence + 3)."""
    d = np.asarray(dates).astype(np.int64)
    return (d + 3) % 7


def pit_universe(data):
    """The gate's point-in-time universe; if none is passed (K-0131), the top 30 listed perps by trailing
    30-day quote volume (sum of dollar_volume over t-29..t), computed from rows <= t only."""
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
    """The card's five raw features and the 'scored' mask; features are NaN where a coin is not scored."""
    close = np.asarray(close, dtype=np.float64)
    c = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    logc = np.log(c)
    # (1) mom28 = log(close_t / close_t-28); (2) ret7 = log(close_t / close_t-7)
    mom28 = logc - _lag(logc, 28)
    ret7 = logc - _lag(logc, 7)
    # (3) basis7 = mean daily basis over t-6..t, at least 5 valid values
    bs, bc = _rolling_sum_count(basis, 7)
    basis7 = np.where(bc >= 5, bs / np.maximum(bc, 1.0), np.nan)
    # (4) volshock = log(mean qv over t-6..t / mean qv over t-89..t)
    q7s, q7c = _rolling_sum_count(qv, 7)
    q90s, q90c = _rolling_sum_count(qv, 90)
    m7 = np.where(q7c > 0, q7s / np.maximum(q7c, 1.0), np.nan)
    m90 = np.where(q90c > 0, q90s / np.maximum(q90c, 1.0), np.nan)
    ok = np.isfinite(m7) & np.isfinite(m90) & (m7 > 0) & (m90 > 0)
    volshock = np.full(c.shape, np.nan)
    volshock[ok] = np.log(m7[ok] / m90[ok])
    # (5) rvol28 = std (ddof 1) of daily log close-to-close returns over t-27..t
    r = logc - _lag(logc, 1)
    rs, rc = _rolling_sum_count(r, 28)
    rss, _ = _rolling_sum_count(r * r, 28)
    var = (rss - rs * rs / np.maximum(rc, 1.0)) / np.maximum(rc - 1.0, 1.0)
    rvol28 = np.where(rc >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    # scored: member on t, >= 80 valid closes in the last 90 rows, >= 5 basis values in the last 7
    _, cc90 = _rolling_sum_count(c, 90)
    scored = np.asarray(member, dtype=bool) & (cc90 >= 80) & (bc >= 5)
    feats = [np.where(scored, f, np.nan) for f in (mom28, ret7, basis7, volshock, rvol28)]
    return feats, scored


def model_inputs(data):
    """Point-in-time universe, z-scored feature stack (T, N, 5) and the ranked 21-row label."""
    member = pit_universe(data)
    basis = np.asarray(data.extras["basis"], dtype=np.float64)       # fails loudly if missing
    qv = np.asarray(data.dollar_volume, dtype=np.float64)            # quote volume (USDT) of the perp
    feats, scored = compute_features(data.close, qv, basis, member)
    X = stack(*[cs_zscore(f, scored) for f in feats])               # z-score across scored members
    y = cs_rank(forward_return(data, HORIZON), member)               # label: next-open 21-row return, ranked
    return member, X, y


def build_book(score, member, decision, n_leg, hold=HORIZON, min_scored=MIN_SCORED):
    """Overlapping cohorts: at each decision row with >= min_scored finite member scores, long the n_leg
    highest and short the n_leg lowest at 1/(6 n_leg) each, held `hold` rows; a coin that leaves the universe
    is closed at the first row outside it and stays in cash until its cohort expires. Weights add up."""
    score = np.asarray(score, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    T, N = score.shape
    W = np.zeros((T, N), dtype=np.float64)
    w = 1.0 / (6.0 * n_leg)
    for t in np.flatnonzero(decision):
        idx = np.flatnonzero(member[t] & np.isfinite(score[t]))
        if idx.size < min_scored:
            continue
        s = score[t, idx]
        longs = idx[np.argsort(-s, kind="stable")[:n_leg]]
        shorts = idx[np.argsort(s, kind="stable")[:n_leg]]
        e = min(t + hold, T)
        for cols, sgn in ((longs, 1.0), (shorts, -1.0)):
            alive = np.cumprod(member[t:e][:, cols].astype(np.int64), axis=0) > 0
            W[t:e, cols] += sgn * w * alive
    gross = np.abs(W).sum(axis=1)
    over = gross > 1.0                       # only float rounding on a gap-free 24/7 calendar
    W[over] = W[over] / gross[over][:, None]
    return W


def gbm_scores(data, params, member, X, y):
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                        retrain_every=REFIT_EVERY, universe=member)


# ----------------------------------------------------------------------------- strategy

def target_weights(data, params):
    member, X, y = model_inputs(data)
    score = gbm_scores(data, params, member, X, y)
    decision = weekday(data.dates) == SUNDAY          # close of the Sunday bar (00:00 UTC Monday)
    return build_book(score, member, decision, int(params["n_leg"]))
