"""H-0144 Momentum life cycle in liquid crypto, learned.

Binance spot top-50 (PIT, framework crypto_top_n), long-only. Weekly decisions on the first row of each
ISO week. Five per-coin features (trend, recent, stage, vtrend, vol90), z-scored across the scored members
of each row; LightGBM (framework gbm) in walk_forward on the 56-row forward-return rank. Book: 8 sleeves of
1/8, one replaced per week; each new sleeve is equal weight over the top `top` fraction (floor, >= 3) of the
scored members, or BTCUSDT when fewer than 15 members are scored / no model fit exists. Sleeve positions in
coins that left the universe are moved to BTCUSDT at the next weekly decision row.
"""
import math

import numpy as np
from lab.framework.api import only_on, period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

# fixed by the card (not in signal.params)
HORIZON = 56            # label horizon and hold (rows)
RETRAIN_EVERY = 56      # ml.retrain_every_days
N_SLEEVES = 8
MIN_SCORED = 15         # fewer scored members -> new sleeve holds BTC
MIN_PICKS = 3
MIN_VALID_BARS = 150    # valid daily bars in t-179..t to be scored
FORMATION = 180
SKIP = 30
VOL_WIN = 90
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 400, "subsample": 0.7,
             "colsample": 1.0}
BTC = "BTCUSDT"


# ----------------------------------------------------------------------------------------------- helpers
def _lag(x, k):
    """x shifted down by k >= 0 rows (row t holds x[t-k]); NaN where t-k < 0."""
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if k < T:
        out[k:] = x[:T - k] if k > 0 else x
    return out


def _window_sum(x, n):
    """Sum over rows t-n+1..t of x (NaN treated as 0); NaN where the window is incomplete (t < n-1)."""
    x0 = np.where(np.isfinite(x), x, 0.0).astype(np.float64)
    T = x0.shape[0]
    c = np.vstack([np.zeros((1,) + x0.shape[1:]), np.cumsum(x0, axis=0)])
    out = np.full(x0.shape, np.nan)
    if n <= T:
        out[n - 1:] = c[n:] - c[:T - n + 1]
    return out


def _window_mean_valid(x, valid, n):
    """Mean of x over the valid rows of the trailing n-row window; NaN if none or the window is incomplete."""
    xs = np.where(valid, x, 0.0)
    s = _window_sum(xs, n)
    c = _window_sum(valid.astype(np.float64), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan), c


def _window_max(x, n):
    """Max of x over rows t-n+1..t ignoring NaN; NaN where the window is incomplete or all NaN."""
    T = x.shape[0]
    m = np.full(x.shape, np.nan)
    for k in range(min(n, T)):
        m = np.fmax(m, _lag(x, k))
    if n - 1 > 0:
        m[:min(n - 1, T)] = np.nan
    return m


def raw_features(ret_co, ret_oc, close, qv, listed):
    """The card's five raw features and the valid-bar count (all (T, N), rows <= t only)."""
    close = np.asarray(close, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    valid = np.isfinite(close) & np.asarray(listed, dtype=bool)
    rco = np.where(np.isfinite(ret_co), ret_co, 0.0).astype(np.float64)
    roc = np.where(np.isfinite(ret_oc), ret_oc, 0.0).astype(np.float64)
    # daily log total return (close to close); 0 where there is no trading; guard against <= -100 %
    lr = np.log1p(np.maximum(rco, -0.9999)) + np.log1p(np.maximum(roc, -0.9999))
    cl = np.cumsum(lr, axis=0)                                   # cumulative log total return at each close
    trend = _lag(cl, SKIP) - _lag(cl, FORMATION)                 # t-180 -> t-30
    recent = cl - _lag(cl, SKIP)                                 # t-30 -> t
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
    """Features z-scored across the scored members of each row, and the scored mask."""
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
        return ps, np.flatnonzero(ps)
    mask = np.zeros(len(dates), dtype=bool)
    mask[ps.astype(int)] = True
    return mask, np.flatnonzero(mask)


def top_set(score_row, top):
    """Columns of the new sleeve: top floor(top x n) (>= 3) scored columns, or None if < 15 scored."""
    ok = np.isfinite(score_row)
    n = int(ok.sum())
    if n < MIN_SCORED:
        return None
    m = max(MIN_PICKS, int(math.floor(float(top) * n + 1e-9)))
    cols = np.flatnonzero(ok)
    order = np.argsort(-score_row[cols], kind="stable")
    return cols[order[:m]]


def sleeve_book(score, member, listed, dec_rows, btc, top):
    """(T, N) book on the decision rows (NaN elsewhere) from the 8-sleeve rule."""
    T, N = score.shape
    out = np.full((T, N), np.nan)
    sleeves = np.zeros((N_SLEEVES, N))
    sleeves[:, btc] = 1.0                       # sleeves not yet formed hold BTC (no model fit before data start)
    ok_cols = np.ones(N, dtype=bool)
    for k, t in enumerate(dec_rows):
        ok_cols = member[t] & listed[t]
        ok_cols[btc] = True                     # BTC is the filler; never moved to itself
        bad = (sleeves != 0.0) & ~ok_cols[None, :]
        if bad.any():
            moved = np.where(bad, sleeves, 0.0).sum(axis=1)
            sleeves[bad] = 0.0
            sleeves[:, btc] += moved
        new = np.zeros(N)
        picks = top_set(score[t], top)
        if picks is None:
            new[btc] = 1.0
        else:
            new[picks] = 1.0 / len(picks)
        sleeves[k % N_SLEEVES] = new
        w = sleeves.sum(axis=0) / N_SLEEVES
        if not member[t, btc]:
            w[btc] = 0.0                        # only members may be held: the BTC filler waits in cash
        g = np.abs(w).sum()
        if g > 1.0:
            w = w / g
        out[t] = w
    return out


def target_weights(data, params):
    score, uni = model_score(data, params)
    btc = data.col(BTC)
    dmask, drows = decision_rows(data.dates)
    book = sleeve_book(score, uni, np.asarray(data.listed, dtype=bool), drows, btc, params["top"])
    return only_on(dmask, book)
