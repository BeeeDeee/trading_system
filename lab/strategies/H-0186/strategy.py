"""H-0186: linear short-term momentum in liquid crypto perps, learned.

Top-30 Binance perps (point-in-time), six features z-scored across the scored members of each date,
walk-forward ridge (alpha 1.0, refit every 63 rows) on the cross-sectional rank of the forward
open-to-open return over `horizon` rows. Every day a dollar-neutral cohort (long the top leg_frac,
short the bottom leg_frac of the scored members, each leg 0.5 / horizon) is formed and held for
`horizon` days; the book is the sum of the live cohorts.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"horizon": 5, "leg_frac": 0.3}

RIDGE_ALPHA = 1.0        # card: fixed, not a grid parameter
RETRAIN_EVERY = 63       # card: ml.retrain_every_days
TOP_N = 30               # card: universe crypto_top_n, n = 30 (fallback only)
UNIVERSE_WINDOW = 30     # fallback universe: 30-day quote volume
MIN_SCORED = 15          # fewer scored members -> no cohort that day
MIN_LEG = 3              # at least 3 names per leg

RET_WINDOW = 28          # t-27..t
MIN_RET_DAYS = 21
VOL_LONG = 60            # t-59..t
VOL_SHORT = 7            # t-6..t
MIN_VOL_DAYS = 45
FUND_WINDOW = 7          # t-6..t
MIN_FUND_DAYS = 5


# ----------------------------------------------------------------------------------------------
# point-in-time rolling helpers (trailing windows that include row t; work for any number of rows)
# ----------------------------------------------------------------------------------------------
def _rolling_sum(x, n):
    """Sum of x over rows t-n+1..t (fewer rows at the start). x must be finite."""
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return c[hi] - c[lo]


def _lag(x, k, fill=np.nan):
    """x shifted down by k >= 0 rows (row t holds x[t-k])."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, fill)
    if k == 0:
        return x.copy()
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------
def daily_returns(data):
    """Daily total return close(t-1) -> close(t) and its validity mask (price on both closes)."""
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    have = np.isfinite(close) & listed
    prev = np.zeros_like(have)
    prev[1:] = have[:-1]
    valid = have & prev
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid &= np.isfinite(r)
    r = np.where(valid, r, 0.0)
    return r, valid


def fallback_universe(data, n=TOP_N, window=UNIVERSE_WINDOW):
    """Top-n perps by trailing `window`-day quote volume (sum over rows t-window+1..t; a missing day
    adds nothing), among perps listed with a price on day t. Ties broken by column order."""
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    vol = _rolling_sum(np.where(np.isfinite(dv) & (dv > 0), dv, 0.0), window)
    ok = listed & np.isfinite(close) & (vol > 0)
    key = np.where(ok, -vol, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    return ok & (rank < n)


def membership(data):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    return fallback_universe(data)


def raw_features(data):
    """F1..F6 (raw, before z-scoring) and the eligibility mask of the card's scoring rule."""
    r, valid = daily_returns(data)

    # F1: total return of day t
    f1 = np.where(valid, r, np.nan)

    # compounded returns, skipping missing days: log-sum of valid daily returns
    lr = np.where(valid, np.log1p(np.maximum(r, -0.999999)), 0.0)
    cum = np.cumsum(lr, axis=0)                       # cum[t] = sum of log returns of rows 0..t
    # F2: close(t-7) -> close(t-1) = returns of rows t-6..t-1
    f2 = np.expm1(_lag(cum, 1) - _lag(cum, 7, fill=0.0))
    # F3: close(t-28) -> close(t-7) = returns of rows t-27..t-7
    f3 = np.expm1(_lag(cum, 7) - _lag(cum, 28, fill=0.0))

    # F4: std of valid daily returns over t-27..t
    n_ret = _rolling_sum(valid.astype(np.float64), RET_WINDOW)
    s1 = _rolling_sum(r, RET_WINDOW)
    s2 = _rolling_sum(r * r, RET_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n_ret) / (n_ret - 1.0)
    f4 = np.sqrt(np.maximum(var, 0.0))
    f4 = np.where(n_ret >= 2, f4, np.nan)

    # F5: log(mean quote volume t-6..t / mean quote volume t-59..t), valid volume days only
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    vvalid = np.isfinite(dv) & (dv > 0)
    dv0 = np.where(vvalid, dv, 0.0)
    n_vs = _rolling_sum(vvalid.astype(np.float64), VOL_SHORT)
    n_vl = _rolling_sum(vvalid.astype(np.float64), VOL_LONG)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_s = _rolling_sum(dv0, VOL_SHORT) / n_vs
        m_l = _rolling_sum(dv0, VOL_LONG) / n_vl
        f5 = np.log(m_s / m_l)
    f5 = np.where((n_vs >= 1) & (n_vl >= 1), f5, np.nan)

    # F6: sum of daily funding over t-6..t, valid funding days only
    fund = np.asarray(data.extras["funding"], dtype=np.float64)
    fvalid = np.isfinite(fund)
    n_f = _rolling_sum(fvalid.astype(np.float64), FUND_WINDOW)
    f6 = _rolling_sum(np.where(fvalid, fund, 0.0), FUND_WINDOW)

    eligible = (n_ret >= MIN_RET_DAYS) & (n_vl >= MIN_VOL_DAYS) & (n_f >= MIN_FUND_DAYS)
    feats = [f1, f2, f3, f4, f5, f6]
    for f in feats:
        eligible &= np.isfinite(f)
    feats = [np.where(eligible, f, np.nan) for f in feats]
    return feats, eligible


def scores(data, params):
    """Walk-forward ridge scores (T, N); NaN where a member is not scored."""
    h = int(params["horizon"])
    member = membership(data)
    feats, eligible = raw_features(data)
    scored = member & eligible
    X = stack(*[cs_zscore(np.where(scored, f, np.nan), scored) for f in feats])
    y = cs_rank(forward_return(data, h), scored)
    s = walk_forward(X, y, horizon=h, model="ridge", params={"alpha": RIDGE_ALPHA},
                     retrain_every=RETRAIN_EVERY, universe=member)
    s = np.asarray(s, dtype=np.float64)
    return np.where(scored & np.isfinite(s), s, np.nan), member


# ----------------------------------------------------------------------------------------------
# book
# ----------------------------------------------------------------------------------------------
def leg_masks(score, leg_frac):
    """Long / short masks of the cohort formed on each row (rows with < MIN_SCORED scored: none)."""
    s = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s)
    n = ok.sum(axis=1)
    k = np.maximum(np.floor(leg_frac * n + 1e-9).astype(int), MIN_LEG)
    form = n >= MIN_SCORED
    key = np.where(ok, -s, np.inf)
    order = np.argsort(key, axis=1, kind="stable")    # best score first, ties by column order
    rank = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    rank[rows, order] = np.arange(s.shape[1])[None, :]
    long_ = ok & form[:, None] & (rank < k[:, None])
    short = ok & form[:, None] & (rank >= (n - k)[:, None])
    return long_, short, k


def cohort_weights(score, horizon, leg_frac):
    """Weights of the cohort formed on each row: each leg 0.5 / horizon, equal weight inside."""
    long_, short, k = leg_masks(score, leg_frac)
    leg = (0.5 / horizon) / k.astype(np.float64)
    return long_ * leg[:, None] - short * leg[:, None]


def book(cohorts, member, horizon):
    """Sum of the cohorts formed on rows t-horizon+1..t, zero outside the point-in-time universe."""
    w = np.zeros_like(cohorts)
    for j in range(horizon):
        w += _lag(cohorts, j, fill=0.0)
    w = np.where(member, w, 0.0)
    gross = np.abs(w).sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / np.maximum(gross, 1e-300), 1.0)   # rounding guard only
    return w * scale[:, None]


def target_weights(data, params):
    h = int(params["horizon"])
    lf = float(params["leg_frac"])
    score, member = scores(data, params)
    return book(cohort_weights(score, h, lf), member, h)
