"""H-0186 mechanism events: a perp enters the long (+1) or short (-1) leg of the cohort formed on day t
from the ridge score, having not been in that leg of the cohort formed on day t-1.

Same pipeline as strategy.py (copied, this file is evaluated on its own)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

RIDGE_ALPHA = 1.0
RETRAIN_EVERY = 63
TOP_N = 30
UNIVERSE_WINDOW = 30
MIN_SCORED = 15
MIN_LEG = 3

RET_WINDOW = 28
MIN_RET_DAYS = 21
VOL_LONG = 60
VOL_SHORT = 7
MIN_VOL_DAYS = 45
FUND_WINDOW = 7
MIN_FUND_DAYS = 5


def _rolling_sum(x, n):
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return c[hi] - c[lo]


def _lag(x, k, fill=np.nan):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, fill)
    if k == 0:
        return x.copy()
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def daily_returns(data):
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
    r, valid = daily_returns(data)
    f1 = np.where(valid, r, np.nan)
    lr = np.where(valid, np.log1p(np.maximum(r, -0.999999)), 0.0)
    cum = np.cumsum(lr, axis=0)
    f2 = np.expm1(_lag(cum, 1) - _lag(cum, 7, fill=0.0))
    f3 = np.expm1(_lag(cum, 7) - _lag(cum, 28, fill=0.0))

    n_ret = _rolling_sum(valid.astype(np.float64), RET_WINDOW)
    s1 = _rolling_sum(r, RET_WINDOW)
    s2 = _rolling_sum(r * r, RET_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n_ret) / (n_ret - 1.0)
    f4 = np.sqrt(np.maximum(var, 0.0))
    f4 = np.where(n_ret >= 2, f4, np.nan)

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


def leg_masks(score, leg_frac):
    s = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s)
    n = ok.sum(axis=1)
    k = np.maximum(np.floor(leg_frac * n + 1e-9).astype(int), MIN_LEG)
    form = n >= MIN_SCORED
    key = np.where(ok, -s, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    rank[rows, order] = np.arange(s.shape[1])[None, :]
    long_ = ok & form[:, None] & (rank < k[:, None])
    short = ok & form[:, None] & (rank >= (n - k)[:, None])
    return long_, short, k


def entry_events(long_, short, member, tradable):
    """New entries into the long / short leg of today's cohort versus yesterday's cohort."""
    prev_long = np.zeros_like(long_)
    prev_short = np.zeros_like(short)
    prev_long[1:] = long_[:-1]
    prev_short[1:] = short[:-1]
    new_long = long_ & ~prev_long
    new_short = short & ~prev_short
    ok = member & tradable
    mask = (new_long | new_short) & ok
    side = np.where(new_long, 1.0, np.where(new_short, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side


def events(data, params):
    lf = float(params["leg_frac"])
    score, member = scores(data, params)
    long_, short, _ = leg_masks(score, lf)
    tradable = np.asarray(data.tradable, dtype=bool)
    return entry_events(long_, short, member, tradable)
