"""H-0111 mechanism event: at a weekly decision date a member enters the top (bottom) `tail` of the gbm
score (it was not in that tail at the previous weekly decision) while its 5-session total return lies
between the 20th and 80th percentile of the scored members on that date. Side +1 top tail, -1 bottom.
The feature / walk_forward / score_weights pipeline is copied from strategy.py (same code)."""
import numpy as np
from lab.framework.api import period_starts, total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
WINDOW = 21
MIN_HALF = 36
RETRAIN = 126
NUM_LEAVES = 15
MIN_SCORED = 100
P_LO = 20.0
P_HI = 80.0


def _lag(x, k, fill):
    if k == 0:
        return x
    out = np.full_like(x, fill)
    out[k:] = x[:-k]
    return out


def valid_sessions(ret_co, ret_oc, close, listed):
    has_px = np.isfinite(close)
    prev_px = _lag(has_px, 1, False)
    with np.errstate(invalid="ignore"):
        ok = (listed.astype(bool) & has_px & prev_px & np.isfinite(ret_co) & np.isfinite(ret_oc)
              & (ret_co > -1.0) & (ret_oc > -1.0))
    return ok


def halfday_features(ret_co, ret_oc, close, listed, window=WINDOW, min_half=MIN_HALF):
    ok = valid_sessions(ret_co, ret_oc, close, listed)
    with np.errstate(invalid="ignore", divide="ignore"):
        co = np.where(ok, np.log1p(np.where(ok, ret_co, 0.0)), 0.0).astype(np.float64)
        oc = np.where(ok, np.log1p(np.where(ok, ret_oc, 0.0)), 0.0).astype(np.float64)
    okf = ok.astype(np.float64)

    n = np.zeros(co.shape, dtype=np.float64)
    s1 = np.zeros(co.shape, dtype=np.float64)
    sq_co = np.zeros(co.shape, dtype=np.float64)
    sq_all = np.zeros(co.shape, dtype=np.float64)
    for k in range(window):
        a = _lag(co, k, 0.0)
        b = _lag(oc, k, 0.0)
        n += 2.0 * _lag(okf, k, 0.0)
        s1 += a + b
        a2 = a * a
        sq_co += a2
        sq_all += a2 + b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
    del s1
    d2 = np.zeros(co.shape, dtype=np.float64)
    d3 = np.zeros(co.shape, dtype=np.float64)
    for k in range(window):
        v = _lag(okf, k, 0.0)
        a = (_lag(co, k, 0.0) - mean) * v
        b = (_lag(oc, k, 0.0) - mean) * v
        d2 += a * a + b * b
        d3 += a * a * a + b * b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = d2 / n
        m3 = d3 / n
        g1 = m3 / m2 ** 1.5
        rskew = g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)
        rvol = np.sqrt(d2 / (n - 1.0))
        onshare = sq_co / sq_all
    good = (n >= min_half) & (m2 > 0) & (sq_all > 0)
    rskew = np.where(good, rskew, np.nan)
    rvol = np.where(good, rvol, np.nan)
    onshare = np.where(good, onshare, np.nan)
    return rskew, rvol, onshare, n


def ea_window(ea, window=WINDOW):
    hit = (np.nan_to_num(np.asarray(ea, dtype=np.float64), nan=0.0) == 1.0).astype(np.int64)
    c = np.cumsum(hit, axis=0)
    prev = _lag(c, window, 0)
    return ((c - prev) > 0).astype(np.float64)


def features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0111 needs the liq_n point-in-time universe (data.universe is None)")
    ea = data.extras["ea_reaction_day"]
    rskew, rvol, onshare, n_half = halfday_features(data.ret_co, data.ret_oc, data.close, data.listed)
    eligible = n_half >= MIN_HALF
    idx = total_return_index(data)
    ret5 = trailing_return(idx, 5)
    ret21 = trailing_return(idx, WINDOW)
    ea21 = ea_window(ea, WINDOW)
    raw = [rskew, rvol, onshare, ret5, ret21, ea21]
    raw = [np.where(eligible, f, np.nan) for f in raw]
    return [cs_zscore(f, uni) for f in raw], ret5


def scores(data, params):
    feats, ret5 = features(data)
    X = stack(*feats)
    uni = data.universe
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm",
                         params={"n_estimators": int(params["n_estimators"]), "num_leaves": NUM_LEAVES},
                         retrain_every=RETRAIN, universe=uni)
    return score, ret5


def tail_events(w_all, score, ret5, days, uni, tradable):
    """Pure event logic on the book (w_all from score_weights), so it can be tested by hand."""
    T, N = w_all.shape
    w = np.nan_to_num(np.asarray(w_all, dtype=np.float64), nan=0.0)
    uni = np.asarray(uni, dtype=bool)
    long_now = w > 0
    short_now = w < 0
    rows = np.arange(T)
    dec = np.asarray(days, dtype=bool)
    # previous weekly decision row for each row (strictly earlier)
    last_incl = np.maximum.accumulate(np.where(dec, rows, -1))
    prev_dec = np.full(T, -1)
    prev_dec[1:] = last_incl[:-1]
    has_prev = prev_dec >= 0
    pl = np.where(has_prev[:, None], long_now[np.maximum(prev_dec, 0)], False)
    ps = np.where(has_prev[:, None], short_now[np.maximum(prev_dec, 0)], False)

    scored = uni & np.isfinite(score) & np.isfinite(ret5)
    r = np.where(scored, ret5, np.nan)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    any_scored = scored.any(axis=1) & dec
    if any_scored.any():
        sub = r[any_scored]
        lo[any_scored] = np.nanpercentile(sub, P_LO, axis=1)
        hi[any_scored] = np.nanpercentile(sub, P_HI, axis=1)
    with np.errstate(invalid="ignore"):
        mid = scored & (r >= lo[:, None]) & (r <= hi[:, None])
    base = dec[:, None] & uni & np.asarray(tradable, dtype=bool) & mid
    enter_long = base & long_now & ~pl
    enter_short = base & short_now & ~ps
    mask = enter_long | enter_short
    side = np.where(enter_short, -1.0, 1.0)
    return mask, side


def events(data, params):
    score, ret5 = scores(data, params)
    tail = float(params["tail"])
    w_all = score_weights(score, data.universe, top=tail, bottom=tail, min_names=MIN_SCORED)
    days = period_starts(data.dates, "W")
    return tail_events(w_all, score, ret5, days, data.universe, data.tradable)
