"""H-0194 mechanism events: exactly the book's entries (side +1).

A LIQ-500 member with a newly usable open-market purchase filing at close t (ins_buy_n_91d rises, no such
candidate in t-5..t-1), member and tradable at t, whose out-of-sample gbm score is at or above the (1 - top)
quantile of the scored events in rows t-252..t-1 (at least 50 of them). The pipeline below is a copy of
strategy.py (features, walk_forward, entry rule); diagnostic.py cannot import strategy.py.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
DEDUP = 5
Q_WINDOW = 252
MIN_PRIOR = 50
GBM_FIXED = {"num_leaves": 8, "min_child_samples": 50, "learning_rate": 0.05}


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    return np.where(listed & np.isfinite(r), r, np.nan)


def _window_sum(x, n):
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cc = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(v, axis=0, out=cs[1:])
    np.cumsum(valid, axis=0, out=cc[1:])
    lo = np.maximum(np.arange(1, T + 1) - n, 0)
    s = cs[1:] - cs[lo]
    c = cc[1:] - cc[lo]
    return s, c


def trailing_compound_return(r, n):
    lr = np.log1p(np.where(np.isfinite(r), np.maximum(r, -0.999999), np.nan))
    s, c = _window_sum(lr, n)
    out = np.expm1(s)
    out[c < n] = np.nan
    return out


def trailing_vol(r, n):
    s, c = _window_sum(r, n)
    s2, _ = _window_sum(r * r, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / c
        var = s2 / c - mean * mean
    out = np.sqrt(np.maximum(var, 0.0))
    out[c < n] = np.nan
    return out


def row_median(x, mask):
    xm = np.where(mask & np.isfinite(x), x, np.nan)
    out = np.full(x.shape[0], np.nan)
    rows = np.isfinite(xm).any(axis=1)
    if rows.any():
        out[rows] = np.nanmedian(xm[rows], axis=1)
    return out


def detect_events(buy_n91, uni, tradable, dedup=DEDUP):
    b = np.asarray(buy_n91, dtype=np.float64)
    T = b.shape[0]
    rise = np.zeros(b.shape, dtype=bool)
    if T > 1:
        with np.errstate(invalid="ignore"):
            rise[1:] = b[1:] > b[:-1]
    cand = rise & np.asarray(uni, dtype=bool) & np.asarray(tradable, dtype=bool)
    cs = np.zeros((T + 1,) + b.shape[1:], dtype=np.int32)
    np.cumsum(cand, axis=0, out=cs[1:])
    t = np.arange(T)
    lo = np.maximum(t - dedup, 0)
    prior = cs[t] - cs[lo]
    return cand & (prior == 0)


def build_features(data, event, uni):
    ex = data.extras
    r = daily_returns(data)
    r21 = trailing_compound_return(r, 21)
    rel21 = r21 - row_median(r21, uni)[:, None]
    del r21
    z6 = np.asarray(cs_zscore(rel21, uni), dtype=np.float32)
    del rel21
    vol63 = trailing_vol(r, 63)
    del r
    z7 = np.asarray(cs_zscore(vol63, uni), dtype=np.float32)
    del vol63

    bv91 = np.asarray(ex["ins_buy_value_91d"], dtype=np.float32)
    bn91 = np.asarray(ex["ins_buy_n_91d"], dtype=np.float32)
    bn365 = np.asarray(ex["ins_buy_n_365d"], dtype=np.float32)
    sv182 = np.asarray(ex["ins_sell_value_182d"], dtype=np.float32)
    mcap = np.asarray(ex["sf1_arq_marketcap"], dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcap = np.where(np.isfinite(mcap) & (mcap > 0), mcap, np.float32(np.nan))
        f1 = np.log1p(np.where(bv91 >= 0, bv91, np.float32(np.nan)))
        f2 = bv91 / mcap
        f4 = bn365 - bn91
        f5 = sv182 / mcap
    del mcap
    feats = [f1, f2, bn91, f4, f5, z6, z7]
    complete = event.copy()
    for f in feats:
        complete &= np.isfinite(f)
    out = []
    for i in range(len(feats)):
        out.append(np.where(complete, feats[i], np.float32(np.nan)).astype(np.float32, copy=False))
        feats[i] = None
    return out, complete


def model_scores(data, params, uni):
    if uni is None:
        raise ValueError("H-0194 needs the point-in-time LIQ-500 universe")
    ex = data.extras
    tradable = np.asarray(data.tradable, dtype=bool)
    event = detect_events(ex["ins_buy_n_91d"], uni, tradable)
    feats, complete = build_features(data, event, uni)
    X = stack(*feats)
    del feats
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    del X, y
    score = np.asarray(score, dtype=np.float64)
    score = np.where(complete & np.isfinite(score), score, np.nan)
    return event, score


def select_entries(score, top, window=Q_WINDOW, min_prior=MIN_PRIOR):
    T = score.shape[0]
    entry = np.zeros(score.shape, dtype=bool)
    rows, cols = np.nonzero(np.isfinite(score))
    if rows.size == 0:
        return entry
    vals = score[rows, cols]
    q = 1.0 - float(top)
    for t in np.unique(rows):
        lo = np.searchsorted(rows, t - window, side="left")
        hi = np.searchsorted(rows, t, side="left")
        if hi - lo < min_prior:
            continue
        thr = np.quantile(vals[lo:hi], q)
        a = np.searchsorted(rows, t, side="left")
        b = np.searchsorted(rows, t, side="right")
        sel = vals[a:b] >= thr
        entry[t, cols[a:b][sel]] = True
    return entry


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0194 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    _, score = model_scores(data, params, uni)
    mask = select_entries(score, params["top"]) & uni & tradable
    return mask, None
