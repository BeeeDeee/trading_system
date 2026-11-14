"""H-0151 mechanism test: fresh long-book entries.

At a monthly decision row (last trading day of the month; same calendar rule as the strategy), a scored LIQ-500
member whose model score is in the top `top` fraction of scored members and that was NOT in the top 2 x `top`
fraction at the previous monthly decision. side = +1. The feature and walk_forward pipeline is a copy of
strategy.py (diagnostic.py must be a separate file).
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 60
RETRAIN_EVERY = 126
VOL_WINDOW = 252
VOL_MIN_VALID = 200
DROA_LAG = 252
MIN_SCORED = 20
WINSOR_LO = 1.0
WINSOR_HI = 99.0
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 1000, "subsample": 0.5,
             "colsample": 1.0}


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    last = nxt.astype("datetime64[M]") != month
    new_month = np.zeros(T, dtype=bool)
    new_month[1:] = month[1:] != month[:-1]
    prev_last = np.zeros(T, dtype=bool)
    prev_last[1:] = last[:-1]
    return last | (new_month & ~prev_last)


def _lag_rows(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def book_to_market(pb, idx):
    pb = np.asarray(pb, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.float64)
    T, N = pb.shape
    prev = np.full_like(pb, np.nan)
    prev[1:] = pb[:-1]
    changed = ~(pb == prev)
    changed[0] = True
    rows = np.arange(T)[:, None]
    anchor = np.maximum.accumulate(np.where(changed, rows, 0), axis=0)
    base_row = np.maximum(anchor - 1, 0)
    base = np.take_along_axis(idx, base_row, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = idx / base
        inv_pb = np.where(pb > 0, 1.0 / pb, np.nan)
        bm = inv_pb / grow
    bm[~np.isfinite(bm) | ~(base > 0)] = np.nan
    return bm


def _ratio(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def realized_vol(r, valid, window=VOL_WINDOW, min_valid=VOL_MIN_VALID):
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(r)
    x = np.where(v, r, 0.0)
    T, N = x.shape
    c1 = np.zeros((T + 1, N))
    c2 = np.zeros((T + 1, N))
    cn = np.zeros((T + 1, N))
    c1[1:] = np.cumsum(x, axis=0)
    c2[1:] = np.cumsum(x * x, axis=0)
    cn[1:] = np.cumsum(v, axis=0)
    lo = np.maximum(np.arange(1, T + 1) - window, 0)
    hi = np.arange(1, T + 1)
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    n = cn[hi] - cn[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    vol = np.sqrt(var)
    vol[n < min_valid] = np.nan
    return vol


def winsorize_rows(x, lo=WINSOR_LO, hi=WINSOR_HI):
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x)
    n = fin.sum(axis=1)
    srt = np.sort(np.where(fin, x, np.inf), axis=1)
    out = x.copy()
    rows = np.nonzero(n > 0)[0]
    if rows.size == 0:
        return out
    nn = n[rows].astype(np.float64)

    def q(p):
        pos = (nn - 1.0) * p / 100.0
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n[rows] - 1)
        frac = pos - i0
        a = srt[rows, i0]
        b = srt[rows, i1]
        return a + (b - a) * frac

    qlo = q(lo)[:, None]
    qhi = q(hi)[:, None]
    out[rows] = np.clip(x[rows], qlo, qhi)
    out[~fin] = np.nan
    return out


def raw_features(data):
    ex = data.extras
    pb = ex["sf1_arq_pb"]
    assets = ex["sf1_arq_assets"]
    gp = ex["sf1_art_gp"]
    netinc = np.asarray(ex["sf1_art_netinc"], dtype=np.float64)
    ncfo = np.asarray(ex["sf1_art_ncfo"], dtype=np.float64)
    idx = total_return_index(data)
    bm = book_to_market(pb, idx)
    gpa = _ratio(gp, assets)
    acc = _ratio(netinc - ncfo, assets)
    roa = _ratio(netinc, assets)
    droa = roa - _lag_rows(roa, DROA_LAG)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    vol = realized_vol(r, valid)
    return [bm, gpa, acc, droa, vol]


def model_inputs(data):
    uni = data.universe
    feats = raw_features(data)
    complete = np.asarray(uni, dtype=bool).copy()
    for f in feats:
        complete &= np.isfinite(f)
    zs = [cs_zscore(winsorize_rows(np.where(complete, f, np.nan)), uni) for f in feats]
    return stack(*zs)


def model_scores(data, params):
    uni = data.universe
    X = model_inputs(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score[~np.asarray(uni, dtype=bool)] = np.nan
    return score


def rank_sets(s, ok, top):
    N = s.shape[0]
    n = int(ok.sum())
    in_top = np.zeros(N, dtype=bool)
    in_keep = np.zeros(N, dtype=bool)
    if n < MIN_SCORED:
        return in_top, in_keep, n
    cols = np.nonzero(ok)[0]
    order = cols[np.lexsort((cols, -s[cols]))]
    n_top = int(np.floor(top * n + 1e-9))
    n_keep = int(np.floor(2.0 * top * n + 1e-9))
    in_top[order[:n_top]] = True
    in_keep[order[:n_keep]] = True
    return in_top, in_keep, n


def fresh_entries(score, member, tradable, dec, top):
    score = np.asarray(score, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    prev_keep = np.zeros(N, dtype=bool)
    for t in np.nonzero(dec)[0]:
        ok = np.isfinite(score[t]) & member[t]
        in_top, in_keep, n = rank_sets(score[t], ok, top)
        mask[t] = in_top & ~prev_keep & member[t] & tradable[t]
        prev_keep = in_keep
    return mask


def events(data, params):
    top = float(params["top"])
    score = model_scores(data, params)
    dec = decision_rows(data.dates)
    mask = fresh_entries(score, data.universe, data.tradable, dec, top)
    side = np.where(mask, 1.0, 0.0)
    return mask, side
