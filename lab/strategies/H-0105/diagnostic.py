"""H-0105 mechanism event: a scored candidate row (large peer-residual overnight gap, no 8-K on t-1..t, no
earnings flag on t-5..t) on which the gbm and the ridge control (same features, rows, label and refit
schedule) take opposite sides, each side against that model's trailing 252-session median score on candidate
rows. side = the gbm's side. The pipeline below is a copy of strategy.py (diagnostic.py is a separate file).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"gap_z": 2.5, "n_estimators": 100}

N_PEERS = 10
PEER_WINDOW = 252
PEER_MIN_COMMON = 200
Z_WINDOW = 63
Z_MIN = 50
PRIOR_WINDOW = 21
RV_WINDOW = 63
DV_WINDOW = 63
SINCE_CAP = 63
EA_LOOKBACK = 5
K8_LOOKBACK = 1
HORIZON = 5
RETRAIN = 63
MEDIAN_WINDOW = 252
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 0.8}
RIDGE_PARAMS = {"alpha": 1.0}


def _shift(x, k, fill):
    if k == 0:
        return x.copy()
    out = np.full(x.shape, fill, dtype=x.dtype)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _prefix(x):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=float)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def _window_stats(x, lo, hi):
    v = np.isfinite(x)
    x0 = np.where(v, x, 0.0)
    c = _prefix(v.astype(float))
    s1 = _prefix(x0)
    s2 = _prefix(x0 * x0)
    return c[hi] - c[lo], s1[hi] - s1[lo], s2[hi] - s2[lo]


def _std_from(n, s1, s2, min_n):
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
    return np.where((n >= min_n) & (sd > 0), sd, np.nan)


def _returns(data):
    co = np.asarray(data.ret_co, dtype=float)
    oc = np.asarray(data.ret_oc, dtype=float)
    close = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    fin = np.isfinite(close)
    valid_cc = listed & fin & _shift(fin, 1, False) & np.isfinite(co) & np.isfinite(oc)
    valid_io = valid_cc & tradable
    cc = (1.0 + co) * (1.0 + oc) - 1.0
    return co, oc, cc, valid_io, valid_cc


def _peers(cc, valid_cc, mem, f):
    lo = max(0, f - PEER_WINDOW + 1)
    vb = valid_cc[lo:f + 1][:, mem]
    V = vb.astype(float)
    R = np.where(vb, cc[lo:f + 1][:, mem], 0.0)
    n = V.T @ V
    sx = R.T @ V
    sxx = (R * R).T @ V
    sxy = R.T @ R
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx * sx / n
        vy = vx.T
        corr = cov / np.sqrt(vx * vy)
    ok = (n >= PEER_MIN_COMMON) & (vx > 0) & (vy > 0) & np.isfinite(corr)
    np.fill_diagonal(ok, False)
    key = np.where(ok, corr, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")[:, :N_PEERS]
    has = ok.sum(axis=1) >= N_PEERS
    return mem[has], mem[order[has]]


def _residual(x, v, rows, js, P):
    xp = x[rows][:, P]
    vp = v[rows][:, P]
    cnt = vp.sum(axis=2)
    s = np.where(vp, xp, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = s / cnt
    own = x[rows][:, js]
    ok = v[rows][:, js] & (cnt > 0)
    return np.where(ok, own - pm, np.nan)


def _peer_features(data, co, oc, cc, valid_io, valid_cc):
    T, N = co.shape
    uni = np.asarray(data.universe, dtype=bool)
    starts = np.flatnonzero(np.asarray(period_starts(data.dates, "M"), dtype=bool))
    zg = np.full((T, N), np.nan)
    zi = np.full((T, N), np.nan)
    prior = np.full((T, N), np.nan)
    for m, f in enumerate(starts):
        b = int(starts[m + 1]) if m + 1 < len(starts) else T
        f = int(f)
        mem = np.flatnonzero(uni[f])
        if mem.size < N_PEERS + 1:
            continue
        js, P = _peers(cc, valid_cc, mem, f)
        if js.size == 0:
            continue
        a = max(0, f - Z_WINDOW)
        rows = slice(a, b)
        G = _residual(co, valid_io, rows, js, P)
        I = _residual(oc, valid_io, rows, js, P)
        C = _residual(cc, valid_cc, rows, js, P)
        t = np.arange(f - a, b - a)
        lo = np.maximum(t - Z_WINDOW, 0)
        n, s1, s2 = _window_stats(G, lo, t)
        zg[f:b, js] = G[t] / _std_from(n, s1, s2, Z_MIN)
        n, s1, s2 = _window_stats(I, lo, t)
        zi[f:b, js] = I[t] / _std_from(n, s1, s2, Z_MIN)
        lo21 = t - PRIOR_WINDOW
        okp = lo21 >= 0
        n, s1, _ = _window_stats(C, np.maximum(lo21, 0), t)
        prior[f:b, js] = np.where(okp[:, None] & (n == PRIOR_WINDOW), s1, np.nan)
    return zg, zi, prior


def _abnormal_dollar_volume(data, uni):
    dv = np.asarray(data.dollar_volume, dtype=float)
    T, N = dv.shape
    out = np.full((T, N), np.nan)
    tt, jj = np.nonzero(uni)
    lags = np.arange(DV_WINDOW, 0, -1)
    chunk = 100000
    for c0 in range(0, tt.size, chunk):
        t = tt[c0:c0 + chunk]
        j = jj[c0:c0 + chunk]
        r = t[:, None] - lags[None, :]
        W = dv[np.maximum(r, 0), j[:, None]]
        W = np.where((r >= 0) & np.isfinite(W) & (W >= 0), W, np.nan)
        n = np.isfinite(W).sum(axis=1)
        Ws = np.sort(W, axis=1)
        k1 = np.maximum((n - 1) // 2, 0)
        k2 = np.maximum(n // 2, 0)
        med = 0.5 * (Ws[np.arange(t.size), k1] + Ws[np.arange(t.size), k2])
        cur = dv[t, j]
        ok = (n >= Z_MIN) & (med > 0) & np.isfinite(cur) & (cur > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t, j] = np.where(ok, np.log(cur / med), np.nan)
    return out


def _realized_vol(cc, valid_cc):
    T, N = cc.shape
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - RV_WINDOW, 0)
    out = np.full((T, N), np.nan, dtype=np.float32)
    for c0 in range(0, N, 256):
        x = np.where(valid_cc[:, c0:c0 + 256], cc[:, c0:c0 + 256], np.nan)
        n, s1, s2 = _window_stats(x, lo, hi)
        out[:, c0:c0 + 256] = _std_from(n, s1, s2, Z_MIN)
    return out


def _since_release(ea, start_row):
    T, N = ea.shape
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(ea == 1, rows, -1), axis=0)
    finite = np.isfinite(ea)
    first = np.where(finite.any(axis=0), np.argmax(finite, axis=0), T)[None, :]
    known_from = np.maximum(first, start_row)
    since = np.where(last >= 0, rows - last, np.where(rows - known_from >= SINCE_CAP, SINCE_CAP, np.nan))
    since = np.minimum(since, SINCE_CAP)
    return np.where(finite, since, np.nan)


def _flags_ok(flag, lookback):
    T = flag.shape[0]
    bad = ~(flag == 0)
    nb = _prefix(bad.astype(float))
    hi = np.arange(1, T + 1)
    lo = hi - (lookback + 1)
    okwin = (lo >= 0)[:, None] & ((nb[hi] - nb[np.maximum(lo, 0)]) == 0)
    started = np.maximum.accumulate((flag == 1).any(axis=1))
    started_lb = _shift(started, lookback, False)
    first = np.flatnonzero(started)
    start_row = int(first[0]) if first.size else T
    return okwin & started_lb[:, None], start_row


def _zscore(x, uni):
    z = cs_zscore(x, uni)
    fin = uni & np.isfinite(x)
    hi = np.where(fin, x, -np.inf).max(axis=1)
    lo = np.where(fin, x, np.inf).min(axis=1)
    flat = (hi == lo)[:, None]
    return np.where(fin & flat & ~np.isfinite(z), 0.0, z).astype(np.float32)


def build_inputs(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=float)
    k8 = np.asarray(data.extras["nonearn_8k_reaction_day"], dtype=float)
    sp = np.asarray(data.extras["sp500_member"], dtype=float)
    co, oc, cc, valid_io, valid_cc = _returns(data)
    zg, zi, prior = _peer_features(data, co, oc, cc, valid_io, valid_cc)
    with np.errstate(invalid="ignore"):
        big = np.abs(zg) >= float(params["gap_z"])
    f_zg = _zscore(zg, uni)
    del zg
    f_zi = _zscore(zi, uni)
    del zi
    f_prior = _zscore(prior, uni)
    del prior
    del co, oc
    f_rv = _zscore(_realized_vol(cc, valid_cc), uni)
    del cc, valid_cc, valid_io
    f_adv = _zscore(_abnormal_dollar_volume(data, uni), uni)
    f_sp = _zscore(sp, uni)
    ea_ok, ea_start = _flags_ok(ea, EA_LOOKBACK)
    k8_ok, _ = _flags_ok(k8, K8_LOOKBACK)
    f_since = _zscore(_since_release(ea, ea_start), uni)
    feats = [f_zg, f_zi, f_adv, f_prior, f_rv, f_sp, f_since]
    complete = np.ones(uni.shape, dtype=bool)
    for z in feats:
        complete &= np.isfinite(z)
    cand = uni & big & ea_ok & k8_ok & complete
    X = stack(*feats)
    X[~cand] = np.nan
    y = cs_rank(forward_return(data, HORIZON), uni)
    return cand, X, y


def model_scores(data, X, y, model, model_params):
    return walk_forward(X, y, horizon=HORIZON, model=model, params=model_params, retrain_every=RETRAIN,
                        universe=np.asarray(data.universe, dtype=bool))


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def sides(score, cand):
    sc = cand & np.isfinite(score)
    out = np.full(score.shape, np.nan)
    rows, cols = np.nonzero(sc)
    if rows.size == 0:
        return out, sc
    vals = np.asarray(score, dtype=float)[rows, cols]
    urows = np.unique(rows)
    lo = np.searchsorted(rows, urows - (MEDIAN_WINDOW - 1), side="left")
    st = np.searchsorted(rows, urows, side="left")
    hi = np.searchsorted(rows, urows, side="right")
    for a, s, h in zip(lo, st, hi):
        med = np.median(vals[a:h])
        out[rows[s:h], cols[s:h]] = np.where(vals[s:h] > med, 1.0, -1.0)
    return out, sc


def events(data, params):
    cand, X, y = build_inputs(data, params)
    side_g, sc_g = sides(model_scores(data, X, y, "gbm", gbm_params(params)), cand)
    side_r, sc_r = sides(model_scores(data, X, y, "ridge", dict(RIDGE_PARAMS)), cand)
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = sc_g & sc_r & (side_g != side_r) & uni & tradable
    side = np.where(mask, side_g, 0.0)
    return mask, side
