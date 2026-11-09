"""H-0101 mechanism event: a book entry (release row whose walk-forward gbm score is at or above the
(1 - top) quantile of all scored release rows in sessions t-62..t), side +1.

Same pipeline as strategy.py (copied: the diagnostic is a separate file)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
LOOKBACK = 63
LOAD_WINDOW = 252
ENTRY_WINDOW = 63
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200}
COL_CHUNK = 256


def _prev_std(x, n):
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T <= n:
        return out
    m = np.isfinite(x)
    xs = np.where(m, x, 0.0)
    z = np.zeros((1,) + x.shape[1:])
    s1 = np.concatenate([z, np.cumsum(xs, axis=0)], axis=0)
    s2 = np.concatenate([z, np.cumsum(xs * xs, axis=0)], axis=0)
    c = np.concatenate([z, np.cumsum(m, axis=0)], axis=0)
    a1 = s1[n:T] - s1[0:T - n]
    a2 = s2[n:T] - s2[0:T - n]
    cnt = c[n:T] - c[0:T - n]
    var = (a2 - a1 * a1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd[cnt < n] = np.nan
    out[n:T] = sd
    return out


def _end_of_week(dates):
    d = dates.astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    gap = (nxt - d).astype(np.int64)
    return (gap > 1).astype(np.float64)


def _release_load_rank(flag, uni):
    T = flag.shape[0]
    is_rel = (flag == 1.0) & uni
    n_t = is_rel.sum(axis=1).astype(np.float64)
    hist = (np.isfinite(flag) & uni).any(axis=1)
    out = np.full(T, np.nan)
    W = LOAD_WINDOW
    if T < W:
        return out
    win = np.lib.stride_tricks.sliding_window_view(n_t, W)
    cur = n_t[W - 1:]
    less = (win < cur[:, None]).sum(axis=1)
    eq = (win == cur[:, None]).sum(axis=1)
    pct = (less + (eq - 1) / 2.0) / (W - 1)
    hcount = np.concatenate([[0], np.cumsum(hist)])
    full = (hcount[W:] - hcount[:T - W + 1]) == W
    out[W - 1:] = np.where(full, pct, np.nan)
    return out


def _features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0101 needs the PIT LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    flag = np.asarray(data.extras["ea_reaction_day"])
    has_px = np.isfinite(np.asarray(data.close))
    dv = np.asarray(data.dollar_volume, dtype=np.float32)
    co = np.asarray(data.ret_co, dtype=np.float32)
    oc = np.asarray(data.ret_oc, dtype=np.float32)
    T, N = uni.shape

    r_all = np.where(has_px, (1.0 + co) * (1.0 + oc) - 1.0, np.nan).astype(np.float32)
    mem_ok = uni & np.isfinite(r_all)
    cnt = mem_ok.sum(axis=1)
    tot = np.where(mem_ok, r_all, 0.0).sum(axis=1)
    mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    f1 = np.full((T, N), np.nan, dtype=np.float32)
    f4 = np.full((T, N), np.nan, dtype=np.float32)
    for a in range(0, N, COL_CHUNK):
        b = min(N, a + COL_CHUNK)
        r = r_all[:, a:b].astype(np.float64)
        ex = r - mkt[:, None]
        sd_ex = _prev_std(ex, LOOKBACK)
        sd_r = _prev_std(r, LOOKBACK)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = ex / sd_ex
        z[~np.isfinite(z)] = np.nan
        f1[:, a:b] = z
        f4[:, a:b] = sd_r
    del r_all

    med = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(LOOKBACK, T):
        cols = np.nonzero(uni[t])[0]
        if cols.size == 0:
            continue
        med[t, cols] = np.median(dv[t - LOOKBACK:t, cols], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.log(dv / med)
        f3 = np.log(med)
    f2[~np.isfinite(f2)] = np.nan
    f3[~np.isfinite(f3)] = np.nan
    del med

    z1 = cs_zscore(f1, uni)
    z2 = cs_zscore(f2, uni)
    z3 = cs_zscore(f3, uni)
    z4 = cs_zscore(f4, uni)
    del f1, f2, f3, f4

    load = _release_load_rank(flag, uni)
    eow = _end_of_week(data.dates)
    f5 = np.broadcast_to(load[:, None], (T, N))
    f6 = np.broadcast_to(eow[:, None], (T, N))

    release = uni & (flag == 1.0) & has_px & np.isfinite(dv) & (dv > 0)
    X = stack(z1, z2, z3, z4, f5, f6)
    ok = release & np.isfinite(X).all(axis=2)
    X[~ok] = np.nan
    return X, ok, uni


def _scores(data, params):
    X, ok, uni = _features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.where(ok, np.asarray(score, dtype=np.float64), np.nan)
    return score, uni


def _entries(score, top):
    T, N = score.shape
    ent = np.zeros((T, N), dtype=bool)
    rows, cols = np.nonzero(np.isfinite(score))
    if rows.size == 0:
        return ent
    vals = score[rows, cols]
    q_level = 1.0 - float(top)
    for t in np.unique(rows):
        lo = np.searchsorted(rows, t - (ENTRY_WINDOW - 1), side="left")
        hi = np.searchsorted(rows, t, side="right")
        q = np.quantile(vals[lo:hi], q_level)
        s = np.searchsorted(rows, t, side="left")
        sel = vals[s:hi] >= q
        ent[t, cols[s:hi][sel]] = True
    return ent


def events(data, params):
    score, uni = _scores(data, params)
    ent = _entries(score, params["top"])
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = ent & uni & tradable
    return mask, None
