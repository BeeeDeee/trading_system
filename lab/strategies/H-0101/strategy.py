"""H-0101 Distraction-conditioned earnings-reaction drift, learned.

LIQ-500 long-only. Release rows (universe member with ea_reaction_day == 1 on session t) are scored by a
walk-forward LightGBM (framework) on six features: four stock-level features z-scored across the universe
members of date t, plus two date-level attention features (release-load percentile rank, end-of-week flag).
A release row is an entry when its score is at or above the (1 - top) quantile of all scored release rows of
sessions t-62..t. Each entry is held for 5 sessions; every active name gets min(0.10, 1 / n_active).
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.3}

# fixed values of the card (not in signal.params, so not varied by G2)
HORIZON = 5            # ml.horizon_days, label horizon and purge
RETRAIN_EVERY = 63     # ml.retrain_every_days
LOOKBACK = 63          # sessions t-63..t-1 for volatility and median dollar volume
LOAD_WINDOW = 252      # sessions t-251..t for the release-load percentile rank
ENTRY_WINDOW = 63      # sessions t-62..t for the entry quantile
HOLD = 5               # sessions an entry is held (bought at open t+1, sold at open t+6)
MAX_WEIGHT = 0.10      # per-name cap
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200}
COL_CHUNK = 256        # column block size (memory)


def _prev_std(x, n):
    """Std (ddof=1) of x over rows t-n..t-1 (strictly before t); NaN unless all n values are finite."""
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T <= n:
        return out
    m = np.isfinite(x)
    xs = np.where(m, x, 0.0)
    z = np.zeros((1,) + x.shape[1:])
    s1 = np.concatenate([z, np.cumsum(xs, axis=0)], axis=0)        # s1[k] = sum of rows < k
    s2 = np.concatenate([z, np.cumsum(xs * xs, axis=0)], axis=0)
    c = np.concatenate([z, np.cumsum(m, axis=0)], axis=0)
    a1 = s1[n:T] - s1[0:T - n]                                     # rows t-n..t-1 for t = n..T-1
    a2 = s2[n:T] - s2[0:T - n]
    cnt = c[n:T] - c[0:T - n]
    var = (a2 - a1 * a1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd[cnt < n] = np.nan
    out[n:T] = sd
    return out


def _end_of_week(dates):
    """1.0 if the session is the last weekday (Mon-Fri) of its calendar week, i.e. a Friday; exchange
    holidays are ignored (a Thursday before a Friday holiday is not flagged)."""
    d = dates.astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    gap = (nxt - d).astype(np.int64)
    return (gap > 1).astype(np.float64)


def _release_load_rank(flag, uni):
    """Date-level percentile rank (ties averaged, in [0, 1]) of n_t = number of universe members with
    flag == 1 on t among the counts of sessions t-251..t; NaN until 252 sessions of flag history exist."""
    T = flag.shape[0]
    is_rel = (flag == 1.0) & uni
    n_t = is_rel.sum(axis=1).astype(np.float64)
    hist = (np.isfinite(flag) & uni).any(axis=1)                    # the flag data covers this session
    out = np.full(T, np.nan)
    W = LOAD_WINDOW
    if T < W:
        return out
    win = np.lib.stride_tricks.sliding_window_view(n_t, W)           # row k = sessions k..k+W-1
    cur = n_t[W - 1:]
    less = (win < cur[:, None]).sum(axis=1)
    eq = (win == cur[:, None]).sum(axis=1)
    pct = (less + (eq - 1) / 2.0) / (W - 1)
    hcount = np.concatenate([[0], np.cumsum(hist)])
    full = (hcount[W:] - hcount[:T - W + 1]) == W                   # all of t-251..t have flag history
    out[W - 1:] = np.where(full, pct, np.nan)
    return out


def _features(data):
    """Returns X (T, N, 6), release mask (T, N), universe (T, N)."""
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

    # close-to-close total return; NaN where there is no price (ret_* are 0 there, which is not a return)
    r_all = np.where(has_px, (1.0 + co) * (1.0 + oc) - 1.0, np.nan).astype(np.float32)
    # equal-weight mean of all universe members with a return on that session
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

    # median dollar volume over t-63..t-1, computed for universe members of t
    # (NaN anywhere in the window -> NaN; zero volumes give -inf logs, set to NaN = not scored)
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

    # release rows: member, flag == 1, price and volume on t
    release = uni & (flag == 1.0) & has_px & np.isfinite(dv) & (dv > 0)
    X = stack(z1, z2, z3, z4, f5, f6)
    ok = release & np.isfinite(X).all(axis=2)
    X[~ok] = np.nan                                                 # train and score only complete release rows
    return X, ok, uni


def _scores(data, params):
    X, ok, uni = _features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)                 # label ranked across the universe per date
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.where(ok, np.asarray(score, dtype=np.float64), np.nan)
    return score, uni


def _entries(score, top):
    """Entry if score >= (1 - top) quantile of all scored release rows of sessions t-62..t."""
    T, N = score.shape
    ent = np.zeros((T, N), dtype=bool)
    rows, cols = np.nonzero(np.isfinite(score))                     # row-major: rows sorted
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


def _book(ent, uni):
    """Hold each entry for HOLD decision rows (t..t+HOLD-1), only while a universe member; equal weight
    min(MAX_WEIGHT, 1 / n_active) per active name, the rest cash."""
    T, N = ent.shape
    held = np.zeros((T, N), dtype=bool)
    for k in range(HOLD):
        held[k:] |= ent[:T - k]
    held &= uni
    n_act = held.sum(axis=1)
    w_row = np.where(n_act > 0, np.minimum(MAX_WEIGHT, 1.0 / np.maximum(n_act, 1)), 0.0)
    return np.where(held, w_row[:, None], 0.0)


def target_weights(data, params):
    score, uni = _scores(data, params)
    ent = _entries(score, params["top"])
    return _book(ent, uni)
