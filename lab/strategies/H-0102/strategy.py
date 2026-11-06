"""H-0102 Sophistication-conditioned earnings-reaction drift, learned.

LIQ-500 long-only, 5-session slots (weight 1/20) in 8-K release reactions whose walk-forward
LightGBM score (six features: reaction_z, f13_io, liquidity, release_volume, runup, idio_vol)
is in the top `top` fraction of the scored release rows of the trailing 126 sessions.
Row t is decided after the close of t with rows 0..t only; the engine fills at the open of t+1.
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

# fixed by the card (not grid parameters)
HORIZON = 5                 # ml.horizon_days: label open t+1 -> open t+6
RETRAIN_EVERY = 63          # ml.retrain_every_days
WINDOW = 63                 # 63 sessions ending t-1 (std, mean dollar volume, median dollar volume, idio vol)
RUNUP_WINDOW = 21           # 21 sessions ending at the close of t-1
QUANTILE_WINDOW = 126       # trailing sessions (including t) of scored release rows for the entry quantile
MIN_SCORED = 100            # fewer scored rows in that window -> no entries
SLOTS = 20                  # weight 1 / SLOTS = 0.05 per entry
HOLD = 5                    # buy at the open of t+1, sell at the open of t+6
GBM_FIXED = {
    "num_leaves": 8,
    "min_child_samples": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample": 1.0,
}


# ----------------------------------------------------------------------------------------------
# point-in-time helpers
# ----------------------------------------------------------------------------------------------
def _lagged_mean_std(x, n):
    """Mean and sample std (ddof=1) of x over the n rows t-n .. t-1 (the window ends at t-1).

    NaN unless all n values of the window are finite. Causal: row t uses rows < t only.
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    mean = np.full((T, N), np.nan)
    std = np.full((T, N), np.nan)
    if T <= n:
        return mean, std
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    z = np.zeros((1, N))
    cs = np.concatenate([z, np.cumsum(v, axis=0)])
    cs2 = np.concatenate([z, np.cumsum(v * v, axis=0)])
    cc = np.concatenate([z, np.cumsum(valid, axis=0, dtype=np.float64)])
    s = cs[n:T] - cs[0:T - n]           # sum of x[t-n:t] for t = n .. T-1
    s2 = cs2[n:T] - cs2[0:T - n]
    c = cc[n:T] - cc[0:T - n]
    full = c == n
    m = s / n
    var = np.maximum((s2 - s * m) / (n - 1), 0.0)
    mean[n:] = np.where(full, m, np.nan)
    std[n:] = np.where(full, np.sqrt(var), np.nan)
    return mean, std


def _cs_median(x, mask):
    """Per-row median of x over the columns where mask is True and x is finite (NaN if none)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    xs = np.sort(np.where(ok, x, np.nan), axis=1)     # NaNs sort last
    cnt = ok.sum(axis=1)
    rows = np.arange(x.shape[0])
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.maximum(cnt // 2, 0)
    hi = np.minimum(hi, x.shape[1] - 1)
    med = 0.5 * (xs[rows, lo] + xs[rows, hi])
    return np.where(cnt > 0, med, np.nan)


def _cs_mean(x, mask):
    """Per-row equal-weight mean of x over the columns where mask is True and x is finite."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def _lagged_median_on(x, mask, n):
    """Median of x over rows t-n .. t-1, computed only where mask[t] (NaN elsewhere or if any NaN)."""
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for t in range(n, T):
        cols = np.nonzero(mask[t])[0]
        if cols.size == 0:
            continue
        out[t, cols] = np.median(x[t - n:t, cols], axis=0)   # NaN in the window -> NaN
    return out


def _lag(x, k):
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


# ----------------------------------------------------------------------------------------------
# card rules
# ----------------------------------------------------------------------------------------------
def release_rows(data):
    """(t, stock) with ea_reaction_day == 1 on t while the stock is a PIT LIQ-500 member on t."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0102 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    return (ea == 1.0) & uni, uni


def raw_features(data, uni):
    """The card's six features for every column, from data up to the close of t."""
    listed = np.asarray(data.listed, dtype=bool)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = np.where(listed, r, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        # (1) reaction_z: release-session close-to-close return minus the members' median, over the
        #     stock's 63-session daily return std ending t-1
        _, sd63 = _lagged_mean_std(r, WINDOW)
        react = (r - _cs_median(r, uni)[:, None]) / sd63
        react = np.where(np.isfinite(react) & (sd63 > 0), react, np.nan)
        # (2) f13_io as delivered by the loader (NaN = unknown / stale, row not scored)
        io = np.asarray(data.extras["f13_io"], dtype=np.float64)
        # (3) liquidity: log mean dollar volume over the 63 sessions ending t-1
        dv = np.asarray(data.dollar_volume, dtype=np.float64)
        mdv, _ = _lagged_mean_std(dv, WINDOW)
        liq = np.where(mdv > 0, np.log(np.where(mdv > 0, mdv, 1.0)), np.nan)
        # (4) release_volume: log(dollar volume on t / median dollar volume over the 63 sessions ending t-1)
        meddv = _lagged_median_on(dv, uni, WINDOW)
        ratio = dv / meddv
        okr = np.isfinite(ratio) & (ratio > 0)
        relvol = np.where(okr, np.log(np.where(okr, ratio, 1.0)), np.nan)
        # (5) runup: total return over the 21 sessions ending at the close of t-1, minus the members' median
        idx = np.asarray(total_return_index(data), dtype=np.float64)
        run = _lag(idx, 1) / _lag(idx, RUNUP_WINDOW + 1) - 1.0
        run = np.where(np.isfinite(run), run, np.nan)
        runup = run - _cs_median(run, uni)[:, None]
        # (6) idio_vol: 63-session std (ending t-1) of the return minus the members' equal-weight mean that day
        resid = r - _cs_mean(r, uni)[:, None]
        _, ivol = _lagged_mean_std(resid, WINDOW)
    return [react, io, liq, relvol, runup, ivol]


def model_inputs(data):
    """Feature tensor (z-scored over all members on t, then kept on release rows only) and label."""
    rel, uni = release_rows(data)
    feats = raw_features(data, uni)
    z = [np.where(rel, cs_zscore(f, uni), np.nan) for f in feats]
    X = stack(*z)
    y = cs_rank(forward_return(data, HORIZON), uni)     # label: ranked across the universe on each date
    return X, y, rel, uni


def model_scores(data, params):
    X, y, rel, uni = model_inputs(data)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score = np.where(rel & np.isfinite(score), score, np.nan)
    return score, rel, uni


def entry_mask(score, top, window=QUANTILE_WINDOW, min_scored=MIN_SCORED):
    """Scored release row at or above the (1 - top) quantile of the scored rows of the trailing window
    (including t); no entries while that window holds fewer than min_scored scored rows."""
    score = np.asarray(score, dtype=np.float64)
    T = score.shape[0]
    fin = np.isfinite(score)
    has = fin.any(axis=1)
    out = np.zeros(score.shape, dtype=bool)
    q = 1.0 - float(top)
    for t in range(T):
        if not has[t]:
            continue
        lo = max(0, t - window + 1)
        pool = score[lo:t + 1][fin[lo:t + 1]]
        if pool.size < min_scored:
            continue
        thr = np.quantile(pool, q)
        out[t] = fin[t] & (score[t] >= thr)
    return out


def slot_book(entry, score, uni, listed, slots=SLOTS, hold=HOLD):
    """Fixed-weight slots: an entry on t holds 1/slots on decision rows t .. t+hold-1 (open t+1 to
    open t+hold+1). New entries are admitted in descending score order while fewer than `slots` are
    open; a held stock keeps its single slot; a stock off the universe (or unlisted) is sold."""
    T, N = entry.shape
    w = np.zeros((T, N))
    held = np.zeros(N, dtype=bool)
    last = np.full(N, -1)
    unit = 1.0 / slots
    for t in range(T):
        held &= (last >= t) & uni[t] & listed[t]
        n_open = int(held.sum())
        cand = np.nonzero(entry[t] & ~held)[0]
        if cand.size and n_open < slots:
            order = cand[np.argsort(-score[t, cand], kind="stable")]
            take = order[: slots - n_open]
            held[take] = True
            last[take] = t + hold - 1
        w[t, held] = unit
    return w


def target_weights(data, params):
    score, rel, uni = model_scores(data, params)
    entry = entry_mask(score, params["top"])
    listed = np.asarray(data.listed, dtype=bool)
    return slot_book(entry, score, uni, listed)
