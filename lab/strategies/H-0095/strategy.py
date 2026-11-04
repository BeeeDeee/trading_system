"""H-0095: learned selection of drifting earnings reactions.

LIQ-500 long-only cap-weighted core plus a 5-session tilt (0.02 per slot, total tilt capped at 0.30)
into 8-K Item 2.02 release reactions whose walk-forward LightGBM score is in the top
(1 - entry_quantile) of the event scores of the previous 252 sessions.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"entry_quantile": 0.7, "n_estimators": 100}

HORIZON = 5            # label horizon and hold (sessions)
VOL_WIN = 63           # abnormal-return std window t-63..t-1
VOLU_WIN = 63          # dollar-volume median window t-63..t-1
RUNUP_WIN = 21         # run-up window t-21..t-1
PREV_WIN = 140         # previous release reaction lookback
LIQ_WIN = 63           # mean dollar volume window t-62..t (liq_rank, core proxy)
MIN_FIT_ROWS = 504     # rows of history after the first non-NaN ea_reaction_day before any event is scored
RETRAIN = 63           # refit every 63 rows
PCT_WIN = 252          # percentile pool: event scores of rows t-252..t-1
PCT_MIN = 126          # sessions of earlier scores needed before any entry
HOLD = 5               # entry at open t+1, exit at open t+6
SLOT_W = 0.02
TILT_CAP = 0.30
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}
MEDIAN_CHUNK = 128


# ----------------------------------------------------------------------------- helpers

def member_mean(x, uni):
    """Equal-weight cross-sectional mean of x over universe members with a finite value, per row."""
    m = uni & np.isfinite(x)
    s = np.where(m, x, 0.0).sum(axis=1)
    c = m.sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


def _prefix(x):
    """Cumulative sums (with a leading zero row) of the finite values of x and of their count."""
    T, N = x.shape
    v = np.isfinite(x)
    cs = np.zeros((T + 1, N))
    cs[1:] = np.cumsum(np.where(v, x, 0.0), axis=0)
    cc = np.zeros((T + 1, N), dtype=np.int64)
    cc[1:] = np.cumsum(v, axis=0)
    return cs, cc


def window_sum_count(x, n, include_current):
    """Sum and count of finite x over rows t-n..t-1 (include_current=False) or t-n+1..t (True).
    Windows that start before row 0 are truncated at row 0 (the count shows how many values exist)."""
    T, N = x.shape
    cs, cc = _prefix(x)
    end = np.arange(T) + (1 if include_current else 0)
    st = np.clip(end - n, 0, None)
    s = cs[end] - cs[st]
    c = cc[end] - cc[st]
    return s, c


def lagged_std(x, n):
    """Sample std (ddof=1) of x over rows t-n..t-1; NaN unless all n values are finite."""
    s, c = window_sum_count(x, n, False)
    s2, _ = window_sum_count(x * x, n, False)
    full = c == n
    var = np.full(x.shape, np.nan)
    var[full] = (s2[full] - s[full] * s[full] / n) / (n - 1)
    var = np.where(var < 0, 0.0, var)
    return np.sqrt(var)


def trailing_mean(x, n):
    """Mean of the finite values of x over rows t-n+1..t (at least one finite value; partial at row 0)."""
    s, c = window_sum_count(x, n, True)
    out = np.full(x.shape, np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


def lagged_median(x, mask, n):
    """Median of the finite values of x over rows t-n..t-1, computed only where mask (members)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    xp = np.vstack([np.full((n, N), np.nan), x])          # xp[t + n] = x[t]
    for a in range(0, T, MEDIAN_CHUNK):
        b = min(a + MEDIAN_CHUNK, T)
        cj = np.flatnonzero(mask[a:b].any(axis=0))
        if cj.size == 0:
            continue
        seg = xp[a:b - 1 + n][:, cj]                       # windows for rows a..b-1
        L = b - a
        win = np.stack([seg[k:k + L] for k in range(n)], axis=2)   # (L, len(cj), n): x rows t-n..t-1
        cnt = np.isfinite(win).sum(axis=2)
        srt = np.sort(win, axis=2)                         # NaN sorted last
        lo = np.clip((cnt - 1) // 2, 0, None)[..., None]
        hi = np.clip(cnt // 2, 0, None)[..., None]
        med = 0.5 * (np.take_along_axis(srt, lo, axis=2)[..., 0] + np.take_along_axis(srt, hi, axis=2)[..., 0])
        med[cnt == 0] = np.nan
        out[a:b, cj] = med
    return out


def shift_down(x, k):
    """x shifted k rows down (row t holds x[t-k]); first k rows NaN."""
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:x.shape[0] - k]
    return out


# ----------------------------------------------------------------------------- features

def raw_features(ret_co, ret_oc, listed, dollar_volume, uni, ea):
    """The card's six raw features for every (date, stock) (NaN where not computable), plus the
    63-session mean dollar volume (used by liq_rank and the core proxy)."""
    T, N = ret_co.shape
    lst = np.asarray(listed, dtype=bool)
    rco = np.where(lst, np.asarray(ret_co, dtype=np.float64), np.nan)
    roc = np.where(lst, np.asarray(ret_oc, dtype=np.float64), np.nan)
    cc = (1.0 + rco) * (1.0 + roc) - 1.0
    abn = cc - member_mean(cc, uni)[:, None]
    sd = lagged_std(abn, VOL_WIN)
    sd = np.where(sd > 0, sd, np.nan)

    f1 = (rco - member_mean(rco, uni)[:, None]) / sd
    f2 = (roc - member_mean(roc, uni)[:, None]) / sd

    dv = np.asarray(dollar_volume, dtype=np.float64)
    dv = np.where(lst & np.isfinite(dv) & (dv > 0), dv, np.nan)
    med = lagged_median(dv, uni, VOLU_WIN)
    med = np.where(med > 0, med, np.nan)
    f3 = np.log(dv / med)

    prod = np.ones((T, N))
    for k in range(1, RUNUP_WIN + 1):
        prod = prod * (1.0 + shift_down(abn, k))         # NaN if any day is missing
    f4 = (prod - 1.0) / (sd * np.sqrt(RUNUP_WIN))

    eaf = np.asarray(ea, dtype=np.float64)
    ev_all = np.isfinite(eaf) & (eaf == 1.0)
    rz = abn / sd                                          # release-session abnormal z on its own date
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(ev_all, rows, -1), axis=0) if T > 0 else np.zeros((0, N), int)
    prev = np.vstack([np.full((1, N), -1), last[:-1]]) if T > 0 else last   # strictly earlier event
    okp = (prev >= 0) & (rows - prev <= PREV_WIN)
    val = rz[np.clip(prev, 0, None), np.arange(N)[None, :]] if T > 0 else rz
    f5 = np.where(okp & np.isfinite(val), val, 0.0)

    mdv = trailing_mean(dv, LIQ_WIN)
    return f1, f2, f3, f4, f5, mdv


def event_mask(ea, uni):
    eaf = np.asarray(ea, dtype=np.float64)
    return np.isfinite(eaf) & (eaf == 1.0) & uni


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def first_fit_row(ea_full):
    """First row allowed to be scored: 504 rows after the first non-NaN ea_reaction_day of any stock."""
    any_ea = np.isfinite(np.asarray(ea_full, dtype=np.float64)).any(axis=1)
    idx = np.flatnonzero(any_ea)
    if idx.size == 0:
        return any_ea.shape[0]
    return int(idx[0]) + MIN_FIT_ROWS


def event_scores(data, params):
    """Walk-forward model scores on event rows (T, N), NaN elsewhere, and the 63-session mean
    dollar volume (T, N) of the stocks that are ever members (NaN for the others)."""
    if data.universe is None:
        raise ValueError("H-0095 needs the liq_n universe")
    uni_full = np.asarray(data.universe, dtype=bool)
    ea_full = data.extras["ea_reaction_day"]
    T, N = uni_full.shape
    cols = np.flatnonzero(uni_full.any(axis=0))
    score_full = np.full((T, N), np.nan)
    mdv_full = np.full((T, N), np.nan)
    if cols.size == 0:
        return score_full, mdv_full
    uni = uni_full[:, cols]
    ea = np.asarray(ea_full, dtype=np.float64)[:, cols]
    f1, f2, f3, f4, f5, mdv = raw_features(np.asarray(data.ret_co)[:, cols], np.asarray(data.ret_oc)[:, cols],
                                           np.asarray(data.listed)[:, cols],
                                           np.asarray(data.dollar_volume)[:, cols], uni, ea)
    f6 = cs_rank(mdv, uni)
    X = stack(cs_zscore(f1, uni), cs_zscore(f2, uni), cs_zscore(f3, uni), cs_zscore(f4, uni),
              cs_zscore(f5, uni), cs_zscore(f6, uni))
    ev = event_mask(ea, uni)
    X[~ev] = np.nan                                        # the model never sees non-event rows
    y = np.array(cs_rank(np.asarray(forward_return(data, HORIZON))[:, cols], uni), dtype=np.float64)
    y[~ev] = np.nan                                        # label ranked over all members, kept on events
    score = np.array(walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params(params),
                                  retrain_every=RETRAIN, universe=uni), dtype=np.float64)
    score[~ev] = np.nan
    score[:min(first_fit_row(ea_full), T)] = np.nan
    score_full[:, cols] = score
    mdv_full[:, cols] = mdv
    return score_full, mdv_full


# ----------------------------------------------------------------------------- entries and book

def entry_flags(score, q):
    """Event (t, j) enters when the share of event scores of rows t-252..t-1 below its score >= q,
    and at least 126 sessions of earlier scores exist."""
    T, N = score.shape
    entry = np.zeros((T, N), dtype=bool)
    r, c = np.nonzero(np.isfinite(score))
    if r.size == 0:
        return entry
    s = score[r, c]
    first = int(r[0])
    starts = np.searchsorted(r, np.arange(T + 1), side="left")
    for t in np.unique(r):
        if t - first < PCT_MIN:
            continue
        lo = starts[max(t - PCT_WIN, 0)]
        hi = starts[t]
        if hi <= lo:
            continue
        pool = s[lo:hi]
        a, b = starts[t], starts[t + 1]
        cur = s[a:b]
        pct = (pool[None, :] < cur[:, None]).sum(axis=1) / pool.size
        entry[t, c[a:b]] = pct >= q
    return entry


def core_weights(mcap, mdv, mem):
    """Cap weights over members: filed marketcap, else 63-session mean dollar volume x median
    marketcap/dollar-volume ratio of members with both; no marketcap at all -> dollar-volume weights."""
    mc = np.where(mem & np.isfinite(mcap) & (mcap > 0), mcap, np.nan)
    dvm = np.where(mem & np.isfinite(mdv) & (mdv > 0), mdv, np.nan)
    has_mc = np.isfinite(mc)
    both = has_mc & np.isfinite(dvm)
    if has_mc.any():
        if both.any():
            ratio = float(np.median(mc[both] / dvm[both]))
            fill = np.where(has_mc, mc, dvm * ratio)
        else:
            fill = mc
    else:
        fill = dvm
    w = np.where(np.isfinite(fill), fill, 0.0)
    s = w.sum()
    return w / s if s > 0 else w


def build_book(entry, uni, mcap, mdv, rebalance):
    """Daily target weights: monthly-fixed core x (1 - total tilt) + open slots."""
    T, N = uni.shape
    W = np.zeros((T, N))
    core = np.zeros(N)
    slot_w = np.zeros(N)
    slot_end = np.full(N, -1)
    for t in range(T):
        mem = uni[t]
        if rebalance[t]:
            core = core_weights(mcap[t], mdv[t], mem)
        slot_w[slot_end <= t] = 0.0                        # hold over: exit at the open of entry + 6
        slot_w[~mem] = 0.0                                 # left the universe: closed now
        new = entry[t] & mem & (slot_w == 0.0)             # an event inside an open hold adds no slot
        k = int(new.sum())
        if k > 0:
            used = float(slot_w.sum())
            if used + SLOT_W * k <= TILT_CAP + 1e-12:
                per = SLOT_W
            else:
                per = max(TILT_CAP - used, 0.0) / k
            if per > 0:
                slot_w[new] = per
                slot_end[new] = t + HOLD
        tilt = float(slot_w.sum())
        cw = np.where(mem, core, 0.0)
        s = cw.sum()
        if s > 0:
            W[t] = cw / s * (1.0 - tilt)
        W[t] += slot_w
    return W


def month_rebalance_days(dates):
    T = len(dates)
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        reb = ps.copy()
    else:
        reb = np.zeros(T, dtype=bool)
        reb[ps.astype(int)] = True
    if T > 0:
        reb[0] = True                                      # initialise the core on the first row
    return reb


def target_weights(data, params):
    q = float(params["entry_quantile"])
    if data.universe is None:
        raise ValueError("H-0095 needs the liq_n universe")
    uni = np.asarray(data.universe, dtype=bool)
    mcap = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    score, mdv = event_scores(data, params)
    entry = entry_flags(score, q)
    return build_book(entry, uni, mcap, mdv, month_rebalance_days(data.dates))
