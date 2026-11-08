"""H-0110 Capital-gains holding-period clock, learned (LIQ-500, market-neutral, weekly).

Row t is decided after the close of t from rows 0..t only; trades happen at the open of t+1.

(1) Spike days: m(j,s) = dollar_volume(j,s) / median(dollar_volume(j, s-63..s-1)), needing >= 50 valid
    values (and a positive median). Spike if m >= spike_mult, ea_reaction_day == 0 on every session
    s-3..s+3 (NaN or 1 fails), and the median of m over the universe members of day s is < 1.5.
    A spike day uses rows up to s+3; it is only ever read at rows t with date(s) <= date(t) - 352 days.
(2) Tax clock: q(s) = date(s) + 366 calendar days. Anchor of j at t: the spike day with
    |date(t) - q(s)| <= 14 and the largest m (ties: the most recent). c = date(t) - q(s).
    Window stock = has an anchor and is a universe member at t.
(3) Features of window stocks (raw): c, g = log TRI(t)/TRI(s), log m(j,s), total return of day s,
    63-session std of daily total returns (1+ret_co)(1+ret_oc)-1 up to t.
(4) walk_forward gbm, label forward_return(data, 5), refit every 63 rows, expanding window.
(5) First session of each week: long the k highest / short the k lowest scored window stocks,
    k = max(2, floor(leg * n_scored)), +-0.5/k each; fewer than 10 scored -> flat. Positions whose
    stock leaves the universe are closed on the first row where it is not a member.
"""
import numpy as np
from lab.framework.api import period_starts, total_return_index, rolling_std
from lab.framework.ml import forward_return, stack, walk_forward

PARAMS = {"leg": 0.2, "n_estimators": 100, "spike_mult": 4}

MED_WINDOW = 63          # sessions s-63..s-1 for the dollar-volume median
MED_MIN_VALID = 50       # at least 50 valid values
EA_HALF = 3              # no earnings release on s-3..s+3
MKT_MED_MAX = 1.5        # median m over universe members must be below this
QUAL_DAYS = 366          # long-term qualification: spike day + 366 calendar days
CLOCK_HALF = 14          # |date(t) - q(s)| <= 14 calendar days
VOL_WINDOW = 63          # realized volatility window
HORIZON = 5              # label and hold
RETRAIN_EVERY = 63       # card ml.retrain_every_days
MIN_SCORED = 10          # fewer scored window stocks -> hold nothing
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}
CHUNK = 100000           # pairs per chunk in the rolling median


def _shift_down(x, k):
    """out[t] = x[t-k] (NaN for t < k); k >= 0."""
    out = np.full(x.shape, np.nan, dtype=x.dtype)
    if k == 0:
        out[:] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rolling_min_prev(x, n):
    """NaN-ignoring min of x over rows t-n..t-1 (NaN if none valid)."""
    a = _shift_down(x, 1)
    span = 1
    while span * 2 <= n:
        a = np.fmin(a, _shift_down(a, span))
        span *= 2
    if span < n:
        a = np.fmin(a, _shift_down(a, n - span))
    return a


def volume_multiple(dv, need):
    """m(j,s) = dv(s) / median(dv over s-63..s-1), >= 50 valid values, positive median.

    Computed only where `need` is True (elsewhere NaN). float32 (T, N)."""
    T, N = dv.shape
    m = np.full((T, N), np.nan, dtype=np.float32)
    if T <= MED_WINDOW:
        return m
    valid = np.isfinite(dv)
    cnt = np.zeros((T + 1, N), dtype=np.int32)
    cnt[1:] = np.cumsum(valid, axis=0, dtype=np.int32)
    nvalid = np.zeros((T, N), dtype=np.int32)
    nvalid[MED_WINDOW:] = cnt[MED_WINDOW:T] - cnt[0:T - MED_WINDOW]
    ok = need & valid & (nvalid >= MED_MIN_VALID)
    ok[:MED_WINDOW] = False
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        return m
    filled = np.where(valid, dv, np.inf).astype(np.float64)
    offs = np.arange(-MED_WINDOW, 0)
    kth = list(range((MED_MIN_VALID - 1) // 2, MED_WINDOW // 2 + 1))
    for a in range(0, rows.size, CHUNK):
        r = rows[a:a + CHUNK]
        c = cols[a:a + CHUNK]
        win = filled[r[:, None] + offs[None, :], c[:, None]]
        win.partition(kth, axis=1)
        k = nvalid[r, c].astype(np.int64)
        lo = np.take_along_axis(win, ((k - 1) // 2)[:, None], axis=1)[:, 0]
        hi = np.take_along_axis(win, (k // 2)[:, None], axis=1)[:, 0]
        med = 0.5 * (lo + hi)
        pos = med > 0
        val = np.full(r.size, np.nan)
        val[pos] = filled[r[pos], c[pos]] / med[pos]
        m[r, c] = val
    return m


def row_nanmedian(x):
    """Median of the finite values of each row (NaN if none)."""
    T = x.shape[0]
    out = np.full(T, np.nan)
    if x.size == 0:
        return out
    s = np.sort(x, axis=1)
    k = np.isfinite(x).sum(axis=1)
    ok = k > 0
    if ok.any():
        sub = s[ok].astype(np.float64)
        kk = k[ok]
        ar = np.arange(sub.shape[0])
        out[ok] = 0.5 * (sub[ar, (kk - 1) // 2] + sub[ar, kk // 2])
    return out


def ea_clear(ea):
    """True at row s if ea_reaction_day == 0 on every row s-3..s+3 (rows outside the view fail)."""
    T, N = ea.shape
    z = (ea == 0)
    w = 2 * EA_HALF + 1
    out = np.zeros((T, N), dtype=bool)
    if T < w:
        return out
    cs = np.zeros((T + 1, N), dtype=np.int32)
    cs[1:] = np.cumsum(z, axis=0, dtype=np.int32)
    # window rows s-3..s+3 for s in [3, T-4]: cs[s+4] - cs[s-3]
    out[EA_HALF:T - EA_HALF] = (cs[w:T + 1] - cs[0:T + 1 - w]) == w
    return out


def spike_days(dv, ea, uni, spike_mult):
    """Returns (spike (T,N) bool, m (T,N) float32)."""
    eaok = ea_clear(ea)
    rmin = _rolling_min_prev(dv.astype(np.float64), MED_WINDOW)
    # exact prefilter: median >= window min, so dv < mult * min can never be a spike
    with np.errstate(invalid="ignore"):
        impossible = dv < spike_mult * rmin
    need = uni | (eaok & ~impossible)
    m = volume_multiple(dv, need)
    mkt = row_nanmedian(np.where(uni, m, np.nan))
    with np.errstate(invalid="ignore"):
        spike = np.isfinite(m) & (m >= spike_mult) & eaok & (mkt < MKT_MED_MAX)[:, None]
    return spike, m


def anchors(day_num, spike, m):
    """Anchor spike row per (t, j): spike s with |day(t) - (day(s) + 366)| <= 14, largest m,
    ties the most recent. Returns brow (T,N) int64, -1 where none."""
    T, N = spike.shape
    brow = np.full((T, N), -1, dtype=np.int64)
    if T == 0:
        return brow
    lo = np.searchsorted(day_num, day_num - (QUAL_DAYS + CLOCK_HALF), side="left")
    hi = np.searchsorted(day_num, day_num - (QUAL_DAYS - CLOCK_HALF), side="right") - 1
    width = hi - lo + 1
    W = int(width.max()) if width.size else 0
    if W <= 0:
        return brow
    key = np.where(spike, m, -np.inf).astype(np.float32)
    best = np.full((T, N), -np.inf, dtype=np.float32)
    for k in range(W):                      # most recent candidate first: strict > keeps it on ties
        r = hi - k
        okr = (r >= lo) & (r >= 0)
        rr = np.where(okr, r, 0)
        cand = key[rr]
        cand[~okr] = -np.inf
        upd = cand > best
        best = np.where(upd, cand, best)
        brow = np.where(upd, rr[:, None], brow)
    return brow


def daily_total_return(data):
    return (1.0 + data.ret_co) * (1.0 + data.ret_oc) - 1.0


def clock_features(data, params):
    """Window mask and raw features c, g, log m, spike-day return, vol (all NaN off-window)."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0110 needs the point-in-time liq_n universe")
    uni = np.asarray(uni, dtype=bool)
    ea = data.extras["ea_reaction_day"]
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = dv.shape
    spike, m = spike_days(dv, ea, uni, float(params["spike_mult"]))
    day_num = data.dates.astype(np.int64)
    brow = anchors(day_num, spike, m)
    window = (brow >= 0) & uni
    br = np.where(window, brow, 0)
    cols = np.broadcast_to(np.arange(N)[None, :], (T, N))

    nan32 = np.float32(np.nan)
    c = np.where(window, (day_num[:, None] - day_num[br] - QUAL_DAYS), np.nan).astype(np.float32)
    tri = np.asarray(total_return_index(data), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(window, np.log(tri / tri[br, cols]), np.nan).astype(np.float32)
        lm = np.where(window, np.log(m[br, cols].astype(np.float64)), np.nan).astype(np.float32)
    del tri
    rday = daily_total_return(data)
    rs = np.where(window, rday[br, cols], nan32).astype(np.float32)
    vol = np.asarray(rolling_std(rday, VOL_WINDOW), dtype=np.float32)
    vol = np.where(window, vol, nan32).astype(np.float32)
    return window, c, g, lm, rs, vol


def weekly_book(score, eligible, decision, leg):
    """Decision rows: +0.5/k on the k highest, -0.5/k on the k lowest eligible scores,
    k = max(2, floor(leg * n)); n < 10 -> all zero. Other rows NaN."""
    T, N = score.shape
    w = np.full((T, N), np.nan)
    for t in np.nonzero(decision)[0]:
        row = np.zeros(N)
        idx = np.nonzero(eligible[t] & np.isfinite(score[t]))[0]
        n = idx.size
        if n >= MIN_SCORED:
            k = max(2, int(np.floor(leg * n + 1e-9)))
            order = idx[np.argsort(-score[t, idx], kind="stable")]
            row[order[:k]] = 0.5 / k
            row[order[-k:]] = -0.5 / k
        w[t] = row
    return w


def close_leavers(w, decision, uni):
    """Zero a held position on every non-decision row where its stock is not a universe member."""
    T, N = w.shape
    if T == 0:
        return w
    last = np.maximum.accumulate(np.where(decision, np.arange(T), -1))
    has = last >= 0
    held = np.zeros((T, N), dtype=bool)
    held[has] = w[last[has]] != 0
    leave = held & ~uni & ~decision[:, None]
    w = w.copy()
    w[leave] = 0.0
    return w


def decision_mask(dates):
    T = dates.shape[0]
    ps = np.asarray(period_starts(dates, "W"))
    if ps.dtype == bool:
        return ps.copy()
    dec = np.zeros(T, dtype=bool)
    dec[ps.astype(np.int64)] = True
    return dec


def target_weights(data, params):
    window, c, g, lm, rs, vol = clock_features(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    X = stack(c, g, lm, rs, vol)
    del c, g, lm, rs, vol
    y = forward_return(data, HORIZON)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=data.universe)
    del X, y
    score = np.asarray(score, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)
    eligible = window & uni & tradable
    decision = decision_mask(data.dates)
    w = weekly_book(score, eligible, decision, float(params["leg"]))
    return close_leavers(w, decision, uni)
