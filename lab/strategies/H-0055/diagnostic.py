"""H-0055 mechanism events: weekly decision dates where the VRP percentile is in the top quintile
(SPY, side +1) or the bottom quintile (SPY, side -1). Same signal code as strategy.py (duplicated,
since strategy files may only import numpy and the lab framework)."""
import numpy as np

VIX_KEY = "fred_macro.VIXCLS"
RV_WINDOW = 21
RV_MIN_PRESENT = 15
VIX_MAX_AGE = 5
MIN_VALID = 252
TOP_Q = 0.8
BOTTOM_Q = 0.2


def _spy_realized_variance(data):
    j = data.col("SPY")
    rco = np.asarray(data.ret_co[:, j], dtype=float)
    roc = np.asarray(data.ret_oc[:, j], dtype=float)
    close = np.asarray(data.close[:, j], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log1p(rco) + np.log1p(roc)
    present = np.isfinite(close) & np.isfinite(r)
    sq = np.where(present, (100.0 * np.where(present, r, 0.0)) ** 2, 0.0)
    T = sq.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    cn = np.concatenate([[0], np.cumsum(present.astype(np.int64))])
    idx = np.arange(T)
    lo = np.maximum(idx - (RV_WINDOW - 1), 0)
    s = cs[idx + 1] - cs[lo]
    n = cn[idx + 1] - cn[lo]
    rv = np.full(T, np.nan)
    ok = n >= RV_MIN_PRESENT
    rv[ok] = s[ok] * RV_WINDOW / n[ok]
    return rv


def _usable_vix(data):
    x = np.asarray(data.series[VIX_KEY], dtype=float).reshape(-1)
    T = x.shape[0]
    idx = np.arange(T)
    valid = np.isfinite(x)
    last_valid = np.maximum.accumulate(np.where(valid, idx, -1))
    ff = np.where(last_valid >= 0, x[np.maximum(last_valid, 0)], np.nan)
    prev_ff = np.concatenate([[np.nan], ff[:-1]])
    fresh = valid & ~(prev_ff == x)
    last_fresh = np.maximum.accumulate(np.where(fresh, idx, -1))
    age = idx - last_fresh
    return np.where((last_fresh >= 0) & (age <= VIX_MAX_AGE), ff, np.nan)


def _lagged_vrp(data):
    vix = _usable_vix(data)
    rv = _spy_realized_variance(data)
    w = np.full(vix.shape[0], np.nan)
    w[1:] = vix[1:] ** 2 / 12.0 - rv[:-1]
    return w


def _decision_schedule(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    di = d.astype(np.int64)
    wk = (di + 3) // 7
    nxt = np.busday_offset(d, 1, roll="forward")
    flagged = ((nxt.astype(np.int64) + 3) // 7) != wk
    new_week = np.concatenate([[True], wk[1:] != wk[:-1]])
    seg = np.cumsum(new_week) - 1
    nseg = int(seg[-1]) + 1
    flag_count = np.bincount(seg, weights=flagged.astype(float), minlength=nseg)
    starts = np.nonzero(new_week)[0]
    fb_rows = starts[1:]
    fb_rows = fb_rows[flag_count[seg[fb_rows] - 1] == 0]
    dec = np.concatenate([np.nonzero(flagged)[0], fb_rows])
    src = np.concatenate([np.nonzero(flagged)[0], fb_rows - 1])
    order = np.argsort(dec, kind="stable")
    return dec[order].astype(np.int64), src[order].astype(np.int64)


def _percentile_ranks(w, rows, rank_days):
    p = np.full(len(rows), np.nan)
    for k, t in enumerate(rows):
        v = w[t]
        if not np.isfinite(v):
            continue
        win = w[max(0, t - rank_days + 1): t + 1]
        win = win[np.isfinite(win)]
        if win.size < MIN_VALID:
            continue
        p[k] = np.count_nonzero(win <= v) / win.size
    return p


def events(data, params):
    rank_days = int(params["rank_days"])
    T = len(data.dates)
    N = len(data.instruments)
    spy = data.col("SPY")
    w = _lagged_vrp(data)
    dec, src = _decision_schedule(data.dates)
    p = _percentile_ranks(w, src, rank_days)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    ok = np.isfinite(p)
    rows, pv = dec[ok], p[ok]
    hi = pv >= TOP_Q
    lo = pv <= BOTTOM_Q
    eligible = np.asarray(data.tradable[rows, spy], dtype=bool)
    if data.universe is not None:
        eligible &= np.asarray(data.universe[rows, spy], dtype=bool)
    sel = (hi | lo) & eligible
    mask[rows[sel], spy] = True
    side[rows[sel], spy] = np.where(hi[sel], 1.0, -1.0)
    return mask, side
