"""H-0168 Hedge-failure rebound.

After a session where SPY falls at least k_sigma trailing daily vols AND IEF also falls, move the
40 % IEF sleeve of a 60/40 SPY/IEF book into SPY, QQQ, IWM, DIA, MDY (8 % each) for hold_days
open-to-open steps. Outside windows the book is the 60/40 benchmark, reset on the first session
of each month.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 5, "k_sigma": 1.5}

VOL_WINDOW = 63      # sessions t-63 .. t-1
VOL_MIN_VALID = 50   # minimum valid returns in the window
SIDE_EQUITY = ("QQQ", "IWM", "DIA", "MDY")
TILT_EACH = 0.08


def _session_return(data, j):
    """Close t-1 -> close t total return and whether instrument j has a bar on t."""
    r = (1.0 + data.ret_co[:, j].astype(np.float64)) * (1.0 + data.ret_oc[:, j].astype(np.float64)) - 1.0
    bar = np.asarray(data.listed[:, j], dtype=bool) & np.isfinite(data.close[:, j]) & np.isfinite(r)
    return r, bar


def _trailing_std(x, valid, window, min_valid):
    """Sample std (ddof=1) of x over rows t-window .. t-1 (row t excluded), valid rows only."""
    T = x.shape[0]
    v = np.where(valid, x, 0.0)
    c = valid.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    t = np.arange(T)
    hi = t                       # rows < t
    lo = np.maximum(t - window, 0)
    n = cc[hi] - cc[lo]
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    out = np.full(T, np.nan)
    ok = n >= min_valid
    var = (s2[ok] - s[ok] * s[ok] / n[ok]) / (n[ok] - 1.0)
    out[ok] = np.sqrt(np.maximum(var, 0.0))
    return out


def hedge_failure_events(data, k_sigma):
    """(T,) bool: hedge-failure event after the close of row t (rows <= t only)."""
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")
    r_spy, bar_spy = _session_return(data, j_spy)
    r_ief, bar_ief = _session_return(data, j_ief)
    sigma = _trailing_std(r_spy, bar_spy, VOL_WINDOW, VOL_MIN_VALID)
    ok = bar_spy & bar_ief & np.isfinite(sigma)
    ev = np.zeros(r_spy.shape[0], dtype=bool)
    ev[ok] = (r_spy[ok] <= -float(k_sigma) * sigma[ok]) & (r_ief[ok] < 0.0)
    return ev


def target_weights(data, params):
    hold = int(params["hold_days"])
    k = float(params["k_sigma"])
    T = len(data.dates)
    N = len(data.instruments)
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")
    j_side = [data.col(name) for name in SIDE_EQUITY]
    data.col("TLT")   # never traded; must exist (peer for the mechanism test)
    data.col("SHY")

    ev = hedge_failure_events(data, k)
    rows = np.arange(T)
    last = np.maximum.accumulate(np.where(ev, rows, -1)) if T else np.zeros(0, dtype=int)
    has = last >= 0
    age = rows - last                         # decision rows since the latest event
    in_window = has & (age < hold)            # tilt held at the open after these rows
    window_end = has & (age == hold)          # back to 60/40 at the open of e+1+hold

    month_start = np.zeros(T, dtype=bool)
    ps = np.asarray(period_starts(data.dates, "M"))
    if ps.dtype == bool:
        month_start[:] = ps
    else:
        month_start[ps.astype(int)] = True
    base_rows = (month_start & ~in_window) | window_end
    if T:
        base_rows[0] = True                   # start invested in the benchmark mix

    w = np.full((T, N), np.nan)
    base = np.zeros(N)
    base[j_spy] = 0.60
    base[j_ief] = 0.40
    w[base_rows] = base

    # tilt rows: 68 % SPY + 8 % each side ETF; a side ETF without a tradable bar -> its 8 % to SPY
    if ev.any():
        er = np.flatnonzero(ev)
        tilt = np.zeros((er.size, N))
        spy_w = np.full(er.size, 0.68)
        for j in j_side:
            avail = (np.asarray(data.tradable[er, j], dtype=bool)
                     & np.asarray(data.listed[er, j], dtype=bool)
                     & np.isfinite(data.close[er, j]))
            tilt[:, j] = np.where(avail, TILT_EACH, 0.0)
            spy_w = spy_w + np.where(avail, 0.0, TILT_EACH)
        tilt[:, j_spy] = spy_w
        w[er] = tilt
    return w
