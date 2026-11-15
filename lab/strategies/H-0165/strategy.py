"""H-0165: 60/40 SPY/IEF tilted by the price- and VIX-residual equity put/call ratio.

Alignment (all in data rows; row t is decided after the close of t, filled at the open of t+1):
- The generic series (cboe_putcall_1d.equity_pc, fred_macro.VIXCLS) have clock next_morning, so the
  framework already places the value stamped on session s at row s+1 ("last value known after the
  close of row t"). Row r of the delivered series therefore carries session r-1, which is the card's
  one-session lag; it is not lagged again.
- SPY's session-s return is in row s (ret_co, ret_oc), so it is shifted by one row to sit next to the
  session-s put/call and VIX values. Everything indexed by "aligned row r" below describes session r-1.
- At decision row t the fit uses sessions t-252..t-1 = aligned rows t-251..t and z_t = residual of
  aligned row t (session t-1) / std of the window's residuals.
"""
import numpy as np
from lab.framework.api import only_on, period_starts

PARAMS = {"hold_days": 5, "z_threshold": 1.5}

# Fixed constants of the card (not tuned, not in the G2 grid)
SMOOTH = 5          # sessions in the mean of ln(PC)
SMOOTH_MIN = 4      # valid values needed in that mean
FIT_WINDOW = 252    # sessions in the OLS window
FIT_MIN = 200       # rows with p, x, v valid needed in the window
BASE_SPY = 0.60
TILT = 0.20
STALE_RUN = 3       # a value repeated on this many consecutive rows (or more) is a carried-forward gap


def fresh_values(pc):
    """ln(PC) per row, NaN where PC is missing, non-positive, or a carried-forward copy.

    The generic loader holds the last known value; a gap in the Cboe series (and everything after the
    end of the delivered Cboe history) would then show up as the same value repeated. A value
    equal to the two previous rows' values (third or later row of an identical run) is treated as
    missing. Uses rows <= r only.
    """
    pc = np.asarray(pc, dtype=float)
    lp = np.full(pc.shape, np.nan)
    ok = np.isfinite(pc) & (pc > 0)
    lp[ok] = np.log(pc[ok])
    stale = np.zeros(pc.shape, dtype=bool)
    if pc.shape[0] >= STALE_RUN:
        same = np.ones(pc.shape[0] - (STALE_RUN - 1), dtype=bool)
        for k in range(1, STALE_RUN):
            a = pc[STALE_RUN - 1:]
            b = pc[STALE_RUN - 1 - k:pc.shape[0] - k]
            same &= np.isfinite(a) & (a == b)
        stale[STALE_RUN - 1:] = same
    lp[stale] = np.nan
    return lp


def trailing_valid_mean(x, n, min_valid):
    """Mean of the valid values of x over rows r-n+1..r; NaN if fewer than min_valid are valid."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    out = np.full(T, np.nan)
    if T == 0:
        return out
    ok = np.isfinite(x)
    s = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    c = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    cnt = c[hi] - c[lo]
    full = (hi - n) >= 0          # the window must lie inside the data
    good = full & (cnt >= min_valid)
    out[good] = (s[hi] - s[lo])[good] / cnt[good]
    return out


def residual_z(pc_aligned, vix_aligned, spy_logret_session):
    """z per decision row, NaN where it cannot be computed.

    pc_aligned[r], vix_aligned[r]: put/call ratio and VIX of session r-1 (as delivered).
    spy_logret_session[r]: SPY log total return of session r (row r's ret_co, ret_oc).
    """
    pc_aligned = np.asarray(pc_aligned, dtype=float)
    T = pc_aligned.shape[0]
    z = np.full(T, np.nan)
    if T < 2:
        return z
    # p: mean ln(PC) over sessions s-4..s (aligned rows r-4..r)
    p = trailing_valid_mean(fresh_values(pc_aligned), SMOOTH, SMOOTH_MIN)
    # v: ln(VIX_s)
    vix = np.asarray(vix_aligned, dtype=float)
    v = np.full(T, np.nan)
    okv = np.isfinite(vix) & (vix > 0)
    v[okv] = np.log(vix[okv])
    # x: SPY log total return over sessions s-4..s = session rows r-5..r-1
    lr = np.asarray(spy_logret_session, dtype=float)
    lr_shift = np.full(T, np.nan)
    lr_shift[1:] = lr[:-1]                       # aligned row r carries session r-1
    x = trailing_valid_mean(lr_shift, SMOOTH, SMOOTH) * SMOOTH   # sum of all five returns
    valid = np.isfinite(p) & np.isfinite(x) & np.isfinite(v)
    cnt = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    for t in range(T):
        if not valid[t]:
            continue
        lo = max(t - FIT_WINDOW + 1, 0)
        if t - FIT_WINDOW + 1 < 0:
            continue                              # window not yet complete (warm-up)
        if cnt[t + 1] - cnt[lo] < FIT_MIN:
            continue
        sl = slice(lo, t + 1)
        m = valid[sl]
        xs = x[sl][m]
        vs = v[sl][m]
        ps = p[sl][m]
        A = np.column_stack([np.ones(xs.shape[0]), xs, vs])
        coef, _, rank, _ = np.linalg.lstsq(A, ps, rcond=None)
        if rank < 3:
            continue
        e = ps - A @ coef
        sd = float(np.std(e, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        z[t] = e[-1] / sd                         # the last valid row is row t itself
    return z


def tilt_targets(z, hold_days, z_threshold):
    """SPY target per row: 0.80 after fear, 0.40 after complacency for hold_days rows, else 0.60."""
    z = np.asarray(z, dtype=float)
    T = z.shape[0]
    H = int(hold_days)
    k = float(z_threshold)
    spy = np.full(T, BASE_SPY)
    sign = 0
    left = 0
    for t in range(T):
        zt = z[t]
        if np.isfinite(zt) and zt >= k:
            sign, left = 1, H                     # latest signal wins, same sign extends
        elif np.isfinite(zt) and zt <= -k:
            sign, left = -1, H
        if left > 0:
            spy[t] = BASE_SPY + sign * TILT
            left -= 1
        else:
            sign = 0
    return spy


def spy_session_logret(data):
    j = data.col("SPY")
    rco = np.asarray(data.ret_co[:, j], dtype=float)
    roc = np.asarray(data.ret_oc[:, j], dtype=float)
    return np.log1p(rco) + np.log1p(roc)


def compute_z(data):
    pc = np.asarray(data.series["cboe_putcall_1d.equity_pc"], dtype=float)
    vix = np.asarray(data.series["fred_macro.VIXCLS"], dtype=float)
    return residual_z(pc, vix, spy_session_logret(data))


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    js = data.col("SPY")
    ji = data.col("IEF")
    z = compute_z(data)
    spy = tilt_targets(z, params["hold_days"], params["z_threshold"])
    w = np.zeros((T, N))
    w[:, js] = spy
    w[:, ji] = 1.0 - spy
    # decide only when the target changes, and re-set on the first session of each month
    change = np.ones(T, dtype=bool)
    if T > 1:
        change[1:] = spy[1:] != spy[:-1]
    days = change | month_start_mask(data.dates)
    return only_on(days, w)


def month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.reshape(T)
    m = np.zeros(T, dtype=bool)
    m[ps.astype(np.int64)] = True
    return m
