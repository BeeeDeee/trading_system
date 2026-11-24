"""H-0173: oil-specific WTI shocks into country equity ETFs.

After a large 5-session WTI move net of the world equity factor (z-scored), go long the n_leg country ETFs with the
highest (lowest, for z < 0) estimated oil exposure and short the n_leg with the lowest (highest), 0.5 / n_leg per
name, hold exactly 5 sessions, then flat until the next event. Events only fire when the book is flat.
"""
import numpy as np

UNIVERSE = (
    "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO", "EWP", "EWQ", "EWS", "EWT",
    "EWU", "EWW", "EWY", "EWZ", "EZA", "FXI", "ECH", "TUR", "THD", "EIDO", "EPOL", "RSX", "NORW", "EPU", "EIS", "INDA",
)
OIL_KEY = "fred_macro.DCOILWTICO"

WINDOW = 252        # sessions of every rolling estimate (b, std of e, exposures)
MIN_OBS = 200       # joint observations required by every rolling estimate
SHOCK_DAYS = 5      # shock window t-5..t-1
HOLD = 5            # sessions held: enter at open t+1, exit at open t+6
MIN_ELIGIBLE = 10   # eligible ETFs needed for an event
WINSOR = 0.20       # winsorization of the daily oil return

PARAMS = {"n_leg": 3, "shock_z": 1.5}


# ----------------------------------------------------------------------------------------------------------------
# rolling helpers (trailing windows ending at the row, built from prefix sums: row i uses rows <= i only)
# ----------------------------------------------------------------------------------------------------------------
def _window_sum(x, length):
    """Sum of x over rows i-length+1..i (truncated at the start). x must contain no NaN."""
    t = x.shape[0]
    c = np.zeros((t + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    hi = np.arange(1, t + 1)
    lo = np.maximum(hi - length, 0)
    return c[hi] - c[lo]


def _rolling_slope(y, x, length, min_obs):
    """OLS slope of y on x (with intercept) over the trailing window ending at each row; NaN with < min_obs
    joint finite observations or no variance in x. x and y broadcast to a common shape."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y, x = np.broadcast_arrays(y, x)
    m = np.isfinite(y) & np.isfinite(x)
    xv = np.where(m, x, 0.0)
    yv = np.where(m, y, 0.0)
    n = _window_sum(m.astype(np.float64), length)
    sx = _window_sum(xv, length)
    sy = _window_sum(yv, length)
    sxx = _window_sum(xv * xv, length)
    sxy = _window_sum(xv * yv, length)
    den = n * sxx - sx * sx
    num = n * sxy - sx * sy
    ok = (n >= min_obs) & (den > 1e-18)
    out = np.full(y.shape, np.nan)
    out[ok] = num[ok] / den[ok]
    return out


def _rolling_std(x, length, min_obs):
    """Sample standard deviation (ddof 1) of the finite values of x over the trailing window."""
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    xv = np.where(m, x, 0.0)
    n = _window_sum(m.astype(np.float64), length)
    sx = _window_sum(xv, length)
    sxx = _window_sum(xv * xv, length)
    ok = n >= max(min_obs, 2)
    out = np.full(x.shape, np.nan)
    var = (sxx[ok] - sx[ok] * sx[ok] / n[ok]) / (n[ok] - 1.0)
    out[ok] = np.sqrt(np.maximum(var, 0.0))
    return out


def _shift1(x):
    """Value of the previous row (row 0 -> NaN)."""
    out = np.full(x.shape, np.nan)
    out[1:] = x[:-1]
    return out


# ----------------------------------------------------------------------------------------------------------------
# signal pieces
# ----------------------------------------------------------------------------------------------------------------
def _oil_row_returns(p):
    """Rule (1) on the delivered oil series (row r = last WTI value known after the close of r): simple return
    between consecutive available (non-NaN) values, missing when either price is <= 0 or missing, winsorized."""
    p = np.asarray(p, dtype=np.float64)
    t = p.shape[0]
    valid = np.isfinite(p)
    idx = np.where(valid, np.arange(t), -1)
    last = np.maximum.accumulate(idx) if t else idx
    prev = np.full(t, -1, dtype=np.int64)
    prev[1:] = last[:-1]
    out = np.full(t, np.nan)
    has_prev = prev >= 0
    pp = np.where(has_prev, p[np.maximum(prev, 0)], np.nan)
    ok = valid & has_prev & (p > 0) & (pp > 0)
    out[ok] = p[ok] / pp[ok] - 1.0
    return np.clip(out, -WINSOR, WINSOR)


def _oil_session_returns(p):
    """o(d) for XNYS session d. The series has clock next_morning, so the WTI value of session d is first visible at
    row d+1: o(d) = row return at d+1. The last session's o is not yet known (NaN)."""
    r = _oil_row_returns(p)
    o = np.full(r.shape, np.nan)
    o[:-1] = r[1:]
    return o


def _etf_returns(ret_co, ret_oc, close, listed):
    """Close-to-close total return of each ETF and whether it exists (a close at d and at d-1, listed at d)."""
    rc = np.nan_to_num(np.asarray(ret_co, dtype=np.float64), nan=0.0)
    ro = np.nan_to_num(np.asarray(ret_oc, dtype=np.float64), nan=0.0)
    r = (1.0 + rc) * (1.0 + ro) - 1.0
    px = np.isfinite(np.asarray(close, dtype=np.float64))
    valid = px & np.asarray(listed, dtype=bool)
    valid[1:] &= px[:-1]
    valid[0] = False
    return np.where(valid, r, np.nan)


def _signals(r, oil_price):
    """z(t) and exposure(t, j) for decisions after the close of t (inputs of sessions <= t-1 only for oil)."""
    t = r.shape[0]
    cnt = np.isfinite(r).sum(axis=1)
    w = np.full(t, np.nan)
    has = cnt > 0
    w[has] = np.nansum(r[has], axis=1) / cnt[has]                       # rule (2)
    o = _oil_session_returns(oil_price)                                  # rule (1)
    b = _rolling_slope(o, w, WINDOW, MIN_OBS)                            # rule (3): window ending at d
    e = o - b * w
    e_fill = np.where(np.isfinite(e), e, 0.0)
    s_sum = _shift1(_window_sum(e_fill, SHOCK_DAYS))                     # rule (4): sessions t-5..t-1
    sd = _shift1(_rolling_std(e, WINDOW, MIN_OBS))                       # std over the 252 sessions ending t-1
    z = np.full(t, np.nan)
    okz = np.isfinite(sd) & (sd > 0)
    z[okz] = s_sum[okz] / (np.sqrt(SHOCK_DAYS) * sd[okz])
    y = r - w[:, None]
    expo = _rolling_slope(y, e[:, None], WINDOW, MIN_OBS)               # rule (5): window ending at t-1
    expo_prev = np.full(expo.shape, np.nan)
    expo_prev[1:] = expo[:-1]
    return z, expo_prev


def _build_book(z, expo, eligible, has_price, n_leg, shock_z):
    """Rule (6): sequential event book. Returns weights (T, M) with NaN = keep, and the list of events
    (t, z, long_cols, short_cols)."""
    t_rows, m = expo.shape
    wts = np.full((t_rows, m), np.nan)
    events = []
    next_ok = 0
    for t in range(t_rows):
        if t < next_ok:
            continue
        zt = z[t]
        if not np.isfinite(zt) or abs(zt) < shock_z:
            continue
        el = np.flatnonzero(eligible[t])
        if el.size < MIN_ELIGIBLE or el.size < 2 * n_leg:
            continue
        order = el[np.lexsort((el, expo[t, el]))]                       # ascending exposure, ties by column
        low = order[:n_leg]
        high = order[-n_leg:]
        if zt > 0:
            longs, shorts = high, low
        else:
            longs, shorts = low, high
        row = np.zeros(m)
        row[longs] = 0.5 / n_leg
        row[shorts] = -0.5 / n_leg
        wts[t] = row
        held = row != 0
        for k in range(1, HOLD):                                         # rows t+1..t+4: keep, close lost names
            s = t + k
            if s >= t_rows:
                break
            lost = held & ~has_price[s]
            if lost.any():
                wts[s, lost] = 0.0
                held &= ~lost
        if t + HOLD < t_rows:
            wts[t + HOLD] = 0.0                                          # exit at the open of t+6
        next_ok = t + HOLD + 1                                           # book flat only from decision t+6
        events.append((t, float(zt), longs, shorts))
    return wts, events


def _compute(data, params):
    n_leg = int(params["n_leg"])
    shock_z = float(params["shock_z"])
    cols = np.array([data.col(name) for name in UNIVERSE], dtype=np.int64)
    oil = np.asarray(data.series[OIL_KEY], dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    r = _etf_returns(np.asarray(data.ret_co)[:, cols], np.asarray(data.ret_oc)[:, cols], close, listed)
    z, expo = _signals(r, oil)
    has_price = np.isfinite(close) & listed
    eligible = np.isfinite(expo) & has_price
    if data.universe is not None:
        eligible &= np.asarray(data.universe, dtype=bool)[:, cols]
    wts, events = _build_book(z, expo, eligible, has_price, n_leg, shock_z)
    return cols, wts, events


def target_weights(data, params):
    cols, wts, _ = _compute(data, params)
    out = np.full((len(data.dates), len(data.instruments)), np.nan)
    full_rows = np.isfinite(wts).all(axis=1)
    out[full_rows] = 0.0                                                 # full decision rows: nothing else is held
    sub = out[:, cols]
    fin = np.isfinite(wts)
    sub[fin] = wts[fin]
    out[:, cols] = sub
    return out
