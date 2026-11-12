"""H-0132: oil-specific WTI shocks into single-country equity ETFs.

Decision after the close of row t, execution at the open of t+1.

Session alignment: fred_macro arrives aligned by its next_morning clock, so row t of the oil series is the
last WTI value dated before session t (normally session t-1). The oil return in row t therefore belongs to
session t-1 and is paired with the world factor of session t-1 (row t-1). Every rolling window below is
expressed in rows of that "known at t" alignment.
"""
import numpy as np

PARAMS = {"n_leg": 3, "shock_z": 1.5}

UNIVERSE = (
    "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO", "EWP", "EWQ",
    "EWS", "EWT", "EWU", "EWW", "EWY", "EWZ", "EZA", "FXI", "ECH", "TUR", "THD", "EIDO", "EPOL", "RSX",
    "NORW", "GXG", "EPU", "EIS", "INDA",
)
OIL_KEY = "fred_macro.DCOILWTICO"
WIN = 252          # sessions in every regression / std window
MIN_OBS = 200      # joint observations required in a window
SHOCK_LEN = 5      # sessions in the shock sum (t-5..t-1)
HOLD = 5           # sessions held: enter open t+1, exit open t+6
MIN_ELIGIBLE = 10  # eligible ETFs needed for an event
WINSOR = 0.20      # oil daily return winsorized at +-20 %


def lag(x, k=1):
    """Row t holds x[t-k]; first k rows NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def rsum(x, n):
    """Trailing sum over the last n rows (current row included); NaN counts as 0."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0), axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def rolling_slope(y, x, n, min_obs):
    """Trailing OLS slope of y on x (with intercept) over n rows, joint non-NaN observations >= min_obs."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim < y.ndim:
        x = np.broadcast_to(x.reshape(x.shape + (1,) * (y.ndim - x.ndim)), y.shape)
    m = np.isfinite(y) & np.isfinite(x)
    xm = np.where(m, x, 0.0)
    ym = np.where(m, y, 0.0)
    cnt = rsum(m.astype(float), n)
    sx = rsum(xm, n)
    sy = rsum(ym, n)
    sxx = rsum(xm * xm, n)
    sxy = rsum(xm * ym, n)
    den = cnt * sxx - sx * sx
    num = cnt * sxy - sx * sy
    good = (cnt >= min_obs) & (den > 1e-12 * np.maximum(cnt * sxx, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(good, num / np.where(good, den, 1.0), np.nan)
    return b


def rolling_sd(x, n, min_obs):
    """Trailing sample standard deviation (ddof=1) over n rows, non-NaN observations >= min_obs."""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    xm = np.where(m, x, 0.0)
    cnt = rsum(m.astype(float), n)
    s1 = rsum(xm, n)
    s2 = rsum(xm * xm, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1.0)
    var = np.where(cnt >= max(min_obs, 2), np.maximum(var, 0.0), np.nan)
    return np.sqrt(var)


def oil_returns(p):
    """Simple return between consecutive available WTI values; missing if either price <= 0 or missing.

    A row whose value is missing or merely repeats the previous row (forward-filled, no new FRED print)
    is not an available value: its return is missing and the next new print is measured against the last
    available one. Winsorized at +-WINSOR.
    """
    p = np.asarray(p, dtype=float)
    T = p.shape[0]
    fin = np.isfinite(p)
    idx = np.where(fin, np.arange(T), -1)
    idx = np.maximum.accumulate(idx) if T else idx
    pf = np.where(idx >= 0, p[np.maximum(idx, 0)], np.nan)   # forward-filled, causal
    prev = lag(pf)                                           # last available value before row t
    fresh = fin & ~(p == prev)
    ok = fresh & np.isfinite(prev) & (p > 0) & (prev > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.where(ok, p / np.where(ok, prev, 1.0) - 1.0, np.nan)
    return np.clip(o, -WINSOR, WINSOR)


def signals(close, listed, ret_co, ret_oc, oil, member=None):
    """Returns z (T,), exposure beta (T, N), eligible (T, N) bool, alive (T, N) bool (price at t)."""
    close = np.asarray(close, dtype=float)
    listed = np.asarray(listed, dtype=bool)
    alive = np.isfinite(close) & listed
    # daily close-to-close total return of each ETF, only where a price exists at t and t-1
    has_ret = alive & np.vstack([np.zeros((1, close.shape[1]), bool), alive[:-1]])
    r = np.where(has_ret, (1.0 + np.asarray(ret_co, float)) * (1.0 + np.asarray(ret_oc, float)) - 1.0, np.nan)
    # (2) world factor: equal-weight mean of ETFs with a return that session
    cnt = has_ret.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(cnt > 0, np.where(has_ret, r, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    # (1) oil return known at row t = session t-1
    o = oil_returns(oil)
    w_l = lag(w)                                   # world factor of session t-1, paired with o
    # (3) oil-specific return e(t-1) = o - b * w, b over the 252 sessions ending t-1
    b = rolling_slope(o, w_l, WIN, MIN_OBS)
    e = o - b * w_l                                # row t holds e of session t-1
    # (4) shock over sessions t-5..t-1 and its z-score with sd over the 252 sessions ending t-1
    n_e = rsum(np.isfinite(e).astype(float), SHOCK_LEN)
    s = np.where(n_e >= 1, rsum(e, SHOCK_LEN), np.nan)
    sd = rolling_sd(e, WIN, MIN_OBS)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, s / (np.sqrt(SHOCK_LEN) * sd), np.nan)
    # (5) exposure: slope of (r_j - w) on e over the 252 sessions ending t-1
    y = lag(r - w[:, None])
    beta = rolling_slope(y, e, WIN, MIN_OBS)
    elig = np.isfinite(beta) & alive
    if member is not None:
        elig &= np.asarray(member, dtype=bool)
    return z, beta, elig, alive


def book(z, beta, elig, alive, n_leg, shock_z):
    """Event book. Returns weights (T, N) (NaN = keep) and a list of (event_row, sign) pairs."""
    T, N = beta.shape
    W = np.full((T, N), np.nan)
    events = []
    t0 = -1
    held = None
    need = max(MIN_ELIGIBLE, 2 * n_leg)
    for t in range(T):
        if held is not None:
            k = t - t0
            if k < HOLD:
                lost = held & ~alive[t]
                if lost.any():
                    W[t, lost] = 0.0          # name lost its price: close it, weight to cash
                    held = held & ~lost
                continue
            # k == HOLD: exit at the open of t0+6; the book is still open at this decision
            W[t] = 0.0
            held = None
            continue
        zt = z[t]
        if not np.isfinite(zt) or abs(zt) < shock_z:
            continue
        idx = np.flatnonzero(elig[t])
        if idx.size < need:
            continue
        order = idx[np.argsort(beta[t, idx], kind="stable")]
        low = order[:n_leg]
        high = order[-n_leg:]
        sgn = 1.0 if zt > 0 else -1.0
        W[t] = 0.0
        W[t, high] = sgn * 0.5 / n_leg
        W[t, low] = -sgn * 0.5 / n_leg
        held = W[t] != 0.0
        t0 = t
        events.append((t, sgn))
    return W, events


def target_weights(data, params):
    n_leg = int(params["n_leg"])
    shock_z = float(params["shock_z"])
    cols = np.array([data.col(s) for s in UNIVERSE], dtype=int)
    oil = np.asarray(data.series[OIL_KEY], dtype=float)
    member = None
    if data.universe is not None:
        member = np.asarray(data.universe, dtype=bool)[:, cols]
    z, beta, elig, alive = signals(
        np.asarray(data.close, float)[:, cols],
        np.asarray(data.listed, bool)[:, cols],
        np.asarray(data.ret_co, float)[:, cols],
        np.asarray(data.ret_oc, float)[:, cols],
        oil,
        member,
    )
    Wsub, _ = book(z, beta, elig, alive, n_leg, shock_z)
    T = len(data.dates)
    out = np.full((T, len(data.instruments)), np.nan)
    out[:, cols] = Wsub
    return out
