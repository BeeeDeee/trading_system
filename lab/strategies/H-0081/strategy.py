"""H-0081 - TradFi-stress-gated liquidity provision in crypto perps.

When the last known VIX close is at or above vix_min, rank the eligible top-30 perps (BTC excluded) by their
1-day BTC-beta residual return; long the bottom fifth, short the top fifth, dollar-neutral cohort of gross
1/hold_days, held hold_days decision rows (overlapping cohorts net into one book).
"""
import numpy as np

PARAMS = {"hold_days": 3, "vix_min": 25}

BTC = "BTCUSDT.P"          # market factor only, never ranked or traded
VIX_KEY = "fred_macro.VIXCLS"
BETA_WINDOW = 60           # card: OLS over the 60 days ending t (inclusive)
BETA_MIN_OBS = 30          # card: at least 30 joint observations
MIN_ELIGIBLE = 15          # card: at least 15 eligible coins
LEG_FRACTION = 0.2         # card: bottom / top fifth
MIN_LEG = 3                # card: at least 3 coins per leg
TOP_N = 30                 # card universe: crypto_top_n, n = 30
VOL_WINDOW = 30            # card: trailing 30-day median daily quote volume


def _universe(data):
    """Point-in-time top-30 membership.

    Uses data.universe when the gate supplies it. The gate passes no crypto_top_n universe on the daily perp
    data (G0 crash, K-0054), so otherwise it is rebuilt from the card's definition: on row t, the 30 perps with
    the largest median daily dollar (quote) volume over rows t-29..t (all 30 days priced), rows <= t only.
    """
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = dv.shape
    uni = np.zeros((T, N), dtype=bool)
    for t in range(VOL_WINDOW - 1, T):
        med = np.median(dv[t - VOL_WINDOW + 1:t + 1], axis=0)   # NaN if any day of the window is unpriced
        ok = np.nonzero(np.isfinite(med))[0]
        if ok.size == 0:
            continue
        order = ok[np.argsort(-med[ok], kind="stable")]
        uni[t, order[:TOP_N]] = True
    return uni


def _daily_returns(data):
    """Close-to-close total return of row t and its validity (close known at t-1 and t)."""
    close = np.asarray(data.close, dtype=np.float64)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    fin = np.isfinite(close)
    valid = np.zeros_like(fin)
    valid[1:] = fin[1:] & fin[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def _window_sum(x, n):
    """Trailing sum over rows t-n+1..t (inclusive), rows <= t only."""
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return cs[hi] - cs[lo]


def _residuals(data):
    """e_j,t = r_j,t - b_j,t * r_BTC,t with b the trailing 60-day OLS slope (>= 30 joint obs), NaN if not eligible."""
    r, valid = _daily_returns(data)
    b = data.col(BTC)
    x = r[:, b]
    xv = valid[:, b]
    joint = valid & xv[:, None]
    X = np.where(joint, x[:, None], 0.0)
    Y = np.where(joint, r, 0.0)
    n = _window_sum(joint.astype(np.float64), BETA_WINDOW)
    sx = _window_sum(X, BETA_WINDOW)
    sy = _window_sum(Y, BETA_WINDOW)
    sxy = _window_sum(X * Y, BETA_WINDOW)
    sxx = _window_sum(X * X, BETA_WINDOW)
    den = n * sxx - sx * sx
    ok = (n >= BETA_MIN_OBS) & (den > 0) & joint
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(ok, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)
    e = np.where(ok, r - beta * x[:, None], np.nan)
    e[:, b] = np.nan      # BTC is never ranked
    return e


def _vix_gate(data, vix_min):
    """V_t = last VIXCLS value the data layer exposes at row t (carried forward); gate ON when V_t >= vix_min."""
    v = np.asarray(data.series[VIX_KEY], dtype=np.float64).copy()
    for t in range(1, v.shape[0]):           # carry the last known value (weekends / US holidays)
        if not np.isfinite(v[t]):
            v[t] = v[t - 1]
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v >= vix_min)


def cohort_legs(data, params):
    """(T, N) bool long leg, (T, N) bool short leg, (T,) n_leg of the cohort formed after the close of each row."""
    vix_min = float(params["vix_min"])
    uni = _universe(data)
    e = _residuals(data)
    elig = np.isfinite(e) & uni
    gate = _vix_gate(data, vix_min)
    T, N = e.shape
    long_leg = np.zeros((T, N), dtype=bool)
    short_leg = np.zeros((T, N), dtype=bool)
    n_leg = np.zeros(T, dtype=np.int64)
    for t in np.nonzero(gate)[0]:
        cols = np.nonzero(elig[t])[0]
        m = cols.size
        if m < MIN_ELIGIBLE:
            continue
        k = max(MIN_LEG, int(np.floor(LEG_FRACTION * m)))
        order = cols[np.argsort(e[t, cols], kind="stable")]
        long_leg[t, order[:k]] = True
        short_leg[t, order[m - k:]] = True
        n_leg[t] = k
    return long_leg, short_leg, n_leg


def target_weights(data, params):
    hold = int(params["hold_days"])
    long_leg, short_leg, n_leg = cohort_legs(data, params)
    uni = _universe(data)
    T, N = long_leg.shape
    unit = np.zeros(T)
    has = n_leg > 0
    unit[has] = 1.0 / (2.0 * hold * n_leg[has])
    cohort = (long_leg.astype(np.float64) - short_leg.astype(np.float64)) * unit[:, None]
    w = np.zeros((T, N))
    alive = np.ones((T, N), dtype=bool)
    # a cohort formed after the close of s is held in the targets of rows s .. s+hold-1 (entered at the open
    # of s+1, closed at the open of s+1+hold); a coin leaving the universe is closed for good at the first row
    # outside it.
    for k in range(hold):
        if k >= T:
            break
        alive[: T - k] &= uni[k:]
        w[k:] += cohort[: T - k] * alive[: T - k]
    w[:, data.col(BTC)] = 0.0
    return w
