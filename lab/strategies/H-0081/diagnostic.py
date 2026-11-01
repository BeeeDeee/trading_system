"""H-0081 mechanism events: on high-VIX days, the bottom fifth (side +1) and top fifth (side -1) of the eligible
top-30 perps (BTC excluded, >= 15 eligible) by 1-day BTC-beta residual return. Same legs as the strategy."""
import numpy as np

BTC = "BTCUSDT.P"
VIX_KEY = "fred_macro.VIXCLS"
BETA_WINDOW = 60
BETA_MIN_OBS = 30
MIN_ELIGIBLE = 15
LEG_FRACTION = 0.2
MIN_LEG = 3
TOP_N = 30
VOL_WINDOW = 30


def _universe(data):
    """Same as the strategy: data.universe if supplied, else the PIT top 30 by median dollar volume of rows
    t-29..t (all 30 days priced)."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = dv.shape
    uni = np.zeros((T, N), dtype=bool)
    for t in range(VOL_WINDOW - 1, T):
        med = np.median(dv[t - VOL_WINDOW + 1:t + 1], axis=0)
        ok = np.nonzero(np.isfinite(med))[0]
        if ok.size == 0:
            continue
        order = ok[np.argsort(-med[ok], kind="stable")]
        uni[t, order[:TOP_N]] = True
    return uni


def _daily_returns(data):
    close = np.asarray(data.close, dtype=np.float64)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    fin = np.isfinite(close)
    valid = np.zeros_like(fin)
    valid[1:] = fin[1:] & fin[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def _window_sum(x, n):
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return cs[hi] - cs[lo]


def _residuals(data):
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
    e[:, b] = np.nan
    return e


def _vix_gate(data, vix_min):
    v = np.asarray(data.series[VIX_KEY], dtype=np.float64).copy()
    for t in range(1, v.shape[0]):
        if not np.isfinite(v[t]):
            v[t] = v[t - 1]
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v >= vix_min)


def events(data, params):
    vix_min = float(params["vix_min"])
    uni = _universe(data)
    e = _residuals(data)
    elig = np.isfinite(e) & uni
    gate = _vix_gate(data, vix_min)
    T, N = e.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.nonzero(gate)[0]:
        cols = np.nonzero(elig[t])[0]
        m = cols.size
        if m < MIN_ELIGIBLE:
            continue
        k = max(MIN_LEG, int(np.floor(LEG_FRACTION * m)))
        order = cols[np.argsort(e[t, cols], kind="stable")]
        mask[t, order[:k]] = True
        side[t, order[:k]] = 1.0
        mask[t, order[m - k:]] = True
        side[t, order[m - k:]] = -1.0
    mask &= uni & np.asarray(data.tradable, dtype=bool)
    side = np.where(mask, side, 0.0)
    return mask, side
