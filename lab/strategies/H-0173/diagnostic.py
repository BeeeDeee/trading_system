"""H-0173 mechanism test: on each fired event date t (|z(t)| >= shock_z, book flat), the n_leg eligible ETFs with
the highest oil exposure get side sign(z), the n_leg with the lowest get side -sign(z) (= the strategy's long leg
+1, short leg -1). Same pipeline as strategy.py (copied: separate file)."""
import numpy as np

UNIVERSE = (
    "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO", "EWP", "EWQ", "EWS", "EWT",
    "EWU", "EWW", "EWY", "EWZ", "EZA", "FXI", "ECH", "TUR", "THD", "EIDO", "EPOL", "RSX", "NORW", "EPU", "EIS", "INDA",
)
OIL_KEY = "fred_macro.DCOILWTICO"

WINDOW = 252
MIN_OBS = 200
SHOCK_DAYS = 5
HOLD = 5
MIN_ELIGIBLE = 10
WINSOR = 0.20


def _window_sum(x, length):
    t = x.shape[0]
    c = np.zeros((t + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    hi = np.arange(1, t + 1)
    lo = np.maximum(hi - length, 0)
    return c[hi] - c[lo]


def _rolling_slope(y, x, length, min_obs):
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
    out = np.full(x.shape, np.nan)
    out[1:] = x[:-1]
    return out


def _oil_row_returns(p):
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
    r = _oil_row_returns(p)
    o = np.full(r.shape, np.nan)
    o[:-1] = r[1:]
    return o


def _etf_returns(ret_co, ret_oc, close, listed):
    rc = np.nan_to_num(np.asarray(ret_co, dtype=np.float64), nan=0.0)
    ro = np.nan_to_num(np.asarray(ret_oc, dtype=np.float64), nan=0.0)
    r = (1.0 + rc) * (1.0 + ro) - 1.0
    px = np.isfinite(np.asarray(close, dtype=np.float64))
    valid = px & np.asarray(listed, dtype=bool)
    valid[1:] &= px[:-1]
    valid[0] = False
    return np.where(valid, r, np.nan)


def _signals(r, oil_price):
    t = r.shape[0]
    cnt = np.isfinite(r).sum(axis=1)
    w = np.full(t, np.nan)
    has = cnt > 0
    w[has] = np.nansum(r[has], axis=1) / cnt[has]
    o = _oil_session_returns(oil_price)
    b = _rolling_slope(o, w, WINDOW, MIN_OBS)
    e = o - b * w
    e_fill = np.where(np.isfinite(e), e, 0.0)
    s_sum = _shift1(_window_sum(e_fill, SHOCK_DAYS))
    sd = _shift1(_rolling_std(e, WINDOW, MIN_OBS))
    z = np.full(t, np.nan)
    okz = np.isfinite(sd) & (sd > 0)
    z[okz] = s_sum[okz] / (np.sqrt(SHOCK_DAYS) * sd[okz])
    y = r - w[:, None]
    expo = _rolling_slope(y, e[:, None], WINDOW, MIN_OBS)
    expo_prev = np.full(expo.shape, np.nan)
    expo_prev[1:] = expo[:-1]
    return z, expo_prev


def _event_list(z, expo, eligible, n_leg, shock_z):
    """Same sequential rule as the strategy's book: an event at t blocks decisions t+1..t+5 (position open)."""
    t_rows = expo.shape[0]
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
        order = el[np.lexsort((el, expo[t, el]))]
        low = order[:n_leg]
        high = order[-n_leg:]
        if zt > 0:
            longs, shorts = high, low
        else:
            longs, shorts = low, high
        next_ok = t + HOLD + 1
        events.append((t, float(zt), longs, shorts))
    return events


def events(data, params):
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
    member = np.ones(close.shape, dtype=bool)
    if data.universe is not None:
        member = np.asarray(data.universe, dtype=bool)[:, cols]
        eligible &= member
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]

    t_rows, n_all = len(data.dates), len(data.instruments)
    mask = np.zeros((t_rows, n_all), dtype=bool)
    side = np.zeros((t_rows, n_all), dtype=np.float64)
    for t, _zt, longs, shorts in _event_list(z, expo, eligible, n_leg, shock_z):
        for leg, sgn in ((longs, 1.0), (shorts, -1.0)):
            keep = leg[member[t, leg] & tradable[t, leg]]
            mask[t, cols[keep]] = True
            side[t, cols[keep]] = sgn
    return mask, side
