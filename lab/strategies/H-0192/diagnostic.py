"""H-0192 mechanism events: same event rule as strategy.py (copied, separate file).

Event after the close of an 8-K Item 2.02 reaction session t for an S&P 500 member:
  side +1: z >= z_min and TRI_t >= (1 - band) * max TRI over t-252..t-1
  side -1: z <= -z_min and TRI_t <= (1 + band) * min TRI over t-252..t-1
z = (r_t - median r_t of members with data) / std of the stock's residuals over t-63..t-1 (>= 40).
Marked only where the stock is a member and tradable on t.
"""
import numpy as np

VOL_WIN = 63
VOL_MIN = 40
ANCHOR_WIN = 252
ANCHOR_MIN = 240


def _shift(x, k, fill):
    out = np.empty_like(x)
    if k == 0:
        out[:] = x
        return out
    if k >= x.shape[0]:
        out[:] = fill
        return out
    out[:k] = fill
    out[k:] = x[:-k]
    return out


def _rolling_max_incl(x, w):
    T, N = x.shape
    L = T + w - 1
    nb = -(-L // w)
    pad = np.full((nb * w, N), -np.inf)
    pad[w - 1:w - 1 + T] = x
    blocks = pad.reshape(nb, w, N)
    g = np.maximum.accumulate(blocks, axis=1).reshape(nb * w, N)
    h = np.maximum.accumulate(blocks[:, ::-1, :], axis=1)[:, ::-1, :].reshape(nb * w, N)
    idx = np.arange(T)
    return np.maximum(h[idx], g[idx + w - 1])


def _lagged_window_sum(x, w):
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:])
    cs[1:] = np.cumsum(x, axis=0)
    hi = np.arange(T)
    lo = np.maximum(hi - w, 0)
    return cs[hi] - cs[lo]


def _lagged_extreme(xm, valid, w, kind):
    if kind == "max":
        src = np.where(valid, xm, -np.inf)
        r = _rolling_max_incl(src, w)
    else:
        src = np.where(valid, -xm, -np.inf)
        r = -_rolling_max_incl(src, w)
    r = _shift(r, 1, np.nan)
    r = np.where(np.isfinite(r), r, np.nan)
    cnt = _lagged_window_sum(valid.astype(np.float64), w)
    return r, cnt


def events(data, params):
    band = float(params["band"])
    z_min = float(params["z_min"])
    if data.universe is None:
        raise ValueError("H-0192 needs the point-in-time sp500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    ea_all = data.extras["ea_reaction_day"]
    T, N_all = uni.shape
    cols = np.flatnonzero(uni.any(axis=0))       # only ever-members can have events
    member = uni[:, cols]
    ret_co = np.asarray(data.ret_co)[:, cols].astype(np.float64)
    ret_oc = np.asarray(data.ret_oc)[:, cols].astype(np.float64)
    close = np.asarray(data.close)[:, cols].astype(np.float64)
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    ea = np.asarray(ea_all)[:, cols].astype(np.float64)
    # total-return index = cumulative product of (1 + ret_co)(1 + ret_oc), as in strategy.py
    tri = np.cumprod((1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc)), axis=0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = listed & np.isfinite(close) & np.isfinite(r) & np.isfinite(tri) & (tri > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        msk = valid & member
        X = np.where(msk, r, np.nan)
        has = msk.any(axis=1)
        X[~has] = 0.0
        med = np.nanmedian(X, axis=1) if X.shape[1] > 0 else np.zeros(X.shape[0])
        med[~has] = np.nan
        e = np.where(valid, r - med[:, None], np.nan)
        ok = np.isfinite(e)
        e0 = np.where(ok, e, 0.0)
        n = _lagged_window_sum(ok.astype(np.float64), VOL_WIN)
        s1 = _lagged_window_sum(e0, VOL_WIN)
        s2 = _lagged_window_sum(e0 * e0, VOL_WIN)
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        var = np.where(var > 0, var, 0.0)
        sd = np.sqrt(var)
        sd_ok = (n >= VOL_MIN) & (sd > 0)
        z = np.where(sd_ok, e / np.where(sd_ok, sd, 1.0), np.nan)

        trim = np.where(valid, tri, np.nan)
        H, cnt = _lagged_extreme(trim, valid, ANCHOR_WIN, "max")
        Lo, _ = _lagged_extreme(trim, valid, ANCHOR_WIN, "min")
        anchor_ok = (cnt >= ANCHOR_MIN) & np.isfinite(H) & np.isfinite(Lo)
        near_hi = anchor_ok & (trim >= (1.0 - band) * H)
        near_lo = anchor_ok & (trim <= (1.0 + band) * Lo)

        base = (ea == 1.0) & member & valid & np.isfinite(z)
        pos = base & (z >= z_min) & near_hi
        neg = base & (z <= -z_min) & near_lo

    mask = np.zeros((T, N_all), dtype=bool)
    mask[:, cols] = (pos | neg) & member & tradable
    side = np.ones((T, N_all))
    side[:, cols] = np.where(neg, -1.0, 1.0)
    return mask, side
