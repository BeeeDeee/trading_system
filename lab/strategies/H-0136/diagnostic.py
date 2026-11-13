import numpy as np

# H-0136 mechanism event: a LIQ-500 member's two-session earnings-release reaction z >= z_threshold
# (the event that puts or keeps the stock in the long book). Same computation as strategy.py
# (copied: diagnostic.py is a separate file). Side +1 for every event.

VOL_WINDOW = 60
VOL_MIN_VALID = 40


def _daily_returns(ret_co, ret_oc, listed, close):
    r = (1.0 + ret_co.astype(np.float64)) * (1.0 + ret_oc.astype(np.float64)) - 1.0
    ok = listed.astype(bool) & np.isfinite(close) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = np.where(valid, r, np.nan)
    return r, valid


def _member_mean(x, mask):
    cnt = mask.sum(axis=1)
    tot = np.where(mask, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    has = cnt > 0
    out[has] = tot[has] / cnt[has]
    return out


def _reaction_z(r, valid, uni, ea):
    T, N = r.shape
    mkt1 = _member_mean(r, valid & uni)
    d = r - mkt1[:, None]
    r2 = np.full((T, N), np.nan)
    v2 = np.zeros((T, N), dtype=bool)
    if T > 1:
        v2[1:] = valid[1:] & valid[:-1]
        r2[1:] = (1.0 + np.nan_to_num(r[1:])) * (1.0 + np.nan_to_num(r[:-1])) - 1.0
    r2 = np.where(v2, r2, np.nan)
    mkt2 = _member_mean(r2, v2 & uni)
    a = r2 - mkt2[:, None]

    rel = (ea == 1.0) & uni
    tt, jj = np.nonzero(rel)
    z = np.full((T, N), np.nan)
    ev = np.zeros((T, N), dtype=bool)
    if tt.size == 0:
        return z, ev
    rows = tt[:, None] - (VOL_WINDOW + 1) + np.arange(VOL_WINDOW)[None, :]
    inside = rows >= 0
    win = d[np.clip(rows, 0, None), jj[:, None]]
    win = np.where(inside, win, np.nan)
    n = np.isfinite(win).sum(axis=1)
    mu = np.where(n > 0, np.nansum(win, axis=1) / np.maximum(n, 1), np.nan)
    dev = np.where(np.isfinite(win), win - mu[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(n - 1, 1)
    s = np.sqrt(var)
    ae = a[tt, jj]
    ok = (n >= VOL_MIN_VALID) & np.isfinite(ae) & (s > 0)
    z[tt[ok], jj[ok]] = ae[ok] / (s[ok] * np.sqrt(2.0))
    ev[tt[ok], jj[ok]] = True
    return z, ev


def events(data, params):
    thr = float(params["z_threshold"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0136 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    r, valid = _daily_returns(np.asarray(data.ret_co), np.asarray(data.ret_oc), listed,
                              np.asarray(data.close, dtype=np.float64))
    z, ev = _reaction_z(r, valid, uni, ea)
    mask = ev & (z >= thr) & uni & np.asarray(data.tradable, dtype=bool)
    return mask, None
