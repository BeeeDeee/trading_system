"""H-0024 mechanism event: a concurrent SF1 filing (usable from row t, filed on t-1) of a LIQ-500 member whose
dollar volume on t-1 or t is >= vol_mult x its 63-session median (rows t-64 .. t-2, >= 40 values), with the
surprise S = change of TTM net income / filing-date market value in the top (side +1) or bottom (side -1)
`quantile` of all filings of universe members in the last 252 sessions (>= 200 filings).
Same code as strategy.compute_events (duplicated because the diagnostic may import only numpy / lab.framework).
"""
import numpy as np

POOL_SESSIONS = 252
MIN_POOL = 200
VOL_FIRST = 64
VOL_LAST = 2
MIN_VOL_OBS = 40


def _as_bool(x):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    return np.nan_to_num(a.astype(np.float64), nan=0.0) != 0


def _eligible(data):
    if data.universe is None:
        raise ValueError("H-0024 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = _as_bool(data.universe)
    ok = uni & _as_bool(data.listed) & _as_bool(data.tradable) & ~_as_bool(data.delisting)
    return uni, ok


def _detect_filings(data, uni):
    ni = np.asarray(data.extras["sf1_art_netinc"])
    mc = np.asarray(data.extras["sf1_art_marketcap"])
    T, N = ni.shape
    new = np.zeros((T, N), dtype=bool)
    if T > 1:
        fin = np.isfinite(ni[1:]) & np.isfinite(ni[:-1]) & np.isfinite(mc[1:]) & np.isfinite(mc[:-1])
        changed = (ni[1:] != ni[:-1]) | (mc[1:] != mc[:-1])
        new[1:] = fin & changed
    with np.errstate(invalid="ignore"):
        new &= mc > 0
    new &= uni
    tt, jj = np.nonzero(new)
    s = (ni[tt, jj].astype(np.float64) - ni[tt - 1, jj].astype(np.float64)) / mc[tt, jj].astype(np.float64)
    return tt, jj, s


def _concurrent(data, tt, jj, vol_mult):
    dv = np.asarray(data.dollar_volume)
    out = np.zeros(tt.shape[0], dtype=bool)
    if tt.shape[0] == 0:
        return out
    offs = np.arange(-VOL_FIRST, -VOL_LAST + 1)
    rows = tt[:, None] + offs[None, :]
    inside = rows >= 0
    win = dv[np.clip(rows, 0, None), jj[:, None]].astype(np.float64)
    win[~inside] = np.nan
    nvalid = np.isfinite(win).sum(axis=1)
    enough = nvalid >= MIN_VOL_OBS
    if not enough.any():
        return out
    med = np.nanmedian(win[enough], axis=1)
    thr = vol_mult * med
    d_prev = dv[tt[enough] - 1, jj[enough]].astype(np.float64)
    d_cur = dv[tt[enough], jj[enough]].astype(np.float64)
    with np.errstate(invalid="ignore"):
        hit = (d_prev >= thr) | (d_cur >= thr)
    out[enough] = hit
    return out


def _event_side(data, params):
    q = float(params["quantile"])
    vol_mult = float(params["vol_mult"])
    uni, ok = _eligible(data)
    T, N = uni.shape
    side = np.zeros((T, N), dtype=np.int8)
    tt, jj, s = _detect_filings(data, uni)
    if tt.shape[0] == 0:
        return side
    cand_idx = np.nonzero(ok[tt, jj])[0]
    if cand_idx.shape[0] == 0:
        return side
    conc = _concurrent(data, tt[cand_idx], jj[cand_idx], vol_mult)
    cand_idx = cand_idx[conc]
    if cand_idx.shape[0] == 0:
        return side
    for d in np.unique(tt[cand_idx]):
        lo = np.searchsorted(tt, d - (POOL_SESSIONS - 1), side="left")
        hi = np.searchsorted(tt, d, side="right")
        if hi - lo < MIN_POOL:
            continue
        q_lo, q_hi = np.quantile(s[lo:hi], [q, 1.0 - q])
        today = cand_idx[tt[cand_idx] == d]
        sv = s[today]
        is_long = sv >= q_hi
        is_short = sv <= q_lo
        both = is_long & is_short
        side[d, jj[today[is_long & ~both]]] = 1
        side[d, jj[today[is_short & ~both]]] = -1
    return side


def events(data, params):
    ev = _event_side(data, params)
    mask = ev != 0
    side = np.where(ev < 0, -1.0, 1.0)
    return mask, side
