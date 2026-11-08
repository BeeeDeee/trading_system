"""H-0110 mechanism event: a universe member crosses the long-term qualification date of its anchor
spike after the close of t (t is the first session with c >= 0 for that anchor, i.e. date(t-1) < q(s)
<= date(t)); side -1 if g > 0 (appreciated), +1 if g < 0 (depreciated), g == 0 no event.
Spike and anchor rules are copied from strategy.py (same code, same parameters)."""
import numpy as np
from lab.framework.api import total_return_index

MED_WINDOW = 63
MED_MIN_VALID = 50
EA_HALF = 3
MKT_MED_MAX = 1.5
QUAL_DAYS = 366
CLOCK_HALF = 14
CHUNK = 100000


def _shift_down(x, k):
    out = np.full(x.shape, np.nan, dtype=x.dtype)
    if k == 0:
        out[:] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rolling_min_prev(x, n):
    a = _shift_down(x, 1)
    span = 1
    while span * 2 <= n:
        a = np.fmin(a, _shift_down(a, span))
        span *= 2
    if span < n:
        a = np.fmin(a, _shift_down(a, n - span))
    return a


def volume_multiple(dv, need):
    T, N = dv.shape
    m = np.full((T, N), np.nan, dtype=np.float32)
    if T <= MED_WINDOW:
        return m
    valid = np.isfinite(dv)
    cnt = np.zeros((T + 1, N), dtype=np.int32)
    cnt[1:] = np.cumsum(valid, axis=0, dtype=np.int32)
    nvalid = np.zeros((T, N), dtype=np.int32)
    nvalid[MED_WINDOW:] = cnt[MED_WINDOW:T] - cnt[0:T - MED_WINDOW]
    ok = need & valid & (nvalid >= MED_MIN_VALID)
    ok[:MED_WINDOW] = False
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        return m
    filled = np.where(valid, dv, np.inf).astype(np.float64)
    offs = np.arange(-MED_WINDOW, 0)
    kth = list(range((MED_MIN_VALID - 1) // 2, MED_WINDOW // 2 + 1))
    for a in range(0, rows.size, CHUNK):
        r = rows[a:a + CHUNK]
        c = cols[a:a + CHUNK]
        win = filled[r[:, None] + offs[None, :], c[:, None]]
        win.partition(kth, axis=1)
        k = nvalid[r, c].astype(np.int64)
        lo = np.take_along_axis(win, ((k - 1) // 2)[:, None], axis=1)[:, 0]
        hi = np.take_along_axis(win, (k // 2)[:, None], axis=1)[:, 0]
        med = 0.5 * (lo + hi)
        pos = med > 0
        val = np.full(r.size, np.nan)
        val[pos] = filled[r[pos], c[pos]] / med[pos]
        m[r, c] = val
    return m


def row_nanmedian(x):
    T = x.shape[0]
    out = np.full(T, np.nan)
    if x.size == 0:
        return out
    s = np.sort(x, axis=1)
    k = np.isfinite(x).sum(axis=1)
    ok = k > 0
    if ok.any():
        sub = s[ok].astype(np.float64)
        kk = k[ok]
        ar = np.arange(sub.shape[0])
        out[ok] = 0.5 * (sub[ar, (kk - 1) // 2] + sub[ar, kk // 2])
    return out


def ea_clear(ea):
    T, N = ea.shape
    z = (ea == 0)
    w = 2 * EA_HALF + 1
    out = np.zeros((T, N), dtype=bool)
    if T < w:
        return out
    cs = np.zeros((T + 1, N), dtype=np.int32)
    cs[1:] = np.cumsum(z, axis=0, dtype=np.int32)
    out[EA_HALF:T - EA_HALF] = (cs[w:T + 1] - cs[0:T + 1 - w]) == w
    return out


def spike_days(dv, ea, uni, spike_mult):
    eaok = ea_clear(ea)
    rmin = _rolling_min_prev(dv.astype(np.float64), MED_WINDOW)
    with np.errstate(invalid="ignore"):
        impossible = dv < spike_mult * rmin
    need = uni | (eaok & ~impossible)
    m = volume_multiple(dv, need)
    mkt = row_nanmedian(np.where(uni, m, np.nan))
    with np.errstate(invalid="ignore"):
        spike = np.isfinite(m) & (m >= spike_mult) & eaok & (mkt < MKT_MED_MAX)[:, None]
    return spike, m


def anchors(day_num, spike, m):
    T, N = spike.shape
    brow = np.full((T, N), -1, dtype=np.int64)
    if T == 0:
        return brow
    lo = np.searchsorted(day_num, day_num - (QUAL_DAYS + CLOCK_HALF), side="left")
    hi = np.searchsorted(day_num, day_num - (QUAL_DAYS - CLOCK_HALF), side="right") - 1
    width = hi - lo + 1
    W = int(width.max()) if width.size else 0
    if W <= 0:
        return brow
    key = np.where(spike, m, -np.inf).astype(np.float32)
    best = np.full((T, N), -np.inf, dtype=np.float32)
    for k in range(W):
        r = hi - k
        okr = (r >= lo) & (r >= 0)
        rr = np.where(okr, r, 0)
        cand = key[rr]
        cand[~okr] = -np.inf
        upd = cand > best
        best = np.where(upd, cand, best)
        brow = np.where(upd, rr[:, None], brow)
    return brow


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0110 needs the point-in-time liq_n universe")
    uni = np.asarray(uni, dtype=bool)
    ea = data.extras["ea_reaction_day"]
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = dv.shape
    spike, m = spike_days(dv, ea, uni, float(params["spike_mult"]))
    day_num = data.dates.astype(np.int64)
    brow = anchors(day_num, spike, m)
    window = (brow >= 0) & uni
    br = np.where(window, brow, 0)
    cols = np.broadcast_to(np.arange(N)[None, :], (T, N))
    c = day_num[:, None] - day_num[br] - QUAL_DAYS            # calendar days from q(s)
    gap = np.zeros(T, dtype=np.int64)
    gap[1:] = day_num[1:] - day_num[:-1]
    first_row = np.zeros(T, dtype=bool)
    first_row[0] = True
    crossing = (c >= 0) & ((c - gap[:, None]) < 0) & ~first_row[:, None]
    tri = np.asarray(total_return_index(data), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(tri / tri[br, cols])
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = window & crossing & tradable & np.isfinite(g) & (g != 0)
    side = np.where(g > 0, -1.0, 1.0)
    side = np.where(mask, side, 0.0)
    return mask, side
