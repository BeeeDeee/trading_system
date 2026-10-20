"""H-0067 mechanism diagnostic: the same events the strategy opens slots on (before any gross scaling).

Self-contained copy of the event logic of strategy.py (diagnostic may only import numpy / lab.framework).
"""
import numpy as np

WIN_FAR = 68
WIN_NEAR = 6
MIN_VALID = 50
POOL_SESSIONS = 252
MIN_BIN = 30


def _window_sum(x, ti, tj):
    T, N = x.shape
    c = np.zeros((T + 1, N), dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    a = np.clip(ti - WIN_FAR, 0, None)
    b = np.clip(ti - WIN_NEAR + 1, 0, None)
    b = np.maximum(a, b)
    return c[b, tj] - c[a, tj]


def _dedupe(rt, rj, hold):
    keep = np.zeros(rt.shape[0], dtype=bool)
    order = np.lexsort((rt, rj))
    last_j = -1
    last_t = 0
    for k in order:
        j = int(rj[k])
        t = int(rt[k])
        if j != last_j or t - last_t > hold:
            keep[k] = True
            last_j = j
            last_t = t
    return keep


def _sides_by_bin(ct, absz, pos, av, first_row, q):
    n = ct.shape[0]
    side = np.zeros(n, dtype=np.int8)
    if n == 0:
        return side
    days = np.unique(ct)
    day_lo = np.searchsorted(ct, days, side="left")
    day_hi = np.searchsorted(ct, days, side="right")
    pool_lo = np.searchsorted(ct, days - POOL_SESSIONS, side="left")
    for k in range(days.shape[0]):
        d = int(days[k])
        if d - POOL_SESSIONS < first_row:
            continue
        lo, hi = int(pool_lo[k]), int(day_lo[k])
        if hi - lo < MIN_BIN:
            continue
        pz = absz[lo:hi]
        cuts = np.quantile(pz, [1.0 / 3.0, 2.0 / 3.0])
        pbin = np.searchsorted(cuts, pz, side="left") + 3 * pos[lo:hi]
        pav = av[lo:hi]
        cache = {}
        for i in range(int(day_lo[k]), int(day_hi[k])):
            b = int(np.searchsorted(cuts, absz[i], side="left") + 3 * pos[i])
            if b not in cache:
                sel = pav[pbin == b]
                if sel.shape[0] < MIN_BIN:
                    cache[b] = None
                else:
                    cache[b] = (np.quantile(sel, 1.0 - q), np.quantile(sel, q))
            th = cache[b]
            if th is None:
                continue
            up = av[i] >= th[0]
            dn = av[i] <= th[1]
            if up and not dn:
                side[i] = 1
            elif dn and not up:
                side[i] = -1
    return side


def _event_table(data, params):
    hold = int(params["hold_days"])
    q = float(params["leg_quantile"])
    if data.universe is None:
        raise ValueError("H-0067 needs the LIQ-500 point-in-time universe")
    U = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8))

    has_8k = np.isfinite(ea).any(axis=1)
    if not has_8k.any():
        return empty
    first_row = int(np.argmax(has_8k))

    with np.errstate(invalid="ignore"):
        raw = (ea == 1.0) & U & trad
    rt, rj = np.nonzero(raw)
    keep = _dedupe(rt, rj, hold)
    ct, cj = rt[keep], rj[keep]
    order = np.lexsort((cj, ct))
    ct, cj = ct[order], cj[order]
    if ct.shape[0] == 0:
        return empty

    with np.errstate(invalid="ignore"):
        dv_ok = listed & np.isfinite(dv) & (dv > 0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    r[~dv_ok] = np.nan
    r[~np.isfinite(r)] = np.nan
    mem = U & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    e = r - mkt[:, None]
    del r
    e_ok = np.isfinite(e)
    e0 = np.where(e_ok, e, 0.0)

    n_e = _window_sum(e_ok.astype(np.float64), ct, cj)
    s1 = _window_sum(e0, ct, cj)
    s2 = _window_sum(e0 * e0, ct, cj)
    del e0
    e_t = e[ct, cj]
    del e
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.maximum(n_e, 1.0)) / np.maximum(n_e - 1.0, 1.0)
        sigma = np.sqrt(np.maximum(var, 0.0))
        z = e_t / sigma
    ok = (n_e >= MIN_VALID) & (var > 0) & np.isfinite(z)

    n_v = _window_sum(dv_ok.astype(np.float64), ct, cj)
    sv = _window_sum(np.where(dv_ok, dv, 0.0), ct, cj)
    dv_t = dv[ct, cj]
    with np.errstate(invalid="ignore", divide="ignore"):
        av = np.log(dv_t) - np.log(sv / np.maximum(n_v, 1.0))
    ok &= (n_v >= MIN_VALID) & dv_ok[ct, cj] & np.isfinite(av)

    vt, vj, vz, vav = ct[ok], cj[ok], z[ok], av[ok]
    side = _sides_by_bin(vt, np.abs(vz), (vz >= 0).astype(np.int64), vav, first_row, q)
    sel = side != 0
    return vt[sel], vj[sel], side[sel]


def events(data, params):
    et, ej, es = _event_table(data, params)
    U = np.asarray(data.universe, dtype=bool)
    T, N = U.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    mask[et, ej] = True
    side[et, ej] = es.astype(np.float64)
    return mask, side
