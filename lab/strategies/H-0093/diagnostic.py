"""H-0093 mechanism events: the same events the strategy opens (same code, copied: the diagnostic may not
import the strategy module).  mask[t, j] = event decided after the close of t, side = +1 long / -1 short."""
import numpy as np

ANCHOR_LAG = 248
CLUSTER = 40
RECENT = 40
R1_FAR = 120
R1_NEAR = 40
ALIGN_TOL = 5
NEW_QUARTER_MIN = 40
MAX_GAP = 120
HOLD = 5
EPS = 1e-9


def _quarters(x):
    fin = np.isfinite(x)
    idx = np.nonzero(fin)[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    vals = x[idx]
    chg = np.ones(idx.size, dtype=bool)
    chg[1:] = vals[1:] != vals[:-1]
    starts = []
    for r in idx[chg].tolist():
        if not starts or r - starts[-1] >= NEW_QUARTER_MIN:
            starts.append(r)
    starts = np.asarray(starts, dtype=np.int64)
    vfin = np.full(starts.size, np.nan)
    if starts.size > 1:
        pos = np.searchsorted(idx, starts[1:] - 1, side="right") - 1
        vfin[:-1] = vals[pos]
    return starts, vfin


def _season_halfunits(vfin, m):
    rr = np.full(m, -1, dtype=np.int64)
    if m < 5:
        return rr
    c = np.arange(2, m - 2)
    centre = vfin[c]
    tot = np.zeros(c.size, dtype=np.int64)
    for d in (-2, -1, 1, 2):
        nb = vfin[c + d]
        tot += 2 * (centre > nb) + (centre == nb)
    ok = np.isfinite(centre)
    for d in (-2, -1, 1, 2):
        ok &= np.isfinite(vfin[c + d])
    rr[c[ok]] = tot[ok]
    return rr


def _events(data, params):
    thr = float(params["threshold"])
    Y = int(params["years_back"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0093 needs the LIQ-500 point-in-time universe")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"])
    netinc = data.extras["sf1_arq_netinc"]
    T, N = ea.shape

    R = ea == 1.0
    C = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(R, axis=0, out=C[1:])

    def count(a, b):
        a = np.maximum(a, 0)
        return C[b + 1, cols] - C[a, cols]

    tt, cols = np.nonzero(R[: max(T - ANCHOR_LAG, 0)])
    t = tt + ANCHOR_LAG
    ok = count(t - ANCHOR_LAG - CLUSTER + 1, t - ANCHOR_LAG - 1) == 0
    ok &= count(t - RECENT + 1, t) == 0
    t, cols = t[ok], cols[ok]
    ok = uni[t, cols] & tradable[t, cols]
    t, cols = t[ok], cols[ok]
    ok = count(t - R1_FAR, t - R1_NEAR) > 0
    t, cols = t[ok], cols[ok]
    del C

    out_t, out_j, out_s = [], [], []
    order = np.lexsort((t, cols))
    t, cols = t[order], cols[order]
    bounds = np.nonzero(np.diff(cols))[0] + 1
    for seg_t, seg_c in zip(np.split(t, bounds), np.split(cols, bounds)):
        if seg_t.size == 0:
            continue
        j = int(seg_c[0])
        rj = np.nonzero(R[:, j])[0]
        r1 = rj[np.searchsorted(rj, seg_t - R1_NEAR, side="right") - 1]
        starts, vfin = _quarters(np.asarray(netinc[:, j], dtype=np.float64))
        m = starts.size
        if m < 4 * Y + 2:
            continue
        i = np.searchsorted(starts, seg_t, side="right") - 1
        good = i >= 4 * Y + 1
        ic = np.where(good, i, 4 * Y + 1)
        good &= starts[ic] >= r1 - ALIGN_TOL
        bad = np.zeros(m, dtype=np.int64)
        bad[1:] = np.cumsum(np.diff(starts) > MAX_GAP)
        good &= (bad[ic] - bad[ic - 4 * Y - 1]) == 0
        rr = _season_halfunits(vfin, m)
        tot = np.zeros(ic.size, dtype=np.int64)
        for k in range(1, Y + 1):
            rk = rr[ic - 4 * k + 1]
            good &= rk >= 0
            tot += np.maximum(rk, 0)
        S = tot / (8.0 * Y)
        side = np.where(S >= thr - EPS, 1.0, np.where(S <= 1.0 - thr + EPS, -1.0, 0.0))
        good &= side != 0
        for tv, sv in zip(seg_t[good].tolist(), side[good].tolist()):
            out_t.append(tv)
            out_j.append(j)
            out_s.append(sv)
    return out_t, out_j, out_s, uni, T, N


def _open_events(data, params):
    ev_t, ev_j, ev_s, uni, T, N = _events(data, params)
    opened = []
    open_until = {}
    for t, j, s in zip(ev_t, ev_j, ev_s):
        if t <= open_until.get(j, -1):
            continue
        end = min(t + HOLD - 1, T - 1)
        seg = uni[t + 1: end + 1, j]
        if seg.size and not seg.all():
            end = t + int(np.argmin(seg))
        opened.append((t, j, s, end))
        open_until[j] = end
    return opened, uni, T, N


def events(data, params):
    opened, uni, T, N = _open_events(data, params)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t, j, s, end in opened:
        mask[t, j] = True
        side[t, j] = s
    return mask, side
