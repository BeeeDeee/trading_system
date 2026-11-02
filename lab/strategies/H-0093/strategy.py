"""H-0093 earnings-seasonality surprise, LIQ-500 market-neutral.

Long stocks entering a seasonally strong quarter, short stocks entering a seasonally weak one, for the 5
sessions centred on the predicted 8-K Item 2.02 release (last year's same-quarter reaction session + ~251
sessions); the net dollar exposure is hedged with the 100 most liquid LIQ-500 members not in an event.
"""
import numpy as np

PARAMS = {"threshold": 0.75, "years_back": 5}

# Fixed by the card's description (not tunable parameters).
ANCHOR_LAG = 248        # last year's same-quarter reaction session is t - 248
CLUSTER = 40            # the anchor is the first reaction session within the 40 sessions up to t - 248
RECENT = 40             # no reaction session in t-39 .. t
R1_FAR = 120            # R1 = most recent reaction session in t-120 .. t-40
R1_NEAR = 40
ALIGN_TOL = 5           # latest quarter usable no earlier than R1 - 5
NEW_QUARTER_MIN = 40    # a value change >= 40 sessions after the last quarter start opens a new quarter
MAX_GAP = 120           # consecutive quarter starts more than 120 sessions apart void the score
HOLD = 5                # decision rows t .. t+4 hold the position (sessions t+1 .. t+5)
SLOT = 0.05
N_HEDGE = 100
EPS = 1e-9


def _quarters(x):
    """Accepted quarter starts and finalized values of one stock's daily netinc series.

    starts[q]: row where quarter q first became usable.  vfin[q] (q < m-1): the quarter's value after its
    last amendment, i.e. the series value on the row before quarter q+1 started.  vfin[m-1] is NaN (the
    latest quarter can still be amended; the score never uses it).
    """
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
        # else: amendment of the latest accepted quarter (its value is overwritten, its start is kept)
    starts = np.asarray(starts, dtype=np.int64)
    vfin = np.full(starts.size, np.nan)
    if starts.size > 1:
        pos = np.searchsorted(idx, starts[1:] - 1, side="right") - 1
        vfin[:-1] = vals[pos]
    return starts, vfin


def _season_halfunits(vfin, m):
    """rr[c] = 2 * (# of the 4 neighbours c-2, c-1, c+1, c+2 that quarter c exceeds) + (# ties), in 0..8."""
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
    """List of (t, j, side) candidate events in row order per stock (before the open-event rule)."""
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

    R = ea == 1.0                         # NaN (no EDGAR coverage yet) counts as "no reaction session"
    C = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(R, axis=0, out=C[1:])       # C[i] = reactions in rows < i

    def count(a, b):                      # reactions in rows a..b inclusive (a, b arrays of row indices)
        a = np.maximum(a, 0)
        return C[b + 1, cols] - C[a, cols]

    # anchor: reaction on t-248 that is the first in t-287 .. t-248
    tt, cols = np.nonzero(R[: max(T - ANCHOR_LAG, 0)])
    t = tt + ANCHOR_LAG
    ok = count(t - ANCHOR_LAG - CLUSTER + 1, t - ANCHOR_LAG - 1) == 0
    ok &= count(t - RECENT + 1, t) == 0               # no reaction in t-39 .. t
    t, cols = t[ok], cols[ok]
    ok = uni[t, cols] & tradable[t, cols]             # LIQ-500 member on t, has a bar on t
    t, cols = t[ok], cols[ok]
    # R1: most recent reaction in t-120 .. t-40 (none in t-39 .. t, so it is the last one <= t-40)
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
        i = np.searchsorted(starts, seg_t, side="right") - 1      # latest accepted quarter (lag 1) on t
        good = i >= 4 * Y + 1                                       # lags up to 4Y+2 exist
        ic = np.where(good, i, 4 * Y + 1)
        good &= starts[ic] >= r1 - ALIGN_TOL                        # alignment guard
        # gap rule over the span lag 4Y+2 .. lag 1 (quarters ic-4Y-1 .. ic)
        bad = np.zeros(m, dtype=np.int64)
        bad[1:] = np.cumsum(np.diff(starts) > MAX_GAP)
        good &= (bad[ic] - bad[ic - 4 * Y - 1]) == 0
        rr = _season_halfunits(vfin, m)
        tot = np.zeros(ic.size, dtype=np.int64)
        for k in range(1, Y + 1):
            rk = rr[ic - 4 * k + 1]                                 # same quarter k years earlier = lag 4k
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
    """Events that open (the stock is not already in an open event), with the last row holding weight.

    An event decided on t holds decision rows t .. t+4 (sessions t+1 .. t+5, closed at the open of t+6);
    it is closed early at the first decision row on which the stock is not a LIQ-500 member.
    """
    ev_t, ev_j, ev_s, uni, T, N = _events(data, params)
    opened = []
    open_until = {}
    for t, j, s in zip(ev_t, ev_j, ev_s):
        if t <= open_until.get(j, -1):
            continue                                   # already in an open event
        end = min(t + HOLD - 1, T - 1)
        seg = uni[t + 1: end + 1, j]
        if seg.size and not seg.all():
            end = t + int(np.argmin(seg))
        opened.append((t, j, s, end))
        open_until[j] = end
    return opened, uni, T, N


def target_weights(data, params):
    opened, uni, T, N = _open_events(data, params)
    liq = np.asarray(data.extras["liq_rank"], dtype=np.float64)
    W = np.zeros((T, N), dtype=np.float64)
    for t, j, s, end in opened:
        W[t: end + 1, j] = s * SLOT

    # daily hedge of the net exposure with the 100 most liquid members not in an open event
    rows = np.nonzero(np.any(W != 0.0, axis=1))[0]
    for t in rows.tolist():
        e = W[t]
        net = float(e.sum())
        cand = np.nonzero(uni[t] & (e == 0.0) & np.isfinite(liq[t]))[0]
        if cand.size > N_HEDGE:
            cand = cand[np.argsort(liq[t, cand], kind="stable")[:N_HEDGE]]
        if cand.size and net != 0.0:
            W[t, cand] = -net / cand.size
        gross = float(np.abs(W[t]).sum())
        if gross > 1.0:
            W[t] /= gross
    return W
