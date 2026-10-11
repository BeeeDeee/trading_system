"""H-0019 Trading-window net flow (LIQ-500, dollar-neutral, event book).

After a new quarterly SF1 filing becomes usable (ART marketcap changes from one non-NaN value to another),
go long the stock if its net window-flow score F = (-ncfcommon - sbcomp) / marketcap is in the top
`quantile` of the LIQ-500 cross-section of F, short it if in the bottom `quantile`; hold `hold_days`
trading days. Daily target book: equal weights +0.5 on the active longs, -0.5 on the active shorts,
flat if either leg has fewer than 5 names.
"""
import numpy as np

PARAMS = {"hold_days": 5, "quantile": 0.2}

MIN_LEG = 5  # card: "If either leg has fewer than 5 active names the whole book is flat that day"


def flow_score(data):
    """F[t, j] = (-ncfcommon - sbcomp) / marketcap from the latest usable ART filing; NaN if unknown."""
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=np.float64)
    sbc = np.asarray(data.extras["sf1_art_sbcomp"], dtype=np.float64)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=np.float64)
    valid = np.isfinite(ncf) & np.isfinite(sbc) & np.isfinite(mcap) & (mcap > 0)
    F = np.full(mcap.shape, np.nan)
    F[valid] = (-ncf[valid] - sbc[valid]) / mcap[valid]
    return F, mcap


def new_filing(mcap):
    """True on row t when marketcap[t] and marketcap[t-1] are both non-NaN and differ."""
    nf = np.zeros(mcap.shape, dtype=bool)
    if mcap.shape[0] > 1:
        cur, prev = mcap[1:], mcap[:-1]
        nf[1:] = np.isfinite(cur) & np.isfinite(prev) & (cur != prev)
    return nf


def mid_rank_pct(vals):
    """Percentile of each value in its cross-section, (average rank - 0.5) / n, ties share the rank."""
    n = vals.shape[0]
    u, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg_rank = cum - (cnt - 1) / 2.0  # 1-based average rank of each unique value
    return (avg_rank[inv.reshape(-1)] - 0.5) / n


def compute_events(data, params):
    """(T, N) bool event mask and (T, N) side (+1 long / -1 short / 0), rows <= t only."""
    q = float(params["quantile"])
    if data.universe is None:
        raise ValueError("H-0019 needs the LIQ-500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    F, mcap = flow_score(data)
    nf = new_filing(mcap)
    T, N = F.shape
    cand = nf & member & tradable & np.isfinite(F)
    side = np.zeros((T, N), dtype=np.float64)
    rows = np.nonzero(cand.any(axis=1))[0]
    for t in rows:
        pool = member[t] & np.isfinite(F[t])  # cross-section: all members with a valid F on t
        idx = np.nonzero(pool)[0]
        if idx.size == 0:
            continue
        pct = mid_rank_pct(F[t, idx])
        row_side = np.zeros(N)
        row_side[idx[pct >= 1.0 - q]] = 1.0
        row_side[idx[pct <= q]] = -1.0
        side[t] = np.where(cand[t], row_side, 0.0)
    mask = side != 0.0
    return mask, side


def target_weights(data, params):
    h = int(params["hold_days"])
    mask, side = compute_events(data, params)
    T, N = mask.shape
    rows = np.arange(T).reshape(-1, 1)

    # latest event row per stock (a new event restarts the clock with the new side)
    last_ev = np.maximum.accumulate(np.where(mask, rows, -1), axis=0)
    has_ev = last_ev >= 0
    ev_side = np.take_along_axis(side, np.maximum(last_ev, 0), axis=0)

    # a held stock that leaves LIQ-500 or stops trading (not listed / not tradable) is closed for good
    member = np.asarray(data.universe, dtype=bool)
    ok = member & np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
    last_bad = np.maximum.accumulate(np.where(~ok, rows, -1), axis=0)

    active = has_ev & (rows - last_ev < h) & (last_bad < last_ev)
    longs = active & (ev_side > 0)
    shorts = active & (ev_side < 0)
    nL = longs.sum(axis=1)
    nS = shorts.sum(axis=1)
    on = (nL >= MIN_LEG) & (nS >= MIN_LEG)

    w = np.zeros((T, N), dtype=np.float64)
    wl = np.where(on, 0.5 / np.maximum(nL, 1), 0.0).reshape(-1, 1)
    ws = np.where(on, 0.5 / np.maximum(nS, 1), 0.0).reshape(-1, 1)
    w = np.where(longs, wl, w)
    w = np.where(shorts, -ws, w)
    return w
