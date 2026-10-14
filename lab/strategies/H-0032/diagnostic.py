"""H-0032 mechanism diagnostic: the same events and sides as the strategy.

Event after the close of t = P - half_window - 1 (P = last year's Item 2.02 reaction session
+ 364 calendar days), member of LIQ-500 at t, no release in the 30 sessions up to t, composite
score in the top (+1) or bottom (-1) leg_quantile of LIQ-500 members on t. Also tradable at t.
"""
import numpy as np

DEDUP_SESSIONS = 10
PREDICT_DAYS = 364
NO_RECENT_SESSIONS = 30
AG_LAG = 252
MIN_MEMBERS = 100


def _pct_rank(x):
    n = x.shape[0]
    if n < 2:
        return np.full(n, 0.5)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    start = np.concatenate((np.zeros(1), np.cumsum(counts)[:-1].astype(np.float64)))
    avg = start + (counts - 1) / 2.0
    return avg[inv] / (n - 1)


def _event_sides(data, params):
    hw = int(params["half_window"])
    q = float(params["leg_quantile"])
    ex = data.extras
    ea = np.asarray(ex["ea_reaction_day"], dtype=np.float64)
    netinc = ex["sf1_art_netinc"]
    ncfo = ex["sf1_art_ncfo"]
    assets = ex["sf1_art_assets"]
    gp = ex["sf1_art_gp"]
    if data.universe is None:
        raise ValueError("H-0032 needs the LIQ-500 universe mask")
    U = np.asarray(data.universe, dtype=bool)
    T, N = ea.shape
    side = np.zeros((T, N), dtype=np.float64)
    if T == 0:
        return side

    rel = ea == 1.0
    cpad = np.zeros((T + 1, N), dtype=np.int64)
    cpad[1:] = np.cumsum(rel, axis=0, dtype=np.int64)
    idx = np.arange(T)
    prev = cpad[idx] - cpad[np.maximum(idx - DEDUP_SESSIONS, 0)]
    keep = rel & (prev == 0)
    recent = cpad[idx + 1] - cpad[np.maximum(idx + 1 - NO_RECENT_SESSIONS, 0)]

    dint = data.dates.astype(np.int64)
    qdate = np.busday_offset(data.dates, hw + 1, roll="forward").astype(np.int64)
    rr, jj = np.nonzero(keep)
    pdate = dint[rr] + PREDICT_DAYS
    te = np.searchsorted(qdate, pdate, side="left")
    ok = te < T
    cand = np.zeros((T, N), dtype=bool)
    cand[te[ok], jj[ok]] = True
    cand &= U & (recent == 0)

    rows = np.nonzero(cand.any(axis=1))[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in rows:
            if t < AG_LAG:
                continue
            a = np.asarray(assets[t], dtype=np.float64)
            a0 = np.asarray(assets[t - AG_LAG], dtype=np.float64)
            acc = (np.asarray(netinc[t], dtype=np.float64) - np.asarray(ncfo[t], dtype=np.float64)) / a
            gpa = np.asarray(gp[t], dtype=np.float64) / a
            ag = a / a0 - 1.0
            valid = (U[t] & np.isfinite(acc) & np.isfinite(gpa) & np.isfinite(ag)
                     & (a > 0) & (a0 > 0))
            n = int(valid.sum())
            if n < MIN_MEMBERS:
                continue
            score = (1.0 - _pct_rank(acc[valid]) + _pct_rank(gpa[valid])
                     + 1.0 - _pct_rank(ag[valid])) / 3.0
            hi = np.quantile(score, 1.0 - q)
            lo = np.quantile(score, q)
            s_full = np.full(N, np.nan)
            s_full[valid] = score
            evt = cand[t] & valid
            is_long = evt & (s_full >= hi)
            is_short = evt & (s_full <= lo) & ~is_long
            side[t, is_long] = 1.0
            side[t, is_short] = -1.0
    return side


def events(data, params):
    side = _event_sides(data, params)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (side != 0.0) & tradable
    side = np.where(mask, side, 0.0)
    return mask, side
