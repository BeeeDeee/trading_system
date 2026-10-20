"""H-0064 mechanism events: same event definition and side as strategy.py (copied, not imported)."""
import numpy as np
from lab.framework.api import total_return_index

LEAD_SESSIONS = 3
LOOKBACK_DAYS = 364
RECENT_SESSIONS = 20


def _rolling_count(x, n):
    c = np.cumsum(x.astype(np.int64), axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def _predicted_candidates(data):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    T, N = ea.shape
    cand = np.zeros((T, N), dtype=bool)
    if T == 0:
        return cand
    keys = np.busday_offset(dates, LEAD_SESSIONS, roll="forward")
    L_rows, L_cols = np.nonzero(ea == 1.0)
    if L_rows.size == 0:
        return cand
    shifted = (dates[L_rows].astype(np.int64) + LOOKBACK_DAYS).astype("datetime64[D]")
    targets = np.busday_offset(shifted, 0, roll="forward")
    t_fire = np.searchsorted(keys, targets, side="left")
    ok = (t_fire < T) & (t_fire > L_rows)
    cand[t_fire[ok], L_cols[ok]] = True
    return cand


def _formation_return(data, k):
    tri = np.asarray(total_return_index(data), dtype=np.float64)
    T, N = tri.shape
    r = np.full((T, N), np.nan)
    if k >= T:
        return r
    with np.errstate(divide="ignore", invalid="ignore"):
        r[k:] = tri[k:] / tri[:-k] - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    bad = ~(np.isfinite(close) & listed)
    nbad = _rolling_count(bad, k + 1)
    r[nbad > 0] = np.nan
    r[:k] = np.nan
    r[~np.isfinite(r)] = np.nan
    return r


def events(data, params):
    k = int(params["formation_days"])
    q = float(params["leg_quantile"])
    universe = data.universe
    if universe is None:
        raise ValueError("H-0064 needs the point-in-time LIQ-500 universe (data.universe)")
    universe = np.asarray(universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)

    cand = _predicted_candidates(data)
    recent = _rolling_count(ea == 1.0, RECENT_SESSIONS) > 0
    r = _formation_return(data, k)
    cand &= universe & tradable & ~recent & np.isfinite(r)

    T, N = r.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.nonzero(cand.any(axis=1))[0]:
        peers = universe[t] & np.isfinite(r[t])
        if not peers.any():
            continue
        x = r[t, peers]
        q_lo = np.quantile(x, q)
        q_hi = np.quantile(x, 1.0 - q)
        c = cand[t]
        lo = c & (r[t] <= q_lo)
        hi = c & (r[t] >= q_hi) & ~lo
        mask[t] = lo | hi
        side[t, lo] = 1.0
        side[t, hi] = -1.0
    return mask, side
