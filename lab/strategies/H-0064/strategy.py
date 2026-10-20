"""H-0064 Pre-announcement liquidity-provision reversal (LIQ-500).

After the close of session t (rows 0..t only), a LIQ-500 stock whose predicted earnings-release
session P (first session on or after a past 8-K Item 2.02 reaction day + 364 calendar days) is
exactly `lead_sessions` = 3 sessions ahead, with no reaction day in the last 20 sessions, and whose
compounded total return over the last `formation_days` sessions is in the bottom (top)
`leg_quantile` of the LIQ-500 cross-section, opens a fixed slot of +0.05 (-0.05) at the open of
t+1, closed at the open of t+1+hold_days. Gross is scaled down to 1.0 on days it would exceed 1.0.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"formation_days": 3, "hold_days": 5, "leg_quantile": 0.2}

LEAD_SESSIONS = 3        # card: t = P - 3 sessions (fixed in the card, not a grid parameter)
LOOKBACK_DAYS = 364      # card: 52 weeks, same weekday
RECENT_SESSIONS = 20     # card: no reaction day in the 20 sessions up to and including t
SLOT_WEIGHT = 0.05       # card: each slot is 5 % of equity
MAX_GROSS = 1.0


def _rolling_count(x, n):
    """Number of True values in rows t-n+1..t (point-in-time, uses rows <= t only)."""
    c = np.cumsum(x.astype(np.int64), axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def predicted_candidates(data):
    """(T, N) bool: row t is LEAD_SESSIONS sessions before a predicted release session of stock j.

    Future sessions are not knowable from the data, so the sessions after t are counted on the
    weekday calendar (np.busday_offset, no exchange holidays): key_t = 3rd weekday after date t.
    Each past reaction day L gives target D = first weekday on/after date(L) + 364 days; the
    candidate row is the first row t with key_t >= D (so a target is never skipped when a holiday
    lies between consecutive rows). If D itself is a holiday this is exactly P-3; if a holiday lies
    strictly between t and P the row is P-2 (one session late).
    """
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


def formation_return(data, k):
    """(T, N) compounded total return from the close of t-k to the close of t; NaN without data."""
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
    nbad = _rolling_count(bad, k + 1)          # closes t-k..t must all exist
    r[nbad > 0] = np.nan
    r[:k] = np.nan
    r[~np.isfinite(r)] = np.nan
    return r


def events(data, params):
    """Loser (+1) and winner (-1) events after the close of each row: (mask, side)."""
    k = int(params["formation_days"])
    q = float(params["leg_quantile"])
    universe = data.universe
    if universe is None:
        raise ValueError("H-0064 needs the point-in-time LIQ-500 universe (data.universe)")
    universe = np.asarray(universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)

    cand = predicted_candidates(data)
    recent = _rolling_count(ea == 1.0, RECENT_SESSIONS) > 0
    r = formation_return(data, k)
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


def target_weights(data, params):
    hold = int(params["hold_days"])
    mask, side = events(data, params)
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = mask.shape
    w = np.zeros((T, N), dtype=np.float64)

    slot_side = np.zeros(N)                 # +1 / -1 for an open slot, 0 if none
    slot_end = np.full(N, -1, dtype=np.int64)   # row whose decision closes the slot
    for t in range(T):
        # scheduled exit: decided at t = entry row + hold_days (trades at the open of entry+1+hold)
        expired = (slot_side != 0) & (slot_end <= t)
        # forced exit: left the universe, not tradable or not listed (engine refuses non-members)
        forced = (slot_side != 0) & ~(universe[t] & tradable[t] & listed[t])
        close_now = expired | forced
        slot_side[close_now] = 0.0
        slot_end[close_now] = -1
        # new slots for events on stocks without an open slot
        new = mask[t] & (slot_side == 0)
        if new.any():
            slot_side[new] = side[t, new]
            slot_end[new] = t + hold
        row = slot_side * SLOT_WEIGHT
        g = np.abs(row).sum()
        if g > MAX_GROSS:
            row = row * (MAX_GROSS / g) * (1.0 - 1e-12)
        w[t] = row
    return w
