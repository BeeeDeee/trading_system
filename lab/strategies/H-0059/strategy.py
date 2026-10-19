"""H-0059: R&D-increase underreaction, LIQ-500 long-only, monthly rebalance.

EVENT (after the close of session t, stock j a LIQ-500 member and tradable at t):
  new filing: sf1_art_rnd or sf1_art_assets differs from session t-1 (NaN -> number counts);
  R0, A0 at t, R1, A1 at t-252 (sessions), all finite, R1 > 0, A0 > 0, A1 > 0;
  R0/A0 >= 0.05; R0/R1 - 1 >= growth_min; (R0/A0)/(R1/A1) - 1 >= 0.05;
  no event for j in the previous 252 sessions.
BOOK: at each month-end close, active = event in the last hold_months*21 sessions (incl. that day)
  and member and tradable; weight 1/max(N, 20) each, rest cash. Between month-ends a held stock
  that is out of the universe or not tradable at a close is set to 0 (sold at the next open).
"""
import numpy as np

PARAMS = {"growth_min": 0.1, "hold_months": 12}

LAG = 252               # sessions back for the year-ago trailing figures (fixed by the card)
REFIRE = 252            # no event in the previous 252 sessions (fixed)
MIN_INTENSITY = 0.05    # R0/A0 >= 0.05 (fixed)
MIN_INT_GROWTH = 0.05   # (R0/A0)/(R1/A1) - 1 >= 0.05 (fixed)
SESSIONS_PER_MONTH = 21
MIN_SLOTS = 20          # weight 1/max(N, 20)


def _changed(x):
    """(T, N) bool: x[t] differs from x[t-1]; NaN -> number counts, number -> NaN does not."""
    fin = np.isfinite(x)
    out = np.zeros(x.shape, dtype=bool)
    if x.shape[0] > 1:
        prev_fin = fin[:-1]
        cur_fin = fin[1:]
        with np.errstate(invalid="ignore"):
            diff = x[1:] != x[:-1]
        out[1:] = cur_fin & (~prev_fin | (prev_fin & diff))
    # row 0: the previous session is unknown, and row 0 can never fire (needs t-252)
    return out


def _lag(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def compute_events(rnd, assets, eligible, growth_min):
    """(T, N) bool event matrix; rows <= t only."""
    rnd = np.asarray(rnd, dtype=np.float64)
    assets = np.asarray(assets, dtype=np.float64)
    T, N = rnd.shape
    new_filing = _changed(rnd) | _changed(assets)
    R0, A0 = rnd, assets
    R1, A1 = _lag(rnd, LAG), _lag(assets, LAG)
    with np.errstate(invalid="ignore", divide="ignore"):
        fin = np.isfinite(R0) & np.isfinite(A0) & np.isfinite(R1) & np.isfinite(A1)
        pos = fin & (R1 > 0) & (A0 > 0) & (A1 > 0)
        safe_A0 = np.where(pos, A0, 1.0)
        safe_A1 = np.where(pos, A1, 1.0)
        safe_R1 = np.where(pos, R1, 1.0)
        int0 = np.where(pos, R0, 0.0) / safe_A0
        int1 = safe_R1 / safe_A1
        cond_b = int0 >= MIN_INTENSITY
        cond_c = (np.where(pos, R0, 0.0) / safe_R1 - 1.0) >= growth_min
        cond_d = (int0 / int1 - 1.0) >= MIN_INT_GROWTH
    cand = new_filing & pos & cond_b & cond_c & cond_d & eligible
    # (e) at most one event per stock in any 252 sessions: sequential over candidate rows
    events = np.zeros((T, N), dtype=bool)
    last = np.full(N, -(10 ** 9), dtype=np.int64)
    for t in np.flatnonzero(cand.any(axis=1)):
        fire = cand[t] & ((t - last) > REFIRE)
        events[t] = fire
        last[fire] = t
    return events


def month_end_flags(dates):
    """(T,) bool decision days. A row is the last session of its month if the next weekday
    (np.busday_offset, exchange holidays ignored) falls in another month. If that weekday rule missed
    the month-end (the last weekday was a holiday), the first session of the new month decides instead."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    m = d.astype("datetime64[M]")
    is_me = nxt.astype("datetime64[M]") != m
    flags = is_me.copy()
    if T > 1:
        new_month = m[1:] != m[:-1]
        flags[1:] |= new_month & ~is_me[:-1]
    return flags


def build_weights(events, member, tradable, decision, hold_sessions):
    T, N = events.shape
    elig = member & tradable
    cs = np.zeros((T + 1, N), dtype=np.int64)
    cs[1:] = np.cumsum(events, axis=0)
    lo = np.maximum(np.arange(T) - hold_sessions + 1, 0)
    recent = (cs[1:] - cs[lo]) > 0          # event in rows t-H+1..t
    active = recent & elig
    n = active.sum(axis=1)
    w_dec = active / np.maximum(n, MIN_SLOTS)[:, None]
    w = np.full((T, N), np.nan, dtype=np.float64)   # NaN = no decision, keep positions
    w[decision] = w_dec[decision]
    # between month-ends: a held stock that is out of the universe or not tradable at a close -> 0
    idx = np.where(decision, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    has = idx >= 0
    held = np.zeros((T, N), dtype=bool)
    held[has] = w_dec[idx[has]] > 0
    exit_mask = held & ~elig & ~decision[:, None]
    w[exit_mask] = 0.0
    return w


def target_weights(data, params):
    growth_min = float(params["growth_min"])
    hold_sessions = int(params["hold_months"]) * SESSIONS_PER_MONTH
    rnd = np.asarray(data.extras["sf1_art_rnd"], dtype=np.float64)
    assets = np.asarray(data.extras["sf1_art_assets"], dtype=np.float64)
    if data.universe is None:
        raise ValueError("H-0059 needs the point-in-time LIQ-500 universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    events = compute_events(rnd, assets, member & tradable, growth_min)
    decision = month_end_flags(data.dates)
    return build_weights(events, member, tradable, decision, hold_sessions)
