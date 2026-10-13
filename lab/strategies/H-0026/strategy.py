"""H-0026 Overdue earnings release - LIQ-500 market-neutral.

Short stocks whose Item 2.02 earnings release is `delay_sessions` sessions past the expected date
E(A) = first session on/after (last year's same-quarter reaction session A) + 364 calendar days,
hedged with an equal-weight long in all other tradable LIQ-500 members, hold `hold_days` sessions.

Rows of data.dates are the XNYS sessions (sharadar_sep calendar); "sessions" are counted in rows.
"""
import numpy as np

PARAMS = {"delay_sessions": 3, "hold_days": 5}

ANNIV_DAYS = 364          # 52 weeks, keeps the weekday
QUIET_FROM = 25           # (i) no release (and no NaN) from E-25 through t
PREV_LO = 125             # (ii) at least one release in E-125 .. E-26
PREV_HI = 26
DEDUPE_SESSIONS = 40      # (iv) no other overdue event in the previous 40 sessions
SHORT_PER_NAME = 0.125    # fixed by the card (not a parameter)
MAX_SHORT_GROSS = 0.5


def _inputs(data):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    if data.universe is None:
        raise ValueError("H-0026 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    if ea.shape != uni.shape or ea.shape != trad.shape or ea.shape[0] != dates.shape[0]:
        raise ValueError("shape mismatch between ea_reaction_day, universe, tradable and dates")
    return ea, uni, trad, dates


def overdue_events(data, delay_sessions):
    """(T, N) bool: stock j becomes overdue after the close of row t (rows <= t only)."""
    ea, uni, trad, dates = _inputs(data)
    T, N = ea.shape
    d = int(delay_sessions)
    events = np.zeros((T, N), dtype=bool)
    if T == 0 or N == 0:
        return events

    is_release = ea == 1.0                 # NaN compares False
    not_zero = ~(ea == 0.0)                # 1, NaN (or anything else) breaks the quiet window
    cs_rel = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(is_release, axis=0, out=cs_rel[1:])
    cs_nz = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(not_zero, axis=0, out=cs_nz[1:])

    # anchors A: every reaction session of every stock
    a_row, a_col = np.nonzero(is_release)
    if a_row.size == 0:
        return events
    # E(A): first session on or after A + 364 calendar days (row index; == T if not yet in the data)
    e_row = np.searchsorted(dates, dates[a_row] + np.timedelta64(ANNIV_DAYS, "D"), side="left")
    t_row = e_row + d                      # E(A) is session 0
    keep = t_row < T
    a_col, e_row, t_row = a_col[keep], e_row[keep], t_row[keep]
    if t_row.size == 0:
        return events

    # (i) ea == 0 on every session E-25 .. t
    q_lo = np.maximum(e_row - QUIET_FROM, 0)
    quiet = (cs_nz[t_row + 1, a_col] - cs_nz[q_lo, a_col]) == 0
    # (ii) at least one release in E-125 .. E-26
    p_lo = np.maximum(e_row - PREV_LO, 0)
    p_hi = e_row - PREV_HI
    valid_prev = p_hi >= 0
    p_hi_c = np.maximum(p_hi, 0)
    prev = valid_prev & ((cs_rel[p_hi_c + 1, a_col] - cs_rel[p_lo, a_col]) > 0)
    # (iii) LIQ-500 member and tradable at t
    elig = uni[t_row, a_col] & trad[t_row, a_col]

    ok = quiet & prev & elig
    cand = np.zeros((T, N), dtype=bool)
    cand[t_row[ok], a_col[ok]] = True

    # (iv) dedupe: no other (recorded) overdue event of j in the previous 40 sessions
    c_row, c_col = np.nonzero(cand)        # row-major: time order
    last = np.full(N, -(10 ** 9), dtype=np.int64)
    for r, j in zip(c_row.tolist(), c_col.tolist()):
        if r - last[j] > DEDUPE_SESSIONS:
            events[r, j] = True
            last[j] = r
    return events


def target_weights(data, params):
    hold = int(params["hold_days"])
    d = int(params["delay_sessions"])
    ev = overdue_events(data, d)
    _, uni, trad, _ = _inputs(data)
    elig = uni & trad
    T, N = ev.shape

    W = np.full((T, N), np.nan)
    if T > 0:
        W[0] = 0.0                              # explicit flat start (book in cash until the first event)
    remaining = np.zeros(N, dtype=np.int64)    # rows left in S, counting the current row
    s_prev = np.zeros(N, dtype=bool)
    held = np.zeros(N, dtype=bool)              # non-zero target in the last full decision
    for r in range(T):
        remaining = np.maximum(remaining - 1, 0)
        remaining[~elig[r]] = 0                 # left the universe / not tradable: close at next open
        remaining[ev[r]] = hold                 # new overdue stock enters S after the close of r
        s_now = remaining > 0
        if (s_now != s_prev).any():
            w = np.zeros(N)
            ns = int(s_now.sum())
            if ns > 0:
                per = SHORT_PER_NAME if ns * SHORT_PER_NAME <= MAX_SHORT_GROSS else MAX_SHORT_GROSS / ns
                hedge = elig[r] & ~s_now
                nh = int(hedge.sum())
                if nh > 0:
                    w[s_now] = -per
                    w[hedge] = per * ns / nh
            W[r] = w
            held = w != 0
        else:
            drop = held & ~uni[r]               # hedge names that left the universe: sell only those
            if drop.any():
                W[r, drop] = 0.0
                held = held & ~drop
        s_prev = s_now
    return W
