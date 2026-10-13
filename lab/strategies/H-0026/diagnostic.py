"""H-0026 mechanism test: overdue Item 2.02 release (same event rule as strategy.py, side -1)."""
import numpy as np

ANNIV_DAYS = 364          # 52 weeks, keeps the weekday
QUIET_FROM = 25           # (i) no release (and no NaN) from E-25 through t
PREV_LO = 125             # (ii) at least one release in E-125 .. E-26
PREV_HI = 26
DEDUPE_SESSIONS = 40      # (iv) no other overdue event in the previous 40 sessions


def _overdue(data, delay_sessions):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    if data.universe is None:
        raise ValueError("H-0026 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    T, N = ea.shape
    d = int(delay_sessions)
    events = np.zeros((T, N), dtype=bool)
    if T == 0 or N == 0:
        return events

    is_release = ea == 1.0
    not_zero = ~(ea == 0.0)
    cs_rel = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(is_release, axis=0, out=cs_rel[1:])
    cs_nz = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(not_zero, axis=0, out=cs_nz[1:])

    a_row, a_col = np.nonzero(is_release)
    if a_row.size == 0:
        return events
    e_row = np.searchsorted(dates, dates[a_row] + np.timedelta64(ANNIV_DAYS, "D"), side="left")
    t_row = e_row + d
    keep = t_row < T
    a_col, e_row, t_row = a_col[keep], e_row[keep], t_row[keep]
    if t_row.size == 0:
        return events

    q_lo = np.maximum(e_row - QUIET_FROM, 0)
    quiet = (cs_nz[t_row + 1, a_col] - cs_nz[q_lo, a_col]) == 0
    p_lo = np.maximum(e_row - PREV_LO, 0)
    p_hi = e_row - PREV_HI
    valid_prev = p_hi >= 0
    p_hi_c = np.maximum(p_hi, 0)
    prev = valid_prev & ((cs_rel[p_hi_c + 1, a_col] - cs_rel[p_lo, a_col]) > 0)
    elig = uni[t_row, a_col] & trad[t_row, a_col]

    ok = quiet & prev & elig
    cand = np.zeros((T, N), dtype=bool)
    cand[t_row[ok], a_col[ok]] = True

    c_row, c_col = np.nonzero(cand)
    last = np.full(N, -(10 ** 9), dtype=np.int64)
    for r, j in zip(c_row.tolist(), c_col.tolist()):
        if r - last[j] > DEDUPE_SESSIONS:
            events[r, j] = True
            last[j] = r
    return events


def events(data, params):
    mask = _overdue(data, params["delay_sessions"])
    side = np.full(mask.shape, -1.0)    # every event is on the short side
    return mask, side
