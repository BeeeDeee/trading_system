"""H-0045: FOMC-cycle equity tilt on a 60/40 SPY/IEF book.

Cycle weeks (Cieslak, Morse, Vissing-Jorgensen) are computed from the pre-announced FOMC schedule
series macro_release_calendar_1d.next_fomc_sessions (n[d] = XNYS sessions from d to the next scheduled
statement day strictly after d, known at the close of d). Rows of the data are XNYS sessions.
"""
import numpy as np

PARAMS = {"offset_sessions": 0, "tilt": 0.3}

SERIES_KEY = "macro_release_calendar_1d.next_fomc_sessions"


def _check_coverage(n):
    """Fail loudly on a missing value after the first valid one (an internal gap).

    Trailing NaNs (rows after the last published schedule value) are not an error here: the
    strategy keeps its positions there (NaN decision rows)."""
    valid = np.isfinite(n)
    if not valid.any():
        return valid
    idx = np.flatnonzero(valid)
    first, last = idx[0], idx[-1]
    if not valid[first:last + 1].all():
        bad = first + int(np.flatnonzero(~valid[first:last + 1])[0])
        raise ValueError(f"next_fomc_sessions missing at row {bad} after the first valid value")
    return valid


def sessions_since_fomc(n):
    """k(d): sessions since the most recent FOMC statement day (0 on the statement day), NaN before the
    first statement day. Session s is a statement day iff n[s-1] == 1 (past information only)."""
    T = n.shape[0]
    is_fomc = np.zeros(T, dtype=bool)
    if T > 1:
        prev = n[:-1]
        is_fomc[1:] = np.isfinite(prev) & (prev == 1)
    rows = np.arange(T)
    last = np.maximum.accumulate(np.where(is_fomc, rows, -1))
    k = np.where(last >= 0, rows - last, np.nan).astype(float)
    return k


def week_from_k(k):
    """Step 2 week from k (excluding the 'session before a statement day' rule): NaN = week none."""
    w = np.full(k.shape, np.nan)
    ok = np.isfinite(k)
    kk = np.where(ok, k, -1.0)
    w[ok & (kk >= 0) & (kk <= 3)] = 0.0
    mid = ok & (kk >= 4) & (kk <= 33)
    w[mid] = 1.0 + np.floor((kk[mid] - 4.0) / 5.0)
    return w


def cycle_week_ahead(n, h):
    """week(d + h) for every row d, using only n[d] and k(d) (values known at the close of d).

    h >= 0 sessions ahead (h must be smaller than the gap between two statement days, true for h <= 2).
    Returns a float array, NaN = week none (also where n[d] is NaN)."""
    if h < 0:
        raise ValueError("h must be >= 0")
    n = np.asarray(n, dtype=float)
    k = sessions_since_fomc(n)
    m = n
    valid = np.isfinite(m)
    mm = np.where(valid, m, 0.0)
    # If the next statement day d+m is at or before d+h, k(d+h) = h - m; otherwise k(d+h) = k(d) + h
    # (NaN if no statement day has been seen yet: week none before the first statement day).
    passed = valid & (h >= mm)
    k_target = np.where(passed, h - mm, k + h)
    w = week_from_k(k_target)
    # d+h is the session before a statement day: n(d+h) = m - h == 1 -> week 0, but only once a statement
    # day has been seen (sessions before the first statement day in the data have week none).
    pre = valid & (mm - h == 1) & np.isfinite(k)
    w[pre] = 0.0
    w[~valid] = np.nan
    return w


def target_weights(data, params):
    T, N = len(data.dates), len(data.instruments)
    offset = int(params["offset_sessions"])
    tilt = float(params["tilt"])
    spy = data.col("SPY")
    ief = data.col("IEF")
    for name in ("TLT", "SHY", "LQD"):
        data.col(name)  # universe check: fail loudly if missing; never held

    n = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    if n.shape[0] != T:
        raise ValueError("next_fomc_sessions length does not match the calendar")
    valid = _check_coverage(n)

    # Step 3: the target decided at the close of d (for session d+1) uses week(d + 1 + offset).
    h = 1 + offset
    week = cycle_week_ahead(n, h)

    # Step 4: weights.
    spy_w = np.full(T, 0.6)
    ief_w = np.full(T, 0.4)
    is_week = np.isfinite(week)
    wk = np.where(is_week, week, -1.0)
    even = is_week & (np.mod(wk, 2) == 0)
    odd = is_week & (np.mod(wk, 2) == 1)
    spy_w[even] = 0.6 + tilt
    ief_w[even] = 0.4 - tilt
    spy_w[odd] = 0.6 - tilt
    ief_w[odd] = 0.4 + tilt

    w = np.zeros((T, N), dtype=float)
    w[:, spy] = spy_w
    w[:, ief] = ief_w

    # No decision (keep previous weights) where SPY or IEF has no bar, or where the schedule value is
    # missing after the last published value (trailing NaN). Leading NaN rows (before the series starts)
    # have week none -> 60/40.
    close = np.asarray(data.close, dtype=float)
    no_bar = ~np.isfinite(close[:, spy]) | ~np.isfinite(close[:, ief])
    seen = np.maximum.accumulate(valid) if T else valid
    trailing_missing = seen & ~valid
    w[no_bar | trailing_missing] = np.nan
    return w
