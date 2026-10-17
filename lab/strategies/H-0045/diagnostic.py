"""H-0045 mechanism test: mark SPY at the close of the session before each FOMC-cycle week starts.

Same cycle-week definition as strategy.py (Step 2, primary offset 0): week(d+1) is computed from n[d] and
k(d) only. A week starts on session d+1 when week(d+1) is not none and differs from week(d). Side +1 for an
even starting week (0, 2, 4, 6), -1 for an odd one (1, 3, 5). Only SPY is ever marked.
"""
import numpy as np

SERIES_KEY = "macro_release_calendar_1d.next_fomc_sessions"


def _sessions_since_fomc(n):
    T = n.shape[0]
    is_fomc = np.zeros(T, dtype=bool)
    if T > 1:
        prev = n[:-1]
        is_fomc[1:] = np.isfinite(prev) & (prev == 1)
    rows = np.arange(T)
    last = np.maximum.accumulate(np.where(is_fomc, rows, -1))
    return np.where(last >= 0, rows - last, np.nan).astype(float)


def _week_from_k(k):
    w = np.full(k.shape, np.nan)
    ok = np.isfinite(k)
    kk = np.where(ok, k, -1.0)
    w[ok & (kk >= 0) & (kk <= 3)] = 0.0
    mid = ok & (kk >= 4) & (kk <= 33)
    w[mid] = 1.0 + np.floor((kk[mid] - 4.0) / 5.0)
    return w


def _cycle_week_ahead(n, h):
    n = np.asarray(n, dtype=float)
    k = _sessions_since_fomc(n)
    valid = np.isfinite(n)
    mm = np.where(valid, n, 0.0)
    passed = valid & (h >= mm)
    k_target = np.where(passed, h - mm, k + h)
    w = _week_from_k(k_target)
    pre = valid & (mm - h == 1) & np.isfinite(k)
    w[pre] = 0.0
    w[~valid] = np.nan
    return w


def events(data, params):
    T, N = len(data.dates), len(data.instruments)
    spy = data.col("SPY")
    for name in ("IEF", "TLT", "SHY", "LQD"):
        data.col(name)  # peers must exist (fail loudly)
    n = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    if n.shape[0] != T:
        raise ValueError("next_fomc_sessions length does not match the calendar")

    week_today = _cycle_week_ahead(n, 0)   # week(d)
    week_next = _cycle_week_ahead(n, 1)    # week(d+1), primary offset 0
    starts = np.isfinite(week_next) & ~(np.isfinite(week_today) & (week_today == week_next))

    ok = starts & np.asarray(data.tradable[:, spy], dtype=bool) & np.isfinite(np.asarray(data.close[:, spy], dtype=float))
    if data.universe is not None:
        ok &= np.asarray(data.universe[:, spy], dtype=bool)

    mask = np.zeros((T, N), dtype=bool)
    mask[:, spy] = ok
    side = np.ones((T, N), dtype=float)
    wk = np.where(np.isfinite(week_next), week_next, 0.0)
    side[:, spy] = np.where(np.mod(wk, 2) == 0, 1.0, -1.0)
    return mask, side
