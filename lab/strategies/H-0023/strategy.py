"""H-0023 Earnings-announcement premium on real 8-K Item 2.02 dates.

LIQ-500, dollar-neutral. Long a window of `window_days` sessions centred on the predicted
announcement session p = first session on or after (last year's primary Item 2.02 reaction
session + 364 calendar days); short the other tradable LIQ-500 members.

Calendar approximation: future exchange holidays are not knowable point-in-time, so sessions
are counted with numpy weekday arithmetic (np.busday_offset, Mon-Fri, no holidays). The event
fires on the first data row on or after the weekday that lies h weekdays before p (so if that
weekday is an exchange holiday the event fires on the next session), and only if that row is
still before p.
"""
import numpy as np

PARAMS = {"window_days": 5}

_PRIMARY_GAP_DAYS = 45     # card rule (1): no other reaction session in the 45 calendar days before a
_RECENT_DAYS = 45          # card rule (3): no reaction session in the 45 calendar days up to and including t
_YEAR_DAYS = 364           # card rule (2): 52 weeks, same weekday
_MISSING_CHECK_ROWS = 504  # input check only: an all-NaN panel this long means the dataset is missing


def _half_window(params):
    wd = int(params["window_days"])
    if wd < 1 or wd % 2 != 1:
        raise ValueError("window_days must be a positive odd integer")
    return wd, (wd + 1) // 2


def _inputs(data):
    dates = data.dates
    T = len(dates)
    N = len(data.instruments)
    E = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)   # raises if the dataset is missing
    if E.shape != (T, N):
        raise ValueError("ea_reaction_day has the wrong shape")
    # A panel truncated before the first filings is legitimately all NaN; a panel of two years or
    # more without a single known value means the dataset is missing (K-0010): raise.
    if T >= _MISSING_CHECK_ROWS and not np.isfinite(E).any():
        raise ValueError("sec_8k_item202 ea_reaction_day is missing (all NaN) for the whole panel")
    U = data.universe
    if U is None:
        raise ValueError("the card needs the LIQ-500 point-in-time universe")
    U = np.asarray(U, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ok = U & tradable
    return dates, E, U, ok


def _events(data, h):
    """(T, N) bool: after the close of row t, stock j starts its announcement window."""
    dates, E, U, ok = _inputs(data)
    T, N = E.shape
    event = np.zeros((T, N), dtype=bool)
    if T == 0:
        return event, U, ok
    flag = E > 0.5                                   # NaN -> False (no known filing)
    d_int = dates.astype(np.int64)
    d_rel = d_int - d_int[0] + 1000                  # positive day numbers
    big = np.int64(1 << 26)
    cols, rows = np.nonzero(flag.T)                  # sorted by stock, then by row (date)
    cols = cols.astype(np.int64)
    rows = rows.astype(np.int64)
    keys = cols * big + d_rel[rows]

    def count(c, lo, hi):
        # number of reaction sessions of stock c with day number in [lo, hi]
        return (np.searchsorted(keys, c * big + hi, side="right")
                - np.searchsorted(keys, c * big + lo, side="left"))

    if len(keys) == 0:
        return event, U, ok

    # (1) primary announcements: no other reaction session in the 45 calendar days before a
    da = d_rel[rows]
    primary = count(cols, da - _PRIMARY_GAP_DAYS, da - 1) == 0
    a_rows = rows[primary]
    a_cols = cols[primary]
    if len(a_rows) == 0:
        return event, U, ok

    # (2) predicted session p: first weekday on or after a + 364 calendar days
    p = (d_int[a_rows] + _YEAR_DAYS).astype(dates.dtype)
    p = np.busday_offset(p, 0, roll="forward")
    # (3) trigger t with t + h = p: the weekday h weekdays before p, or the next data row if it is a holiday
    d_star = np.busday_offset(p, -h, roll="forward")
    trig = np.searchsorted(dates, d_star, side="left")
    valid = trig < T
    trig_c = np.minimum(trig, T - 1)
    valid &= dates[trig_c] < p
    t_rows = trig[valid].astype(np.int64)
    t_cols = a_cols[valid]
    if len(t_rows) == 0:
        return event, U, ok
    # no Item 2.02 reaction session in the 45 calendar days up to and including t
    dt = d_rel[t_rows]
    not_yet = count(t_cols, dt - (_RECENT_DAYS - 1), dt) == 0
    # LIQ-500 member and tradable at t
    sel = not_yet & ok[t_rows, t_cols]
    event[t_rows[sel], t_cols[sel]] = True
    return event, U, ok


def target_weights(data, params):
    wd, h = _half_window(params)
    event, U, ok = _events(data, h)
    T, N = event.shape
    W = np.full((T, N), np.nan)
    if T == 0:
        return W

    # (4) open windows: rows t .. t + wd - 1 (bought at open t+1, sold at open t+wd+1);
    #     closed early (and not reopened) on the first row j is not a member or not tradable
    alive = event.copy()
    inwin = event.copy()
    for _ in range(1, wd):
        nxt = np.zeros_like(alive)
        nxt[1:] = alive[:-1] & ok[1:]
        alive = nxt
        inwin |= alive

    # (5) re-weight only on rows where the set of window names changes
    change = np.zeros(T, dtype=bool)
    change[0] = inwin[0].any()
    change[1:] = (inwin[1:] != inwin[:-1]).any(axis=1)
    idx = np.nonzero(change)[0]
    if len(idx):
        iw = inwin[idx]
        sh = ok[idx] & ~iw
        nl = iw.sum(axis=1).astype(np.float64)
        ns = sh.sum(axis=1).astype(np.float64)
        wl = np.where(nl > 0, np.minimum(0.5 / np.maximum(nl, 1.0), 0.05), 0.0)
        wl = np.where(ns > 0, wl, 0.0)               # no short leg possible -> stay in cash
        ws = nl * wl / np.maximum(ns, 1.0)
        W[idx] = np.where(iw, wl[:, None], np.where(sh, -ws[:, None], 0.0))

    # rows without a change keep positions (NaN), but never hold a stock outside the universe
    keep = ~change
    W[keep] = np.where(U[keep], np.nan, 0.0)
    return W
