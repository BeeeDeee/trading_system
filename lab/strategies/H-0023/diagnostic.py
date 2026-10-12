"""H-0023 mechanism diagnostic: the predicted-announcement event (same rule as the strategy).

After the close of row t, stock j (LIQ-500 member, tradable at t) is an event if its predicted
announcement session p (first weekday on or after the reaction session of its primary 8-K Item
2.02 filing + 364 calendar days) is h = (window_days + 1) / 2 sessions ahead (t + h = p), and it
has had no Item 2.02 reaction session in the 45 calendar days up to and including t. Side +1.
Sessions are counted with weekday arithmetic (np.busday_offset, no holidays); if the trigger
weekday is an exchange holiday the event fires on the next session, provided it is still before p.
"""
import numpy as np

_PRIMARY_GAP_DAYS = 45
_RECENT_DAYS = 45
_YEAR_DAYS = 364
_MISSING_CHECK_ROWS = 504


def events(data, params):
    wd = int(params["window_days"])
    if wd < 1 or wd % 2 != 1:
        raise ValueError("window_days must be a positive odd integer")
    h = (wd + 1) // 2

    dates = data.dates
    T = len(dates)
    N = len(data.instruments)
    E = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    if E.shape != (T, N):
        raise ValueError("ea_reaction_day has the wrong shape")
    if T >= _MISSING_CHECK_ROWS and not np.isfinite(E).any():
        raise ValueError("sec_8k_item202 ea_reaction_day is missing (all NaN) for the whole panel")
    U = data.universe
    if U is None:
        raise ValueError("the card needs the LIQ-500 point-in-time universe")
    ok = np.asarray(U, dtype=bool) & np.asarray(data.tradable, dtype=bool)

    mask = np.zeros((T, N), dtype=bool)
    if T == 0:
        return mask, None
    flag = E > 0.5
    d_int = dates.astype(np.int64)
    d_rel = d_int - d_int[0] + 1000
    big = np.int64(1 << 26)
    cols, rows = np.nonzero(flag.T)
    cols = cols.astype(np.int64)
    rows = rows.astype(np.int64)
    keys = cols * big + d_rel[rows]
    if len(keys) == 0:
        return mask, None

    def count(c, lo, hi):
        return (np.searchsorted(keys, c * big + hi, side="right")
                - np.searchsorted(keys, c * big + lo, side="left"))

    da = d_rel[rows]
    primary = count(cols, da - _PRIMARY_GAP_DAYS, da - 1) == 0
    a_rows = rows[primary]
    a_cols = cols[primary]
    if len(a_rows) == 0:
        return mask, None
    p = (d_int[a_rows] + _YEAR_DAYS).astype(dates.dtype)
    p = np.busday_offset(p, 0, roll="forward")
    d_star = np.busday_offset(p, -h, roll="forward")
    trig = np.searchsorted(dates, d_star, side="left")
    valid = trig < T
    trig_c = np.minimum(trig, T - 1)
    valid &= dates[trig_c] < p
    t_rows = trig[valid].astype(np.int64)
    t_cols = a_cols[valid]
    if len(t_rows) == 0:
        return mask, None
    dt = d_rel[t_rows]
    not_yet = count(t_cols, dt - (_RECENT_DAYS - 1), dt) == 0
    sel = not_yet & ok[t_rows, t_cols]
    mask[t_rows[sel], t_cols[sel]] = True
    return mask, None
