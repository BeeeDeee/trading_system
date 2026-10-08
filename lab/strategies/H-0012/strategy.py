"""H-0012 Non-routine insider purchase drift (LIQ-500, market-neutral).

EVENT on row t for stock j: member[t, j] and tradable[t, j], ins_buy_n_<Q>d[t-1, j] == 0 and
ins_buy_n_91d[t, j] > 0 (NaN in either field -> no event). Long set L_t = stocks with an event on any of
the last H rows (t-H+1 .. t) that are tradable (and LIQ-500 members) on t, each weighted
min(0.05, 0.5 / |L_t|). Short leg: equal-weighted short of all other tradable members, total = -long total.
Flat when L_t is empty.
"""
import numpy as np

PARAMS = {"hold_days": 5, "quiet_window_days": 182}

QUIET_FIELDS = {91: "ins_buy_n_91d", 182: "ins_buy_n_182d", 365: "ins_buy_n_365d"}

MAX_NAME_WEIGHT = 0.05
LONG_LEG_TOTAL = 0.5


def insider_events(member, tradable, quiet, buy91):
    """(T, N) bool: first purchase filing usable on t after a quiet window ending t-1."""
    member = np.asarray(member, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    quiet = np.asarray(quiet, dtype=np.float64)
    buy91 = np.asarray(buy91, dtype=np.float64)
    T, N = buy91.shape
    quiet_prev = np.full((T, N), np.nan)
    if T > 1:
        quiet_prev[1:] = quiet[:-1]
    with np.errstate(invalid="ignore"):
        quiet_ok = quiet_prev == 0.0        # NaN -> False
        new_buy = buy91 > 0.0               # NaN -> False
    return member & tradable & quiet_ok & new_buy


def recent_any(events, hold_days):
    """(T, N) bool: an event on any of rows t-H+1 .. t."""
    H = int(hold_days)
    if H < 1:
        raise ValueError("hold_days must be >= 1")
    c = np.cumsum(events.astype(np.int64), axis=0)
    lagged = np.zeros_like(c)
    if c.shape[0] > H:
        lagged[H:] = c[:-H]
    return (c - lagged) > 0


def book(member, tradable, events, hold_days):
    member = np.asarray(member, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    eligible = member & tradable
    long_set = recent_any(events, hold_days) & eligible
    short_set = eligible & ~long_set
    n_long = long_set.sum(axis=1).astype(np.float64)
    n_short = short_set.sum(axis=1).astype(np.float64)
    T, N = long_set.shape
    w = np.zeros((T, N), dtype=np.float64)
    active = (n_long > 0) & (n_short > 0)   # dollar neutrality needs a short leg
    per_long = np.zeros(T)
    per_long[active] = np.minimum(MAX_NAME_WEIGHT, LONG_LEG_TOTAL / n_long[active])
    long_total = per_long * n_long
    per_short = np.zeros(T)
    per_short[active] = long_total[active] / n_short[active]
    w = np.where(long_set, per_long[:, None], w)
    w = np.where(short_set, -per_short[:, None], w)
    return w


def target_weights(data, params):
    H = int(params["hold_days"])
    Q = int(params["quiet_window_days"])
    if Q not in QUIET_FIELDS:
        raise ValueError("quiet_window_days must be one of 91, 182, 365")
    if data.universe is None:
        raise ValueError("card requires a LIQ-500 universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    quiet = data.extras[QUIET_FIELDS[Q]]
    buy91 = data.extras["ins_buy_n_91d"]
    ev = insider_events(member, tradable, quiet, buy91)
    return book(member, tradable, ev, H)
