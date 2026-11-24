"""H-0191 Post-split ex-date demand drift, LIQ-500 market-neutral event slots.

Event (after the close of t, rows <= t only): stock j is a LIQ-500 member and
tradable on t, close_u valid (finite, > 0) on t-1 and t, ret_co / ret_oc finite on t;
r = (1 + ret_co[t]) * (1 + ret_oc[t]) - 1, f = close_u[t-1] * (1 + r) / close_u[t];
event if |f / R - 1| <= 0.03 for a standard ratio R >= min_ratio. At most one event
per stock in any 21 sessions (a trigger less than 21 rows after the previous
accepted event is ignored).

Book: each event opens a slot on decision row t (filled at the open of t+1):
+0.25 on the split stock, -0.25 spread equally over the other LIQ-500 members
tradable on t that are not in an open long slot. The slot is held on decision rows
t .. t+hold_days-1 and removed on row t+hold_days (closed at that row's next open).
If the split stock is not a member or not tradable on a later row s inside the hold,
the slot is removed from row s on. Slots add up; rows with gross > 1 are scaled down
proportionally to gross 1. No slot = all zeros (flat).
"""
import numpy as np

PARAMS = {"hold_days": 5, "min_ratio": 1.5}

SPLIT_RATIOS = (1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
TOLERANCE = 0.03
GAP_SESSIONS = 21
LONG_WEIGHT = 0.25
HEDGE_WEIGHT = 0.25


def split_factor(data):
    """Implied split factor f (T, N); NaN where any input is missing or invalid."""
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    T, N = close.shape
    prev = np.full((T, N), np.nan)
    if T > 1:
        prev[1:] = close[:-1]
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = (np.isfinite(prev) & (prev > 0) & np.isfinite(close) & (close > 0)
             & np.isfinite(ret_co) & np.isfinite(ret_oc))
    f = np.full((T, N), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        f[valid] = prev[valid] * (1.0 + r[valid]) / close[valid]
    return f


def matched_ratio(f, min_ratio):
    """The standard ratio R >= min_ratio that f matches within 3 % (NaN if none)."""
    out = np.full(f.shape, np.nan)
    for R in SPLIT_RATIOS:
        if R < float(min_ratio) - 1e-12:
            continue
        with np.errstate(invalid="ignore"):
            hit = np.isfinite(f) & (np.abs(f / R - 1.0) <= TOLERANCE)
        out[hit & np.isnan(out)] = R
    return out


def split_events(data, params):
    """(T, N) bool: forward-split ex-date events after the close of t."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0191 needs the LIQ-500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    f = split_factor(data)
    ratio = matched_ratio(f, params["min_ratio"])
    cand = uni & tradable & np.isfinite(ratio)
    events = np.zeros(cand.shape, dtype=bool)
    rows, cols = np.nonzero(cand)
    order = np.lexsort((rows, cols))          # by column, then row
    last = {}
    for k in order:
        t, j = int(rows[k]), int(cols[k])
        p = last.get(j)
        if p is not None and t - p < GAP_SESSIONS:
            continue                          # inside 21 sessions of an earlier event: ignored
        events[t, j] = True
        last[j] = t
    return events


def target_weights(data, params):
    hold = int(params["hold_days"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0191 needs the LIQ-500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = uni.shape
    events = split_events(data, params)
    ok = uni & tradable                       # split stock still a member and tradable

    ev_t, ev_j = np.nonzero(events)           # sorted by row
    ends = np.empty(len(ev_t), dtype=np.int64)
    long_active = np.zeros((T, N), dtype=bool)
    for e in range(len(ev_t)):
        t, j = int(ev_t[e]), int(ev_j[e])
        end = min(t + hold, T)
        if t + 1 < end:
            bad = np.nonzero(~ok[t + 1:end, j])[0]
            if len(bad):
                end = t + 1 + int(bad[0])     # closed from the first row it drops out
        ends[e] = end
        long_active[t:end, j] = True

    W = np.zeros((T, N), dtype=np.float64)
    for e in range(len(ev_t)):
        t, j, end = int(ev_t[e]), int(ev_j[e]), int(ends[e])
        W[t:end, j] += LONG_WEIGHT
        hedge = ok[t] & ~long_active[t]
        hedge[j] = False
        k = int(hedge.sum())
        if k == 0:
            continue
        hcols = np.nonzero(hedge)[0]
        # constant slot weights; a hedge name that leaves LIQ-500 is dropped (members only)
        W[t:end, hcols] -= (HEDGE_WEIGHT / k) * uni[t:end][:, hcols]

    gross = np.abs(W).sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / np.maximum(gross, 1e-300), 1.0)
    W *= scale[:, None]
    W[~uni] = 0.0
    return W
