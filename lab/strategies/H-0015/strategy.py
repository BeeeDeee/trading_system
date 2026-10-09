"""H-0015 Filing-anchored earnings-announcement premium (LIQ-500, dollar-neutral).

Step 1  filing days: sf1_arq_revenue or sf1_arq_netinc changes vs the previous row (NaN->value counts,
        NaN->NaN and value->NaN do not); filings < 40 rows after the last kept filing are dropped.
Step 2  reaction day s*: max dollar_volume over the 30 rows ending on F; valid only if
        dv[s*] >= min_spike * median(dv over the 63 rows ending at s*-1).
Step 3  predicted reaction day r: first weekday on/after date(s*) + 364 calendar days.
Step 4  event at t: r is lead_days trading days ahead of t (weekday arithmetic, see _event_matrix),
        stock is a LIQ-500 member and tradable at t.
Step 5  long every active event name for 5 trading days (decision rows t..t+4), weight
        min(0.05, 0.5/n_long); equal-weight short of all other tradable members with the same total.
"""
import numpy as np

PARAMS = {"lead_days": 3, "min_spike": 2.0}

HOLD_DAYS = 5          # decision rows t..t+4 hold the position (exit decided after the close of t+5)
FILING_GAP = 40        # trading days; closer filings are amendments, keep the first
SPIKE_WINDOW = 30      # trading days ending on F (F included)
MEDIAN_WINDOW = 63     # trading days ending the day before s*
YEAR_DAYS = 364        # same weekday one year later
MAX_NAME_W = 0.05
LEG_CAP = 0.5
GROSS_SAFETY = 1.0 - 1e-12   # keeps gross <= 1 under floating-point summation


def _filing_days(rev, ni):
    """Return (rows, cols) of kept quarterly filing days."""
    T, N = rev.shape
    changed = np.zeros((T, N), dtype=bool)
    if T > 1:
        for x in (rev, ni):
            cur = x[1:]
            prev = x[:-1]
            cur_ok = ~np.isnan(cur)
            with np.errstate(invalid="ignore"):
                diff = cur != prev
            changed[1:] |= cur_ok & (np.isnan(prev) | diff)
    jj, tt = np.nonzero(changed.T)          # sorted by stock, then by row
    jl = jj.tolist()
    tl = tt.tolist()
    keep = np.zeros(len(tl), dtype=bool)
    last_j = -1
    last_t = 0
    for k in range(len(tl)):
        j = jl[k]
        t = tl[k]
        if j != last_j or t - last_t >= FILING_GAP:
            keep[k] = True
            last_j = j
            last_t = t
    return tt[keep], jj[keep]


def _reaction_days(dv, F, J, min_spike):
    """For each filing (F, J): s* and whether it passes the volume-spike test."""
    K = len(F)
    if K == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    ar = np.arange(K)
    rows = F[:, None] + np.arange(-SPIKE_WINDOW + 1, 1)[None, :]
    inside = rows >= 0
    vals = dv[np.clip(rows, 0, None), J[:, None]].astype(np.float64)
    vals[~inside | np.isnan(vals)] = -np.inf
    am = np.argmax(vals, axis=1)              # first maximum on ties
    peak = vals[ar, am]
    s = rows[ar, am]
    has_peak = np.isfinite(peak)

    rows2 = s[:, None] + np.arange(-MEDIAN_WINDOW, 0)[None, :]
    inside2 = rows2 >= 0
    v2 = dv[np.clip(rows2, 0, None), J[:, None]].astype(np.float32)
    v2[~inside2] = np.nan
    v2.sort(axis=1)                           # NaN last
    n = np.sum(~np.isnan(v2), axis=1)
    lo = np.clip((n - 1) // 2, 0, MEDIAN_WINDOW - 1)
    hi = np.clip(n // 2, 0, MEDIAN_WINDOW - 1)
    med = 0.5 * (v2[ar, lo].astype(np.float64) + v2[ar, hi].astype(np.float64))
    has_med = n > 0
    with np.errstate(invalid="ignore"):
        spike = peak >= min_spike * np.where(has_med, med, np.inf)
    valid = has_peak & has_med & spike
    return s, valid


def _event_matrix(data, params):
    lead = int(params["lead_days"])
    min_spike = float(params["min_spike"])
    rev = np.asarray(data.extras["sf1_arq_revenue"])
    ni = np.asarray(data.extras["sf1_arq_netinc"])
    dv = np.asarray(data.dollar_volume)
    if data.universe is None:
        raise ValueError("H-0015 needs the LIQ-500 point-in-time universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = dv.shape

    F, J = _filing_days(rev, ni)
    s, valid = _reaction_days(dv, F, J, min_spike)
    F = F[valid]
    J = J[valid]
    s = s[valid]

    dates = np.asarray(data.dates).astype("datetime64[D]")
    d_int = dates.astype(np.int64)
    # a[t]: the date lead_days weekdays after row t (weekday approximation of t + lead_days trading days)
    a = np.busday_offset(dates, lead, roll="forward").astype(np.int64)
    target = (d_int[s] + YEAR_DAYS).astype("datetime64[D]")
    r = np.busday_offset(target, 0, roll="forward").astype(np.int64)
    # event row: the first row t with a[t] >= r (i.e. r <= t + lead weekdays and r > (t-1) + lead weekdays),
    # with r strictly after the date of t; with no holidays this is exactly r == a[t].
    te = np.searchsorted(a, r, side="left")
    ok = te < T
    te_c = np.minimum(te, T - 1)
    ok &= r > d_int[te_c]
    ok &= te_c >= F                          # filing known at the event row (always true, ~1 year ago)
    ev = np.zeros((T, N), dtype=bool)
    ev[te_c[ok], J[ok]] = True
    ev &= member & tradable
    return ev, member, tradable


def target_weights(data, params):
    ev, member, tradable = _event_matrix(data, params)
    T, N = ev.shape
    ok_hold = member & tradable
    active = ev.copy()
    run = ok_hold.copy()                      # run[u] = ok[u] & ok[u-1] & ... & ok[u-k]
    for k in range(1, HOLD_DAYS):
        if k >= T:
            break
        run[k:] &= ok_hold[: T - k]
        run[:k] = False
        active[k:] |= ev[: T - k] & run[k:]

    n_long = active.sum(axis=1)
    short_el = ok_hold & ~active
    n_short = short_el.sum(axis=1)
    go = (n_long > 0) & (n_short > 0)
    nl = np.maximum(n_long, 1).astype(np.float64)
    ns = np.maximum(n_short, 1).astype(np.float64)
    w_long = np.where(go, np.minimum(MAX_NAME_W, LEG_CAP / nl), 0.0)
    total = np.where(go, np.minimum(LEG_CAP, MAX_NAME_W * n_long), 0.0)
    w_short = np.where(go, total / ns, 0.0)
    w = active * w_long[:, None] - short_el * w_short[:, None]
    w = w.astype(np.float64) * GROSS_SAFETY
    return w
