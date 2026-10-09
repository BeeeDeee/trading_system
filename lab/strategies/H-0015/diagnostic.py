"""H-0015 mechanism-test events: same event definition as strategy.py (code duplicated on purpose,
the diagnostic may only import numpy). Event at the close of t: the predicted reaction day
(first weekday on/after date(s*) + 364, s* = SF1-anchored, volume-validated reaction day one year
earlier) is lead_days trading days ahead (weekday arithmetic); member of LIQ-500 and tradable at t.
side = +1 (long leg)."""
import numpy as np

FILING_GAP = 40
SPIKE_WINDOW = 30
MEDIAN_WINDOW = 63
YEAR_DAYS = 364


def _filing_days(rev, ni):
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
    jj, tt = np.nonzero(changed.T)
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
    K = len(F)
    if K == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    ar = np.arange(K)
    rows = F[:, None] + np.arange(-SPIKE_WINDOW + 1, 1)[None, :]
    inside = rows >= 0
    vals = dv[np.clip(rows, 0, None), J[:, None]].astype(np.float64)
    vals[~inside | np.isnan(vals)] = -np.inf
    am = np.argmax(vals, axis=1)
    peak = vals[ar, am]
    s = rows[ar, am]
    has_peak = np.isfinite(peak)

    rows2 = s[:, None] + np.arange(-MEDIAN_WINDOW, 0)[None, :]
    inside2 = rows2 >= 0
    v2 = dv[np.clip(rows2, 0, None), J[:, None]].astype(np.float32)
    v2[~inside2] = np.nan
    v2.sort(axis=1)
    n = np.sum(~np.isnan(v2), axis=1)
    lo = np.clip((n - 1) // 2, 0, MEDIAN_WINDOW - 1)
    hi = np.clip(n // 2, 0, MEDIAN_WINDOW - 1)
    med = 0.5 * (v2[ar, lo].astype(np.float64) + v2[ar, hi].astype(np.float64))
    has_med = n > 0
    with np.errstate(invalid="ignore"):
        spike = peak >= min_spike * np.where(has_med, med, np.inf)
    valid = has_peak & has_med & spike
    return s, valid


def events(data, params):
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
    a = np.busday_offset(dates, lead, roll="forward").astype(np.int64)
    target = (d_int[s] + YEAR_DAYS).astype("datetime64[D]")
    r = np.busday_offset(target, 0, roll="forward").astype(np.int64)
    te = np.searchsorted(a, r, side="left")
    ok = te < T
    te_c = np.minimum(te, T - 1)
    ok &= r > d_int[te_c]
    ok &= te_c >= F
    mask = np.zeros((T, N), dtype=bool)
    mask[te_c[ok], J[ok]] = True
    mask &= member & tradable
    return mask, None
