"""H-0183 Term-premium duration timing on a 60/40 core.

Bond sleeve (40 %) split into H = cohort_months monthly cohorts; each cohort holds TLT if the
10y-3m slope (FRED T10Y3M, monthly mean) was above its trailing median_months median when the
cohort was formed, SHY if it was not, IEF while no state exists. SPY is always 60 %. LQD is never held.
"""
import numpy as np

PARAMS = {"cohort_months": 12, "median_months": 60}

SERIES_KEY = "fred_macro.T10Y3M"
BOOK = ("SPY", "IEF", "TLT", "SHY")


# ----------------------------------------------------------------------------- calendar
def _parts(d):
    """month id (months since the epoch), year, month, day of month, weekday (Monday = 0)."""
    m64 = d.astype("datetime64[M]")
    mid = m64.astype(np.int64)
    year = mid // 12 + 1970
    month = mid % 12 + 1
    dom = (d - m64.astype("datetime64[D]")).astype(np.int64) + 1
    wd = (d.astype(np.int64) + 3) % 7
    return mid, year, month, dom, wd


def _month_end_holiday(d):
    """NYSE holidays that can fall on the last weekday(s) of a month: Memorial Day (last Monday of
    May) and Good Friday (end of March). Other NYSE holidays never do."""
    _, year, month, dom, wd = _parts(d)
    memorial = (month == 5) & (wd == 0) & (dom >= 25)
    # Easter Sunday, anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    dd = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - dd - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * l_) // 451
    n = h + l_ - 7 * mm + 114
    em = n // 31
    ed = n % 31 + 1
    easter_from_mar = np.where(em == 3, ed, 31 + ed)        # day count from 1 March (Mar 1 = 1)
    cand = np.where(month == 3, dom, np.where(month == 4, 31 + dom, -100))
    good_friday = cand == (easter_from_mar - 2)
    return memorial | good_friday


def last_session_flag(dates):
    """True where the row's date is the last XNYS session of its month, from the date alone:
    next weekday (np.busday_offset), skipping Memorial Day and Good Friday, lies in another month."""
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size == 0:
        return np.zeros(0, dtype=bool)
    nb1 = np.busday_offset(d, 1, roll="forward")
    hol = _month_end_holiday(nb1)
    nb = np.where(hol, np.busday_offset(nb1, 1, roll="forward"), nb1)
    return nb.astype("datetime64[M]") != d.astype("datetime64[M]")


def decisions(dates):
    """Monthly decision rows: [(row, month_id)], one per calendar month, at the last session of the
    month. If no row of a month was flagged (unforeseen closure), the decision is made at the first
    row of the next month (one session late)."""
    d = np.asarray(dates).astype("datetime64[D]")
    mid = d.astype("datetime64[M]").astype(np.int64)
    flag = last_session_flag(d)
    out = []
    seen = set()
    for t in range(len(d)):
        if t > 0 and mid[t] != mid[t - 1] and int(mid[t - 1]) not in seen:
            out.append((t, int(mid[t - 1])))
            seen.add(int(mid[t - 1]))
        if flag[t] and int(mid[t]) not in seen:
            out.append((t, int(mid[t])))
            seen.add(int(mid[t]))
    return out


# ----------------------------------------------------------------------------- signal
def monthly_states(data, median_months):
    """Returns (decision list [(row, month_id)], dict month_id -> x_m, dict month_id -> S_m in {+1,-1,0})."""
    L = int(median_months)
    if L < 1:
        raise ValueError("median_months must be >= 1")
    d = np.asarray(data.dates).astype("datetime64[D]")
    T = len(d)
    s = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    if s.shape[0] != T:
        raise ValueError("T10Y3M series length does not match the view")
    mid = d.astype("datetime64[M]").astype(np.int64)
    # row r holds the last value dated strictly before dates[r] (clock next_morning); it is a value
    # dated in month(r) when the previous row is in the same month (rows 1.. of the month)
    contrib = np.zeros(T, dtype=bool)
    if T > 1:
        contrib[1:] = (mid[1:] == mid[:-1]) & np.isfinite(s[1:])
    dec = decisions(d)
    xs = {}
    states = {}
    prev = 0
    for t, m in dec:
        start = int(np.searchsorted(mid, m, side="left"))
        end = min(t, int(np.searchsorted(mid, m, side="right")) - 1)
        sel = contrib[start:end + 1]
        vals = s[start:end + 1][sel]
        x = float(np.mean(vals)) if vals.size > 0 else np.nan
        xs[m] = x
        S = prev
        if np.isfinite(x):
            win = np.array([xs.get(mm, np.nan) for mm in range(m - L + 1, m + 1)], dtype=float)
            if np.all(np.isfinite(win)):
                S = 1 if x > float(np.median(win)) else -1
        states[m] = S
        prev = S
    return dec, xs, states


# ----------------------------------------------------------------------------- book
def target_weights(data, params):
    H = int(params["cohort_months"])
    L = int(params["median_months"])
    if H < 1:
        raise ValueError("cohort_months must be >= 1")
    T = len(data.dates)
    N = len(data.instruments)
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")
    j_tlt = data.col("TLT")
    j_shy = data.col("SHY")
    data.col("LQD")  # in the universe as an unmarked peer only, never held

    # the book starts once SPY, IEF, TLT and SHY have all had a price
    close = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    w = np.full((T, N), np.nan)
    if T == 0:
        return w
    started = np.ones(T, dtype=bool)
    for j in (j_spy, j_ief, j_tlt, j_shy):
        started &= np.maximum.accumulate(listed[:, j] & np.isfinite(close[:, j]))

    dec, _, states = monthly_states(data, L)

    first_start = int(np.argmax(started)) if started.any() else T
    w[:first_start] = 0.0                       # before the book starts: no position
    for t, m in dec:
        row = np.zeros(N)
        if started[t]:
            slots = [states.get(mm, 0) for mm in range(m - H + 1, m + 1)]
            steep = sum(1 for v in slots if v == 1)
            flat = sum(1 for v in slots if v == -1)
            w_tlt = 0.4 * steep / H
            w_shy = 0.4 * flat / H
            w_ief = 0.4 * (H - steep - flat) / H
            row[j_spy] = 0.6
            row[j_tlt] = w_tlt
            row[j_shy] = w_shy
            row[j_ief] = w_ief
        w[t] = row
    return w
