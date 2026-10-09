"""H-0016 Turn-of-month beta spread (LIQ-500, dollar-neutral).

Entry: after the close of the (PRE_DAYS+1)-th-to-last scheduled XNYS trading day of the month, long the top
beta quintile (+0.5 in total, equal weights) and short the bottom beta quintile (-0.5 in total), among stocks
in the universe, tradable on that day, with a valid trailing beta. Exit: after the close of the post_days-th
scheduled XNYS trading day of the new month, all weights 0. NaN (hold, drift) in between, flat outside.

The XNYS calendar is built from the exchange's published holiday rules (known in advance), never from the
data's own rows, so truncating the data cannot change which day is the k-th(-to-last) trading day.
"""
import numpy as np

PARAMS = {"beta_lookback": 252, "post_days": 3}

PRE_DAYS = 2            # fixed by the card (not a parameter): enter at the open of the 2nd-to-last day
MIN_OBS_SHARE = 0.8     # at least 0.8 * beta_lookback valid paired observations
QUANTILES = 5           # quintiles


# ----------------------------------------------------------------------------------------------- calendar
def _days(x):
    return np.asarray(x, dtype=np.int64).astype("timedelta64[D]")


def _months(x):
    return np.asarray(x, dtype=np.int64).astype("timedelta64[M]")


def _month_start(years, month):
    """First calendar day of `month` (1..12, may be 13 = January of the next year) of each integer year."""
    ys = (np.asarray(years, dtype=np.int64) - 1970).astype("datetime64[Y]").astype("datetime64[M]")
    return (ys + _months(month - 1)).astype("datetime64[D]")


def _weekday(d):
    """Monday = 0 ... Sunday = 6 (day number 0 of the epoch was a Thursday)."""
    return (d.astype(np.int64) + 3) % 7


def _nth_weekday(years, month, weekday, n):
    first = _month_start(years, month)
    return first + _days((weekday - _weekday(first)) % 7 + 7 * (n - 1))


def _last_weekday(years, month, weekday):
    last = _month_start(years, month + 1) - _days(1)
    return last - _days((_weekday(last) - weekday) % 7)


def _observed_sat_fri_sun_mon(d):
    wd = _weekday(d)
    return d + _days(np.where(wd == 5, -1, np.where(wd == 6, 1, 0)))


def _easter(years):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    y = np.asarray(years, dtype=np.int64)
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    ms = (y - 1970).astype("datetime64[Y]").astype("datetime64[M]") + _months(month - 1)
    return ms.astype("datetime64[D]") + _days(day - 1)


def xnys_holidays(y0, y1):
    """Scheduled NYSE full-day holidays for the integer years y0..y1 (current rule set, MLK day included).

    Unscheduled closures (the September 2001 attacks, state funerals, hurricane Sandy) are not included:
    they were not known in advance, so a point-in-time calendar cannot contain them.
    """
    years = np.arange(int(y0), int(y1) + 1, dtype=np.int64)
    out = []
    # New Year's Day: Sunday -> Monday; Saturday -> no holiday (NYSE does not close the Friday before)
    ny = _month_start(years, 1)
    wd = _weekday(ny)
    out.append(ny[wd < 5])
    out.append((ny + _days(1))[wd == 6])
    out.append(_nth_weekday(years, 1, 0, 3))          # Martin Luther King Jr. Day
    out.append(_nth_weekday(years, 2, 0, 3))          # Washington's Birthday
    out.append(_easter(years) - _days(2))             # Good Friday
    out.append(_last_weekday(years, 5, 0))            # Memorial Day
    jy = years[years >= 2022]                         # Juneteenth (NYSE holiday from the year 2022)
    if jy.size:
        out.append(_observed_sat_fri_sun_mon(_month_start(jy, 6) + _days(18)))
    out.append(_observed_sat_fri_sun_mon(_month_start(years, 7) + _days(3)))     # Independence Day
    out.append(_nth_weekday(years, 9, 0, 1))          # Labor Day
    out.append(_nth_weekday(years, 11, 3, 4))         # Thanksgiving
    out.append(_observed_sat_fri_sun_mon(_month_start(years, 12) + _days(24)))   # Christmas
    hol = np.concatenate(out).astype("datetime64[D]")
    hol = hol[np.is_busday(hol)]
    return np.unique(hol)


def xnys_busdaycalendar(dates):
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    return np.busdaycalendar(holidays=xnys_holidays(years.min() - 1, years.max() + 1))


def turn_of_month_flags(dates, pre_days, post_days):
    """Entry and exit rows from the date of each row alone and the scheduled XNYS calendar.

    entry[t]: row t is the scheduled (pre_days+1)-th-to-last trading day of its month (pre_days scheduled
              trading days remain after it in the month).
    exit[t]:  row t is the first row of its month on or after the scheduled post_days-th trading day.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    if T == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    cal = xnys_busdaycalendar(dates)
    mon = dates.astype("datetime64[M]")
    m_start = mon.astype("datetime64[D]")
    m_next = (mon + _months(1)).astype("datetime64[D]")
    nxt = dates + _days(1)
    sched = np.is_busday(dates, busdaycal=cal)
    remaining = np.busday_count(nxt, m_next, busdaycal=cal)      # scheduled days after t in its month
    kth = np.busday_count(m_start, nxt, busdaycal=cal)           # scheduled days up to t in its month
    entry = sched & (remaining == pre_days)
    reached = kth >= post_days
    prev_reached = np.zeros(T, bool)
    same_month = np.zeros(T, bool)
    prev_reached[1:] = reached[:-1]
    same_month[1:] = mon[1:] == mon[:-1]
    exit_ = reached & ~(same_month & prev_reached)
    return entry, exit_


# --------------------------------------------------------------------------------------------------- beta
def quintile_sides(data, beta_lookback, rows):
    """For each decision row t in `rows`: +1 top beta quintile, -1 bottom quintile, 0 otherwise.

    r[t,j] = (1+ret_co)(1+ret_oc)-1 where the stock has a bar (listed, tradable, finite); m[t] = equal-weighted
    mean of r over universe members with a valid r. Beta = OLS slope of r[.,j] on m over the last beta_lookback
    rows ending at t inclusive, with at least 0.8*beta_lookback valid pairs.
    """
    U = data.universe
    if U is None:
        raise ValueError("H-0016 needs the liq_n point-in-time universe (data.universe is None)")
    U = np.asarray(U, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    r = np.array(data.ret_co, dtype=np.float64)          # copy
    T, N = r.shape
    L = int(beta_lookback)
    min_obs = MIN_OBS_SHARE * L

    r += 1.0
    r *= 1.0 + np.asarray(data.ret_oc, dtype=np.float64)
    r -= 1.0
    valid = listed & tradable & np.isfinite(r)
    r[~valid] = 0.0
    mv = valid & U
    cnt = mv.sum(axis=1)
    m = np.full(T, np.nan)
    ok = cnt > 0
    m[ok] = np.sum(r, axis=1, where=mv)[ok] / cnt[ok]
    m_ok = np.isfinite(m)
    m0 = np.where(m_ok, m, 0.0)

    rows = np.asarray(rows, dtype=np.int64)
    S = np.zeros((rows.shape[0], N), dtype=np.int8)
    for i, t in enumerate(rows):
        cand = np.flatnonzero(U[t] & tradable[t] & listed[t])
        if cand.size == 0:
            continue
        lo = max(0, t - L + 1)
        v = valid[lo:t + 1][:, cand] & m_ok[lo:t + 1, None]
        y = r[lo:t + 1][:, cand]
        x = m0[lo:t + 1, None]
        vf = v.astype(np.float64)
        n = vf.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (vf * x).sum(axis=0) / n
            my = (vf * y).sum(axis=0) / n
            dx = (x - mx) * vf
            dy = (y - my) * vf
            sxx = (dx * dx).sum(axis=0)
            sxy = (dx * dy).sum(axis=0)
            beta = sxy / sxx
        good = (n >= min_obs) & (sxx > 0) & np.isfinite(beta)
        idx = cand[good]
        b = beta[good]
        k = idx.size // QUANTILES
        if k == 0:
            continue
        order = np.argsort(b, kind="stable")      # ascending beta; ties by column order
        S[i, idx[order[-k:]]] = 1
        S[i, idx[order[:k]]] = -1
    return S


# ----------------------------------------------------------------------------------------------- strategy
def target_weights(data, params):
    beta_lookback = int(params["beta_lookback"])
    post_days = int(params["post_days"])
    T = len(data.dates)
    N = len(data.instruments)
    entry, exit_ = turn_of_month_flags(data.dates, PRE_DAYS, post_days)
    W = np.full((T, N), np.nan)
    W[exit_] = 0.0
    rows = np.flatnonzero(entry)
    S = quintile_sides(data, beta_lookback, rows)
    for i, t in enumerate(rows):
        w = np.zeros(N)
        longs = S[i] > 0
        shorts = S[i] < 0
        nl = int(longs.sum())
        ns = int(shorts.sum())
        if nl > 0 and ns > 0:
            w[longs] = 0.5 / nl
            w[shorts] = -0.5 / ns
        W[t] = w
    return W
