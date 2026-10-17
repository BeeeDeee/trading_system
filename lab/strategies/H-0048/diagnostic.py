"""H-0048 mechanism-test events.

After the close of the session t with rem(t) == window_sessions (rem = XNYS sessions after t
in the same calendar month, from a rule-based holiday calendar), TLT is marked +1 and IEF -1.
SPY, SHY, LQD, AGG, TIP are never marked (peer set). The calendar code is a copy of the
one in strategy.py (the diagnostic is loaded on its own).
"""
import numpy as np

JUNETEENTH_FIRST_YEAR = 2022


def days_from_civil(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    y = y - (m <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = np.where(m > 2, m - 3, m + 9)
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(z):
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    y = y + (m <= 2)
    return y, m, d


def weekday(z):
    return np.mod(np.asarray(z, dtype=np.int64) + 3, 7)


def _nth_weekday(y, m, wd, n):
    first = days_from_civil(y, m, 1)
    return first + np.mod(wd - weekday(first), 7) + 7 * (n - 1)


def _last_weekday(y, m, wd):
    ny = np.where(m == 12, y + 1, y)
    nm = np.where(m == 12, 1, m + 1)
    last = days_from_civil(ny, nm, 1) - 1
    return last - np.mod(weekday(last) - wd, 7)


def _easter(y):
    y = np.asarray(y, dtype=np.int64)
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
    l = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * mm + 114) // 31
    day = (h + l - 7 * mm + 114) % 31 + 1
    return days_from_civil(y, month, day)


def _observed(z):
    wd = weekday(z)
    return z - (wd == 5) + (wd == 6)


def xnys_holidays(years):
    years = np.asarray(years, dtype=np.int64)
    out = []
    ny = days_from_civil(years, 1, 1)
    ny_wd = weekday(ny)
    out.append(ny[ny_wd < 5])
    out.append((ny + 1)[ny_wd == 6])
    out.append(_nth_weekday(years, 1, 0, 3))
    out.append(_nth_weekday(years, 2, 0, 3))
    out.append(_easter(years) - 2)
    out.append(_last_weekday(years, 5, 0))
    jy = years[years >= JUNETEENTH_FIRST_YEAR]
    out.append(_observed(days_from_civil(jy, 6, 19)))
    out.append(_observed(days_from_civil(years, 7, 4)))
    out.append(_nth_weekday(years, 9, 0, 1))
    out.append(_nth_weekday(years, 11, 3, 4))
    out.append(_observed(days_from_civil(years, 12, 25)))
    return np.unique(np.concatenate(out))


def sessions_remaining_in_month(dates):
    z = np.asarray(dates).astype(np.int64)
    if z.size == 0:
        return np.zeros(0, dtype=np.int64)
    y, m, _ = civil_from_days(z)
    ny = np.where(m == 12, y + 1, y)
    nm = np.where(m == 12, 1, m + 1)
    next_first = days_from_civil(ny, nm, 1)
    lo = int(z.min())
    hi = int(next_first.max())
    grid = np.arange(lo, hi, dtype=np.int64)
    gy, _, _ = civil_from_days(grid)
    hol = xnys_holidays(np.arange(int(gy.min()), int(gy.max()) + 1))
    is_session = (weekday(grid) < 5) & ~np.isin(grid, hol)
    cum = np.cumsum(is_session.astype(np.int64))
    return cum[next_first - 1 - lo] - cum[z - lo]


def events(data, params):
    window = int(params["window_sessions"])
    T = len(data.dates)
    N = len(data.instruments)
    i_ief = data.col("IEF")
    i_tlt = data.col("TLT")

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    if T == 0:
        return mask, side

    rem = sessions_remaining_in_month(data.dates)
    day = rem == window
    ok = np.asarray(data.tradable, dtype=bool).copy()
    if data.universe is not None:
        ok &= np.asarray(data.universe, dtype=bool)

    mask[:, i_tlt] = day & ok[:, i_tlt]
    mask[:, i_ief] = day & ok[:, i_ief]
    side[:, i_tlt] = 1.0
    side[:, i_ief] = -1.0
    return mask, side
