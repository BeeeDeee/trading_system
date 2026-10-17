"""H-0048 Month-end Treasury index-extension tilt.

60/40 SPY/IEF book that holds its bond sleeve in TLT instead of IEF over the last
`window_sessions` XNYS sessions of each calendar month.

rem(t) = number of XNYS sessions after t in the same calendar month, computed from a
rule-based XNYS holiday calendar (integer civil-date arithmetic on the date of row t),
never from which data rows exist after t.
"""
import numpy as np

PARAMS = {"window_sessions": 2}

W_EQ = 0.60
W_BOND = 0.40

# Juneteenth is an NYSE holiday from this year on.
JUNETEENTH_FIRST_YEAR = 2022


# ----------------------------------------------------------------------------------------
# integer civil-date arithmetic (days since 1970-01-01, proleptic Gregorian)
# ----------------------------------------------------------------------------------------
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
    """Monday = 0 ... Sunday = 6 (day 0 of the epoch was a Thursday)."""
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
    """Saturday -> Friday before, Sunday -> Monday after."""
    wd = weekday(z)
    return z - (wd == 5) + (wd == 6)


def xnys_holidays(years):
    """Regular NYSE full-day holidays (days since epoch) for the given years."""
    years = np.asarray(years, dtype=np.int64)
    out = []
    # New Year's Day: Sunday -> Monday; Saturday -> not observed (NYSE rule)
    ny = days_from_civil(years, 1, 1)
    ny_wd = weekday(ny)
    out.append(ny[ny_wd < 5])
    out.append((ny + 1)[ny_wd == 6])
    out.append(_nth_weekday(years, 1, 0, 3))          # Martin Luther King Jr. Day
    out.append(_nth_weekday(years, 2, 0, 3))          # Washington's Birthday
    out.append(_easter(years) - 2)                    # Good Friday
    out.append(_last_weekday(years, 5, 0))            # Memorial Day
    jy = years[years >= JUNETEENTH_FIRST_YEAR]
    out.append(_observed(days_from_civil(jy, 6, 19)))  # Juneteenth
    out.append(_observed(days_from_civil(years, 7, 4)))   # Independence Day
    out.append(_nth_weekday(years, 9, 0, 1))          # Labor Day
    out.append(_nth_weekday(years, 11, 3, 4))         # Thanksgiving
    out.append(_observed(days_from_civil(years, 12, 25)))  # Christmas
    return np.unique(np.concatenate(out))


def sessions_remaining_in_month(dates):
    """rem(t): XNYS sessions strictly after date t that fall in t's calendar month.

    Uses only the date of each row and the rule-based holiday calendar.
    """
    z = np.asarray(dates).astype(np.int64)
    if z.size == 0:
        return np.zeros(0, dtype=np.int64)
    y, m, _ = civil_from_days(z)
    ny = np.where(m == 12, y + 1, y)
    nm = np.where(m == 12, 1, m + 1)
    next_first = days_from_civil(ny, nm, 1)          # first calendar day of the next month
    lo = int(z.min())
    hi = int(next_first.max())                       # exclusive
    grid = np.arange(lo, hi, dtype=np.int64)
    gy, _, _ = civil_from_days(grid)
    hol = xnys_holidays(np.arange(int(gy.min()), int(gy.max()) + 1))
    is_session = (weekday(grid) < 5) & ~np.isin(grid, hol)
    cum = np.cumsum(is_session.astype(np.int64))
    return cum[next_first - 1 - lo] - cum[z - lo]


# ----------------------------------------------------------------------------------------
def target_weights(data, params):
    window = int(params["window_sessions"])
    T = len(data.dates)
    N = len(data.instruments)
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")
    i_tlt = data.col("TLT")
    # SHY, LQD, AGG, TIP are never traded (peers for the mechanism test only).

    w = np.full((T, N), np.nan, dtype=float)
    if T == 0:
        return w

    rem = sessions_remaining_in_month(data.dates)
    tradable = np.asarray(data.tradable, dtype=bool)

    base = np.zeros(N)
    base[i_spy] = W_EQ
    base[i_ief] = W_BOND
    tilt = np.zeros(N)
    tilt[i_spy] = W_EQ
    tilt[i_tlt] = W_BOND

    # Exit / monthly rebalance to 60/40: after the close of the last session of the month.
    exit_rows = rem == 0
    # Entry: after the close of the session with rem == window. "If TLT has no bar at the
    # entry open the sleeve stays in IEF": the next open is unknown at t, so TLT having a
    # bar (tradable) on row t is the point-in-time proxy; otherwise no entry this month.
    entry_rows = (rem == window) & tradable[:, i_tlt]

    # Base state on the first row (unless it is itself a decision row).
    if not (exit_rows[0] or entry_rows[0]):
        w[0] = base
    w[exit_rows] = base
    w[entry_rows] = tilt
    return w
