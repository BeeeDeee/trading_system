"""H-0001 Turn-of-month equity timing: SPY inside the turn-of-month window, IEF otherwise.

Row t is decided after the close of row t and executed at the next open, so the weights of row t are the
target of the NEXT trading day n(t). Window membership of n(t) is computed from the date of row t alone:

* n(t) = first business day after date[t] on a rule-based NYSE holiday calendar (weekends + regular NYSE
  holidays computed from rules; unscheduled closures cannot be known in advance and are not modelled).
* Position from the month end: n(t) is day -pos_end with pos_end = business days in [n(t), end of month]
  on that calendar (the future part of the calendar, which is known in advance).
* Position from the month start: n(t) is day +pos_start; if n(t) is in the same month as row t this is
  (observed trading rows of the month up to t) + 1, otherwise 1 (the past part is observed exactly).
* In window <=> pos_end <= days_before or pos_start <= days_after  ->  100 % SPY, else 100 % IEF.
"""
import numpy as np

PARAMS = {"days_before": 2, "days_after": 3}

_EPOCH_YEAR = 1970  # datetime64 epoch, used to convert year numbers <-> datetime64[Y]


def _days(n):
    return np.asarray(n, dtype=np.int64).astype("timedelta64[D]")


def _months(n):
    return np.asarray(n, dtype=np.int64).astype("timedelta64[M]")


def _weekday(d):
    """Monday = 0 ... Sunday = 6 for a datetime64[D] array (day 0 of the epoch is a Thursday)."""
    return (d.astype(np.int64) + 3) % 7


def _ymd(years, month, day):
    """datetime64[D] array from integer years and scalar month/day (no date literals)."""
    y = (np.asarray(years, dtype=np.int64) - _EPOCH_YEAR).astype("datetime64[Y]")
    return (y.astype("datetime64[M]") + _months(int(month) - 1)).astype("datetime64[D]") + _days(int(day) - 1)


def _observed(d):
    """Fixed-date holiday observance: Saturday -> Friday before, Sunday -> Monday after."""
    wd = _weekday(d)
    return np.where(wd == 5, d - _days(1), np.where(wd == 6, d + _days(1), d))


def easter_sunday(years):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm), vectorized over integer years."""
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
    l = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * l) // 451
    x = h + l - 7 * mm + 114
    month = x // 31
    day = x % 31 + 1
    base = (y - _EPOCH_YEAR).astype("datetime64[Y]").astype("datetime64[M]")
    return (base + _months(month - 1)).astype("datetime64[D]") + _days(day - 1)


def nyse_holidays(first_year, last_year):
    """Regular NYSE full-day holidays for the given years, computed from the published rules.

    New Year (Sunday -> Monday; a Saturday New Year is not observed on Dec 31), MLK day, Presidents day,
    Good Friday, Memorial day, Independence day, Labor day, Thanksgiving, Christmas. Juneteenth (since 2022)
    is omitted: June 19 can never be among the first 4 or last 3 trading days of June, so it cannot change
    window membership. Unscheduled closures (weather, state funerals) are not knowable in advance.
    """
    years = np.arange(int(first_year), int(last_year) + 1, dtype=np.int64)
    parts = []
    nyd = _ymd(years, 1, 1)
    wd = _weekday(nyd)
    parts.append(nyd[wd < 5])
    parts.append((nyd + _days(1))[wd == 6])
    parts.append(np.busday_offset(_ymd(years, 1, 1), 2, roll="forward", weekmask="Mon"))   # MLK
    parts.append(np.busday_offset(_ymd(years, 2, 1), 2, roll="forward", weekmask="Mon"))   # Presidents
    parts.append(easter_sunday(years) - _days(2))                                                # Good Friday
    parts.append(np.busday_offset(_ymd(years, 5, 31), 0, roll="backward", weekmask="Mon"))  # Memorial
    parts.append(_observed(_ymd(years, 7, 4)))                                               # Independence
    parts.append(np.busday_offset(_ymd(years, 9, 1), 0, roll="forward", weekmask="Mon"))   # Labor
    parts.append(np.busday_offset(_ymd(years, 11, 1), 3, roll="forward", weekmask="Thu"))  # Thanksgiving
    parts.append(_observed(_ymd(years, 12, 25)))                                             # Christmas
    hol = np.unique(np.concatenate(parts).astype("datetime64[D]"))
    return hol[_weekday(hol) < 5]


def window_flags(dates, days_before, days_after):
    """For each row t: (in_window of the next trading day n(t), month of n(t), month of row t)."""
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    years = dates.astype("datetime64[Y]").astype(np.int64) + _EPOCH_YEAR
    hol = nyse_holidays(years.min(), years.max() + 1)
    nxt = np.busday_offset(dates + _days(1), 0, roll="forward", holidays=hol)
    month_n = nxt.astype("datetime64[M]")
    month_t = dates.astype("datetime64[M]")
    # days from n(t) to the end of its month, n(t) included -> n(t) is day -pos_end
    pos_end = np.busday_count(nxt, (month_n + _months(1)).astype("datetime64[D]"), holidays=hol)
    # observed trading rows of the current month up to and including row t
    idx = np.arange(T)
    new_month = np.ones(T, dtype=bool)
    new_month[1:] = month_t[1:] != month_t[:-1]
    rows_in_month = idx - np.maximum.accumulate(np.where(new_month, idx, 0)) + 1
    pos_start = np.where(month_n == month_t, rows_in_month + 1, 1)  # n(t) is day +pos_start
    in_window = (pos_end <= int(days_before)) | (pos_start <= int(days_after))
    return in_window, month_n, month_t


def target_weights(data, params):
    k = int(params["days_before"])
    m = int(params["days_after"])
    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)
    if T == 0:
        return w
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")

    in_window, month_n, month_t = window_flags(data.dates, k, m)

    # Start: first full month after both instruments have prices; before that hold nothing.
    close = np.asarray(data.close, dtype=float)
    priced = np.isfinite(close[:, i_spy]) & np.isfinite(close[:, i_ief])
    if not priced.any():
        return w
    fb = int(np.argmax(priced))
    idx = np.arange(T)
    active = (idx >= fb) & (month_n > month_t[fb])

    # Missing data: switch only when both legs can fill at the next open; otherwise keep the current
    # position and switch at the first later open where both are tradable (a window that has ended by
    # then is skipped because the target is IEF again).
    tradable = np.asarray(data.tradable, dtype=bool)
    decide = active & tradable[:, i_spy] & tradable[:, i_ief]
    last = np.maximum.accumulate(np.where(decide, idx, -1))
    held = last >= 0
    spy = np.zeros(T, dtype=bool)
    spy[held] = in_window[last[held]]
    w[held, i_spy] = np.where(spy[held], 1.0, 0.0)
    w[held, i_ief] = np.where(spy[held], 0.0, 1.0)
    return w
