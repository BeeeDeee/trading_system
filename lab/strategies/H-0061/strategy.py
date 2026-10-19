"""H-0061 Default-spread business-conditions tilt on a 60/40 SPY/IEF book.

Decision after the close of the last XNYS session of each calendar month:
    DS = DBAA - DAAA (latest usable values, series already lagged by the framework clock next_morning)
    p  = (# window obs strictly below DS + 0.5 x # equal) / # window obs
         window = the trailing window_days valid DS observations ending with the latest one (inclusive)
    w_SPY = 0.60 + tilt x (2p - 1), w_IEF = 1 - w_SPY, TLT/SHY/AGG = 0
Plain 60/40 if the signal is missing (no value or latest value more than 5 sessions old) or the window holds
fewer than 756 valid observations. All other rows are NaN (hold, no drift rebalancing).
"""
import numpy as np

PARAMS = {"tilt": 0.2, "window_days": 1260}

BASE_SPY = 0.60
MIN_OBS = 756          # fixed in the card (not a grid parameter)
MAX_AGE_SESSIONS = 5   # "more than 5 sessions old" -> missing
DS_DECIMALS = 4        # spreads are quoted to 0.01 pp; rounding removes float32 noise so equal spreads tie

SERIES_AAA = "fred_moody_yields_1d.DAAA"
SERIES_BAA = "fred_moody_yields_1d.DBAA"


# ----------------------------------------------------------------------------------------------- calendar
def _ymd_to_days(y, m, d):
    """int arrays year, month, day -> datetime64[D] (no datetime64 constructor, no date strings)."""
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    first = months.astype("datetime64[M]").astype("datetime64[D]")
    return first + (d - 1).astype("timedelta64[D]")


def _weekday(days):
    """Monday = 0 ... Sunday = 6 (day 0 of the epoch was a Thursday)."""
    return (days.astype(np.int64) + 3) % 7


def _nth_weekday(y, m, wd, n):
    first = _ymd_to_days(y, m, np.ones_like(y))
    off = (wd - _weekday(first)) % 7 + 7 * (n - 1)
    return first + off.astype("timedelta64[D]")


def _last_weekday(y, m, wd):
    nxt_m = m % 12 + 1
    nxt_y = y + (m == 12)
    last = _ymd_to_days(nxt_y, nxt_m, np.ones_like(y)) - np.timedelta64(1, "D")
    off = (_weekday(last) - wd) % 7
    return last - off.astype("timedelta64[D]")


def _observed(days, saturday_to_friday=True):
    wd = _weekday(days)
    out = days.copy()
    if saturday_to_friday:
        out = np.where(wd == 5, out - np.timedelta64(1, "D"), out)
    out = np.where(wd == 6, out + np.timedelta64(1, "D"), out)
    return out


def _easter(y):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
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
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return _ymd_to_days(y, month, day)


def nyse_holidays(years):
    """Regular NYSE full-day holiday rules for the given years (unscheduled closures are not known)."""
    y = np.asarray(years, dtype=np.int64)
    one = np.ones_like(y)
    hol = []
    # New Year: Sunday -> Monday; Saturday -> no holiday (NYSE does not close the Friday before)
    ny = _ymd_to_days(y, one, one)
    ny_wd = _weekday(ny)
    ny_obs = np.where(ny_wd == 6, ny + np.timedelta64(1, "D"), ny)
    hol.append(ny_obs[ny_wd != 5])
    mlk = y >= 1998
    hol.append(_nth_weekday(y[mlk], one[mlk], 0, 3))                       # MLK day, 3rd Monday of January
    hol.append(_nth_weekday(y, 2 * one, 0, 3))                              # Presidents day
    hol.append(_easter(y) - np.timedelta64(2, "D"))                         # Good Friday
    hol.append(_last_weekday(y, 5 * one, 0))                                # Memorial day
    jt = y >= 2022
    hol.append(_observed(_ymd_to_days(y[jt], 6 * one[jt], 19 * one[jt])))   # Juneteenth
    hol.append(_observed(_ymd_to_days(y, 7 * one, 4 * one)))                # Independence day
    hol.append(_nth_weekday(y, 9 * one, 0, 1))                              # Labor day
    hol.append(_nth_weekday(y, 11 * one, 3, 4))                             # Thanksgiving
    hol.append(_observed(_ymd_to_days(y, 12 * one, 25 * one)))              # Christmas
    out = np.concatenate(hol).astype("datetime64[D]")
    return np.unique(out)


def next_session(dates):
    """Next weekday that is not a regular NYSE holiday, computed from each row's own date."""
    dates = np.asarray(dates).astype("datetime64[D]")
    if len(dates) == 0:
        return dates
    yrs = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(yrs.min() - 1, yrs.max() + 2)
    hol = nyse_holidays(years)
    # roll backward: a row on a rule holiday (absent from real XNYS data) is treated as the previous session
    return np.busday_offset(dates, 1, roll="backward", holidays=hol)


def month_end_mask(dates):
    """True where row t is the last XNYS session of its calendar month (by rule, from the date of t)."""
    dates = np.asarray(dates).astype("datetime64[D]")
    nxt = next_session(dates)
    return dates.astype("datetime64[M]") != nxt.astype("datetime64[M]")


def week_end_mask(dates):
    """True where row t is the last XNYS session of its calendar (Mon-Sun) week."""
    dates = np.asarray(dates).astype("datetime64[D]")
    nxt = next_session(dates)
    wk = (dates.astype(np.int64) + 3) // 7       # week index with weeks starting Monday
    wk_n = (nxt.astype(np.int64) + 3) // 7
    return wk != wk_n


# ----------------------------------------------------------------------------------------------- signal
def spread_percentile(daaa, dbaa, rows, window_days, min_obs=MIN_OBS, max_age=MAX_AGE_SESSIONS):
    """Percentile p_t of the latest usable default spread within its trailing valid observations.

    daaa, dbaa: (T,) yields per session row, as delivered (row t = latest value known after the close of t).
    rows: row indices at which p is wanted. Returns (len(rows),) float, NaN where the signal is missing or
    the window has fewer than min_obs valid observations.
    """
    daaa = np.asarray(daaa, dtype=np.float64)
    dbaa = np.asarray(dbaa, dtype=np.float64)
    T = len(daaa)
    valid = np.isfinite(daaa) & np.isfinite(dbaa)
    ds = np.where(valid, np.round(dbaa - daaa, DS_DECIMALS), np.nan)
    vals = ds[valid]
    kcount = np.cumsum(valid)                       # number of valid observations in rows 0..t
    # staleness: the row where the current value first appeared (works whether the framework forward-fills
    # the series or leaves NaN on rows without a new value)
    prev_valid = np.concatenate([[False], valid[:-1]])
    prev_aaa = np.concatenate([[np.nan], daaa[:-1]])
    prev_baa = np.concatenate([[np.nan], dbaa[:-1]])
    new_val = valid & (~prev_valid | (daaa != prev_aaa) | (dbaa != prev_baa))
    last_new = np.maximum.accumulate(np.where(new_val, np.arange(T), -1)) if T else np.zeros(0, dtype=np.int64)

    out = np.full(len(rows), np.nan)
    W = int(window_days)
    for i, t in enumerate(rows):
        k = int(kcount[t])
        if k == 0 or last_new[t] < 0:
            continue
        # a value that first appears on row r was stamped on or before session r-1: its age at t is t-r+1
        if t - last_new[t] + 1 > max_age:
            continue
        n = min(W, k)
        if n < min_obs:
            continue
        win = vals[k - n:k]
        cur = win[-1]
        out[i] = (np.sum(win < cur) + 0.5 * np.sum(win == cur)) / n
    return out


def target_weights(data, params):
    tilt = float(params["tilt"])
    window_days = int(params["window_days"])
    T = len(data.dates)
    N = len(data.instruments)
    spy = data.col("SPY")
    ief = data.col("IEF")
    daaa = np.asarray(data.series[SERIES_AAA], dtype=np.float64)
    dbaa = np.asarray(data.series[SERIES_BAA], dtype=np.float64)

    w = np.full((T, N), np.nan)
    if T == 0:
        return w
    dec_rows = np.flatnonzero(month_end_mask(data.dates))
    p = spread_percentile(daaa, dbaa, dec_rows, window_days)
    w_spy = np.where(np.isfinite(p), BASE_SPY + tilt * (2.0 * p - 1.0), BASE_SPY)
    w[dec_rows, :] = 0.0                       # TLT, SHY, AGG always 0
    w[dec_rows, spy] = w_spy
    w[dec_rows, ief] = 1.0 - w_spy
    return w
