"""H-0061 mechanism test events.

After the close of the last XNYS session of each calendar week, SPY is marked when the latest usable default
spread DBAA - DAAA is in the top third of its trailing window_days valid observations (p >= 2/3, side +1) or in
the bottom third (p <= 1/3, side -1). Middle third, missing signal or fewer than 756 valid observations: no
event. IEF, TLT, SHY and AGG are never marked (peer basket). Same percentile rule as the strategy.
"""
import numpy as np

MIN_OBS = 756
MAX_AGE_SESSIONS = 5
DS_DECIMALS = 4
SERIES_AAA = "fred_moody_yields_1d.DAAA"
SERIES_BAA = "fred_moody_yields_1d.DBAA"


def _ymd_to_days(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    first = months.astype("datetime64[M]").astype("datetime64[D]")
    return first + (d - 1).astype("timedelta64[D]")


def _weekday(days):
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


def _observed(days):
    wd = _weekday(days)
    out = np.where(wd == 5, days - np.timedelta64(1, "D"), days)
    out = np.where(wd == 6, out + np.timedelta64(1, "D"), out)
    return out


def _easter(y):
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


def _nyse_holidays(years):
    y = np.asarray(years, dtype=np.int64)
    one = np.ones_like(y)
    hol = []
    ny = _ymd_to_days(y, one, one)
    ny_wd = _weekday(ny)
    ny_obs = np.where(ny_wd == 6, ny + np.timedelta64(1, "D"), ny)
    hol.append(ny_obs[ny_wd != 5])
    mlk = y >= 1998
    hol.append(_nth_weekday(y[mlk], one[mlk], 0, 3))
    hol.append(_nth_weekday(y, 2 * one, 0, 3))
    hol.append(_easter(y) - np.timedelta64(2, "D"))
    hol.append(_last_weekday(y, 5 * one, 0))
    jt = y >= 2022
    hol.append(_observed(_ymd_to_days(y[jt], 6 * one[jt], 19 * one[jt])))
    hol.append(_observed(_ymd_to_days(y, 7 * one, 4 * one)))
    hol.append(_nth_weekday(y, 9 * one, 0, 1))
    hol.append(_nth_weekday(y, 11 * one, 3, 4))
    hol.append(_observed(_ymd_to_days(y, 12 * one, 25 * one)))
    return np.unique(np.concatenate(hol).astype("datetime64[D]"))


def _week_end_mask(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)
    yrs = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    hol = _nyse_holidays(np.arange(yrs.min() - 1, yrs.max() + 2))
    nxt = np.busday_offset(dates, 1, roll="backward", holidays=hol)
    return (dates.astype(np.int64) + 3) // 7 != (nxt.astype(np.int64) + 3) // 7


def _spread_percentile(daaa, dbaa, rows, window_days):
    daaa = np.asarray(daaa, dtype=np.float64)
    dbaa = np.asarray(dbaa, dtype=np.float64)
    T = len(daaa)
    valid = np.isfinite(daaa) & np.isfinite(dbaa)
    ds = np.where(valid, np.round(dbaa - daaa, DS_DECIMALS), np.nan)
    vals = ds[valid]
    kcount = np.cumsum(valid)
    prev_valid = np.concatenate([[False], valid[:-1]])
    prev_aaa = np.concatenate([[np.nan], daaa[:-1]])
    prev_baa = np.concatenate([[np.nan], dbaa[:-1]])
    new_val = valid & (~prev_valid | (daaa != prev_aaa) | (dbaa != prev_baa))
    last_new = np.maximum.accumulate(np.where(new_val, np.arange(T), -1))
    out = np.full(len(rows), np.nan)
    W = int(window_days)
    for i, t in enumerate(rows):
        k = int(kcount[t])
        if k == 0 or last_new[t] < 0:
            continue
        if t - last_new[t] + 1 > MAX_AGE_SESSIONS:
            continue
        n = min(W, k)
        if n < MIN_OBS:
            continue
        win = vals[k - n:k]
        cur = win[-1]
        out[i] = (np.sum(win < cur) + 0.5 * np.sum(win == cur)) / n
    return out


def events(data, params):
    window_days = int(params["window_days"])
    T = len(data.dates)
    N = len(data.instruments)
    spy = data.col("SPY")
    daaa = np.asarray(data.series[SERIES_AAA], dtype=np.float64)
    dbaa = np.asarray(data.series[SERIES_BAA], dtype=np.float64)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    if T == 0:
        return mask, side
    rows = np.flatnonzero(_week_end_mask(data.dates))
    p = _spread_percentile(daaa, dbaa, rows, window_days)
    hi = np.isfinite(p) & (p >= 2.0 / 3.0)
    lo = np.isfinite(p) & (p <= 1.0 / 3.0)
    ok = np.asarray(data.tradable[rows, spy], dtype=bool)
    if data.universe is not None:
        ok &= np.asarray(data.universe[rows, spy], dtype=bool)
    mask[rows[hi & ok], spy] = True
    side[rows[hi & ok], spy] = 1.0
    mask[rows[lo & ok], spy] = True
    side[rows[lo & ok], spy] = -1.0
    return mask, side
