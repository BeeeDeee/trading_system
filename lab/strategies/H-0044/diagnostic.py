"""H-0044 mechanism test events.

After the close of d = T - (entry_lead + 1) sessions (T - 5 with the primary entry_lead = 4; T = last XNYS
session of the calendar month), every currency ETF that the strategy trades that month (valid signal, at least
3 valid currencies, D_j != 0) is an event with side +1 if D_j > 0 and -1 if D_j < 0, the sign of the strategy's
position. SHY, SHV, BIL, SPY and the country ETFs are never marked. The calendar and signal code is a copy of
strategy.py (only numpy and lab.framework imports are allowed here).
"""
import numpy as np
from lab.framework.api import total_return_index

PAIRS = (
    ("FXE", "EZU"),
    ("FXY", "EWJ"),
    ("FXB", "EWU"),
    ("FXA", "EWA"),
    ("FXC", "EWC"),
    ("FXF", "EWL"),
)
US_EQUITY = "SPY"
MIN_CURRENCIES = 3


def _weekday(daynum):
    return (daynum + 3) % 7                                   # 0 = Monday ... 6 = Sunday


def _month_start_day(month_num):
    return np.asarray(month_num, dtype="int64").astype("datetime64[M]").astype("datetime64[D]").astype("int64")


def _nth_weekday(month_start, weekday, n):
    offset = (weekday - _weekday(month_start)) % 7
    return month_start + (offset + 7 * (n - 1))


def _last_weekday(next_month_start, weekday):
    last_day = next_month_start - 1
    return last_day - ((_weekday(last_day) - weekday) % 7)


def _observed(day):
    wd = _weekday(day)
    return np.where(wd == 5, day - 1, np.where(wd == 6, day + 1, day))


def _easter(years):
    a = years % 19
    b = years // 100
    c = years % 100
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
    return month, day


def nyse_holidays(dates):
    y0 = int(dates[0].astype("datetime64[Y]").astype("int64"))
    y1 = int(dates[-1].astype("datetime64[Y]").astype("int64"))
    yoff = np.arange(y0, y1 + 1, dtype="int64")
    years = yoff + 1970

    def month_start(m):
        return _month_start_day(yoff * 12 + (m - 1))

    hol = []
    jan1 = month_start(1)
    wd_jan1 = _weekday(jan1)
    newyear = np.where(wd_jan1 == 6, jan1 + 1, jan1)
    hol.append(newyear[wd_jan1 != 5])
    hol.append(_nth_weekday(month_start(1), 0, 3))
    hol.append(_nth_weekday(month_start(2), 0, 3))
    em, ed = _easter(years)
    easter = _month_start_day(yoff * 12 + (em - 1)) + (ed - 1)
    hol.append(easter - 2)
    hol.append(_last_weekday(month_start(6), 0))
    hol.append(_observed(month_start(7) + 3))
    hol.append(_nth_weekday(month_start(9), 0, 1))
    hol.append(_nth_weekday(month_start(11), 3, 4))
    hol.append(_observed(month_start(12) + 24))
    return np.unique(np.concatenate(hol)).astype("datetime64[D]")


def events(data, params):
    lead = int(params["entry_lead"])
    dates = data.dates.astype("datetime64[D]")
    T, N = len(dates), len(data.instruments)
    cur_cols = [data.col(c) for c, _ in PAIRS]
    cty_cols = [data.col(e) for _, e in PAIRS]
    us_col = data.col(US_EQUITY)

    holidays = nyse_holidays(dates)
    month_num = dates.astype("datetime64[M]").astype("int64")
    last_day = (_month_start_day(month_num + 1) - 1).astype("datetime64[D]")
    last_session = np.busday_offset(last_day, 0, roll="backward", holidays=holidays)
    is_session = np.is_busday(dates, holidays=holidays)
    n_before = np.busday_count(dates, last_session, holidays=holidays)
    decision = is_session & (n_before == lead + 1)

    months = dates.astype("datetime64[M]")
    new_month = np.ones(T, dtype=bool)
    new_month[1:] = months[1:] != months[:-1]
    first_row_of_month = np.maximum.accumulate(np.where(new_month, np.arange(T), 0))
    prev_end = first_row_of_month - 1

    tri = total_return_index(data)
    has_px = np.isfinite(data.close)

    mask = np.zeros((T, N), dtype=bool)
    side = np.ones((T, N))
    rows = np.nonzero(decision & (prev_end >= 0))[0]
    if rows.size == 0:
        return mask, side
    p = prev_end[rows]
    D = np.zeros((rows.size, len(PAIRS)))
    valid = np.zeros((rows.size, len(PAIRS)), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_us = tri[rows, us_col] / tri[p, us_col] - 1.0
        ok_us = has_px[rows, us_col] & has_px[p, us_col]
        for k, (cc, ec) in enumerate(zip(cur_cols, cty_cols)):
            f = tri[rows, cc] / tri[p, cc] - 1.0
            r_usd = tri[rows, ec] / tri[p, ec] - 1.0
            d_j = r_us - ((1.0 + r_usd) / (1.0 + f) - 1.0)
            v = ok_us & has_px[rows, cc] & has_px[p, cc] & has_px[rows, ec] & has_px[p, ec]
            v &= data.listed[rows, cc].astype(bool) & data.tradable[rows, cc].astype(bool)
            if data.universe is not None:
                v &= data.universe[rows, cc].astype(bool)
            v &= np.isfinite(d_j)
            valid[:, k] = v
            D[:, k] = np.where(v, d_j, 0.0)
    month_ok = valid.sum(axis=1) >= MIN_CURRENCIES
    for k, cc in enumerate(cur_cols):
        ev = valid[:, k] & month_ok & (D[:, k] != 0.0)
        mask[rows, cc] = ev
        side[rows, cc] = np.where(D[:, k] < 0.0, -1.0, 1.0)
    return mask, side
