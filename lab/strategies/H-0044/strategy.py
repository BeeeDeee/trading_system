"""H-0044 Month-end FX hedge-rebalancing flow.

For each calendar month: T = last XNYS session of the month (weekdays minus NYSE holidays computed by rule,
no data rows from the future). Decision after the close of d = T - (entry_lead + 1) sessions; for each currency
ETF j: D_j = R_US - R_j_loc with month-to-date total returns from the last data row of the previous month to d,
R_j_loc = (1 + R_country_usd) / (1 + F_j) - 1. Weights w_j = D_j / sum_k |D_k| over valid currencies (>= 3,
else flat). Held unchanged (NaN rows) until the close of T, where all weights go to zero (exit at the open of
T + 1). Flat on all other rows. SHY, SHV, BIL are never traded; SPY and the country ETFs are signal only.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"entry_lead": 4}

# currency ETF -> country equity ETF (local market of the currency)
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


# Date arithmetic below is done on int64 day numbers (days since 1970-01-01, a Thursday) and int64 month
# numbers (months since 1970-01), converted to datetime64 only at the end.

def _weekday(daynum):
    return (daynum + 3) % 7                                   # 0 = Monday ... 6 = Sunday


def _month_start_day(month_num):
    """int64 month numbers -> int64 day number of the first day of that month."""
    return np.asarray(month_num, dtype="int64").astype("datetime64[M]").astype("datetime64[D]").astype("int64")


def _nth_weekday(month_start, weekday, n):
    """n-th (1-based) given weekday (0 = Monday) of the months starting at day number month_start."""
    offset = (weekday - _weekday(month_start)) % 7
    return month_start + (offset + 7 * (n - 1))


def _last_weekday(next_month_start, weekday):
    """Last given weekday of the month that ends the day before day number next_month_start."""
    last_day = next_month_start - 1
    return last_day - ((_weekday(last_day) - weekday) % 7)


def _observed(day):
    """US federal observance: Saturday -> Friday, Sunday -> Monday."""
    wd = _weekday(day)
    return np.where(wd == 5, day - 1, np.where(wd == 6, day + 1, day))


def _easter(years):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm): int years -> (month 1..12, day 1..31)."""
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
    """Regular NYSE full-day holidays for every calendar year spanned by `dates` (rules known in advance).

    New Year's Day (Sunday -> Monday; a Saturday New Year is not observed), MLK Day, Washington's Birthday,
    Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas (Saturday -> Friday,
    Sunday -> Monday). Juneteenth (from 2022) is omitted: it always falls before the last 7 sessions of June,
    so it can never change T or any session count used here. Unscheduled closures are not known in advance
    and are not included.
    """
    y0 = int(dates[0].astype("datetime64[Y]").astype("int64"))       # years since 1970
    y1 = int(dates[-1].astype("datetime64[Y]").astype("int64"))
    yoff = np.arange(y0, y1 + 1, dtype="int64")
    years = yoff + 1970

    def month_start(m):           # m = 1..12 -> int64 day numbers, one per year
        return _month_start_day(yoff * 12 + (m - 1))

    hol = []
    jan1 = month_start(1)
    wd_jan1 = _weekday(jan1)
    newyear = np.where(wd_jan1 == 6, jan1 + 1, jan1)
    hol.append(newyear[wd_jan1 != 5])
    hol.append(_nth_weekday(month_start(1), 0, 3))                    # MLK
    hol.append(_nth_weekday(month_start(2), 0, 3))                    # Washington's Birthday
    em, ed = _easter(years)
    easter = _month_start_day(yoff * 12 + (em - 1)) + (ed - 1)
    hol.append(easter - 2)                                            # Good Friday
    hol.append(_last_weekday(month_start(6), 0))                      # Memorial Day
    hol.append(_observed(month_start(7) + 3))                         # Independence Day
    hol.append(_nth_weekday(month_start(9), 0, 1))                    # Labor Day
    hol.append(_nth_weekday(month_start(11), 3, 4))                   # Thanksgiving
    hol.append(_observed(month_start(12) + 24))                       # Christmas
    return np.unique(np.concatenate(hol)).astype("datetime64[D]")


def session_schedule(dates, holidays):
    """For each row: is it an XNYS session, and the number of sessions in [date, T) with T = last session of
    the row's calendar month (0 on T itself)."""
    dates = dates.astype("datetime64[D]")
    month_num = dates.astype("datetime64[M]").astype("int64")
    last_day = (_month_start_day(month_num + 1) - 1).astype("datetime64[D]")
    last_session = np.busday_offset(last_day, 0, roll="backward", holidays=holidays)
    is_session = np.is_busday(dates, holidays=holidays)
    n_before_t = np.busday_count(dates, last_session, holidays=holidays)
    return is_session, n_before_t


def compute_signal(data, params):
    """Decision rows and the signal D (T, 6) with validity mask, for the currency ETFs in PAIRS order."""
    lead = int(params["entry_lead"])
    dates = data.dates.astype("datetime64[D]")
    T = len(dates)
    cur_cols = [data.col(c) for c, _ in PAIRS]
    cty_cols = [data.col(e) for _, e in PAIRS]
    us_col = data.col(US_EQUITY)

    holidays = nyse_holidays(dates)
    is_session, n_before = session_schedule(dates, holidays)
    decision = is_session & (n_before == lead + 1)
    hold = is_session & (n_before >= 1) & (n_before <= lead)
    # every other session (including T itself, n_before == 0) is a flat row: zero weights at the close of T
    # close all positions at the open of T + 1

    # last data row of the previous calendar month (point-in-time: rows <= t only)
    months = dates.astype("datetime64[M]")
    new_month = np.ones(T, dtype=bool)
    new_month[1:] = months[1:] != months[:-1]
    first_row_of_month = np.maximum.accumulate(np.where(new_month, np.arange(T), 0))
    prev_end = first_row_of_month - 1                                 # -1 if the month is the first in the data

    tri = total_return_index(data)
    close = data.close
    has_px = np.isfinite(close)

    D = np.full((T, len(PAIRS)), np.nan)
    valid = np.zeros((T, len(PAIRS)), dtype=bool)
    rows = np.nonzero(decision & (prev_end >= 0))[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        _fill_signal(data, rows, prev_end, tri, has_px, us_col, cur_cols, cty_cols, D, valid)
    return is_session, decision, hold, D, valid, cur_cols


def _fill_signal(data, rows, prev_end, tri, has_px, us_col, cur_cols, cty_cols, D, valid):
    if rows.size:
        p = prev_end[rows]

        def mtd(col):
            return tri[rows, col] / tri[p, col] - 1.0

        def ok(col):
            return has_px[rows, col] & has_px[p, col]

        r_us = mtd(us_col)
        ok_us = ok(us_col)
        for k, (cc, ec) in enumerate(zip(cur_cols, cty_cols)):
            f = mtd(cc)
            r_usd = mtd(ec)
            r_loc = (1.0 + r_usd) / (1.0 + f) - 1.0
            d_j = r_us - r_loc
            v = ok_us & ok(cc) & ok(ec)
            # tradability at d + 1 is not knowable after the close of d; proxy: listed and tradable at d
            v &= data.listed[rows, cc].astype(bool) & data.tradable[rows, cc].astype(bool)
            if data.universe is not None:
                v &= data.universe[rows, cc].astype(bool)
            v &= np.isfinite(d_j)
            D[rows, k] = np.where(v, d_j, np.nan)
            valid[rows, k] = v


def target_weights(data, params):
    T, N = len(data.dates), len(data.instruments)
    is_session, decision, hold, D, valid, cur_cols = compute_signal(data, params)

    w = np.zeros((T, N))
    n_valid = valid.sum(axis=1)
    Dz = np.where(valid, D, 0.0)
    denom = np.abs(Dz).sum(axis=1)
    trade = decision & (n_valid >= MIN_CURRENCIES) & (denom > 0)
    safe = np.where(denom > 0, denom, 1.0)
    for k, cc in enumerate(cur_cols):
        w[:, cc] = np.where(trade, Dz[:, k] / safe, 0.0)

    # rows strictly between d and T: no decision, the positions taken at d are held unchanged
    w[hold] = np.nan
    # non-session rows (not expected on an XNYS calendar): no decision, keep positions
    w[~is_session] = np.nan
    return w
