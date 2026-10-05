"""H-0002: front-running month-end stock/bond rebalancing with a dollar-neutral SPY/IEF spread.

Rules (card.signal.description) and where they live:
  * Window = last K = window_days XNYS trading days of month m (L-K+1 .. L); decision day t = the
    (K+1)-th last trading day of m.  -> `_remaining_trading_days` + `is_decision`.
    Point-in-time: "trading days left in the month after t" is computed from the date of row t alone
    with np.busday_count and a rule-based NYSE regular-holiday calendar (`_nyse_holidays`).
    Unscheduled closures (e.g. Hurricane Sandy 2012) cannot be known in advance and are not modelled.
  * After the close of t: month-to-date compounded total return of SPY and IEF from the close of the
    last row of month m-1 to the close of t; D = R_SPY - R_IEF.   -> loop over decision rows.
  * D > 0: SPY -0.5, IEF +0.5; D < 0: SPY +0.5, IEF -0.5; D == 0 or a missing return on any day of
    month m up to t (or no price on the anchor close of m-1): flat for the month.
  * Entry at the open of t+1 (engine executes row t at the next open).
  * Either leg not tradable on the entry open: flat for the month. tradable[t+1] is not known at t,
    so row t+1 (first window day, after its close) is set to 0 -> any partial fill is closed at the
    next open (point-in-time approximation).
  * Weights held fixed through the window (NaN rows = keep positions, no daily rebalancing).
  * Exit: the engine cannot trade at the close, so the card's fallback is used: row L is set to 0,
    which closes the position at the open of the first trading day of month m+1.
  * Flat (0) on every other row.
"""
import numpy as np
from lab.framework.api import only_on, period_starts, total_return_index, trailing_return, rolling_mean, rolling_std

PARAMS = {"window_days": 5}

_EQUITY = "SPY"
_BOND = "IEF"


def _shift(d, k, unit):
    """d (datetime64[unit]) + k units, done in integer space (no generic-unit timedelta)."""
    return (d.astype(np.int64) + np.asarray(k, dtype=np.int64)).astype("datetime64[" + unit + "]")


def _ymd(years, month, day):
    """datetime64[D] array for (year array, month int/array, day int/array)."""
    y = np.asarray(years, dtype=np.int64)
    months = (y - 1970) * 12 + (np.asarray(month, dtype=np.int64) - 1)
    first = months.astype("datetime64[M]").astype("datetime64[D]")
    return _shift(first, np.asarray(day, dtype=np.int64) - 1, "D")


def _weekday(d):
    """0 = Monday .. 6 = Sunday (day 0 of the epoch was a Thursday)."""
    return (d.astype(np.int64) - 4) % 7


def _observed(d):
    """Saturday holiday -> Friday before, Sunday holiday -> Monday after (NYSE rule)."""
    wd = _weekday(d)
    return _shift(d, np.where(wd == 5, -1, np.where(wd == 6, 1, 0)), "D")


def _nyse_holidays(years):
    """Rule-based NYSE regular full-day holidays for the given years (no date literals)."""
    y = np.asarray(years, dtype=np.int64)
    out = []
    # New Year's Day: Sunday -> Monday; Saturday -> no holiday (NYSE does not close on Dec 31).
    ny = _ymd(y, 1, 1)
    wd = _weekday(ny)
    out.append(_shift(ny, np.where(wd == 6, 1, 0), "D")[wd != 5])
    # Martin Luther King Jr. Day (3rd Monday of January), Presidents' Day (3rd Monday of February).
    out.append(np.busday_offset(_ymd(y, 1, 1), 2, roll="forward", weekmask="Mon"))
    out.append(np.busday_offset(_ymd(y, 2, 1), 2, roll="forward", weekmask="Mon"))
    # Good Friday: Easter Sunday (anonymous Gregorian algorithm) - 2 days.
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
    n = h + l - 7 * mm + 114
    out.append(_shift(_ymd(y, n // 31, n % 31 + 1), -2, "D"))
    # Memorial Day: last Monday of May.
    out.append(np.busday_offset(_ymd(y, 6, 1), -1, roll="forward", weekmask="Mon"))
    # Independence Day (observed).
    out.append(_observed(_ymd(y, 7, 4)))
    # Labor Day: 1st Monday of September.
    out.append(np.busday_offset(_ymd(y, 9, 1), 0, roll="forward", weekmask="Mon"))
    # Thanksgiving: 4th Thursday of November.
    out.append(np.busday_offset(_ymd(y, 11, 1), 3, roll="forward", weekmask="Thu"))
    # Christmas (observed).
    out.append(_observed(_ymd(y, 12, 25)))
    return np.unique(np.concatenate(out).astype("datetime64[D]"))


def _remaining_trading_days(dates):
    """Number of (rule-based) XNYS trading days strictly after each date within its calendar month."""
    dates = dates.astype("datetime64[D]")
    month = dates.astype("datetime64[M]")
    next_month_start = _shift(month, 1, "M").astype("datetime64[D]")
    yr = month.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(int(yr.min()), int(yr.max()) + 1)
    hol = _nyse_holidays(years)
    return np.busday_count(_shift(dates, 1, "D"), next_month_start, holidays=hol), month.astype(np.int64)


def target_weights(data, params):
    K = int(params["window_days"])
    dates = np.asarray(data.dates).astype("datetime64[D]")
    T = dates.shape[0]
    N = len(data.instruments)
    W = np.zeros((T, N), dtype=float)
    if T == 0:
        return W

    ce = data.col(_EQUITY)
    cb = data.col(_BOND)
    cols = [ce, cb]

    rem, mid = _remaining_trading_days(dates)

    # First row of each calendar-month segment and anchor (= last row of the previous month).
    first = np.ones(T, dtype=bool)
    first[1:] = mid[1:] != mid[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(T), 0))

    # Decision row: (K+1)-th last trading day of the month = K trading days left after it.
    # Only the first such row of a month decides (guards against a data row on a holiday).
    is_dec = rem == K
    prev_dec_same_month = np.zeros(T, dtype=bool)
    prev_dec_same_month[1:] = is_dec[:-1] & (mid[1:] == mid[:-1])
    hold_dup = is_dec & prev_dec_same_month
    is_dec = is_dec & ~prev_dec_same_month

    # Window rows (positions fixed): NaN = keep positions. Row L (0 days left) and all other rows: 0.
    in_window = ((rem >= 1) & (rem < K)) | hold_dup
    W[in_window, :] = np.nan

    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    tradable = np.asarray(data.tradable, dtype=bool)
    missing = ~tradable | ~np.isfinite(ret_co) | ~np.isfinite(ret_oc)
    if data.listed is not None:
        missing = missing | ~np.asarray(data.listed, dtype=bool)
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)

    for t in np.flatnonzero(is_dec):
        s = int(start[t])
        a = s - 1  # anchor row: close of the last trading day of month m-1
        w_e = 0.0
        w_b = 0.0
        ok = a >= 0 and mid[a] != mid[s]
        if ok:
            ok = not missing[a:t + 1][:, cols].any()
        if ok and universe is not None:
            ok = bool(universe[t, ce] and universe[t, cb])
        if ok:
            g_e = np.prod((1.0 + ret_co[s:t + 1, ce]) * (1.0 + ret_oc[s:t + 1, ce]))
            g_b = np.prod((1.0 + ret_co[s:t + 1, cb]) * (1.0 + ret_oc[s:t + 1, cb]))
            D = (g_e - 1.0) - (g_b - 1.0)
            if D > 0:
                w_e, w_b = -0.5, 0.5
            elif D < 0:
                w_e, w_b = 0.5, -0.5
        W[t, :] = 0.0
        W[t, ce] = w_e
        W[t, cb] = w_b
        # Entry-day tradability: known only after row t+1; close any partial fill at the next open.
        j = t + 1
        if (w_e != 0.0) and j < T and mid[j] == mid[t] and not (tradable[j, ce] and tradable[j, cb]):
            W[j, :] = 0.0

    return W
