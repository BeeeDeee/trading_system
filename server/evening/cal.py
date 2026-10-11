"""NYSE trading calendar (regular holidays and early closes) computed by rule, 2020-2040."""
import datetime as dt

D = dt.date


def _nth_weekday(y, m, wd, n):
    d = D(y, m, 1)
    d += dt.timedelta(days=(wd - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = D(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - wd) % 7)


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    mo = (h + l - 7 * m + 90) // 25
    return D(y, mo, (h + l - 7 * m + 33 * mo + 19) % 32)


def _observed(d):
    """Saturday holiday -> Friday, Sunday -> Monday (NYSE rule)."""
    return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d


def holidays(y):
    h = {
        _nth_weekday(y, 1, 0, 3): "MLK",
        _nth_weekday(y, 2, 0, 3): "Presidents",
        _easter(y) - dt.timedelta(days=2): "Good Friday",
        _last_weekday(y, 5, 0): "Memorial",
        _observed(D(y, 6, 19)): "Juneteenth",
        _observed(D(y, 7, 4)): "Independence",
        _nth_weekday(y, 9, 0, 1): "Labor",
        _nth_weekday(y, 11, 3, 4): "Thanksgiving",
        _observed(D(y, 12, 25)): "Christmas",
    }
    ny = D(y, 1, 1)
    if ny.weekday() != 5:                  # a Saturday New Year's Day is NOT observed on the Friday before
        h[_observed(ny)] = "New Year"
    return h


def is_holiday(d):
    return d in holidays(d.year)


def is_trading_day(d):
    return d.weekday() < 5 and not is_holiday(d)


def is_early_close(d):
    """13:00 ET close: day after Thanksgiving, Dec 24, Jul 3 (when Jul 4 is a weekday and Jul 3 is open)."""
    if not is_trading_day(d):
        return False
    if d.month == 11 and d == _nth_weekday(d.year, 11, 3, 4) + dt.timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24:
        return True
    return d.month == 7 and d.day == 3 and D(d.year, 7, 4).weekday() < 5


def prev_trading_day(d):
    d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def trading_days_between(a, b):
    """Trading days strictly after a, up to and including b."""
    out, d = [], a + dt.timedelta(days=1)
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out
