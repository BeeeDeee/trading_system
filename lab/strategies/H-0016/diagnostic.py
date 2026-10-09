"""H-0016 mechanism events: after the close of the third-to-last scheduled XNYS trading day of a month, a stock
in the universe (tradable, valid trailing beta) in the top beta quintile (+1) or bottom beta quintile (-1).
Same calendar and ranking code as the strategy's entry (copied, so the file is self-contained)."""
import numpy as np

PRE_DAYS = 2
MIN_OBS_SHARE = 0.8
QUANTILES = 5


def _days(x):
    return np.asarray(x, dtype=np.int64).astype("timedelta64[D]")


def _months(x):
    return np.asarray(x, dtype=np.int64).astype("timedelta64[M]")


def _month_start(years, month):
    ys = (np.asarray(years, dtype=np.int64) - 1970).astype("datetime64[Y]").astype("datetime64[M]")
    return (ys + _months(month - 1)).astype("datetime64[D]")


def _weekday(d):
    return (d.astype(np.int64) + 3) % 7


def _nth_weekday(years, month, weekday, n):
    first = _month_start(years, month)
    return first + _days((weekday - _weekday(first)) % 7 + 7 * (n - 1))


def _last_weekday(years, month, weekday):
    last = _month_start(years, month + 1) - _days(1)
    return last - _days((_weekday(last) - weekday) % 7)


def _observed(d):
    wd = _weekday(d)
    return d + _days(np.where(wd == 5, -1, np.where(wd == 6, 1, 0)))


def _easter(years):
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


def _holidays(y0, y1):
    years = np.arange(int(y0), int(y1) + 1, dtype=np.int64)
    out = []
    ny = _month_start(years, 1)
    wd = _weekday(ny)
    out.append(ny[wd < 5])
    out.append((ny + _days(1))[wd == 6])
    out.append(_nth_weekday(years, 1, 0, 3))
    out.append(_nth_weekday(years, 2, 0, 3))
    out.append(_easter(years) - _days(2))
    out.append(_last_weekday(years, 5, 0))
    jy = years[years >= 2022]
    if jy.size:
        out.append(_observed(_month_start(jy, 6) + _days(18)))
    out.append(_observed(_month_start(years, 7) + _days(3)))
    out.append(_nth_weekday(years, 9, 0, 1))
    out.append(_nth_weekday(years, 11, 3, 4))
    out.append(_observed(_month_start(years, 12) + _days(24)))
    hol = np.concatenate(out).astype("datetime64[D]")
    return np.unique(hol[np.is_busday(hol)])


def _entry_rows(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    if dates.shape[0] == 0:
        return np.zeros(0, np.int64)
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    cal = np.busdaycalendar(holidays=_holidays(years.min() - 1, years.max() + 1))
    m_next = (dates.astype("datetime64[M]") + _months(1)).astype("datetime64[D]")
    sched = np.is_busday(dates, busdaycal=cal)
    remaining = np.busday_count(dates + _days(1), m_next, busdaycal=cal)
    return np.flatnonzero(sched & (remaining == PRE_DAYS))


def events(data, params):
    L = int(params["beta_lookback"])
    U = data.universe
    if U is None:
        raise ValueError("H-0016 needs the liq_n point-in-time universe (data.universe is None)")
    U = np.asarray(U, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    r = np.array(data.ret_co, dtype=np.float64)
    T, N = r.shape
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

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in _entry_rows(data.dates):
        cand = np.flatnonzero(U[t] & tradable[t] & listed[t])
        if cand.size == 0:
            continue
        lo = max(0, t - L + 1)
        vf = (valid[lo:t + 1][:, cand] & m_ok[lo:t + 1, None]).astype(np.float64)
        y = r[lo:t + 1][:, cand]
        x = m0[lo:t + 1, None]
        n = vf.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (vf * x).sum(axis=0) / n
            my = (vf * y).sum(axis=0) / n
            dx = (x - mx) * vf
            dy = (y - my) * vf
            sxx = (dx * dx).sum(axis=0)
            beta = (dx * dy).sum(axis=0) / sxx
        good = (n >= MIN_OBS_SHARE * L) & (sxx > 0) & np.isfinite(beta)
        idx = cand[good]
        k = idx.size // QUANTILES
        if k == 0:
            continue
        order = np.argsort(beta[good], kind="stable")
        top = idx[order[-k:]]
        bot = idx[order[:k]]
        mask[t, top] = True
        side[t, top] = 1.0
        mask[t, bot] = True
        side[t, bot] = -1.0
    return mask, side
