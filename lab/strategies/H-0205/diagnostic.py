"""H-0205 mechanism test: mark the long leg (top-50 pending share increase g among eligible S&P 500
members, only in quarters with >= 20 eligible, as in the strategy) on each quarterly decision date d
(W sessions before the rebalance R); side +1.
Same pipeline as strategy.py (copied; this file is evaluated separately)."""
import numpy as np
from lab.framework.api import total_return_index

CORE = "SPY"
MAX_NAMES = 50
MIN_NAMES = 20
G_MIN = 0.005
G_MAX = 0.25
SPLIT_TOL = 0.20
STALE_DAYS = 400
GUARD_PRE_SESSIONS = 5


def _ymd(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _weekday(days):
    return (np.asarray(days, dtype=np.int64) + 3) % 7


def _nth_weekday(y, m, wd, n):
    first = _ymd(y, m, 1)
    return first + (wd - _weekday(first)) % 7 + 7 * (n - 1)


def _last_weekday(y, m, last_day, wd):
    last = _ymd(y, m, last_day)
    return last - (_weekday(last) - wd) % 7


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
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return _ymd(y, month, day)


def _observed(days):
    wd = _weekday(days)
    return days + np.where(wd == 5, -1, np.where(wd == 6, 1, 0))


def _xnys_holidays(y0, y1):
    years = np.arange(int(y0), int(y1) + 1, dtype=np.int64)
    ny = _ymd(years, 1, 1)
    ny_wd = _weekday(ny)
    keep = ny_wd != 5
    new_year = ny[keep] + np.where(ny_wd[keep] == 6, 1, 0)
    mlk = _nth_weekday(years[years >= 1998], 1, 0, 3)
    presidents = _nth_weekday(years, 2, 0, 3)
    good_friday = _easter(years) - 2
    memorial = _last_weekday(years, 5, 31, 0)
    juneteenth = _observed(_ymd(years[years >= 2022], 6, 19))
    july4 = _observed(_ymd(years, 7, 4))
    labor = _nth_weekday(years, 9, 0, 1)
    thanksgiving = _nth_weekday(years, 11, 3, 4)
    christmas = _observed(_ymd(years, 12, 25))
    hol = np.concatenate([new_year, mlk, presidents, good_friday, memorial, juneteenth,
                          july4, labor, thanksgiving, christmas])
    return np.unique(hol).astype("datetime64[D]")


def _schedule(dates, w):
    T = len(dates)
    if T == 0:
        return []
    dates = np.asarray(dates).astype("datetime64[D]")
    yrs = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    y0, y1 = int(yrs[0]) - 1, int(yrs[-1]) + 1
    hol = _xnys_holidays(y0 - 1, y1 + 1)
    years = np.repeat(np.arange(y0, y1 + 1, dtype=np.int64), 4)
    months = np.tile(np.array([3, 6, 9, 12], dtype=np.int64), y1 - y0 + 1)
    fri3 = _nth_weekday(years, months, 4, 3).astype("datetime64[D]")
    reb = np.busday_offset(fri3, 0, roll="backward", holidays=hol)
    dec = np.busday_offset(reb, -int(w), roll="backward", holidays=hol)
    out = []
    for q in range(1, len(reb)):
        e = int(np.searchsorted(dates, dec[q], side="left"))
        rp = int(np.searchsorted(dates, reb[q - 1], side="right")) - 1
        if e >= T or rp < 0 or rp >= e or not dates[e] < reb[q]:
            continue
        out.append((e, rp))
    return out


def _share_panels(data):
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    pr = np.asarray(data.extras["sf1_arq_price"], dtype=np.float64)
    T, N = mc.shape
    valid = np.isfinite(mc) & np.isfinite(pr) & (mc > 0) & (pr > 0)
    shares = np.where(valid, mc / np.where(valid, pr, 1.0), np.nan)
    chg = np.zeros((T, N), dtype=bool)
    chg[0] = valid[0]
    if T > 1:
        same = (mc[1:] == mc[:-1]) & (pr[1:] == pr[:-1]) & valid[:-1]
        chg[1:] = valid[1:] & ~same
    rows = np.arange(T, dtype=np.int64)[:, None]
    last_filing = np.maximum.accumulate(np.where(chg, rows, -1), axis=0)
    return shares, valid, last_filing


def _split_flag_cumsum(data):
    close = np.asarray(data.close, dtype=np.float64)
    idx = np.asarray(total_return_index(data), dtype=np.float64)
    T, N = close.shape
    vc = np.isfinite(close) & (close > 0) & np.isfinite(idx) & (idx > 0)
    rows = np.arange(T, dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(vc, rows, -1), axis=0)
    prev = np.vstack([np.full((1, N), -1, dtype=np.int64), last[:-1]])
    pc = np.clip(prev, 0, None)
    c_prev = np.take_along_axis(close, pc, axis=0)
    i_prev = np.take_along_axis(idx, pc, axis=0)
    ok = vc & (prev >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (close / np.where(ok, c_prev, 1.0)) / (idx / np.where(ok, i_prev, 1.0))
    flag = ok & np.isfinite(rel) & (np.abs(rel - 1.0) > SPLIT_TOL)
    return np.cumsum(flag, axis=0, dtype=np.int32)


def _select_names(data, e, rp, shares, valid, last_filing, cflag, spy):
    dates = np.asarray(data.dates).astype("datetime64[D]")
    N = shares.shape[1]
    uni = np.asarray(data.universe[e], dtype=bool)
    trd = np.asarray(data.tradable[e], dtype=bool)
    ok = uni & trd & valid[e] & valid[rp]
    ok[spy] = False
    lf_new = last_filing[e]
    lf_old = last_filing[rp]
    ok &= lf_old >= 0
    ok &= lf_new > rp
    dd = dates.astype(np.int64)
    ok &= (dd[e] - dd[np.clip(lf_new, 0, None)]) <= STALE_DAYS
    ok &= (dd[rp] - dd[np.clip(lf_old, 0, None)]) <= STALE_DAYS
    a = np.clip(lf_old - GUARD_PRE_SESSIONS - 1, 0, None)
    before = np.where(a > 0, cflag[np.clip(a - 1, 0, None), np.arange(N)], 0)
    ok &= (cflag[e] - before) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = shares[e] / shares[rp] - 1.0
    ok &= np.isfinite(g) & (g >= G_MIN) & (g <= G_MAX)
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return cand
    order = np.argsort(-g[cand], kind="stable")
    return cand[order][:MAX_NAMES]


def events(data, params):
    w = int(params["window_sessions"])
    spy = data.col(CORE)
    if data.universe is None:
        raise ValueError("card universe sp500 required")
    T, N = len(data.dates), len(data.instruments)
    shares, valid, last_filing = _share_panels(data)
    cflag = _split_flag_cumsum(data)
    mask = np.zeros((T, N), dtype=bool)
    for e, rp in _schedule(data.dates, w):
        sel = _select_names(data, e, rp, shares, valid, last_filing, cflag, spy)
        if len(sel) < MIN_NAMES:
            continue
        mask[e, sel] = True
    return mask, None
