"""H-0183 mechanism test: at every monthly decision where a slope state S_m exists, mark TLT and SHY.
S_m = +1 (steep): TLT +1, SHY -1. S_m = -1 (flat): TLT -1, SHY +1. SPY, IEF, LQD never marked.
The calendar and state code is a copy of strategy.py (separate file by contract)."""
import numpy as np

SERIES_KEY = "fred_macro.T10Y3M"


def _parts(d):
    m64 = d.astype("datetime64[M]")
    mid = m64.astype(np.int64)
    year = mid // 12 + 1970
    month = mid % 12 + 1
    dom = (d - m64.astype("datetime64[D]")).astype(np.int64) + 1
    wd = (d.astype(np.int64) + 3) % 7
    return mid, year, month, dom, wd


def _month_end_holiday(d):
    _, year, month, dom, wd = _parts(d)
    memorial = (month == 5) & (wd == 0) & (dom >= 25)
    a = year % 19
    b = year // 100
    c = year % 100
    dd = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - dd - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * l_) // 451
    n = h + l_ - 7 * mm + 114
    em = n // 31
    ed = n % 31 + 1
    easter_from_mar = np.where(em == 3, ed, 31 + ed)
    cand = np.where(month == 3, dom, np.where(month == 4, 31 + dom, -100))
    good_friday = cand == (easter_from_mar - 2)
    return memorial | good_friday


def _last_session_flag(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size == 0:
        return np.zeros(0, dtype=bool)
    nb1 = np.busday_offset(d, 1, roll="forward")
    hol = _month_end_holiday(nb1)
    nb = np.where(hol, np.busday_offset(nb1, 1, roll="forward"), nb1)
    return nb.astype("datetime64[M]") != d.astype("datetime64[M]")


def _decisions(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    mid = d.astype("datetime64[M]").astype(np.int64)
    flag = _last_session_flag(d)
    out = []
    seen = set()
    for t in range(len(d)):
        if t > 0 and mid[t] != mid[t - 1] and int(mid[t - 1]) not in seen:
            out.append((t, int(mid[t - 1])))
            seen.add(int(mid[t - 1]))
        if flag[t] and int(mid[t]) not in seen:
            out.append((t, int(mid[t])))
            seen.add(int(mid[t]))
    return out


def _monthly_states(data, median_months):
    L = int(median_months)
    if L < 1:
        raise ValueError("median_months must be >= 1")
    d = np.asarray(data.dates).astype("datetime64[D]")
    T = len(d)
    s = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    if s.shape[0] != T:
        raise ValueError("T10Y3M series length does not match the view")
    mid = d.astype("datetime64[M]").astype(np.int64)
    contrib = np.zeros(T, dtype=bool)
    if T > 1:
        contrib[1:] = (mid[1:] == mid[:-1]) & np.isfinite(s[1:])
    dec = _decisions(d)
    xs = {}
    states = {}
    prev = 0
    for t, m in dec:
        start = int(np.searchsorted(mid, m, side="left"))
        end = min(t, int(np.searchsorted(mid, m, side="right")) - 1)
        sel = contrib[start:end + 1]
        vals = s[start:end + 1][sel]
        x = float(np.mean(vals)) if vals.size > 0 else np.nan
        xs[m] = x
        S = prev
        if np.isfinite(x):
            win = np.array([xs.get(mm, np.nan) for mm in range(m - L + 1, m + 1)], dtype=float)
            if np.all(np.isfinite(win)):
                S = 1 if x > float(np.median(win)) else -1
        states[m] = S
        prev = S
    return dec, xs, states


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    j_tlt = data.col("TLT")
    j_shy = data.col("SHY")
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    if T == 0:
        return mask, side
    dec, _, states = _monthly_states(data, int(params["median_months"]))
    for t, m in dec:
        S = states[m]
        if S == 0:
            continue
        mask[t, j_tlt] = True
        mask[t, j_shy] = True
        side[t, j_tlt] = float(S)
        side[t, j_shy] = float(-S)
    ok = np.asarray(data.tradable, dtype=bool)
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)
    mask &= ok
    side = np.where(mask, side, 0.0)
    return mask, side
