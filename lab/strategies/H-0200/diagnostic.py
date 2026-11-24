"""H-0200 mechanism test: top-k (+1) and bottom-k (-1) of the weekly new-filer pool by
trailing 91-day insider purchase value / anchored market cap. Same pipeline as strategy.py (copied)."""
import numpy as np
from lab.framework.api import total_return_index

NEW_FILER_LAG = 5
MIN_POOL = 3


def _days_from_ymd(years, months, days):
    years = np.asarray(years, dtype=np.int64)
    months = np.asarray(months, dtype=np.int64)
    days = np.asarray(days, dtype=np.int64)
    m = ((years - 1970) * 12 + (months - 1)).astype("datetime64[M]")
    return m.astype("datetime64[D]").astype(np.int64) + (days - 1)


def _weekday(dnum):
    return (np.asarray(dnum, dtype=np.int64) + 3) % 7


def _good_friday(years):
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
    q = h + l - 7 * mm + 114
    easter = _days_from_ymd(y, q // 31, (q % 31) + 1)
    return easter - 2


def _observed(dnum, saturday_to_friday=True):
    wd = _weekday(dnum)
    out = np.where(wd == 6, dnum + 1, dnum)
    if saturday_to_friday:
        return np.where(wd == 5, dnum - 1, out)
    return out[wd != 5]


def _nyse_holidays(years):
    years = np.asarray(years, dtype=np.int64)
    one = np.ones_like(years)
    parts = [
        _observed(_days_from_ymd(years, one, one), saturday_to_friday=False),
        _good_friday(years),
        _observed(_days_from_ymd(years, 7 * one, 4 * one)),
        _observed(_days_from_ymd(years, 12 * one, 25 * one)),
    ]
    return np.unique(np.concatenate(parts))


def _week_end_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size == 0:
        return np.zeros(0, dtype=bool)
    dn = d.astype(np.int64)
    yrs = d.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(int(yrs.min()) - 1, int(yrs.max()) + 2)
    hol = _nyse_holidays(years).astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward", holidays=hol).astype(np.int64)
    return (nxt + 3) // 7 != (dn + 3) // 7


def _anchor_index(mc_raw):
    mc = np.asarray(mc_raw)
    T, N = mc.shape
    fin = np.isfinite(mc)
    changed = np.zeros((T, N), dtype=bool)
    if T > 1:
        changed[1:] = fin[1:] & ~(mc[1:] == mc[:-1])
    rows = np.arange(T, dtype=np.int32)[:, None]
    anchor = np.where(changed, rows, np.int32(-1)).astype(np.int32)
    np.maximum.accumulate(anchor, axis=0, out=anchor)
    return anchor


def _mcap_at(t, mc_raw, tri, anchor):
    N = anchor.shape[1]
    a = anchor[t]
    ok = a >= 0
    base = np.full(N, np.nan)
    cols = np.flatnonzero(ok)
    base[cols] = tri[a[cols], cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = np.asarray(mc_raw[t], dtype=np.float64) * np.asarray(tri[t], dtype=np.float64) / base
    good = ok & np.isfinite(mc) & (mc > 0)
    return np.where(good, mc, np.nan)


def _pool_and_ratio(t, member, bn, bv, mc_t):
    N = member.shape[0]
    if t < NEW_FILER_LAG:
        return np.zeros(N, dtype=bool), np.full(N, np.nan)
    n_now = np.asarray(bn[t], dtype=np.float64)
    n_old = np.asarray(bn[t - NEW_FILER_LAG], dtype=np.float64)
    val = np.asarray(bv[t], dtype=np.float64)
    fin = np.isfinite(n_now) & np.isfinite(n_old) & np.isfinite(val) & np.isfinite(mc_t)
    with np.errstate(invalid="ignore"):
        new = fin & (n_now > n_old) & (val > 0)
    pool = member & new
    ratio = np.full(N, np.nan)
    ratio[pool] = val[pool] / mc_t[pool]
    return pool, ratio


def _cohort_size(n, top_frac):
    if n < MIN_POOL:
        return 0
    k = max(1, int(np.floor(top_frac * n + 0.5 + 1e-9)))
    return min(k, n // 2)


def _id_rank(names):
    def key(j):
        s = names[j]
        if len(s) > 1 and s[0] == "E" and s[1:].isdigit():
            return (0, int(s[1:]), s)
        return (1, 0, s)
    order = sorted(range(len(names)), key=key)
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    return rank


def _cohort(pool, ratio, top_frac, idrank):
    idx = np.flatnonzero(pool)
    k = _cohort_size(idx.size, top_frac)
    if k == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    order = np.lexsort((idrank[idx], -ratio[idx]))
    ranked = idx[order]
    return ranked[:k], ranked[-k:]


def events(data, params):
    top_frac = float(params["top_frac"])
    if data.universe is None:
        raise ValueError("H-0200 needs the point-in-time LIQ-500 universe (data.universe)")
    uni = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    bv = data.extras["ins_buy_value_91d"]
    bn = data.extras["ins_buy_n_91d"]
    mc_raw = data.extras["sf1_arq_marketcap"]
    T, N = uni.shape

    tri = np.asarray(total_return_index(data), dtype=np.float64)
    anchor = _anchor_index(mc_raw)
    idrank = _id_rank(tuple(data.instruments))

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(_week_end_rows(data.dates)):
        member = uni[t] & listed[t]
        mc_t = _mcap_at(t, mc_raw, tri, anchor)
        pool, ratio = _pool_and_ratio(t, member, bn, bv, mc_t)
        top, bottom = _cohort(pool, ratio, top_frac, idrank)
        ok = uni[t] & tradable[t]
        top = top[ok[top]]
        bottom = bottom[ok[bottom]]
        mask[t, top] = True
        side[t, top] = 1.0
        mask[t, bottom] = True
        side[t, bottom] = -1.0
    return mask, side
