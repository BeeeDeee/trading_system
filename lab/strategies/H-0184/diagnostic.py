"""H-0184 mechanism test: at each book month-end decision date with state HIGH, TLT and TLH +1, SHY and IEI -1;
with state LOW the reverse; NEUTRAL months no events. IEF, SPY, AGG and LQD are never marked (peers; AGG and
LQD are peer-only instruments added in card v2). Same pipeline as strategy.py (copied: diagnostic.py is a
separate file)."""
import numpy as np

DS = "treasury_auctions_1d"
YEAR = 252
STEP = 21


def _easter_good_friday(years):
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
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    first = ((y - 1970) * 12 + (month - 1)).astype("datetime64[M]").astype("datetime64[D]")
    return first + _days(day - 3)


def _days(k):
    return np.asarray(k, dtype=np.int64).astype("timedelta64[D]")


def _memorial_day(years):
    y = np.asarray(years, dtype=np.int64)
    may31 = ((y - 1970) * 12 + 4).astype("datetime64[M]").astype("datetime64[D]") + _days(30)
    return np.busday_offset(may31, 0, roll="backward", weekmask="Mon")


def _holidays(d):
    yrs = d.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(int(yrs.min()) - 1, int(yrs.max()) + 2)
    return np.sort(np.concatenate([_easter_good_friday(years), _memorial_day(years)]))


def _month_end_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d + _days(1), 0, roll="forward", holidays=_holidays(d))
    mi = d.astype("datetime64[M]").astype(np.int64)
    is_me = nxt.astype("datetime64[M]").astype(np.int64) != mi
    dec = is_me.copy()
    if T > 1:
        idx = np.where(is_me, np.arange(T), -1)
        last_me = np.maximum.accumulate(idx)
        prev_last = np.concatenate([[-1], last_me[:-1]])
        prev_month = np.concatenate([[mi[0]], mi[:-1]])
        prev_has = (prev_last >= 0) & (mi[np.maximum(prev_last, 0)] == prev_month)
        new_month = np.concatenate([[False], mi[1:] != mi[:-1]])
        dec |= new_month & ~prev_has
    return dec


def _surge(data, base_years):
    last_sess = np.asarray(data.series[DS + ".last_coupon5_sessions"], dtype=np.float64)
    next_sess = np.asarray(data.series[DS + ".next_coupon5_sessions"], dtype=np.float64)
    next_off = np.asarray(data.series[DS + ".next_coupon5_offering_usd"], dtype=np.float64)
    T = last_sess.shape[0]
    auction = last_sess == 0
    prev_ok = np.zeros(T, dtype=bool)
    prev_off = np.full(T, np.nan)
    if T > 1:
        prev_ok[1:] = next_sess[:-1] == 1
        prev_off[1:] = next_off[:-1]
    valid = auction & prev_ok & np.isfinite(prev_off)
    off = np.where(valid, prev_off, 0.0)
    S = np.full(T, np.nan)
    if T >= YEAR:
        cs = np.concatenate([[0.0], np.cumsum(off)])
        S[YEAR - 1:] = cs[YEAR:] - cs[:T - YEAR + 1]
    lags = np.arange(YEAR, YEAR * (1 + int(base_years)) + 1, STEP)
    acc = np.zeros(T)
    for L in lags:
        sh = np.full(T, np.nan)
        if L < T:
            sh[L:] = S[:T - L]
        acc = acc + sh
    B = acc / len(lags)
    ok = np.isfinite(S) & np.isfinite(B) & (S > 0) & (B > 0)
    g = np.full(T, np.nan)
    g[ok] = np.log(S[ok] / B[ok])
    return g


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    spy, shy, ief, tlt = data.col("SPY"), data.col("SHY"), data.col("IEF"), data.col("TLT")
    iei, tlh = data.col("IEI"), data.col("TLH")
    close = np.asarray(data.close, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)

    # book decision dates: month-ends from the first one at which SPY, SHY, IEF and TLT all have a price
    dec = _month_end_rows(data.dates)
    priced4 = np.all(np.isfinite(close[:, [spy, shy, ief, tlt]]), axis=1)
    cand = np.flatnonzero(dec & priced4)
    book = np.zeros(T, dtype=bool)
    if cand.size:
        book = dec & (np.arange(T) >= cand[0])

    g = _surge(data, int(params["base_years"]))
    cut = float(params["supply_cut"])
    fin = np.isfinite(g)
    state = np.zeros(T)
    state[fin & (g >= cut)] = 1.0
    state[fin & (g <= -cut)] = -1.0

    seen = np.maximum.accumulate(np.isfinite(close), axis=0)    # from the instrument's first price on
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    active = book & (state != 0)
    uni = data.universe
    for j, sgn in ((tlt, 1.0), (tlh, 1.0), (shy, -1.0), (iei, -1.0)):
        m = active & seen[:, j] & tradable[:, j]
        if uni is not None:
            m = m & np.asarray(uni, dtype=bool)[:, j]
        mask[:, j] = m
        side[:, j] = np.where(m, sgn * state, 0.0)
    return mask, side
