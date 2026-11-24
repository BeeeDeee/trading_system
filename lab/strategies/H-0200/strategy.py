"""H-0200: insider buying relative to firm size (LIQ-500, weekly).

Rules of the card and where they live:
  decision day      last session of the calendar week, from the date of row t alone   -> week_end_rows
  (1) market cap    sf1_arq_marketcap at its last change, grown by total return        -> anchor_index, mcap_at
  (2) new-filer pool member, ins_buy_n_91d(t) > ins_buy_n_91d(t-5), ins_buy_value_91d > 0 -> pool_and_ratio
  (3) ratio         ins_buy_value_91d / market cap (purchases only)                     -> pool_and_ratio
  (4) cohort        n >= 3, k = max(1, round(top_frac n)), 2k <= n, top k by r           -> cohort
  (5) long leg      held hold_weeks rebalances, clock restarts, dropped when not member  -> active_sets
                    w = min(0.40 / N, 0.05)
  (6) hedge         EW short of the other members, notional sum w_i clip(beta_i,.5,1.5) -> betas, target_weights
  (7) flat when no name is active.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"hold_weeks": 4, "top_frac": 0.2}

LONG_CAP = 0.40          # total long leg
NAME_CAP = 0.05          # per long name
BETA_WINDOW = 252        # sessions
BETA_MIN_OBS = 126
BETA_LO = 0.5
BETA_HI = 1.5
NEW_FILER_LAG = 5        # sessions
MIN_POOL = 3


# ----------------------------------------------------------------------------------------------
# calendar: last session of the calendar week, known from the date of row t alone
# ----------------------------------------------------------------------------------------------
def _days_from_ymd(years, months, days):
    """int arrays (year, month, day) -> int64 days since the numpy epoch."""
    years = np.asarray(years, dtype=np.int64)
    months = np.asarray(months, dtype=np.int64)
    days = np.asarray(days, dtype=np.int64)
    m = ((years - 1970) * 12 + (months - 1)).astype("datetime64[M]")
    return m.astype("datetime64[D]").astype(np.int64) + (days - 1)


def _weekday(dnum):
    """0 = Monday ... 6 = Sunday (the numpy epoch day was a Thursday)."""
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


def nyse_holidays(years):
    """Rule-based NYSE full-day holidays that can remove a Friday session (int days).

    New Year (Saturday not observed), Good Friday, Independence Day and Christmas (Saturday ->
    Friday, Sunday -> Monday). Monday and mid-week holidays never move the last session of a week,
    so they are not needed. Unscheduled closures and Juneteenth (from 2022) are not modelled.
    """
    years = np.asarray(years, dtype=np.int64)
    one = np.ones_like(years)
    parts = [
        _observed(_days_from_ymd(years, one, one), saturday_to_friday=False),
        _good_friday(years),
        _observed(_days_from_ymd(years, 7 * one, 4 * one)),
        _observed(_days_from_ymd(years, 12 * one, 25 * one)),
    ]
    return np.unique(np.concatenate(parts))


def week_end_rows(dates):
    """True where row t is the last session of its calendar (Mon-Sun) week.

    Decided from the date of row t alone: the next weekday after t that is not a rule-based NYSE
    holiday lies in a later calendar week.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size == 0:
        return np.zeros(0, dtype=bool)
    dn = d.astype(np.int64)
    yrs = d.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(int(yrs.min()) - 1, int(yrs.max()) + 2)
    hol = nyse_holidays(years).astype("datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward", holidays=hol).astype(np.int64)
    return (nxt + 3) // 7 != (dn + 3) // 7


# ----------------------------------------------------------------------------------------------
# market cap anchored at the filing and grown by total return (K-0168)
# ----------------------------------------------------------------------------------------------
def anchor_index(mc_raw):
    """(T, N) int32: row of the last change of sf1_arq_marketcap at or before t, -1 if unknown.

    A value already present on the first row of the view has an unknown filing session
    (left-censored), so it stays unknown until the first change inside the view.
    """
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


def mcap_at(t, mc_raw, tri, anchor):
    """(N,) market cap at row t: anchored value x TRI[t] / TRI[anchor]; NaN if unknown."""
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


# ----------------------------------------------------------------------------------------------
# pool, ratio, cohort
# ----------------------------------------------------------------------------------------------
def pool_and_ratio(t, member, bn, bv, mc_t):
    """New-filer pool at row t and the ratio r = ins_buy_value_91d / market cap."""
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


def cohort_size(n, top_frac):
    """k = max(1, round(top_frac n)) (half up), capped so that 2k <= n; 0 if n < 3."""
    if n < MIN_POOL:
        return 0
    k = max(1, int(np.floor(top_frac * n + 0.5 + 1e-9)))
    return min(k, n // 2)


def id_rank(names):
    """Rank of instrument ids: E<number> numerically, anything else after them by name."""
    def key(j):
        s = names[j]
        if len(s) > 1 and s[0] == "E" and s[1:].isdigit():
            return (0, int(s[1:]), s)
        return (1, 0, s)
    order = sorted(range(len(names)), key=key)
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    return rank


def cohort(pool, ratio, top_frac, idrank):
    """(top, bottom) column indices: ordered by larger r first, then lower id."""
    idx = np.flatnonzero(pool)
    k = cohort_size(idx.size, top_frac)
    if k == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    order = np.lexsort((idrank[idx], -ratio[idx]))
    ranked = idx[order]
    return ranked[:k], ranked[-k:]


# ----------------------------------------------------------------------------------------------
# holding clock
# ----------------------------------------------------------------------------------------------
def active_sets(entries, eligible, hold):
    """(R, N) bool active long names per rebalance.

    entries: list of R index arrays (the cohort of each rebalance); eligible: (R, N) bool
    (member and listed at the rebalance). A name is active from its entry through hold - 1 later
    rebalances; re-entry restarts the clock; a held name that is not eligible is dropped.
    """
    R, N = eligible.shape
    never = -(10 ** 9)
    last = np.full(N, never, dtype=np.int64)
    act = np.zeros((R, N), dtype=bool)
    for r in range(R):
        held = (r - last) < hold
        last[held & ~eligible[r]] = never
        last[np.asarray(entries[r], dtype=np.int64)] = r
        act[r] = ((r - last) < hold) & eligible[r]
    return act


# ----------------------------------------------------------------------------------------------
# betas against the equal-weight LIQ-500 member return
# ----------------------------------------------------------------------------------------------
def close_returns(tri, listed, close):
    """(T, N) close-to-close total return, NaN where the stock did not trade on both closes."""
    tri = np.asarray(tri, dtype=np.float64)
    T, N = tri.shape
    r = np.full((T, N), np.nan)
    if T < 2:
        return r
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = tri[1:] / tri[:-1] - 1.0
    cl = np.isfinite(np.asarray(close, dtype=np.float64))
    ok = np.zeros((T, N), dtype=bool)
    ok[1:] = listed[1:] & listed[:-1] & cl[1:] & cl[:-1]
    ok &= np.isfinite(r)
    r[~ok] = np.nan
    return r


def market_return(r, member):
    """(T,) equal-weight return of the members with a valid return on that row."""
    m = member & np.isfinite(r)
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    pos = cnt > 0
    out[pos] = s[pos] / cnt[pos]
    return out


def betas(t, cols, r, mkt):
    """OLS beta of r[:, cols] on mkt over rows t-251..t; 1.0 with fewer than 126 observations."""
    cols = np.asarray(cols, dtype=np.int64)
    lo = max(0, t - BETA_WINDOW + 1)
    y = r[lo:t + 1][:, cols]
    x = np.asarray(mkt[lo:t + 1], dtype=np.float64)[:, None]
    ok = np.isfinite(y) & np.isfinite(x)
    n = ok.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(ok, x, 0.0).sum(axis=0) / n
        my = np.where(ok, y, 0.0).sum(axis=0) / n
        dx = np.where(ok, x - mx, 0.0)
        dy = np.where(ok, y - my, 0.0)
        var = (dx * dx).sum(axis=0)
        cov = (dx * dy).sum(axis=0)
        b = cov / var
    good = (n >= BETA_MIN_OBS) & (var > 0) & np.isfinite(b)
    return np.where(good, b, 1.0)


# ----------------------------------------------------------------------------------------------
def target_weights(data, params):
    hold = int(params["hold_weeks"])
    top_frac = float(params["top_frac"])
    if hold < 1:
        raise ValueError("hold_weeks must be >= 1")
    if data.universe is None:
        raise ValueError("H-0200 needs the point-in-time LIQ-500 universe (data.universe)")
    uni = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    bv = data.extras["ins_buy_value_91d"]
    bn = data.extras["ins_buy_n_91d"]
    mc_raw = data.extras["sf1_arq_marketcap"]
    T, N = uni.shape

    tri = np.asarray(total_return_index(data), dtype=np.float64)
    anchor = anchor_index(mc_raw)
    r = close_returns(tri, listed, data.close)
    mkt = market_return(r, uni)
    idrank = id_rank(tuple(data.instruments))

    dec = np.flatnonzero(week_end_rows(data.dates))
    R = dec.size
    entries = []
    eligible = np.zeros((R, N), dtype=bool)
    for i, t in enumerate(dec):
        member = uni[t] & listed[t]
        eligible[i] = member
        mc_t = mcap_at(t, mc_raw, tri, anchor)
        pool, ratio = pool_and_ratio(t, member, bn, bv, mc_t)
        top, _ = cohort(pool, ratio, top_frac, idrank)
        entries.append(top)
    act = active_sets(entries, eligible, hold)

    W = np.full((T, N), np.nan)
    for i, t in enumerate(dec):
        row = np.zeros(N)
        longs = np.flatnonzero(act[i])
        if longs.size > 0:
            w = min(LONG_CAP / longs.size, NAME_CAP)
            row[longs] = w
            b = np.clip(betas(t, longs, r, mkt), BETA_LO, BETA_HI)
            hedge_notional = float(w * b.sum())
            hedge = eligible[i] & ~act[i]
            m = int(hedge.sum())
            if m > 0:
                row[hedge] = -hedge_notional / m
            g = np.abs(row).sum()
            if g > 1.0:
                row /= g          # only float rounding can push 0.40 + 0.60 above 1
        W[t] = row
    return W
