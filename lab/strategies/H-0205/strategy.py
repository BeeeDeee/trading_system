"""H-0205: quarterly S&P 500 share-update pressure as a long-only tilt on an SPY core.

All year: 100 % SPY. At the decision date d (W XNYS sessions before the quarterly rebalance R =
third Friday of Mar/Jun/Sep/Dec, or the last session before it) move 20 % of NAV into the (up to)
50 eligible S&P 500 members with the largest pending share increase g = S_new / S_old - 1
(shares = sf1_arq_marketcap / sf1_arq_price), equal weight; back to 100 % SPY after the close of R.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"window_sessions": 20}

# fixed by the card / owner (not in the grid)
CORE = "SPY"
TILT = 0.20
MAX_NAMES = 50
MIN_NAMES = 20
G_MIN = 0.005
G_MAX = 0.25
SPLIT_TOL = 0.20
STALE_DAYS = 400
GUARD_PRE_SESSIONS = 5


# ----------------------------------------------------------------------------------------------
# XNYS calendar from rules (regular full-day holidays), integer days since 1970-01-01.
# Pure date arithmetic, so it is point-in-time.
# ----------------------------------------------------------------------------------------------
def _ymd(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _weekday(days):
    """0 = Monday ... 6 = Sunday (integer days)."""
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


def xnys_holidays(y0, y1):
    """Regular NYSE full-day holidays of the years y0..y1 as datetime64[D]
    (unscheduled one-off closures are not included)."""
    years = np.arange(int(y0), int(y1) + 1, dtype=np.int64)
    ny = _ymd(years, 1, 1)
    ny_wd = _weekday(ny)
    keep = ny_wd != 5                      # New Year on a Saturday is not observed
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


def rebalance_calendar(y0, y1, w):
    """(R, d) per quarter of the years y0..y1: R = third Friday of Mar/Jun/Sep/Dec rolled back to
    the last session, d = the session w sessions before R."""
    hol = xnys_holidays(int(y0) - 1, int(y1) + 1)
    years = np.repeat(np.arange(int(y0), int(y1) + 1, dtype=np.int64), 4)
    months = np.tile(np.array([3, 6, 9, 12], dtype=np.int64), int(y1) - int(y0) + 1)
    fri3 = _nth_weekday(years, months, 4, 3).astype("datetime64[D]")
    reb = np.busday_offset(fri3, 0, roll="backward", holidays=hol)
    dec = np.busday_offset(reb, -int(w), roll="backward", holidays=hol)
    return reb, dec


def schedule(dates, w):
    """List of (e, x, rp): decision row e (first row on/after d), exit-decision row x (first row
    on/after R; len(dates) if not reached yet) and rp, the row of the close of R_prev."""
    T = len(dates)
    if T == 0:
        return []
    dates = np.asarray(dates).astype("datetime64[D]")
    yrs = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    y0, y1 = int(yrs[0]) - 1, int(yrs[-1]) + 1
    reb, dec = rebalance_calendar(y0, y1, w)
    out = []
    for q in range(1, len(reb)):
        e = int(np.searchsorted(dates, dec[q], side="left"))
        x = int(np.searchsorted(dates, reb[q], side="left"))
        rp = int(np.searchsorted(dates, reb[q - 1], side="right")) - 1
        if e >= T or rp < 0 or rp >= e or not dates[e] < reb[q]:
            continue
        out.append((e, x, rp))
    return out


# ----------------------------------------------------------------------------------------------
# share counts, filings, split guard
# ----------------------------------------------------------------------------------------------
def share_panels(data):
    """shares = marketcap / price (both frozen at the filing), validity, and the row on which the
    latest filing became visible (-1 if none)."""
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    pr = np.asarray(data.extras["sf1_arq_price"], dtype=np.float64)
    T, N = mc.shape
    valid = np.isfinite(mc) & np.isfinite(pr) & (mc > 0) & (pr > 0)
    shares = np.where(valid, mc / np.where(valid, pr, 1.0), np.nan)
    # a new filing becomes visible where marketcap or price change (or data reappear)
    chg = np.zeros((T, N), dtype=bool)
    chg[0] = valid[0]
    if T > 1:
        same = (mc[1:] == mc[:-1]) & (pr[1:] == pr[:-1]) & valid[:-1]
        chg[1:] = valid[1:] & ~same
    rows = np.arange(T, dtype=np.int64)[:, None]
    last_filing = np.maximum.accumulate(np.where(chg, rows, -1), axis=0)
    return shares, valid, last_filing


def split_flag_cumsum(data):
    """Cumulative count of sessions on which close_u(t)/close_u(previous priced row) differs from
    the total-return ratio over the same step by more than SPLIT_TOL."""
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


def select_names(data, e, rp, shares, valid, last_filing, cflag, spy):
    """Columns of the (up to MAX_NAMES) eligible members with the largest g at decision row e,
    and g itself."""
    dates = np.asarray(data.dates).astype("datetime64[D]")
    N = shares.shape[1]
    uni = np.asarray(data.universe[e], dtype=bool)
    trd = np.asarray(data.tradable[e], dtype=bool)
    ok = uni & trd & valid[e] & valid[rp]
    ok[spy] = False
    lf_new = last_filing[e]
    lf_old = last_filing[rp]
    ok &= lf_old >= 0
    # the latest filing as of d became visible after the close of R_prev (pending update)
    ok &= lf_new > rp
    # staleness: at most STALE_DAYS calendar days since the filing became visible
    dd = dates.astype(np.int64)
    ok &= (dd[e] - dd[np.clip(lf_new, 0, None)]) <= STALE_DAYS
    ok &= (dd[rp] - dd[np.clip(lf_old, 0, None)]) <= STALE_DAYS
    # split guard over [5 sessions before the S_old filing date, d]; the filing date is taken
    # as the session before the filing became visible
    a = np.clip(lf_old - GUARD_PRE_SESSIONS - 1, 0, None)
    before = np.where(a > 0, cflag[np.clip(a - 1, 0, None), np.arange(N)], 0)
    ok &= (cflag[e] - before) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = shares[e] / shares[rp] - 1.0
    ok &= np.isfinite(g) & (g >= G_MIN) & (g <= G_MAX)
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return cand, g
    order = np.argsort(-g[cand], kind="stable")
    return cand[order][:MAX_NAMES], g


def target_weights(data, params):
    w = int(params["window_sessions"])
    spy = data.col(CORE)
    if data.universe is None:
        raise ValueError("card universe sp500 required")
    T, N = len(data.dates), len(data.instruments)
    universe = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    delisting = np.asarray(data.delisting, dtype=bool)
    ret_co = np.nan_to_num(np.asarray(data.ret_co, dtype=np.float64))
    ret_oc = np.nan_to_num(np.asarray(data.ret_oc, dtype=np.float64))

    shares, valid, last_filing = share_panels(data)
    cflag = split_flag_cumsum(data)

    # default: 100 % SPY, nothing else (outside the windows the book is SPY)
    W = np.zeros((T, N), dtype=np.float64)
    W[:, spy] = 1.0
    for e, x, rp in schedule(data.dates, w):
        sel, _ = select_names(data, e, rp, shares, valid, last_filing, cflag, spy)
        k = len(sel)
        if k < MIN_NAMES:
            continue
        row = np.zeros(N)
        row[spy] = 1.0 - TILT
        row[sel] = TILT / k
        W[e] = row
        end = min(x, T)
        W[e + 1:end] = np.nan            # hold: no intra-window rebalancing
        # drifted values (fractions of NAV at the fill) to route exit proceeds to SPY
        v_spy = 1.0 - TILT
        v = np.full(k, TILT / k)
        active = np.ones(k, dtype=bool)
        for t in range(e + 1, end):
            if t == e + 1:
                v_spy *= 1.0 + ret_oc[t, spy]
                v *= 1.0 + ret_oc[t, sel]
            else:
                v_spy *= (1.0 + ret_co[t, spy]) * (1.0 + ret_oc[t, spy])
                v *= (1.0 + ret_co[t, sel]) * (1.0 + ret_oc[t, sel])
            out = active & (~universe[t, sel] | delisting[t, sel] | ~listed[t, sel])
            if out.any():
                nav = v_spy + v[active].sum()
                proceeds = v[out].sum()
                W[t, sel[out]] = 0.0
                W[t, spy] = min(1.0, (v_spy + proceeds) / nav) if nav > 0 else 1.0
                v_spy += proceeds
                v[out] = 0.0
                active &= ~out
        # row x (decided after the close of R) keeps the default: 100 % SPY, all names 0
    return W
