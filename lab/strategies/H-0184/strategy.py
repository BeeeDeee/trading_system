"""H-0184: Treasury duration-supply surge -> bond sleeve of a 60/40 SPY/bond book in 12 monthly cohorts.

Rules of the card and where they are implemented:
 (1) day offerings of 5y-30y coupon auctions ............ auction_offerings()
 (2) trailing supply S_t over 252 sessions ............... trailing_supply()
 (3) base B_t = mean S at t-252, t-273, ..., t-252(1+by) .. base_supply()
 (4) g_t = ln(S_t / B_t), state HIGH / LOW / NEUTRAL ..... supply_state()
 month-end decisions (last XNYS session of the month) .... month_end_rows()
 12 monthly cohorts, sleeve TLT / SHY / IEF, SPY 0.60 .... cohort_counts(), target_weights()
 Card v2: AGG and LQD are peer-only instruments (event-study controls); the book holds 0 in them.
"""
import numpy as np

PARAMS = {"base_years": 3, "supply_cut": 0.1}

DS = "treasury_auctions_1d"
YEAR = 252            # sessions in the trailing supply window and the base offset
STEP = 21             # sessions between base sampling points
COHORTS = 12          # cohort hold, fixed by the card (not a parameter)
W_SPY = 0.60
W_BOND = 0.40


# ----------------------------------------------------------------------------- calendar
def _easter_good_friday(years):
    """Good Friday (datetime64[D]) of each Gregorian year (anonymous Gregorian computus)."""
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
    easter = first + _days(day - 1)
    return easter - _days(2)


def _days(k):
    return np.asarray(k, dtype=np.int64).astype("timedelta64[D]")


def _memorial_day(years):
    """Last Monday of May (datetime64[D]) of each year."""
    y = np.asarray(years, dtype=np.int64)
    may31 = ((y - 1970) * 12 + 4).astype("datetime64[M]").astype("datetime64[D]") + _days(30)
    return np.busday_offset(may31, 0, roll="backward", weekmask="Mon")


def nyse_month_end_holidays(dates):
    """The two NYSE holidays that can fall on the last weekday of a month: Good Friday, Memorial Day."""
    d = np.asarray(dates).astype("datetime64[D]")
    yrs = d.astype("datetime64[Y]").astype(np.int64) + 1970
    years = np.arange(int(yrs.min()) - 1, int(yrs.max()) + 2)
    return np.sort(np.concatenate([_easter_good_friday(years), _memorial_day(years)]))


def month_end_rows(dates):
    """(T,) bool: row t is the last XNYS session of its month, decided from the date of row t only.

    The next session after d is the next weekday that is not Good Friday or Memorial Day (the only NYSE
    holidays that can be the last weekday of a month); d is a month-end if that session is in another month.
    Fallback for unscheduled closures: if a month ended without a flagged row, the first row of the next
    month is the decision row for that month.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    hol = nyse_month_end_holidays(d)
    nxt = np.busday_offset(d + _days(1), 0, roll="forward", holidays=hol)
    mi = d.astype("datetime64[M]").astype(np.int64)
    is_me = nxt.astype("datetime64[M]").astype(np.int64) != mi
    dec = is_me.copy()
    if T > 1:
        idx = np.where(is_me, np.arange(T), -1)
        last_me = np.maximum.accumulate(idx)
        prev_last = np.concatenate([[-1], last_me[:-1]])           # last flagged row <= t-1
        prev_month = np.concatenate([[mi[0]], mi[:-1]])
        prev_has = (prev_last >= 0) & (mi[np.maximum(prev_last, 0)] == prev_month)
        new_month = np.concatenate([[False], mi[1:] != mi[:-1]])
        dec |= new_month & ~prev_has
    return dec


# ----------------------------------------------------------------------------- supply signal
def auction_offerings(last_sess, next_sess, next_off):
    """Day offerings (T,) and the number of auction days whose offering is missing.

    Auction day d: last_coupon5_sessions(d) == 0. Offering = next_coupon5_offering_usd on the previous
    session p if next_coupon5_sessions(p) == 1, else missing -> 0.
    """
    last_sess = np.asarray(last_sess, dtype=np.float64)
    next_sess = np.asarray(next_sess, dtype=np.float64)
    next_off = np.asarray(next_off, dtype=np.float64)
    T = last_sess.shape[0]
    auction = last_sess == 0                                  # NaN -> False
    prev_ok = np.zeros(T, dtype=bool)
    prev_off = np.full(T, np.nan)
    if T > 1:
        prev_ok[1:] = next_sess[:-1] == 1
        prev_off[1:] = next_off[:-1]
    valid = auction & prev_ok & np.isfinite(prev_off)
    off = np.where(valid, prev_off, 0.0)
    n_missing = int(np.sum(auction & ~valid))
    return off, n_missing


def trailing_supply(off, window=YEAR):
    """S_t = sum of day offerings over the `window` sessions ending at t; NaN before a full window."""
    off = np.asarray(off, dtype=np.float64)
    T = off.shape[0]
    S = np.full(T, np.nan)
    if T >= window:
        cs = np.concatenate([[0.0], np.cumsum(off)])
        S[window - 1:] = cs[window:] - cs[:T - window + 1]
    return S


def base_supply(S, base_years):
    """B_t = mean of S at t-252, t-273, ..., t-252*(1+base_years); NaN unless every point is defined."""
    S = np.asarray(S, dtype=np.float64)
    T = S.shape[0]
    lags = np.arange(YEAR, YEAR * (1 + int(base_years)) + 1, STEP)
    acc = np.zeros(T)
    for L in lags:
        sh = np.full(T, np.nan)
        if L < T:
            sh[L:] = S[:T - L]
        acc = acc + sh                                        # NaN propagates: undefined point -> undefined B
    return acc / len(lags)


def supply_surge(S, B):
    ok = np.isfinite(S) & np.isfinite(B) & (S > 0) & (B > 0)
    g = np.full(S.shape[0], np.nan)
    g[ok] = np.log(S[ok] / B[ok])
    return g


def supply_state(g, supply_cut):
    """+1 HIGH (g >= cut), -1 LOW (g <= -cut), 0 NEUTRAL or undefined."""
    st = np.zeros(g.shape[0], dtype=np.int64)
    fin = np.isfinite(g)
    st[fin & (g >= supply_cut)] = 1
    st[fin & (g <= -supply_cut)] = -1
    return st


def surge_signal(data, params):
    last_sess = data.series[DS + ".last_coupon5_sessions"]
    next_sess = data.series[DS + ".next_coupon5_sessions"]
    next_off = data.series[DS + ".next_coupon5_offering_usd"]
    off, _ = auction_offerings(last_sess, next_sess, next_off)
    S = trailing_supply(off)
    B = base_supply(S, int(params["base_years"]))
    return supply_surge(S, B)


# ----------------------------------------------------------------------------- book
def book_decision_rows(data):
    """Month-end rows from the first month-end at which SPY, SHY, IEF and TLT all have a price."""
    dec = month_end_rows(data.dates)
    close = np.asarray(data.close, dtype=np.float64)
    cols = [data.col("SPY"), data.col("SHY"), data.col("IEF"), data.col("TLT")]
    priced = np.all(np.isfinite(close[:, cols]), axis=1)
    cand = np.flatnonzero(dec & priced)
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    rows = np.flatnonzero(dec)
    return rows[rows >= cand[0]]


def cohort_counts(states):
    """For each decision j: (#HIGH, #LOW, #NEUTRAL) among the cohorts opened at decisions j-11..j.
    Cohorts that would have started before the first decision count as NEUTRAL (IEF)."""
    st = np.asarray(states, dtype=np.int64)
    n = st.shape[0]
    hi = np.concatenate([[0], np.cumsum(st == 1)])
    lo = np.concatenate([[0], np.cumsum(st == -1)])
    j = np.arange(n)
    start = np.maximum(j + 1 - COHORTS, 0)
    n_hi = hi[j + 1] - hi[start]
    n_lo = lo[j + 1] - lo[start]
    n_ne = COHORTS - n_hi - n_lo
    return n_hi, n_lo, n_ne


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    spy, shy, ief, tlt = data.col("SPY"), data.col("SHY"), data.col("IEF"), data.col("TLT")
    g = surge_signal(data, params)
    st = supply_state(g, float(params["supply_cut"]))
    rows = book_decision_rows(data)
    W = np.full((T, N), np.nan)
    if rows.size == 0:
        return W
    n_hi, n_lo, n_ne = cohort_counts(st[rows])
    W[rows, :] = 0.0         # IEI, TLH (mechanism-test only) and AGG, LQD (peer-only, card v2) are never held
    W[rows, spy] = W_SPY
    W[rows, tlt] = W_BOND * n_hi / COHORTS
    W[rows, shy] = W_BOND * n_lo / COHORTS
    W[rows, ief] = W_BOND * n_ne / COHORTS
    return W
