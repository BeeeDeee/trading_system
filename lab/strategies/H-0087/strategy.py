"""H-0087 Bond-to-equity cross-asset momentum.

15 country ETFs. At each month-end (last XNYS session, approximated by weekday arithmetic plus
Good Friday / Memorial Day), score each country by the vol-scaled change of its OECD monthly 10y
yield over L months (score = -dy / (std of monthly changes over 36 months * sqrt(L))), long the
k = floor(N/4) highest, short the k lowest, legs weighted 1/sigma (63-session daily log total
return vol) to +0.5/H and -0.5/H, book = sum of the H most recent monthly cohort slots.
"""
import math

import numpy as np

PARAMS = {"hold_months": 6, "lookback_months": 12}

# ETF -> key of fred_oecd_10y_yields_1d (catalog fields)
COUNTRY_KEY = {
    "SPY": "US", "EWJ": "JP", "EWU": "GB", "EWC": "CA", "EWA": "AU", "EWL": "CH", "EWD": "SE",
    "EWG": "DE", "EWY": "KR", "EWW": "MX", "EZA": "ZA", "ECH": "CL", "EIS": "IL", "EPOL": "PL",
    "ENZL": "NZ",
}
DATASET = "fred_oecd_10y_yields_1d"

VOL_WINDOW = 63          # sessions for sigma_i and minimum history
STD_MONTHS = 36          # usable months in the change window ending at y(0)
MIN_CHANGES = 24         # minimum month-over-month changes for s_i
MIN_ELIGIBLE = 8         # fewer eligible instruments: no new cohort
PUB_LAG_MONTHS = 2       # value of month m is usable from month m+2 (already stamped so by the data)


# ----------------------------------------------------------------------------------------------
# calendar
# ----------------------------------------------------------------------------------------------
def _month_index(dates):
    return dates.astype("datetime64[M]").astype(np.int64)


def _good_friday_day(years):
    """Day number (days since the numpy epoch) of Good Friday for integer years (Gregorian computus)."""
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
    n = h + l - 7 * m + 114
    month = n // 31
    day = n % 31 + 1
    month_idx = (y - 1970) * 12 + (month - 1)
    first = month_idx.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    return first + day - 1 - 2


def last_session_of_month(dates):
    """Approximate last XNYS session of the month of each date, from the date alone.

    Last weekday of the month (np.busday_offset), moved one weekday earlier when that weekday is
    Memorial Day (a Monday May 31) or Good Friday (March 30/31). Other closures are ignored.
    Returns int64 day numbers.
    """
    mon = dates.astype("datetime64[M]")
    next_first = (mon.astype(np.int64) + 1).astype("datetime64[M]").astype("datetime64[D]")
    e = np.busday_offset(next_first, -1, roll="forward")
    e_int = e.astype(np.int64)
    month_of_year = mon.astype(np.int64) % 12          # 0 = January
    year = mon.astype(np.int64) // 12 + 1970
    weekday = (e_int + 3) % 7                         # 0 = Monday
    memorial = (month_of_year == 4) & (weekday == 0)
    good_friday = (month_of_year == 2) & (e_int == _good_friday_day(year))
    hol = memorial | good_friday
    if hol.any():
        e = np.where(hol, np.busday_offset(e, -1, roll="forward"), e)
        e_int = e.astype(np.int64)
    return e_int


def decision_rows(dates):
    """(T,) bool: month-end decision rows, point-in-time (uses the date of row t and earlier rows).

    Row t decides if its date is the (approximate) last session of its month. Fallback for an
    unforeseen month-end closure: if a calendar month had rows but no decision row, the first row
    of the next month decides.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = len(dates)
    if T == 0:
        return np.zeros(0, dtype=bool)
    is_dec = dates.astype(np.int64) == last_session_of_month(dates)
    mon = _month_index(dates)
    out = is_dec.copy()
    had_dec = False
    for t in range(T):
        if t > 0 and mon[t] != mon[t - 1]:
            if not had_dec:
                out[t] = True     # previous month ended without a decision row
            had_dec = False
        if is_dec[t]:
            had_dec = True
    return out


# ----------------------------------------------------------------------------------------------
# bond score
# ----------------------------------------------------------------------------------------------
def monthly_samples(Y, mon):
    """Value of the step series at the last row of each calendar month present in the data.

    Y (T, K), mon (T,) month index. Returns (m0, samp (M, K)) with NaN for months without rows.
    """
    m0 = int(mon[0])
    M = int(mon[-1]) - m0 + 1
    samp = np.full((M, Y.shape[1]), np.nan)
    rel = (mon - m0).astype(np.int64)
    last = np.full(M, -1, dtype=np.int64)
    last[rel] = np.arange(len(mon))       # later rows overwrite earlier ones -> last row of month
    has = last >= 0
    samp[has] = Y[last[has]]
    return m0, samp


def bond_scores(Y, mon, samp, m0, t, L):
    """Score per key at decision row t.

    The usable value at row t (Y[t]) is the latest monthly yield, y(0). Earlier data months are
    the values at the last rows of earlier calendar months. A monthly value is missing if it is
    NaN or identical to the previous month's value (the step series carried it: no new value was
    published for that month; no filling). Returns (K,) scores, NaN where not defined.
    """
    ci = int(mon[t]) - m0
    nwin = max(STD_MONTHS, L + 1) + 1
    K = Y.shape[1]
    val = np.full((nwin, K), np.nan)
    inview = np.zeros(nwin, dtype=bool)
    for n in range(nwin):
        c = ci - n
        if c < 0:
            break
        inview[n] = True
        val[n] = Y[t] if n == 0 else samp[c]
    fin = np.isfinite(val)
    valid = fin.copy()
    for n in range(nwin - 1):
        if inview[n + 1]:
            carried = fin[n] & fin[n + 1] & (val[n] == val[n + 1])
            valid[n] &= ~carried
    # lookback change
    dy = np.where(valid[0] & valid[L], val[0] - val[L], np.nan)
    # month-over-month changes over the STD_MONTHS usable months ending at y(0)
    ch = val[: STD_MONTHS - 1] - val[1:STD_MONTHS]
    chv = valid[: STD_MONTHS - 1] & valid[1:STD_MONTHS]
    ch = np.where(chv, ch, np.nan)
    cnt = chv.sum(axis=0)
    score = np.full(K, np.nan)
    for j in range(K):
        if cnt[j] < MIN_CHANGES or not np.isfinite(dy[j]):
            continue
        x = ch[chv[:, j], j]
        sd = float(np.std(x, ddof=1))
        if not (sd > 0.0):
            continue
        score[j] = -dy[j] / (sd * math.sqrt(L))
    return score


# ----------------------------------------------------------------------------------------------
# cohort
# ----------------------------------------------------------------------------------------------
def select_legs(scores, tickers):
    """scores: dict ticker -> score of the eligible instruments. Returns (longs, shorts) lists.

    k = floor(N/4) highest are long, k lowest short; ties by larger |score|, then ticker.
    Returns empty lists if fewer than MIN_ELIGIBLE eligible.
    """
    names = [n for n in tickers if n in scores]
    N = len(names)
    if N < MIN_ELIGIBLE:
        return [], []
    k = N // 4
    longs = sorted(names, key=lambda n: (-scores[n], -abs(scores[n]), n))[:k]
    shorts = sorted(names, key=lambda n: (scores[n], -abs(scores[n]), n))[:k]
    return longs, shorts


def leg_weights(names, sigma, total):
    inv = np.array([1.0 / sigma[n] for n in names])
    w = total * inv / inv.sum()
    return dict(zip(names, w))


def _daily_log_returns(data):
    lr = np.log1p(np.asarray(data.ret_co, dtype=float)) + np.log1p(np.asarray(data.ret_oc, dtype=float))
    return lr


def formation(data, H, L):
    """Run the monthly formation. Returns (tickers, cols, bar, uni, list of (t, longs, shorts, cohort)).

    cohort is the (15,) weight vector of the cohort formed at decision row t (zeros if none formed).
    """
    dates = np.asarray(data.dates).astype("datetime64[D]")
    tickers = sorted(COUNTRY_KEY)                     # alphabetical
    cols = [data.col(tk) for tk in tickers]           # raises if an instrument is missing
    Y = np.column_stack([np.asarray(data.series[DATASET + "." + COUNTRY_KEY[tk]], dtype=float)
                         for tk in tickers])          # raises if a series is missing
    close = np.asarray(data.close, dtype=float)[:, cols]
    bar = np.isfinite(close)
    lr = _daily_log_returns(data)[:, cols]
    nbars = np.cumsum(bar, axis=0)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)[:, cols]

    mon = _month_index(dates)
    m0, samp = monthly_samples(Y, mon)
    dec = decision_rows(dates)

    out = []
    for t in np.flatnonzero(dec):
        score = bond_scores(Y, mon, samp, m0, t, L)
        elig = {}
        sigma = {}
        for j, tk in enumerate(tickers):
            if not bar[t, j] or nbars[t, j] < VOL_WINDOW or not np.isfinite(score[j]):
                continue
            if uni is not None and not uni[t, j]:
                continue
            rows = np.flatnonzero(bar[: t + 1, j])[-VOL_WINDOW:]
            x = lr[rows, j]
            if not np.all(np.isfinite(x)):
                continue
            sd = float(np.std(x, ddof=1))
            if not (sd > 0.0):
                continue
            elig[tk] = float(score[j])
            sigma[tk] = sd
        longs, shorts = select_legs(elig, tickers)
        cohort = np.zeros(len(tickers))
        if longs:
            for n, w in leg_weights(longs, sigma, 0.5 / H).items():
                cohort[tickers.index(n)] += w
            for n, w in leg_weights(shorts, sigma, -0.5 / H).items():
                cohort[tickers.index(n)] += w
        out.append((int(t), longs, shorts, cohort))
    return tickers, cols, bar, uni, out


def target_weights(data, params):
    H = int(params["hold_months"])
    L = int(params["lookback_months"])
    T = len(data.dates)
    Ninst = len(data.instruments)
    W = np.full((T, Ninst), np.nan)
    if T == 0:
        return W
    tickers, cols, bar, uni, formed = formation(data, H, L)
    cohorts = []                                      # one (15,) array per decision row (slot)
    for t, _longs, _shorts, cohort in formed:
        cohorts.append(cohort.copy())
        # an ETF without a price bar (or out of the universe) on a rebalance day: weight 0 in
        # every live cohort from that day on
        gone = ~bar[t]
        if uni is not None:
            gone = gone | ~uni[t]
        live = cohorts[-H:]
        for c in live:
            c[gone] = 0.0
        book = np.sum(live, axis=0)
        gross = float(np.abs(book).sum())
        if gross > 1.0:
            book = book / gross
        row = np.zeros(Ninst)
        row[cols] = book
        W[t] = row
    return W
