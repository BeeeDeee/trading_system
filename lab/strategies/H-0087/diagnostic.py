"""H-0087 mechanism event: members of each newly formed month-end cohort.

Copy of the formation code of strategy.py (separate file by contract). Side +1 for the long leg
(largest vol-scaled local yield fall), -1 for the short leg, at the decision row t where the cohort
is formed (orders fill at the open of t+1). Only universe members that are tradable on t are marked.
"""
import math

import numpy as np

COUNTRY_KEY = {
    "SPY": "US", "EWJ": "JP", "EWU": "GB", "EWC": "CA", "EWA": "AU", "EWL": "CH", "EWD": "SE",
    "EWG": "DE", "EWY": "KR", "EWW": "MX", "EZA": "ZA", "ECH": "CL", "EIS": "IL", "EPOL": "PL",
    "ENZL": "NZ",
}
DATASET = "fred_oecd_10y_yields_1d"

VOL_WINDOW = 63
STD_MONTHS = 36
MIN_CHANGES = 24
MIN_ELIGIBLE = 8


def _month_index(dates):
    return dates.astype("datetime64[M]").astype(np.int64)


def _good_friday_day(years):
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
    mon = dates.astype("datetime64[M]")
    next_first = (mon.astype(np.int64) + 1).astype("datetime64[M]").astype("datetime64[D]")
    e = np.busday_offset(next_first, -1, roll="forward")
    e_int = e.astype(np.int64)
    month_of_year = mon.astype(np.int64) % 12
    year = mon.astype(np.int64) // 12 + 1970
    weekday = (e_int + 3) % 7
    memorial = (month_of_year == 4) & (weekday == 0)
    good_friday = (month_of_year == 2) & (e_int == _good_friday_day(year))
    hol = memorial | good_friday
    if hol.any():
        e = np.where(hol, np.busday_offset(e, -1, roll="forward"), e)
        e_int = e.astype(np.int64)
    return e_int


def decision_rows(dates):
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
                out[t] = True
            had_dec = False
        if is_dec[t]:
            had_dec = True
    return out


def monthly_samples(Y, mon):
    m0 = int(mon[0])
    M = int(mon[-1]) - m0 + 1
    samp = np.full((M, Y.shape[1]), np.nan)
    rel = (mon - m0).astype(np.int64)
    last = np.full(M, -1, dtype=np.int64)
    last[rel] = np.arange(len(mon))
    has = last >= 0
    samp[has] = Y[last[has]]
    return m0, samp


def bond_scores(Y, mon, samp, m0, t, L):
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
    dy = np.where(valid[0] & valid[L], val[0] - val[L], np.nan)
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


def select_legs(scores, tickers):
    names = [n for n in tickers if n in scores]
    N = len(names)
    if N < MIN_ELIGIBLE:
        return [], []
    k = N // 4
    longs = sorted(names, key=lambda n: (-scores[n], -abs(scores[n]), n))[:k]
    shorts = sorted(names, key=lambda n: (scores[n], -abs(scores[n]), n))[:k]
    return longs, shorts


def _daily_log_returns(data):
    return np.log1p(np.asarray(data.ret_co, dtype=float)) + np.log1p(np.asarray(data.ret_oc, dtype=float))


def formation_legs(data, L):
    """List of (t, longs, shorts) for every decision row, plus tickers and their columns."""
    dates = np.asarray(data.dates).astype("datetime64[D]")
    tickers = sorted(COUNTRY_KEY)
    cols = [data.col(tk) for tk in tickers]
    Y = np.column_stack([np.asarray(data.series[DATASET + "." + COUNTRY_KEY[tk]], dtype=float)
                         for tk in tickers])
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
        for j, tk in enumerate(tickers):
            if not bar[t, j] or nbars[t, j] < VOL_WINDOW or not np.isfinite(score[j]):
                continue
            if uni is not None and not uni[t, j]:
                continue
            rows = np.flatnonzero(bar[: t + 1, j])[-VOL_WINDOW:]
            x = lr[rows, j]
            if not np.all(np.isfinite(x)):
                continue
            if not (float(np.std(x, ddof=1)) > 0.0):
                continue
            elig[tk] = float(score[j])
        longs, shorts = select_legs(elig, tickers)
        out.append((int(t), longs, shorts))
    return tickers, cols, out


def events(data, params):
    L = int(params["lookback_months"])
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    if T == 0:
        return mask, side
    tickers, cols, formed = formation_legs(data, L)
    tradable = np.asarray(data.tradable, dtype=bool)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    for t, longs, shorts in formed:
        for names, s in ((longs, 1.0), (shorts, -1.0)):
            for n in names:
                c = cols[tickers.index(n)]
                if not tradable[t, c]:
                    continue
                if uni is not None and not uni[t, c]:
                    continue
                mask[t, c] = True
                side[t, c] = s
    return mask, side
