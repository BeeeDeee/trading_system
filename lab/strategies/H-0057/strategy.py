"""H-0057 hedging-pressure premium: CFTC speculator positioning -> commodity and currency ETFs.

Monthly, dollar-neutral. At the first session of each calendar month on which a new COT report is
released: score each market by the z-score of its recent non-commercial net position (mean of the last
avg_reports reports) against its own last z_window_weeks reports; split the eligible ETFs at the median
of their 63-session total return; within each half long the k highest-z and short the k lowest-z ETFs
(k = 1 if the half has < 8 ETFs, else 2); inverse-volatility weights, +0.5 long / -0.5 short.
"""
import math

import numpy as np

PARAMS = {"avg_reports": 4, "z_window_weeks": 156}

DATASET = "cftc_cot_legacy_1d"

# ETF -> CFTC contract market code (card.signal.description). All currency futures are long the foreign
# currency, the same direction as the currency ETFs, so no sign flips.
CFTC_CODE = {
    "GLD": "088691",
    "SLV": "084691",
    "USO": "067651",
    "UNG": "023651",
    "UGA": "111659",
    "CORN": "002602",
    "WEAT": "001602",
    "SOYB": "005602",
    "CPER": "085692",
    "PPLT": "076651",
    "PALL": "075651",
    "CANE": "080732",
    "FXE": "099741",
    "FXY": "097741",
    "FXB": "096742",
    "FXA": "232741",
    "FXC": "090741",
    "FXF": "092741",
}

RET_SESSIONS = 63        # past-return split, volatility window, minimum price history
MAX_REPORT_AGE_DAYS = 21  # latest report at most 21 calendar days old
MIN_ELIGIBLE = 8          # fewer eligible ETFs -> no positions that month
Z_COVERAGE = 0.75         # at least 75 % of z_window_weeks reports
BIG_HALF = 8              # k = 2 when a half has >= 8 ETFs, else k = 1


def report_rows(nc_long, nc_short, open_interest):
    """Rows on which a new report becomes known, and its S = (nc_long - nc_short) / open_interest.

    Works for a forward-filled series (a new report = any of the three values changes) and for a sparse one
    (a new report = values present after a row without values). Rows <= t only.
    """
    nl = np.asarray(nc_long, dtype=float)
    ns = np.asarray(nc_short, dtype=float)
    oi = np.asarray(open_interest, dtype=float)
    finite = np.isfinite(nl) & np.isfinite(ns) & np.isfinite(oi)
    prev_finite = np.zeros_like(finite)
    prev_finite[1:] = finite[:-1]
    changed = np.zeros_like(finite)
    changed[1:] = (nl[1:] != nl[:-1]) | (ns[1:] != ns[:-1]) | (oi[1:] != oi[:-1])
    new = finite & (~prev_finite | changed)
    positive_oi = np.zeros_like(finite)
    positive_oi[finite] = oi[finite] > 0
    new &= positive_oi
    rows = np.flatnonzero(new)
    s = (nl[rows] - ns[rows]) / oi[rows]
    return rows, s


def market_z(rows, s, day, t, avg_reports, z_window):
    """z of one market after the close of row t from reports released on rows <= t; NaN if not valid."""
    n = int(np.searchsorted(rows, t, side="right"))
    if n < avg_reports or n == 0:
        return np.nan
    if day[t] - day[rows[n - 1]] > MAX_REPORT_AGE_DAYS:
        return np.nan
    window = s[max(0, n - z_window):n]
    if window.size < math.ceil(Z_COVERAGE * z_window) or window.size < 2:
        return np.nan
    sd = float(np.std(window, ddof=1))
    if not sd > 0:
        return np.nan
    p = float(np.mean(s[n - avg_reports:n]))
    return (p - float(np.mean(window))) / sd


def select_weights(z, ret, vol, eligible):
    """Weights over the given columns for one decision date (zeros if fewer than 8 are eligible)."""
    m = z.shape[0]
    w = np.zeros(m)
    idx = np.flatnonzero(eligible)
    n = idx.size
    if n < MIN_ELIGIBLE:
        return w
    order = idx[np.lexsort((idx, ret[idx]))]      # ascending past return, ties by column
    n_low = (n + 1) // 2                           # odd count: the middle ETF goes to the low half
    longs = []
    shorts = []
    for half in (order[:n_low], order[n_low:]):
        k = 1 if half.size < BIG_HALF else 2
        zo = half[np.lexsort((half, z[half]))]     # ascending z, ties by column
        shorts.extend(zo[:k].tolist())
        longs.extend(zo[zo.size - k:].tolist())
    longs = np.array(longs, dtype=int)
    shorts = np.array(shorts, dtype=int)
    inv_l = 1.0 / vol[longs]
    inv_s = 1.0 / vol[shorts]
    w[longs] = 0.5 * inv_l / inv_l.sum()
    w[shorts] = -0.5 * inv_s / inv_s.sum()
    return w


def target_weights(data, params):
    avg_reports = int(params["avg_reports"])
    z_window = int(params["z_window_weeks"])

    dates = data.dates
    T = dates.shape[0]
    N = len(data.instruments)
    day = dates.astype("datetime64[D]").astype(np.int64)
    month = dates.astype("datetime64[M]").astype(np.int64)

    etfs = list(CFTC_CODE.keys())
    cols = np.array([data.col(e) for e in etfs], dtype=int)
    M = len(etfs)

    # COT reports per market (index the series directly: a missing key raises)
    markets = []
    any_report = np.zeros(T, dtype=bool)
    for e in etfs:
        code = CFTC_CODE[e]
        nl = data.series[DATASET + "." + code + ".nc_long"]
        ns = data.series[DATASET + "." + code + ".nc_short"]
        oi = data.series[DATASET + "." + code + ".open_interest"]
        rows, s = report_rows(nl, ns, oi)
        markets.append((rows, s))
        any_report[rows] = True

    # decision rows: first session of the calendar month on which a new report is released
    rep_idx = np.flatnonzero(any_report)
    if rep_idx.size:
        rep_month = month[rep_idx]
        first = np.ones(rep_idx.size, dtype=bool)
        first[1:] = rep_month[1:] != rep_month[:-1]
        decision_rows = rep_idx[first]
    else:
        decision_rows = rep_idx

    # prices of the card's ETFs
    ret_co = np.nan_to_num(np.asarray(data.ret_co, dtype=float)[:, cols], nan=0.0)
    ret_oc = np.nan_to_num(np.asarray(data.ret_oc, dtype=float)[:, cols], nan=0.0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0          # daily total return, 0 without trading
    logcum = np.zeros((T + 1, M))
    logcum[1:] = np.cumsum(np.log1p(r), axis=0)
    close = np.asarray(data.close, dtype=float)[:, cols]
    has_price = np.isfinite(close)
    hist = np.cumsum(has_price, axis=0)                 # sessions of price history up to row t
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    dvol = np.asarray(data.dollar_volume, dtype=float)[:, cols]
    traded = np.zeros((T, M), dtype=bool)
    fin = np.isfinite(dvol)
    traded[fin] = dvol[fin] > 0
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)[:, cols]

    w = np.full((T, N), np.nan)
    L = RET_SESSIONS
    for t in decision_rows:
        row = np.zeros(N)
        if t >= L - 1:
            z = np.array([market_z(rows, s, day, t, avg_reports, z_window) for rows, s in markets])
            a = max(t - L + 1, 0)
            ret = np.exp(logcum[t + 1] - logcum[a]) - 1.0
            vol = np.std(r[a:t + 1], axis=0, ddof=1)
            eligible = (
                (hist[t] >= L)
                & tradable[t]
                & has_price[t]
                & traded[t]
                & np.isfinite(z)
                & np.isfinite(vol)
                & (vol > 0)
            )
            if universe is not None:
                eligible &= universe[t]
            row[cols] = select_weights(z, ret, vol, eligible)
        w[t] = row
    return w
