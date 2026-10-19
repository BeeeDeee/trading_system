"""H-0057 mechanism events: every ETF selected into the long set (+1) or short set (-1) at a monthly decision.

Self-contained copy of the selection logic of strategy.py (same rules, same parameters), so the event set
is exactly the set of non-zero positions the strategy takes on its decision rows.
"""
import math

import numpy as np

DATASET = "cftc_cot_legacy_1d"

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

RET_SESSIONS = 63
MAX_REPORT_AGE_DAYS = 21
MIN_ELIGIBLE = 8
Z_COVERAGE = 0.75
BIG_HALF = 8


def _report_rows(nc_long, nc_short, open_interest):
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


def _market_z(rows, s, day, t, avg_reports, z_window):
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


def _select_sides(z, ret, eligible):
    """+1 for the long set, -1 for the short set, 0 otherwise (all 0 if fewer than 8 eligible)."""
    side = np.zeros(z.shape[0])
    idx = np.flatnonzero(eligible)
    n = idx.size
    if n < MIN_ELIGIBLE:
        return side
    order = idx[np.lexsort((idx, ret[idx]))]
    n_low = (n + 1) // 2
    for half in (order[:n_low], order[n_low:]):
        k = 1 if half.size < BIG_HALF else 2
        zo = half[np.lexsort((half, z[half]))]
        side[zo[:k]] = -1.0
        side[zo[zo.size - k:]] = 1.0
    return side


def events(data, params):
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

    markets = []
    any_report = np.zeros(T, dtype=bool)
    for e in etfs:
        code = CFTC_CODE[e]
        nl = data.series[DATASET + "." + code + ".nc_long"]
        ns = data.series[DATASET + "." + code + ".nc_short"]
        oi = data.series[DATASET + "." + code + ".open_interest"]
        rows, s = _report_rows(nl, ns, oi)
        markets.append((rows, s))
        any_report[rows] = True

    rep_idx = np.flatnonzero(any_report)
    if rep_idx.size:
        rep_month = month[rep_idx]
        first = np.ones(rep_idx.size, dtype=bool)
        first[1:] = rep_month[1:] != rep_month[:-1]
        decision_rows = rep_idx[first]
    else:
        decision_rows = rep_idx

    ret_co = np.nan_to_num(np.asarray(data.ret_co, dtype=float)[:, cols], nan=0.0)
    ret_oc = np.nan_to_num(np.asarray(data.ret_oc, dtype=float)[:, cols], nan=0.0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    logcum = np.zeros((T + 1, M))
    logcum[1:] = np.cumsum(np.log1p(r), axis=0)
    close = np.asarray(data.close, dtype=float)[:, cols]
    has_price = np.isfinite(close)
    hist = np.cumsum(has_price, axis=0)
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    dvol = np.asarray(data.dollar_volume, dtype=float)[:, cols]
    traded = np.zeros((T, M), dtype=bool)
    fin = np.isfinite(dvol)
    traded[fin] = dvol[fin] > 0
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)[:, cols]

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    L = RET_SESSIONS
    for t in decision_rows:
        if t < L - 1:
            continue
        z = np.array([_market_z(rows, s, day, t, avg_reports, z_window) for rows, s in markets])
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
        sd = _select_sides(z, ret, eligible)
        sel = sd != 0
        mask[t, cols[sel]] = True
        side[t, cols[sel]] = sd[sel]
    return mask, side
