"""H-0051 mechanism events: at each weekly rebalance with >= 6 eligible ETFs, the long leg (+1,
speculators sold most) and the short leg (-1, speculators bought most); unmarked eligible ETFs are peers.

The book logic below is a verbatim copy of strategy.py (the diagnostic may not import the strategy).
"""
import math

import numpy as np

DATASET = "cftc_cot_legacy_1d"

MARKET = {
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

MIN_HISTORY = 252
DV_WINDOW = 63
MIN_DV = 5e6
VOL_WINDOW = 63
FRESH_DAYS = 10
STALE_DAYS = 10
MIN_ELIGIBLE = 6


def _ffill(x):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(x.shape[0]), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(x.shape[0], np.nan)
    has = idx >= 0
    out[has] = x[idx[has]]
    return out


def market_signal(asof, nc_long, nc_short, oi, z_window):
    T = asof.shape[0]
    a = np.asarray(asof, dtype=np.float64)
    a_ff = _ffill(a)
    prev = np.concatenate([[np.nan], a_ff[:-1]])
    new = np.isfinite(a) & (~np.isfinite(prev) | (a != prev))
    rows = np.flatnonzero(new)
    R = rows.shape[0]
    z_row = np.full(T, np.nan)
    report_row = np.full(T, -1, dtype=np.int64)
    if R == 0:
        return new, z_row, report_row
    ra = a[rows]
    nl = np.asarray(nc_long, dtype=np.float64)[rows]
    ns = np.asarray(nc_short, dtype=np.float64)[rows]
    o = np.asarray(oi, dtype=np.float64)[rows]
    net = nl - ns
    flow = np.full(R, np.nan)
    if R > 1:
        consecutive = np.abs((ra[1:] - ra[:-1]) - 7.0) < 0.5
        good_oi = np.isfinite(o[1:]) & (o[1:] > 0)
        f = (net[1:] - net[:-1]) / np.where(good_oi, o[1:], np.nan)
        flow[1:] = np.where(consecutive & good_oi, f, np.nan)
    W = int(z_window)
    fin = np.isfinite(flow)
    v = np.where(fin, flow, 0.0)
    c0 = np.concatenate([[0.0], np.cumsum(fin.astype(np.float64))])
    s1 = np.concatenate([[0.0], np.cumsum(v)])
    s2 = np.concatenate([[0.0], np.cumsum(v * v)])
    i = np.arange(R)
    lo = np.maximum(i - W, 0)
    n = c0[i] - c0[lo]
    m1 = s1[i] - s1[lo]
    m2 = s2[i] - s2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (m2 - m1 * m1 / n) / (n - 1.0)
        var = np.where(var > 0, var, np.nan)
        sd = np.sqrt(var)
        z = flow / sd
    z = np.where(n >= W / 2.0, z, np.nan)
    z = np.where(np.isfinite(z), z, np.nan)
    k = np.cumsum(new) - 1
    has = k >= 0
    z_row[has] = z[k[has]]
    report_row[has] = rows[k[has]]
    return new, z_row, report_row


def _book(data, params):
    lf = float(params["leg_fraction"])
    zw = int(params["z_window_weeks"])
    T = data.dates.shape[0]
    N = len(data.instruments)
    day = data.dates.astype(np.int64)

    cols = []
    Z = np.full((T, N), np.nan)
    REP = np.full((T, N), -1, dtype=np.int64)
    any_new = np.zeros(T, dtype=bool)
    for name, code in MARKET.items():
        j = data.col(name)
        cols.append(j)
        key = DATASET + "." + code + "."
        new, z_row, rep = market_signal(
            data.series[key + "asof_date"],
            data.series[key + "nc_long"],
            data.series[key + "nc_short"],
            data.series[key + "open_interest"],
            zw,
        )
        Z[:, j] = z_row
        REP[:, j] = rep
        any_new |= new
    cols = np.array(cols, dtype=np.int64)

    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    tr = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    tr = np.where(listed & np.isfinite(close), tr, np.nan)
    hist = np.cumsum(listed.astype(np.int64), axis=0)
    univ = data.universe

    w = np.full((T, N), np.nan)
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)
    last_reb = None
    for t in range(T):
        if not any_new[t]:
            if last_reb is not None and day[t] - day[last_reb] > STALE_DAYS:
                w[t] = 0.0
            continue
        last_reb = t
        lo = max(0, t - DV_WINDOW + 1)
        lov = max(0, t - VOL_WINDOW + 1)
        z = Z[t, cols]
        rep = REP[t, cols]
        fresh = (rep >= 0) & (day[t] - day[np.maximum(rep, 0)] <= FRESH_DAYS)
        elig = np.isfinite(z) & fresh
        elig &= hist[t, cols] >= MIN_HISTORY
        elig &= tradable[t, cols] & listed[t, cols]
        if univ is not None:
            elig &= np.asarray(univ[t, cols], dtype=bool)
        dvw = dv[lo:t + 1][:, cols]
        dvok = np.isfinite(dvw).any(axis=0)
        med = np.full(cols.shape[0], np.nan)
        if dvok.any():
            med[dvok] = np.nanmedian(dvw[:, dvok], axis=0)
        elig &= np.isfinite(med) & (med >= MIN_DV)
        rw = tr[lov:t + 1][:, cols]
        nobs = np.isfinite(rw).sum(axis=0)
        vol = np.full(cols.shape[0], np.nan)
        vok = nobs >= 2
        if vok.any():
            vol[vok] = np.nanstd(rw[:, vok], axis=0, ddof=1)
        elig &= np.isfinite(vol) & (vol > 0)

        row = np.zeros(N)
        ne = int(elig.sum())
        if ne >= MIN_ELIGIBLE:
            k = max(1, int(math.floor(lf * ne + 0.5 + 1e-9)))
            k = min(k, ne // 2)
            idx = np.flatnonzero(elig)
            order = idx[np.argsort(z[idx], kind="stable")]
            lo_leg = order[:k]
            hi_leg = order[-k:]
            il = 1.0 / vol[lo_leg]
            ih = 1.0 / vol[hi_leg]
            row[cols[lo_leg]] = 0.5 * il / il.sum()
            row[cols[hi_leg]] = -0.5 * ih / ih.sum()
            long_m[t, cols[lo_leg]] = True
            short_m[t, cols[hi_leg]] = True
        w[t] = row
    return w, long_m, short_m


def events(data, params):
    _, long_m, short_m = _book(data, params)
    mask = long_m | short_m
    side = np.where(long_m, 1.0, np.where(short_m, -1.0, 0.0))
    return mask, side
