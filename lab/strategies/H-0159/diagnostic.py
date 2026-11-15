"""H-0159 mechanism events: on each shock-week decision (|z| >= z_news), every stock in a leg (top or bottom vol
quintile within its beta tercile) is an event. Side: widening -1 HIGH / +1 LOW; tightening +1 HIGH / -1 LOW.
Same pipeline as strategy.py (copied: diagnostic.py is a separate file)."""
import math

import numpy as np

WEEK = 5
SHOCK_WINDOW = 104
SHOCK_MIN = 78
BETA_DAYS = 252
BETA_MIN = 200
VOL_MIN_FRAC = 0.8
N_TERCILES = 3
QUINTILE = 5
MIN_LEG = 5
DATASET = "fred_moody_yields_1d"


def _civil(dates):
    d = dates.astype("datetime64[D]")
    days = d.astype(np.int64)
    weekday = (days + 3) % 7
    year = d.astype("datetime64[Y]").astype(np.int64) + 1970
    mstart = d.astype("datetime64[M]")
    month = mstart.astype(np.int64) % 12 + 1
    day = (d - mstart.astype("datetime64[D]")).astype(np.int64) + 1
    return weekday, year, month, day


def _add_days(dates, k):
    return (dates.astype("datetime64[D]").astype(np.int64) + int(k)).astype("datetime64[D]")


def _easter(year):
    y = np.asarray(year, dtype=np.int64)
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
    return month, day


def _friday_holiday(fri):
    _, fy, fm, fd = _civil(fri)
    _, sy, sm, sd = _civil(_add_days(fri, 2))
    em, ed = _easter(sy)
    good_friday = (sm == em) & (sd == ed)
    fixed = (((fm == 1) & (fd == 1))
             | ((fm == 7) & ((fd == 4) | (fd == 3)))
             | ((fm == 12) & ((fd == 25) | (fd == 24)))
             | ((fy >= 2022) & (fm == 6) & ((fd == 19) | (fd == 18))))
    return good_friday | fixed


def decision_days(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    wd, _, _, _ = _civil(dates)
    thursday_last = (wd == 3) & _friday_holiday(_add_days(dates, 1))
    return (wd == 4) | thursday_last


def _ffill(x):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(x.shape[0]), 0)
    idx = np.maximum.accumulate(idx)
    return x[idx]


def shock_z(dS, Rm, window=SHOCK_WINDOW, min_obs=SHOCK_MIN):
    dS = np.asarray(dS, dtype=np.float64)
    Rm = np.asarray(Rm, dtype=np.float64)
    K = dS.shape[0]
    ok = np.isfinite(dS) & np.isfinite(Rm)
    z = np.full(K, np.nan)
    for k in range(K):
        if not ok[k]:
            continue
        lo = max(0, k - window)
        sel = ok[lo:k]
        if sel.sum() < min_obs:
            continue
        x = Rm[lo:k][sel]
        y = dS[lo:k][sel]
        xm = x.mean()
        ym = y.mean()
        sxx = np.sum((x - xm) ** 2)
        if not sxx > 0:
            continue
        b = np.sum((x - xm) * (y - ym)) / sxx
        a = ym - b * xm
        res = y - (a + b * x)
        sd = np.std(res, ddof=1)
        if not sd > 0:
            continue
        z[k] = (dS[k] - (a + b * Rm[k])) / sd
    return z


def legs_from(beta, vol, eligible):
    cols = np.flatnonzero(eligible & np.isfinite(beta) & np.isfinite(vol))
    n = cols.size
    lows, highs = [], []
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = cols[np.lexsort((cols, beta[cols]))]
    grp = (np.arange(n) * N_TERCILES) // n
    for g in range(N_TERCILES):
        sub = order[grp == g]
        q = sub.size // QUINTILE
        if q < MIN_LEG:
            continue
        sv = sub[np.lexsort((sub, vol[sub]))]
        lows.append(sv[:q])
        highs.append(sv[-q:])
    low = np.concatenate(lows) if lows else np.zeros(0, dtype=np.int64)
    high = np.concatenate(highs) if highs else np.zeros(0, dtype=np.int64)
    return low, high


def _returns(data):
    close = np.asarray(data.close, dtype=np.float64)
    priced = np.isfinite(close)
    valid = np.zeros_like(priced)
    valid[1:] = priced[1:] & priced[:-1]
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)
    return r, priced


def _market(r, uni):
    mem = uni & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return mkt


def shock_legs(data, params):
    vol_days = int(params["vol_days"])
    z_news = float(params["z_news"])
    uni = data.universe
    if uni is None:
        raise ValueError("card needs the LIQ-500 point-in-time universe")
    uni = np.asarray(uni, dtype=bool)

    baa = _ffill(data.series[DATASET + ".DBAA"])
    aaa = _ffill(data.series[DATASET + ".DAAA"])
    S = baa - aaa

    r, priced = _returns(data)
    mkt = _market(r, uni)
    logm = np.log1p(mkt)
    lr = np.log1p(r)

    dec = decision_days(data.dates)
    drows = np.flatnonzero(dec)
    dS = np.full(drows.size, np.nan)
    Rm = np.full(drows.size, np.nan)
    has = drows >= WEEK
    tt = drows[has]
    dS[has] = S[tt] - S[tt - WEEK]
    Rm[has] = logm[tt[:, None] + np.arange(-WEEK, 0)[None, :]].sum(axis=1)
    z = shock_z(dS, Rm)

    tradable = np.asarray(data.tradable, dtype=bool)
    vol_min = int(math.ceil(VOL_MIN_FRAC * vol_days - 1e-9))
    out = []
    for k in range(drows.size):
        zk = z[k]
        if not np.isfinite(zk) or abs(zk) < z_news:
            continue
        t = int(drows[k])
        if t + 1 < BETA_MIN or t + 1 < vol_min:
            continue
        lo = max(0, t - BETA_DAYS + 1)
        y = r[lo:t + 1]
        x = mkt[lo:t + 1]
        M = np.isfinite(y) & np.isfinite(x)[:, None]
        xw = np.where(M, x[:, None], 0.0)
        yw = np.where(M, y, 0.0)
        n = M.sum(axis=0).astype(np.float64)
        sx = xw.sum(axis=0)
        sy = yw.sum(axis=0)
        sxx = (xw * xw).sum(axis=0)
        sxy = (xw * yw).sum(axis=0)
        den = n * sxx - sx * sx
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where((n >= BETA_MIN) & (den > 0), (n * sxy - sx * sy) / den, np.nan)
        lo_v = max(0, t - vol_days + 1)
        lw = lr[lo_v:t + 1]
        Mv = np.isfinite(lw)
        nv = Mv.sum(axis=0).astype(np.float64)
        lz = np.where(Mv, lw, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = lz.sum(axis=0) / np.maximum(nv, 1.0)
            ss = (np.where(Mv, lw - mu[None, :], 0.0) ** 2).sum(axis=0)
            vol = np.where((nv >= vol_min) & (nv >= 2), np.sqrt(ss / np.maximum(nv - 1.0, 1.0)), np.nan)
        elig = uni[t] & tradable[t] & priced[t]
        low, high = legs_from(beta, vol, elig)
        sign = 1 if zk >= z_news else -1
        out.append((t, sign, low, high))
    return out, dec


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    legs, _ = shock_legs(data, params)
    for t, sign, low, high in legs:
        if not (low.size and high.size):
            continue
        mask[t, low] = True
        mask[t, high] = True
        side[t, low] = 1.0 if sign > 0 else -1.0     # widening: LOW long (+1); tightening: LOW short (-1)
        side[t, high] = -1.0 if sign > 0 else 1.0    # widening: HIGH short (-1); tightening: HIGH long (+1)
    return mask, side
