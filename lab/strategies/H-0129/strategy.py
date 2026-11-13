"""H-0129 Inflation news into nominal balance sheets.

Weekly decision after the last XNYS session of the calendar week (approximated as the Friday: a date is a
decision date when the next Mon-Fri weekday lies in the next calendar week; weeks whose Friday is an
exchange holiday have no decision). If the residual 5-session TIPS-breakeven shock |z| >= z_shock, long the
real-asset-heavy legs and short the monetary-asset-heavy legs (z > 0; mirror for z < 0) inside leverage
terciles of the eligible LIQ-500 stocks, 0.5 gross per side, hold 5 sessions, otherwise flat.
"""
import numpy as np

PARAMS = {"leg_quantile": 0.2, "z_shock": 1.0}

FRED_KEYS = (
    "fred_macro.T5YIE",
    "fred_macro.DFII10",
    "fred_macro.DCOILWTICO",
    "fred_moody_yields_1d.DBAA",
    "fred_moody_yields_1d.DAAA",
)
SF1 = (
    "sf1_arq_cashneq",
    "sf1_arq_receivables",
    "sf1_arq_investmentsc",
    "sf1_arq_liabilities",
    "sf1_arq_assets",
    "sf1_arq_deposits",
)
WINDOW = 252      # trailing sessions of the regression
MIN_ROWS = 200    # minimum valid rows in the window
CHG = 5           # five-session changes
HOLD = 5          # sessions held (exit at the open of t+6)


def decision_mask(dates):
    """True on the last Mon-Fri weekday of each calendar week (holidays ignored, point-in-time)."""
    d = np.asarray(dates).astype("datetime64[D]")
    days = d.astype(np.int64)
    weekday = (days + 3) % 7                      # Monday = 0
    week = (days + 3) // 7
    nxt = np.busday_offset(d, 1, roll="forward")
    week_next = (nxt.astype(np.int64) + 3) // 7
    return (weekday < 5) & (week_next != week)


def _chg(v, k):
    out = np.full(v.shape, np.nan)
    out[k:] = v[k:] - v[:-k]
    return out


def market_5d(data, uni):
    """Row r: equal-weight mean over LIQ-500 members (at r) of the 5-session log total return ending at r."""
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(rco) + np.log1p(roc)
    lr[~np.isfinite(lr)] = np.nan
    T = lr.shape[0]
    s5 = np.full(lr.shape, np.nan)
    if T >= CHG:
        acc = lr[CHG - 1:].copy()
        for i in range(1, CHG):
            acc += lr[CHG - 1 - i:T - i]
        s5[CHG - 1:] = acc
    ok = uni & np.isfinite(s5)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, s5, 0.0).sum(axis=1)
    m5 = np.full(T, np.nan)
    np.divide(tot, cnt, out=m5, where=cnt > 0)
    return m5


def shock_z(data, rows, uni):
    """Residual z of the 5-session T5YIE change at the last session s with all FRED inputs, for each row."""
    t5, r10, oil, baa, aaa = [np.asarray(data.series[k], dtype=np.float64) for k in FRED_KEYS]
    T = t5.shape[0]
    avail = np.isfinite(t5) & np.isfinite(r10) & np.isfinite(oil) & np.isfinite(baa) & np.isfinite(aaa)
    with np.errstate(divide="ignore", invalid="ignore"):
        loil = np.where(oil > 0, np.log(np.where(oil > 0, oil, 1.0)), np.nan)
    y = _chg(t5, CHG)
    x2 = _chg(r10, CHG)
    x3 = _chg(loil, CHG)
    x4 = _chg(baa - aaa, CHG)
    # FRED clock next_morning: row r holds the values of session r-1, so the equity move over the same
    # five sessions is the member mean ending at row r-1.
    m5 = market_5d(data, uni)
    x1 = np.full(T, np.nan)
    x1[1:] = m5[:-1]
    M = np.column_stack([y, x1, x2, x3, x4])
    valid = np.all(np.isfinite(M), axis=1)
    last_avail = np.maximum.accumulate(np.where(avail, np.arange(T), -1)) if T else np.zeros(0, int)
    z = np.full(len(rows), np.nan)
    for i, t in enumerate(rows):
        s = int(last_avail[t])
        if s < 0 or not valid[s]:
            continue
        lo = max(0, s - WINDOW + 1)
        sel = valid[lo:s + 1]
        if int(sel.sum()) < MIN_ROWS:
            continue
        blk = M[lo:s + 1][sel]
        X = np.column_stack([np.ones(blk.shape[0]), blk[:, 1:]])
        yy = blk[:, 0]
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        res = yy - X @ beta
        sd = float(np.std(res, ddof=1))
        if not sd > 0:
            continue
        z[i] = res[-1] / sd           # the last valid row of the window is s itself
    return z


def legs(ex, uni_t, trad_t, q):
    """Real-asset-heavy and monetary-asset-heavy column indices on one date."""
    cash = ex["sf1_arq_cashneq"]
    rec = ex["sf1_arq_receivables"]
    inv = ex["sf1_arq_investmentsc"]
    liab = ex["sf1_arq_liabilities"]
    assets = ex["sf1_arq_assets"]
    dep = ex["sf1_arq_deposits"]
    with np.errstate(invalid="ignore"):
        elig = (uni_t & trad_t & np.isfinite(assets) & (assets > 0) & np.isfinite(liab)
                & (~np.isfinite(dep) | (dep == 0)))
    idx = np.flatnonzero(elig)
    n = idx.size
    real, mon = [], []
    if n < 3:
        return np.array(real, dtype=int), np.array(mon, dtype=int)
    a = assets[idx]
    ma = (np.nan_to_num(cash[idx], nan=0.0) + np.nan_to_num(rec[idx], nan=0.0)
          + np.nan_to_num(inv[idx], nan=0.0)) / a
    lev = liab[idx] / a
    order = np.lexsort((idx, lev))                    # by leverage, ties by column
    terc = (np.arange(n) * 3) // n
    for k in range(3):
        pos = order[terc == k]
        m = pos.size
        kq = int(np.floor(q * m + 1e-9))
        if kq < 1 or 2 * kq > m:
            continue
        o = pos[np.lexsort((idx[pos], ma[pos]))]     # by MA, ties by column
        real.extend(idx[o[:kq]].tolist())
        mon.extend(idx[o[m - kq:]].tolist())
    return np.array(sorted(real), dtype=int), np.array(sorted(mon), dtype=int)


def book(ex_t, uni_t, trad_t, z, q, N):
    w = np.zeros(N)
    real, mon = legs(ex_t, uni_t, trad_t, q)
    if real.size == 0 or mon.size == 0:
        return w
    lng, sht = (real, mon) if z > 0 else (mon, real)
    w[lng] = 0.5 / lng.size
    w[sht] = -0.5 / sht.size
    return w


def _inputs(data):
    if data.universe is None:
        raise ValueError("H-0129 needs the LIQ-500 universe")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    ex = {f: np.asarray(data.extras[f], dtype=np.float64) for f in SF1}
    return uni, trad, ex


def target_weights(data, params):
    q = float(params["leg_quantile"])
    zs = float(params["z_shock"])
    T, N = len(data.dates), len(data.instruments)
    uni, trad, ex = _inputs(data)
    dec = np.flatnonzero(decision_mask(data.dates))
    z = shock_z(data, dec, uni)
    W = np.full((T, N), np.nan)
    for i, d in enumerate(dec):
        nd = int(dec[i + 1]) if i + 1 < len(dec) else T
        if np.isfinite(z[i]) and abs(z[i]) >= zs:
            ex_t = {f: v[d] for f, v in ex.items()}
            w = book(ex_t, uni[d], trad[d], z[i], q, N)
        else:
            w = np.zeros(N)
        W[d] = w
        held = w != 0
        if not held.any():
            continue
        open_ = held.copy()
        for r in range(d + 1, min(d + HOLD, nd, T)):
            drop = open_ & ~(uni[r] & trad[r])     # left the universe / not tradable: close at next open
            if drop.any():
                W[r, drop] = 0.0
                open_ &= ~drop
        e = d + HOLD
        if e < nd and e < T:
            W[e] = 0.0                              # exit at the open of t+6
    return W
