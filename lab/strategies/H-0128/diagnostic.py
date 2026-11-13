"""H-0128 mechanism test: a LIQ-500 stock enters a leg at a weekly decision in a credit-shock week.

Same pipeline as strategy.py (copied, this file is separate): every member of the long leg on a shock decision
row is marked +1, every member of the short leg -1. Members not in a leg are the unmarked peers.
Widening weeks: long = LOW leverage, short = HIGH; tightening weeks: long = HIGH, short = LOW.
"""
import numpy as np

WIN = 5
REG_SESSIONS = 252
MIN_WINDOWS = 40
BETA_SESSIONS = 252
MIN_BETA_DAYS = 200
STALE_DAYS = 400
N_REG = 4


def weekday(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (d + 3) % 7


def decision_mask(dates):
    return weekday(dates) == 4


def ffill(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    idx = np.where(np.isfinite(x), np.arange(x.size), -1)
    idx = np.maximum.accumulate(idx) if x.size else idx
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def lagged_sum(x, w):
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    good = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(good, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(~good)])
    out = np.full(T, np.nan)
    if T > w:
        r = np.arange(w, T)
        s = cs[r] - cs[r - w]
        bad = (cn[r] - cn[r - w]) > 0
        out[r] = np.where(bad, np.nan, s)
    return out


def diff_lag(x, w):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size > w:
        out[w:] = x[w:] - x[:-w]
    return out


def shock_z(S, Y, M, rows):
    S = ffill(S)
    Y = ffill(Y)
    dS = diff_lag(S, WIN)
    dY = diff_lag(Y, WIN)
    rM = lagged_sum(M, WIN)
    rML = np.full(rM.size, np.nan)
    if rM.size > WIN:
        rML[WIN:] = rM[:-WIN]
    z = np.full(S.size, np.nan)
    n_win = REG_SESSIONS // WIN
    steps = WIN * np.arange(n_win)
    for t in rows:
        ks = t - steps
        ks = ks[ks >= 0]
        y = dS[ks]
        X = np.column_stack([np.ones(ks.size), rM[ks], rML[ks], dY[ks]])
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if not valid[0] or valid.sum() < MIN_WINDOWS:
            continue
        Xv = X[valid]
        yv = y[valid]
        b = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        e = yv - Xv @ b
        dof = yv.size - N_REG
        sd = np.sqrt(np.sum(e * e) / dof)
        if sd > 0 and np.isfinite(sd):
            z[t] = e[0] / sd
    return z


def window_sums(a, rows, win):
    c = np.cumsum(a, axis=0)
    hi = c[rows]
    lo_idx = rows - win
    lo = np.where((lo_idx >= 0)[:, None], c[np.maximum(lo_idx, 0)], 0.0)
    return hi - lo


def betas(r, M, valid, rows):
    v = valid & np.isfinite(r) & np.isfinite(M)[:, None]
    x = np.where(v, M[:, None], 0.0)
    y = np.where(v, r, 0.0)
    n = window_sums(v.astype(np.float64), rows, BETA_SESSIONS)
    sx = window_sums(x, rows, BETA_SESSIONS)
    sy = window_sums(y, rows, BETA_SESSIONS)
    sxy = window_sums(x * y, rows, BETA_SESSIONS)
    sxx = window_sums(x * x, rows, BETA_SESSIONS)
    num = n * sxy - sx * sy
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where((n >= MIN_BETA_DAYS) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)
    return b


def filing_age_ok(fields, dates, rows):
    T, N = fields[0].shape
    changed = np.zeros((T, N), dtype=bool)
    for F in fields:
        F = np.asarray(F, dtype=np.float64)
        prev = np.vstack([np.full((1, N), np.nan), F[:-1]])
        same = (F == prev) | (np.isnan(F) & np.isnan(prev))
        changed |= ~same
    last = np.where(changed, np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)[rows]
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    age = d[rows][:, None] - d[np.maximum(last, 0)]
    return (last >= 0) & (age <= STALE_DAYS)


def leverage(debt, cash, mcap):
    debt = np.asarray(debt, dtype=np.float64)
    cash = np.asarray(cash, dtype=np.float64)
    mcap = np.asarray(mcap, dtype=np.float64)
    den = debt + mcap
    ok = np.isfinite(debt) & np.isfinite(cash) & np.isfinite(mcap) & (mcap > 0) & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(ok, (debt - cash) / np.where(ok, den, 1.0), np.nan)
    return L


def legs(beta, L, elig, q):
    N = beta.size
    out = np.zeros(N, dtype=np.int8)
    idx = np.flatnonzero(elig & np.isfinite(beta) & np.isfinite(L))
    if idx.size < 3:
        return out
    order = idx[np.lexsort((idx, beta[idx]))]
    for g in np.array_split(order, 3):
        k = int(np.floor(q * g.size + 1e-9))
        if k <= 0:
            continue
        gs = g[np.lexsort((g, L[g]))]
        out[gs[:k]] = -1
        out[gs[-k:]] = 1
    return out


def book_row(leg, z, z_news):
    N = leg.size
    w = np.zeros(N)
    if not np.isfinite(z) or abs(z) < z_news:
        return w
    hi = leg == 1
    lo = leg == -1
    nh = int(hi.sum())
    nl = int(lo.sum())
    if nh == 0 or nl == 0:
        return w
    if z >= z_news:
        w[lo] = 0.5 / nl
        w[hi] = -0.5 / nh
    else:
        w[hi] = 0.5 / nh
        w[lo] = -0.5 / nl
    return w


def total_returns(data):
    return (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0


def market_return(r, member):
    m = member & np.isfinite(r)
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _book(data, params):
    q = float(params["leg_quantile"])
    z_news = float(params["z_news"])

    uni = data.universe
    if uni is None:
        raise ValueError("H-0128 needs the LIQ-500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    dates = data.dates
    T = dates.size
    N = len(data.instruments)

    baa = np.asarray(data.series["fred_moody_yields_1d.DBAA"], dtype=np.float64).reshape(-1)
    dgs10 = np.asarray(data.series["fred_macro.DGS10"], dtype=np.float64).reshape(-1)
    if baa.size != T or dgs10.size != T:
        raise ValueError("signal series not aligned to the view")
    debt = data.extras["sf1_arq_debt"]
    cash = data.extras["sf1_arq_cashneq"]
    mcap = data.extras["sf1_arq_marketcap"]

    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)

    r = total_returns(data)
    M = market_return(r, uni & listed)

    rows = np.flatnonzero(decision_mask(dates))
    W = np.full((T, N), np.nan)
    if rows.size == 0:
        return W
    S = ffill(baa) - ffill(dgs10)
    z = shock_z(S, ffill(dgs10), M, rows)

    W[rows] = 0.0
    shock = np.isfinite(z[rows]) & (np.abs(z[rows]) >= z_news)
    srows = rows[shock]
    if srows.size == 0:
        return W

    cols = np.flatnonzero(uni[srows].any(axis=0))
    if cols.size == 0:
        return W
    valid = listed[:, cols] & np.isfinite(close[:, cols])
    B = betas(r[:, cols], M, valid, srows)
    debt_c = np.asarray(debt, dtype=np.float64)[:, cols]
    cash_c = np.asarray(cash, dtype=np.float64)[:, cols]
    mcap_c = np.asarray(mcap, dtype=np.float64)[:, cols]
    fresh = filing_age_ok([debt_c, cash_c, mcap_c], dates, srows)
    L = leverage(debt_c[srows], cash_c[srows], mcap_c[srows])
    elig = (uni[srows][:, cols] & listed[srows][:, cols] & tradable[srows][:, cols] & fresh
            & np.isfinite(L) & np.isfinite(B))

    for i, t in enumerate(srows):
        leg = legs(B[i], L[i], elig[i], q)
        w = book_row(leg, z[t], z_news)
        W[t, cols] = w
    return W


def events(data, params):
    W = _book(data, params)
    held = np.isfinite(W) & (W != 0.0)
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = held & uni & tradable
    side = np.where(mask, np.sign(np.where(np.isfinite(W), W, 0.0)), 0.0)
    return mask, side
