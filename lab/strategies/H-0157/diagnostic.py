"""H-0157 mechanism_test events: the same event pipeline as strategy.py (copied, separate file).

Event at the close of US session t for a LIQ-500 member j whose home exchange X (assigned on the
first session of the month by >= 0.5 trailing-252-session excess-return correlation with a country ETF)
has X.anchored(t) = 0, X.anchored(t-1) = 1, X.closed_next(t) = 1, and |excess(t)| / std(excess over
t-63..t-1) >= z_min. Side = -sign(excess(t)). Only members that are tradable on row t.
"""
import numpy as np
from lab.framework.api import period_starts

ETF_MAP = (
    ("EWJ", "XTKS"),
    ("EWH", "XHKG"),
    ("FXI", "XHKG"),
    ("EWY", "XKRX"),
    ("EWT", "XTAI"),
    ("EWS", "XSES"),
    ("EWM", "XKLS"),
    ("EWA", "XASX"),
)
EXCHANGES = ("XTKS", "XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX")
HOLIDAY_DATASET = "exchange_holidays_1d"

CORR_WINDOW = 252
CORR_MIN_OBS = 200
CORR_THRESHOLD = 0.5
VOL_WINDOW = 63
VOL_MIN_OBS = 50


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0157 needs the LIQ-500 point-in-time universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def _etf_columns(data):
    cols = np.array([data.col(name) for name, _ in ETF_MAP], dtype=np.int64)
    exch = np.array([EXCHANGES.index(x) for _, x in ETF_MAP], dtype=np.int64)
    return cols, exch


def _holiday_flags(data):
    T = len(data.dates)
    anchored = np.empty((T, len(EXCHANGES)), dtype=np.float64)
    closed_next = np.empty((T, len(EXCHANGES)), dtype=np.float64)
    for k, x in enumerate(EXCHANGES):
        a = np.asarray(data.series[HOLIDAY_DATASET + "." + x + ".anchored"], dtype=np.float64).reshape(-1)
        c = np.asarray(data.series[HOLIDAY_DATASET + "." + x + ".closed_next"], dtype=np.float64).reshape(-1)
        if a.shape[0] != T or c.shape[0] != T:
            raise ValueError("holiday flag series for " + x + " is not aligned to the view")
        anchored[:, k] = a
        closed_next[:, k] = c
    return anchored, closed_next


def _excess_returns(data, uni, etf_cols):
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(close) & np.isfinite(r)
    del close, rco, roc
    valid = ok.copy()
    valid[0] = False
    valid[1:] &= ok[:-1]
    del ok
    mem = uni & valid
    mem[:, etf_cols] = False
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    peer = np.full(r.shape[0], np.nan)
    has = cnt > 0
    peer[has] = tot[has] / cnt[has]
    ex = (r - peer[:, None]).astype(np.float32)
    ex[~valid] = np.nan
    ex[~has] = np.nan
    return ex


def _month_starts(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return np.flatnonzero(ps)
    return np.sort(ps.astype(np.int64))


def _home_assignment(data, ex, uni, etf_cols, etf_exch):
    T, N = ex.shape
    home = np.full((T, N), -1, dtype=np.int8)
    etf_ex = ex[:, etf_cols].astype(np.float64)
    etf_valid = np.isfinite(etf_ex)
    anyv = etf_valid.any(axis=0)
    first_valid = np.where(anyv, np.argmax(etf_valid, axis=0), T)
    is_etf = np.zeros(N, dtype=bool)
    is_etf[etf_cols] = True

    starts = _month_starts(data.dates)
    for k, t in enumerate(starts):
        t = int(t)
        end = int(starts[k + 1]) if k + 1 < len(starts) else T
        if t < CORR_WINDOW - 1:
            continue
        lo = t - CORR_WINDOW + 1
        etf_ok = first_valid <= lo
        if not etf_ok.any():
            continue
        members = np.flatnonzero(uni[t] & ~is_etf)
        if members.size == 0:
            continue
        X = ex[lo:t + 1][:, members].astype(np.float64)
        Y = etf_ex[lo:t + 1][:, etf_ok]
        Mx = np.isfinite(X)
        My = np.isfinite(Y)
        X0 = np.where(Mx, X, 0.0)
        Y0 = np.where(My, Y, 0.0)
        Mxf = Mx.astype(np.float64)
        Myf = My.astype(np.float64)
        n = Mxf.T @ Myf
        sx = X0.T @ Myf
        sy = Mxf.T @ Y0
        sxx = (X0 * X0).T @ Myf
        syy = Mxf.T @ (Y0 * Y0)
        sxy = X0.T @ Y0
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            corr = cov / np.sqrt(vx * vy)
        good = (n >= CORR_MIN_OBS) & (vx > 0) & (vy > 0) & np.isfinite(corr)
        corr = np.where(good, corr, -np.inf)
        best = np.argmax(corr, axis=1)
        best_c = corr[np.arange(members.size), best]
        assigned = best_c >= CORR_THRESHOLD
        exch_of_col = etf_exch[etf_ok]
        a = np.full(members.size, -1, dtype=np.int8)
        a[assigned] = exch_of_col[best[assigned]].astype(np.int8)
        home[t:end, members] = a[None, :]
    return home


def _closure_starts(anchored, closed_next):
    T = anchored.shape[0]
    e = np.zeros(anchored.shape, dtype=bool)
    if T > 1:
        e[1:] = (anchored[1:] == 0.0) & (closed_next[1:] == 1.0) & (anchored[:-1] == 1.0)
    return e


def _events(data, params, ex, home, uni, etf_cols):
    z_min = float(params["z_min"])
    T, N = ex.shape
    anchored, closed_next = _holiday_flags(data)
    closure = _closure_starts(anchored, closed_next)
    tradable = np.asarray(data.tradable, dtype=bool)
    is_etf = np.zeros(N, dtype=bool)
    is_etf[etf_cols] = True
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(closure.any(axis=1)):
        h = home[t].astype(np.int64)
        assigned = h >= 0
        hit = np.zeros(N, dtype=bool)
        hit[assigned] = closure[t, h[assigned]]
        cand = np.flatnonzero(hit & uni[t] & tradable[t] & ~is_etf & np.isfinite(ex[t]))
        if cand.size == 0:
            continue
        lo = max(0, t - VOL_WINDOW)
        W = ex[lo:t][:, cand].astype(np.float64)
        V = np.isfinite(W)
        nv = V.sum(axis=0)
        W0 = np.where(V, W, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = W0.sum(axis=0) / nv
            dev = np.where(V, W - mu[None, :], 0.0)
            s = np.sqrt((dev * dev).sum(axis=0) / (nv - 1))
        m = ex[t, cand].astype(np.float64)
        ok = (nv >= VOL_MIN_OBS) & np.isfinite(s) & (s > 0) & (m != 0.0)
        z = np.zeros(cand.size)
        z[ok] = m[ok] / s[ok]
        fire = ok & (np.abs(z) >= z_min)
        mask[t, cand[fire]] = True
        side[t, cand[fire]] = -np.sign(m[fire])
    return mask, side


def events(data, params):
    uni = _universe(data)
    etf_cols, etf_exch = _etf_columns(data)
    ex = _excess_returns(data, uni, etf_cols)
    home = _home_assignment(data, ex, uni, etf_cols, etf_exch)
    mask, side = _events(data, params, ex, home, uni, etf_cols)
    return mask, side
