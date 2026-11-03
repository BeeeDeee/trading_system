"""H-0084 mechanism-test events: leg membership at every month-end on which a cohort is formed.

Same signal code as strategy.py (copied: diagnostic.py is a separate file). Long-leg members are
events with side +1, short-leg members side -1.
"""
import math

import numpy as np

PAIRS = (
    ("SPY", "fred_macro.VIXCLS"),
    ("QQQ", "fred_macro.VXNCLS"),
    ("IWM", "cboe_etf_vol_1d.RVXCLS"),
    ("EEM", "cboe_etf_vol_1d.VXEEMCLS"),
    ("EWZ", "cboe_etf_vol_1d.VXEWZCLS"),
    ("FXI", "cboe_etf_vol_1d.VXFXICLS"),
    ("XLE", "cboe_etf_vol_1d.VXXLECLS"),
    ("GDX", "cboe_etf_vol_1d.VXGDXCLS"),
    ("GLD", "fred_macro.GVZCLS"),
    ("SLV", "cboe_etf_vol_1d.VXSLVCLS"),
    ("USO", "fred_macro.OVXCLS"),
    ("FXE", "cboe_etf_vol_1d.EVZCLS"),
)

MAX_IV_AGE = 5
RV_MIN_SHARE = 0.8
Z_WINDOW = 252
Z_MIN_VALID = 200
SIG_WINDOW = 63
SIG_MIN = 50
LEG_GROSS = 1.0 / 6.0
ANN = math.sqrt(252.0)


def _log_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=float)) * (1.0 + np.asarray(data.ret_oc, dtype=float)) - 1.0
    close = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    has = np.isfinite(close) & listed
    present = np.zeros_like(has)
    present[1:] = has[1:] & has[:-1]
    present &= np.isfinite(r) & (r > -1.0)
    lr = np.where(present, np.log1p(np.where(present, r, 0.0)), 0.0)
    return lr, present


def _window_std(lr, present, rows, n, min_count):
    T, K = lr.shape
    R = len(rows)
    out = np.full((R, K), np.nan)
    if R == 0:
        return out
    offs = np.arange(-n + 1, 1)
    idx = rows[:, None] + offs[None, :]
    inside = idx >= 0
    idxc = np.clip(idx, 0, T - 1)
    v = lr[idxc]
    m = present[idxc] & inside[:, :, None]
    cnt = m.sum(axis=1)
    s = np.where(m, v, 0.0).sum(axis=1)
    mean = s / np.maximum(cnt, 1)
    dev = np.where(m, v - mean[:, None, :], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(cnt - 1, 1)
    ok = (cnt >= min_count) & (cnt >= 2)
    out[ok] = np.sqrt(var[ok])
    return out


def _iv_inputs(data, cols):
    T = len(data.dates)
    K = len(cols)
    iv = np.full((T, K), np.nan)
    sess = np.full((T, K), -1, dtype=np.int64)
    for k, (_, key) in enumerate(PAIRS):
        x = np.asarray(data.series[key], dtype=float).reshape(-1)
        if x.shape[0] != T:
            raise ValueError("series %s has wrong length" % key)
        fin = np.isfinite(x)
        last = np.where(fin, np.arange(T), -1)
        last = np.maximum.accumulate(last)
        ff = np.where(last >= 0, x[np.clip(last, 0, T - 1)], np.nan)
        prev = np.full(T, np.nan)
        prev[1:] = ff[:-1]
        new_obs = fin & ~(x == prev)
        obs_row = np.maximum.accumulate(np.where(new_obs, np.arange(T), -1))
        sess[:, k] = np.where(obs_row >= 0, obs_row - 1, -1)
        iv[:, k] = ff
    t_idx = np.arange(T)[:, None]
    age = t_idx - sess
    valid = np.isfinite(iv) & (iv > 0) & (sess >= 0) & (age <= MAX_IV_AGE)
    return iv, sess, valid


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = len(d)
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    me = nxt.astype("datetime64[M]") != d.astype("datetime64[M]")
    mon = d.astype("datetime64[M]")
    newm = np.zeros(T, dtype=bool)
    newm[1:] = mon[1:] != mon[:-1]
    late = np.zeros(T, dtype=bool)
    late[1:] = newm[1:] & ~me[:-1]
    return me | late


def cohorts(data, params):
    n_leg = int(params["n_per_leg"])
    rv_days = int(params["rv_days"])
    T = len(data.dates)
    N = len(data.instruments)
    cols = np.array([data.col(tk) for tk, _ in PAIRS], dtype=np.int64)
    K = len(cols)
    tickers = [tk for tk, _ in PAIRS]
    tick_rank = np.empty(K, dtype=np.int64)
    tick_rank[np.argsort(np.array(tickers))] = np.arange(K)

    lr_all, pres_all = _log_returns(data)
    lr = lr_all[:, cols]
    pres = pres_all[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]

    iv, sess, iv_ok = _iv_inputs(data, cols)
    min_rv = int(math.ceil(RV_MIN_SHARE * rv_days - 1e-9))
    rv_rows = _window_std(lr, pres, np.arange(T), rv_days, min_rv) * ANN
    sc = np.clip(sess, 0, T - 1)
    rv_at = np.take_along_axis(rv_rows, sc, axis=0)
    rv_at = np.where(sess >= 0, rv_at, np.nan)
    ok = iv_ok & np.isfinite(rv_at) & (rv_at > 0)
    x = np.full((T, K), np.nan)
    x[ok] = np.log(iv[ok] / (100.0 * rv_at[ok]))

    rows = np.nonzero(decision_rows(data.dates))[0]
    D = len(rows)
    W = np.zeros((D, N))
    formed = np.zeros(D, dtype=bool)
    if D == 0:
        return rows, W, formed
    sig = _window_std(lr, pres, rows, SIG_WINDOW, SIG_MIN) * ANN
    xv = np.isfinite(x)
    for i, t in enumerate(rows):
        lo = max(0, t - Z_WINDOW + 1)
        win = x[lo:t + 1]
        m = xv[lo:t + 1]
        cnt = m.sum(axis=0)
        mu = np.where(m, win, 0.0).sum(axis=0) / np.maximum(cnt, 1)
        dv = np.where(m, win - mu, 0.0)
        sd = np.sqrt((dv * dv).sum(axis=0) / np.maximum(cnt - 1, 1))
        z = np.full(K, np.nan)
        zok = xv[t] & (cnt >= Z_MIN_VALID) & (sd > 0)
        z[zok] = (x[t, zok] - mu[zok]) / sd[zok]
        elig = np.isfinite(z) & tradable[t] & np.isfinite(sig[i]) & (sig[i] > 0)
        e_idx = np.nonzero(elig)[0]
        if len(e_idx) < 3 * n_leg + 3:
            continue
        ze = z[e_idx]
        tr = tick_rank[e_idx]
        order = e_idx[np.lexsort((tr, -ze))]
        lg = order[:n_leg]
        sh = order[-n_leg:]
        inv_l = 1.0 / sig[i, lg]
        inv_s = 1.0 / sig[i, sh]
        W[i, cols[lg]] = LEG_GROSS * inv_l / inv_l.sum()
        W[i, cols[sh]] = -LEG_GROSS * inv_s / inv_s.sum()
        formed[i] = True
    return rows, W, formed


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    rows, W, formed = cohorts(data, params)
    tradable = np.asarray(data.tradable, dtype=bool)
    for k in range(len(rows)):
        if not formed[k]:
            continue
        t = rows[k]
        hit = (W[k] != 0) & tradable[t]
        if data.universe is not None:
            hit &= np.asarray(data.universe, dtype=bool)[t]
        mask[t] = hit
        side[t] = np.where(hit, np.sign(W[k]), 0.0)
    return mask, side
