"""H-0084 Cross-asset fear premium.

12 ETFs, each paired with a CBOE 30-day implied-vol index. Monthly (last XNYS session, weekday
approximation with a one-day-late fallback), rank z = standardized ln(IV / RV) against its own past
252 sessions; long the n_per_leg richest, short the n_per_leg cheapest, inverse-vol weights per leg
(+1/6 long, -1/6 short per cohort), hold the three most recent monthly cohorts.
"""
import math

import numpy as np

PARAMS = {"n_per_leg": 3, "rv_days": 21}

# ETF -> implied-vol series key (card signal.description)
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

MAX_IV_AGE = 5          # sessions
RV_MIN_SHARE = 0.8
Z_WINDOW = 252
Z_MIN_VALID = 200
SIG_WINDOW = 63
SIG_MIN = 50
LEG_GROSS = 1.0 / 6.0
N_COHORTS = 3
ANN = math.sqrt(252.0)


def _log_returns(data):
    """Daily log total returns (T, N) and a 'present' mask (price on t and t-1)."""
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
    """Std (ddof=1) of the present log returns in the n rows ending at each row in `rows`.

    lr, present: (T, K). rows: (R,) int row indices. Returns (R, K) with NaN where fewer than
    min_count returns are present (or the window would start before row 0 with too few values).
    """
    T, K = lr.shape
    R = len(rows)
    out = np.full((R, K), np.nan)
    if R == 0:
        return out
    offs = np.arange(-n + 1, 1)
    idx = rows[:, None] + offs[None, :]                  # (R, n)
    inside = idx >= 0
    idxc = np.clip(idx, 0, T - 1)
    v = lr[idxc]                                          # (R, n, K)
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
    """IV value available at row t, the row of the session it belongs to, and its validity."""
    T = len(data.dates)
    K = len(cols)
    iv = np.full((T, K), np.nan)
    sess = np.full((T, K), -1, dtype=np.int64)
    for k, (_, key) in enumerate(PAIRS):
        x = np.asarray(data.series[key], dtype=float).reshape(-1)   # missing series raises
        if x.shape[0] != T:
            raise ValueError("series %s has wrong length" % key)
        fin = np.isfinite(x)
        last = np.where(fin, np.arange(T), -1)
        last = np.maximum.accumulate(last)
        ff = np.where(last >= 0, x[np.clip(last, 0, T - 1)], np.nan)
        prev = np.full(T, np.nan)
        prev[1:] = ff[:-1]
        new_obs = fin & ~(x == prev)                     # a new value appeared on this row
        obs_row = np.maximum.accumulate(np.where(new_obs, np.arange(T), -1))
        # clock next_morning: a value that appears on row r is the close of session r - 1
        sess[:, k] = np.where(obs_row >= 0, obs_row - 1, -1)
        iv[:, k] = ff
    t_idx = np.arange(T)[:, None]
    age = t_idx - sess
    valid = np.isfinite(iv) & (iv > 0) & (sess >= 0) & (age <= MAX_IV_AGE)
    return iv, sess, valid


def decision_rows(dates):
    """Month-end decision rows, point-in-time from the dates up to each row.

    Row t is the last XNYS session of its month if the next weekday (np.busday_offset, ignores
    exchange holidays) falls in another month. If that weekday was a holiday (e.g. Memorial Day
    on May 31, Good Friday on Mar 29), no row of the month is flagged; the decision is then made
    on the first session of the next month (one session late).
    """
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
    """Returns (decision row indices, cohort weights (D, N), formed flags (D,))."""
    n_leg = int(params["n_per_leg"])
    rv_days = int(params["rv_days"])
    T = len(data.dates)
    N = len(data.instruments)
    cols = np.array([data.col(tk) for tk, _ in PAIRS], dtype=np.int64)   # missing ticker raises
    K = len(cols)
    tickers = [tk for tk, _ in PAIRS]
    tick_rank = np.empty(K, dtype=np.int64)
    tick_rank[np.argsort(np.array(tickers))] = np.arange(K)

    lr_all, pres_all = _log_returns(data)
    lr = lr_all[:, cols]
    pres = pres_all[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]

    # (1) implied vol and its session
    iv, sess, iv_ok = _iv_inputs(data, cols)
    # (2) realized vol over rv_days ending at each row, then taken at the IV session
    min_rv = int(math.ceil(RV_MIN_SHARE * rv_days - 1e-9))
    rv_rows = _window_std(lr, pres, np.arange(T), rv_days, min_rv) * ANN    # (T, K)
    sc = np.clip(sess, 0, T - 1)
    rv_at = np.take_along_axis(rv_rows, sc, axis=0)
    rv_at = np.where(sess >= 0, rv_at, np.nan)
    # (3) x = ln(IV / (100 RV))
    ok = iv_ok & np.isfinite(rv_at) & (rv_at > 0)
    x = np.full((T, K), np.nan)
    x[ok] = np.log(iv[ok] / (100.0 * rv_at[ok]))

    rows = np.nonzero(decision_rows(data.dates))[0]
    D = len(rows)
    W = np.zeros((D, N))
    formed = np.zeros(D, dtype=bool)
    if D == 0:
        return rows, W, formed
    sig = _window_std(lr, pres, rows, SIG_WINDOW, SIG_MIN) * ANN            # (D, K)
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
        # eligible: valid z, tradable on t (next session approximated by t), valid sigma
        elig = np.isfinite(z) & tradable[t] & np.isfinite(sig[i]) & (sig[i] > 0)
        e_idx = np.nonzero(elig)[0]
        if len(e_idx) < 3 * n_leg + 3:
            continue
        ze = z[e_idx]
        tr = tick_rank[e_idx]
        # one ranking: z descending, ties by ticker (alphabetical); long = first n, short = last n
        order = e_idx[np.lexsort((tr, -ze))]
        lg = order[:n_leg]
        sh = order[-n_leg:]
        inv_l = 1.0 / sig[i, lg]
        inv_s = 1.0 / sig[i, sh]
        W[i, cols[lg]] = LEG_GROSS * inv_l / inv_l.sum()
        W[i, cols[sh]] = -LEG_GROSS * inv_s / inv_s.sum()
        formed[i] = True
    return rows, W, formed


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    out = np.full((T, N), np.nan)
    rows, W, formed = cohorts(data, params)
    D = len(rows)
    if D == 0:
        return out
    tradable = np.asarray(data.tradable, dtype=bool)
    book = np.zeros((T, N))
    for k in range(D):
        if not formed[k]:
            continue
        start = rows[k]
        end = rows[k + N_COHORTS] if k + N_COHORTS < D else T      # dropped at the 3rd next decision
        held = W[k] != 0
        seg = tradable[start:end]
        # an instrument that is not tradable on a row after formation is zeroed until the cohort ends
        dead = np.zeros_like(seg)
        if end - start > 1:
            dead[1:] = np.logical_or.accumulate(~seg[1:], axis=0)
        book[start:end] += np.where(dead & held[None, :], 0.0, W[k][None, :])
    first = rows[0]
    change = np.zeros(T, dtype=bool)
    change[first + 1:] = np.any(book[first + 1:] != book[first:-1], axis=1)
    dec = np.zeros(T, dtype=bool)
    dec[rows] = True
    dec |= change
    out[dec] = book[dec]
    return out
