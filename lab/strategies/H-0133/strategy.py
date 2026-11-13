"""H-0133 Home-holiday delayed anchoring.

Fade an Asia-Pacific country ETF's unanchored US-hours move (since its last home-anchored open) on
sessions after which its home exchange has no new session before the next US open (closed_next = 1),
hedged with the Asian peer ETFs whose home markets reopen (closed_next = 0), EFA as fallback hedge.
Decision after the US close of row t, executed at the open of t+1; the book is recomputed every row.
"""
import numpy as np

PARAMS = {"min_z": 0.5, "vol_days": 63}

DATASET = "exchange_holidays_1d"

# ETF -> home exchange MIC (card signal.description)
HOME = {
    "EWJ": "XTKS",
    "EWH": "XHKG",
    "FXI": "XHKG",
    "EWY": "XKRX",
    "EWT": "XTAI",
    "EWS": "XSES",
    "EWM": "XKLS",
    "EWA": "XASX",
    "THD": "XBKK",
    "EIDO": "XIDX",
    "EPHE": "XPHS",
    "ENZL": "XNZE",
    "EPI": "XBOM",
}
ASIA = tuple(HOME.keys())
FALLBACK = "EFA"
EVENT_GROSS = 0.5
MIN_PRESENT = 0.8


def _flags(data, mic, T):
    """closed_next and anchored of one exchange as (T,) bool.

    A missing calendar row (NaN) is treated as 'never closed': closed_next = 0, anchored = 1.
    """
    cn = np.asarray(data.series[DATASET + "." + mic + ".closed_next"], dtype=float).reshape(-1)
    an = np.asarray(data.series[DATASET + "." + mic + ".anchored"], dtype=float).reshape(-1)
    if cn.shape[0] != T or an.shape[0] != T:
        raise ValueError("calendar series length does not match data.dates for " + mic)
    closed = np.isfinite(cn) & (cn > 0.5)
    anch = (~np.isfinite(an)) | (an > 0.5)
    return closed, anch


def _clean(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, 0.0)


def unanchored_move(ret_co, ret_oc, anch):
    """U[t] = log(1+ret_oc[a]) + sum_{d in (a, t]} [log(1+ret_co[d]) + log(1+ret_oc[d])],
    a = latest row <= t with anch True. NaN before the first anchored row. All (T, K)."""
    lco = np.log1p(_clean(ret_co))
    loc = np.log1p(_clean(ret_oc))
    T, K = loc.shape
    L = np.cumsum(lco + loc, axis=0)          # log total-return index at each close
    base = L - loc                            # level at the open of each row
    rows = np.arange(T)[:, None]
    idx = np.where(np.asarray(anch, dtype=bool), rows, -1)
    idx = np.maximum.accumulate(idx, axis=0)
    safe = np.maximum(idx, 0)
    b = np.take_along_axis(base, safe, axis=0)
    return np.where(idx >= 0, L - b, np.nan)


def rolling_sigma(ret_oc, present, n):
    """Sample std (ddof=1) of ret_oc over the last n rows up to t, using only present rows;
    NaN unless at least 80 % of the n rows are present (rows before the data start count as missing)."""
    x = _clean(ret_oc)
    p = np.asarray(present, dtype=bool)
    T, K = x.shape
    v = np.where(p, x, 0.0)
    c = p.astype(float)
    z = np.zeros((1, K))
    cs = np.vstack([z, np.cumsum(v, axis=0)])
    cs2 = np.vstack([z, np.cumsum(v * v, axis=0)])
    cc = np.vstack([z, np.cumsum(c, axis=0)])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    cnt = cc[hi] - cc[lo]
    need = max(2, int(np.ceil(MIN_PRESENT * n - 1e-9)))
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s * s / cnt) / (cnt - 1.0)
    var = np.maximum(var, 0.0)
    sig = np.sqrt(var)
    return np.where(cnt >= need, sig, np.nan)


def book(U, sigma, ok, closed, efa_ok, min_z):
    """Weights of the Asian ETFs (T, K) and of EFA (T,).

    ok: ETF has a price at t and is tradable at t. Events: closed & ok & sigma available &
    |U/sigma| >= min_z, weight -sign(U) * 0.5 / n. Hedge: -(sum of event weights) equally over the
    ok, sigma-available ETFs with closed = False; fewer than 2 of them -> all into EFA. If neither
    is possible, the row is flat (keeps the book market neutral)."""
    U = np.asarray(U, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    closed = np.asarray(closed, dtype=bool)
    good = np.asarray(ok, dtype=bool) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(U)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(good, U / np.where(good, sigma, 1.0), 0.0)
    ev = good & closed & (np.abs(z) >= min_z)
    n_ev = ev.sum(axis=1)
    per = EVENT_GROSS / np.maximum(n_ev, 1)
    w = np.where(ev, -np.sign(U) * per[:, None], 0.0)
    net = w.sum(axis=1)
    hedge = good & ~closed
    m = hedge.sum(axis=1)
    use_peers = m >= 2
    w = w + np.where(hedge & use_peers[:, None], (-net / np.maximum(m, 1))[:, None], 0.0)
    efa_w = np.where(use_peers, 0.0, -net)
    efa_ok = np.asarray(efa_ok, dtype=bool)
    flat = (n_ev > 0) & (~use_peers) & (~efa_ok)
    w[flat] = 0.0
    efa_w[flat] = 0.0
    return w, efa_w


def target_weights(data, params):
    min_z = float(params["min_z"])
    n = int(params["vol_days"])
    T = len(data.dates)
    N = len(data.instruments)

    cols = [data.col(s) for s in ASIA]
    efa = data.col(FALLBACK)

    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    close = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    present = listed & np.isfinite(close)

    K = len(ASIA)
    closed = np.zeros((T, K), dtype=bool)
    anch = np.zeros((T, K), dtype=bool)
    for k, s in enumerate(ASIA):
        closed[:, k], anch[:, k] = _flags(data, HOME[s], T)

    U = unanchored_move(ret_co[:, cols], ret_oc[:, cols], anch)
    sigma = rolling_sigma(ret_oc[:, cols], present[:, cols], n)
    ok = present[:, cols] & tradable[:, cols]
    efa_ok = present[:, efa] & tradable[:, efa]
    w_asia, w_efa = book(U, sigma, ok, closed, efa_ok, min_z)

    W = np.zeros((T, N), dtype=float)
    W[:, cols] = w_asia
    W[:, efa] = w_efa
    return W
