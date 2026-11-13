"""H-0133 mechanism test events: same event rule and side as strategy.py (code copied, separate file).

Event after the close of t: Asia-Pacific ETF j, home exchange closed_next = 1 at t, price at t, tradable
at t, vol_days sigma available, |U_j,t / sigma_j,t| >= min_z. Side = -sign(U) (fade).
"""
import numpy as np

DATASET = "exchange_holidays_1d"

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
MIN_PRESENT = 0.8


def _flags(data, mic, T):
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


def _unanchored_move(ret_co, ret_oc, anch):
    lco = np.log1p(_clean(ret_co))
    loc = np.log1p(_clean(ret_oc))
    T, K = loc.shape
    L = np.cumsum(lco + loc, axis=0)
    base = L - loc
    rows = np.arange(T)[:, None]
    idx = np.where(np.asarray(anch, dtype=bool), rows, -1)
    idx = np.maximum.accumulate(idx, axis=0)
    safe = np.maximum(idx, 0)
    b = np.take_along_axis(base, safe, axis=0)
    return np.where(idx >= 0, L - b, np.nan)


def _rolling_sigma(ret_oc, present, n):
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


def events(data, params):
    min_z = float(params["min_z"])
    n = int(params["vol_days"])
    T = len(data.dates)
    N = len(data.instruments)
    cols = [data.col(s) for s in ASIA]

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

    U = _unanchored_move(ret_co[:, cols], ret_oc[:, cols], anch)
    sigma = _rolling_sigma(ret_oc[:, cols], present[:, cols], n)
    good = present[:, cols] & tradable[:, cols] & np.isfinite(sigma) & (sigma > 0) & np.isfinite(U)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(good, U / np.where(good, sigma, 1.0), 0.0)
    ev = good & closed & (np.abs(z) >= min_z)

    if data.universe is not None:
        ev = ev & np.asarray(data.universe, dtype=bool)[:, cols]

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    mask[:, cols] = ev
    side[:, cols] = np.where(ev, -np.sign(U), 0.0)
    side[~mask] = 1.0
    return mask, side
