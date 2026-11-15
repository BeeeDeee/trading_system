"""H-0168 mechanism test: hedge-failure sessions mark SPY, QQQ, IWM, DIA, MDY with side +1.

IEF, TLT and SHY are never marked (unmarked peers). Event logic copied from strategy.py.
"""
import numpy as np

VOL_WINDOW = 63
VOL_MIN_VALID = 50
MARKED = ("SPY", "QQQ", "IWM", "DIA", "MDY")


def _session_return(data, j):
    r = (1.0 + data.ret_co[:, j].astype(np.float64)) * (1.0 + data.ret_oc[:, j].astype(np.float64)) - 1.0
    bar = np.asarray(data.listed[:, j], dtype=bool) & np.isfinite(data.close[:, j]) & np.isfinite(r)
    return r, bar


def _trailing_std(x, valid, window, min_valid):
    T = x.shape[0]
    v = np.where(valid, x, 0.0)
    c = valid.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    t = np.arange(T)
    hi = t
    lo = np.maximum(t - window, 0)
    n = cc[hi] - cc[lo]
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    out = np.full(T, np.nan)
    ok = n >= min_valid
    var = (s2[ok] - s[ok] * s[ok] / n[ok]) / (n[ok] - 1.0)
    out[ok] = np.sqrt(np.maximum(var, 0.0))
    return out


def _hedge_failure_events(data, k_sigma):
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")
    r_spy, bar_spy = _session_return(data, j_spy)
    r_ief, bar_ief = _session_return(data, j_ief)
    sigma = _trailing_std(r_spy, bar_spy, VOL_WINDOW, VOL_MIN_VALID)
    ok = bar_spy & bar_ief & np.isfinite(sigma)
    ev = np.zeros(r_spy.shape[0], dtype=bool)
    ev[ok] = (r_spy[ok] <= -float(k_sigma) * sigma[ok]) & (r_ief[ok] < 0.0)
    return ev


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    data.col("TLT")
    data.col("SHY")
    ev = _hedge_failure_events(data, float(params["k_sigma"]))
    mask = np.zeros((T, N), dtype=bool)
    for name in MARKED:
        j = data.col(name)
        ok = (np.asarray(data.tradable[:, j], dtype=bool)
              & np.asarray(data.listed[:, j], dtype=bool)
              & np.isfinite(data.close[:, j]))
        if data.universe is not None:
            ok = ok & np.asarray(data.universe[:, j], dtype=bool)
        mask[:, j] = ev & ok
    side = np.ones((T, N))
    return mask, side
