"""H-0131 mechanism test: rally state of SPY over IEF (same rule as the strategy, copied).

On each rally-state session t, SPY, VTI, DIA and IWM are marked with side -1. IEF, TLT, SHY and AGG
are never marked (peers).
"""
import numpy as np

SHORT_WIN = 5
LONG_WIN = 21
VOL_WIN = 252
VOL_MIN = 200
MARKED = ("SPY", "VTI", "DIA", "IWM")


def _window_sum(x, n):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    t = np.arange(x.shape[0])
    lo = np.maximum(t + 1 - n, 0)
    return c[t + 1] - c[lo]


def _rally_state(data, k):
    js = data.col("SPY")
    ji = data.col("IEF")
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    valid = listed[:, js] & listed[:, ji] & np.isfinite(close[:, js]) & np.isfinite(close[:, ji])
    r_s = (1.0 + data.ret_co[:, js].astype(np.float64)) * (1.0 + data.ret_oc[:, js].astype(np.float64)) - 1.0
    r_i = (1.0 + data.ret_co[:, ji].astype(np.float64)) * (1.0 + data.ret_oc[:, ji].astype(np.float64)) - 1.0
    d = r_s - r_i
    ok = valid & np.isfinite(d)
    d0 = np.where(ok, d, 0.0)
    rel5 = _window_sum(d0, SHORT_WIN)
    rel21 = _window_sum(d0, LONG_WIN)
    n = _window_sum(ok.astype(np.float64), VOL_WIN)
    s1 = _window_sum(d0, VOL_WIN)
    s2 = _window_sum(d0 * d0, VOL_WIN)
    enough = n >= VOL_MIN
    nn = np.where(enough, n, 2.0)
    var = np.maximum((s2 - s1 * s1 / nn) / (nn - 1.0), 0.0)
    s = np.sqrt(var) * np.sqrt(float(SHORT_WIN))
    return ok & enough & (rel5 >= k * s) & (rel21 > 0.0)


def events(data, params):
    k = float(params["k"])
    rally = _rally_state(data, k)
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    for name in MARKED:
        j = data.col(name)
        m = rally & tradable[:, j] & listed[:, j]
        if data.universe is not None:
            m = m & np.asarray(data.universe, dtype=bool)[:, j]
        mask[:, j] = m
    side = np.where(mask, -1.0, 0.0)
    return mask, side
