"""H-0046 mechanism event: the simulated 60/40 SPY/IEF threshold book breaches its band after the close of t.

Marks SPY, VTI, DIA, IWM with side -1 on an overweight breach (d_t = +1) and +1 on an underweight breach
(d_t = -1). The bond ETFs (IEF, TLT, AGG, SHY) are left unmarked as peers. Rules identical to strategy.py.
"""
import numpy as np

TARGET_EQ = 0.60
MARKED = ("SPY", "VTI", "DIA", "IWM")


def _bar_returns(data, j):
    r = (1.0 + np.asarray(data.ret_co[:, j], dtype=float)) * (1.0 + np.asarray(data.ret_oc[:, j], dtype=float)) - 1.0
    close = np.asarray(data.close[:, j], dtype=float)
    has_bar = np.asarray(data.listed[:, j], dtype=bool) & np.isfinite(close) & np.isfinite(r)
    r = np.where(has_bar, r, 0.0)
    return r, has_bar


def _simulate_breaches(r_s, r_b, has_bar, threshold_pp):
    T = len(r_s)
    band = float(threshold_pp) / 100.0
    hi = TARGET_EQ + band
    lo = TARGET_EQ - band
    d = np.zeros(T, dtype=np.int64)
    g_s = 1.0
    g_b = 1.0
    for t in range(T):
        g_s *= 1.0 + r_s[t]
        g_b *= 1.0 + r_b[t]
        w = TARGET_EQ * g_s / (TARGET_EQ * g_s + (1.0 - TARGET_EQ) * g_b)
        if not has_bar[t]:
            continue
        if w > hi:
            d[t] = 1
        elif w < lo:
            d[t] = -1
        if d[t] != 0:
            g_s = 1.0
            g_b = 1.0
    return d


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    r_s, has_s = _bar_returns(data, data.col("SPY"))
    r_b, has_b = _bar_returns(data, data.col("IEF"))
    d = _simulate_breaches(r_s, r_b, has_s & has_b, params["threshold_pp"])

    trig = d != 0
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    tradable = np.asarray(data.tradable, dtype=bool)
    for name in MARKED:
        j = data.col(name)
        ok = trig & tradable[:, j]
        if data.universe is not None:
            ok = ok & np.asarray(data.universe[:, j], dtype=bool)
        mask[:, j] = ok
        side[:, j] = np.where(ok, -d.astype(float), 0.0)
    return mask, side
