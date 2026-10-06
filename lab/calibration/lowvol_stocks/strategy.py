"""Calibration strategy: low-volatility stocks."""
import numpy as np
from lab.framework.api import period_starts, rolling_mean, rolling_std

PARAMS = {"lookback_days": 252, "pct": 0.2}


def target_weights(data, params):
    L, pct = int(params["lookback_days"]), float(params["pct"])
    r = (1.0 + data.ret_co.astype(float)) * (1.0 + data.ret_oc.astype(float)) - 1.0
    vol = rolling_std(r, L)
    seasoned = rolling_mean(data.listed.astype(float), L) >= 0.95
    w = np.full(data.shape, np.nan)
    for t in np.flatnonzero(period_starts(data.dates, "M")):
        ok = np.flatnonzero(data.universe[t] & seasoned[t] & data.tradable[t] & np.isfinite(vol[t]))
        w[t] = 0.0
        if len(ok):
            k = max(1, int(pct * len(ok)))
            w[t, ok[np.argpartition(vol[t, ok], k - 1)[:k]]] = 1.0 / k
    return w
