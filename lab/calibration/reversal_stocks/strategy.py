"""Calibration strategy: short-term reversal."""
import numpy as np
from lab.framework.api import total_return_index, trailing_return

PARAMS = {"lookback_days": 5, "pct": 0.1}


def target_weights(data, params):
    L, pct = int(params["lookback_days"]), float(params["pct"])
    ret = trailing_return(total_return_index(data), L)
    w = np.full(data.shape, np.nan)
    for t in range(L + 1, len(w)):
        ok = np.flatnonzero(data.universe[t] & data.tradable[t] & np.isfinite(ret[t]))
        w[t] = 0.0
        if len(ok):
            k = max(1, int(pct * len(ok)))
            w[t, ok[np.argpartition(ret[t, ok], k - 1)[:k]]] = 1.0 / k
    return w
