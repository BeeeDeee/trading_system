"""Calibration strategy: 12-1 momentum."""
import numpy as np
from lab.framework.api import period_starts, rolling_mean, total_return_index, trailing_return

PARAMS = {"lookback_days": 252, "pct": 0.2}


def target_weights(data, params):
    L, pct = int(params["lookback_days"]), float(params["pct"])
    mom = trailing_return(total_return_index(data), L, 21)
    seasoned = rolling_mean(data.listed.astype(float), L) >= 0.95
    w = np.full(data.shape, np.nan)
    for t in np.flatnonzero(period_starts(data.dates, "M")):
        ok = np.flatnonzero(data.universe[t] & seasoned[t] & data.tradable[t] & np.isfinite(mom[t]))
        w[t] = 0.0
        if len(ok):
            k = max(1, int(pct * len(ok)))
            w[t, ok[np.argpartition(-mom[t, ok], k - 1)[:k]]] = 1.0 / k
    return w
