"""Calibration strategy: BTC trend filter."""
import numpy as np
from lab.framework.api import rolling_mean, total_return_index

PARAMS = {"sma_days": 100}


def target_weights(data, params):
    n = int(params["sma_days"])
    idx = total_return_index(data)
    sma = rolling_mean(idx, n)
    w = np.where(np.isfinite(sma) & (idx > sma), 1.0, 0.0)
    w[: n - 1] = np.nan
    return w
