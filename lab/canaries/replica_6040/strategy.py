"""Benchmark replica (real data): 60/40 SPY/IEF rebalanced monthly, i.e. the benchmark itself. Must fail G1."""

import numpy as np

from lab.framework.api import only_on, period_starts

PARAMS = {"equity": 0.6}


def target_weights(data, params):
    w = np.zeros(data.shape)
    w[:, data.instruments.index("SPY")] = params["equity"]
    w[:, data.instruments.index("IEF")] = 1.0 - params["equity"]
    return only_on(period_starts(data.dates, "M"), w)
