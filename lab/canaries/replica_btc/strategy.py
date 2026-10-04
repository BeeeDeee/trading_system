"""Benchmark replica (real data): hold BTC, i.e. the benchmark itself. Must fail G1."""

import numpy as np

from lab.framework.api import only_on, period_starts

PARAMS = {"weight": 1.0}


def target_weights(data, params):
    w = np.zeros(data.shape)
    w[:, data.instruments.index("BTCUSDT")] = params["weight"]
    return only_on(period_starts(data.dates, "M"), w)
