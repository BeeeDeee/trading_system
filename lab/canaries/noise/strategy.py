"""Noise canary: random weekly weights. Must not pass G1-G3."""

import numpy as np

from lab.framework.api import only_on, period_starts

PARAMS = {"seed": 0}


def target_weights(data, params):
    rng = np.random.default_rng(params["seed"])
    T = data.shape[0]
    cols = [data.instruments.index(f"S{i:02d}") for i in range(5, 25)]
    raw = rng.dirichlet(np.ones(len(cols)), size=T)
    w = np.zeros(data.shape)
    w[:, cols] = raw * data.listed[:, cols]
    return only_on(period_starts(data.dates, "W"), w)
