"""Overfit canary: random weekly market timing (in MKT or in cash) from a generator seeded with `seed`; the
seed with the best dev Sharpe of 2000 is picked. Must fail G2 (neighbors) or G3 (deflated Sharpe)."""

import numpy as np

from lab.framework.api import only_on, period_starts

PARAMS = {"seed": 0}


def target_weights(data, params):
    rng = np.random.default_rng(params["seed"])
    w = np.zeros(data.shape)
    w[:, data.instruments.index("MKT")] = rng.random(data.shape[0]) < 0.5
    return only_on(period_starts(data.dates, "W"), w)
