"""Hindsight canary: the positive control plus a hard-coded crisis date. Must fail the G0 static scan."""

import numpy as np

PARAMS = {"threshold": 0.0}
EDGE = ("S00", "S01", "S02", "S03", "S04")
CRISIS = "2008-09-15"


def target_weights(data, params):
    w = np.zeros(data.shape)
    on = data.series["signal"] > params["threshold"]
    cols = [data.instruments.index(a) for a in EDGE]
    w[np.ix_(on, cols)] = 1.0 / len(cols)
    w[~on, data.instruments.index("MKT")] = 1.0
    w[data.dates.astype(str) == CRISIS] = 0.0
    return w
