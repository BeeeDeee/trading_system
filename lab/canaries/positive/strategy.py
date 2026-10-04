"""Positive control: exploits the edge planted in the synthetic market. Must pass G0-G4."""

import numpy as np

PARAMS = {"threshold": 0.0}
EDGE = ("S00", "S01", "S02", "S03", "S04")


def target_weights(data, params):
    w = np.zeros(data.shape)
    on = data.series["signal"] > params["threshold"]
    cols = [data.instruments.index(a) for a in EDGE]
    w[np.ix_(on, cols)] = 1.0 / len(cols)
    w[~on, data.instruments.index("MKT")] = 1.0
    return w
