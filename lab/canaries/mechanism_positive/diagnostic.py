"""Mechanism test of the positive control: the planted signal on the five edge assets."""

import numpy as np

EDGE = ("S00", "S01", "S02", "S03", "S04")


def events(data, params):
    mask = np.zeros(data.shape, dtype=bool)
    side = np.ones(data.shape)
    on = data.series["signal"] > params["threshold"]
    for a in EDGE:
        j = data.instruments.index(a)
        mask[:, j] = True
        side[:, j] = np.where(on, 1.0, -1.0)
    return mask, side
