"""Mechanism test with no mechanism: random events (point-in-time: row t of the stream does not depend on T)."""

import numpy as np


def events(data, params):
    return np.random.default_rng(7).random(data.shape) < 0.03, None
