"""Look-ahead canary: trades on tomorrow's return. Must fail G0 (truncation/perturbation test)."""

import numpy as np

PARAMS = {"k": 5}


def target_weights(data, params):
    cols = [data.instruments.index(f"S{10 + i:02d}") for i in range(params["k"])]
    tomorrow = np.roll(data.ret_oc, -1, axis=0)
    w = np.zeros(data.shape)
    up = tomorrow[:, cols] > 0
    n = np.maximum(up.sum(axis=1, keepdims=True), 1)
    w[:, cols] = up / n
    return w
