"""Calibration strategy: delta-neutral funding carry."""
import numpy as np
from lab.framework.api import only_on, period_starts

PARAMS = {"gross": 0.9}


def target_weights(data, params):
    leg = float(params["gross"]) / 4.0
    w = np.zeros(data.shape)
    for sym in ("BTCUSDT", "ETHUSDT"):
        s, p = data.col(sym), data.col(sym + ".P")
        both = data.listed[:, s] & data.listed[:, p]          # carry needs both legs
        w[both, s] = leg
        w[both, p] = -leg
    return only_on(period_starts(data.dates, "W"), w)
