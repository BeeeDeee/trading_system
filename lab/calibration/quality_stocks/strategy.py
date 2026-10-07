"""Calibration strategy: exercises SF1, insider and 13F fields (a loader test)."""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"pct": 0.2}


def _rank(x, ok):
    r = np.full(x.shape, 0.5)
    idx = np.flatnonzero(ok & np.isfinite(x))
    if len(idx) > 1:
        r[idx] = np.argsort(np.argsort(x[idx])) / (len(idx) - 1)
    return r


def target_weights(data, params):
    pct = float(params["pct"])
    roe = data.extras["sf1_art_roe"]
    buy = data.extras["ins_buy_value_91d"]
    io = data.extras["f13_io"]
    w = np.full(data.shape, np.nan)
    for t in np.flatnonzero(period_starts(data.dates, "M")):
        w[t] = 0.0
        ok = data.universe[t] & data.tradable[t] & np.isfinite(roe[t])
        idx = np.flatnonzero(ok)
        if len(idx) < 20:
            continue
        score = _rank(roe[t], ok) + _rank(buy[t], ok) + _rank(io[t], ok)
        k = max(1, int(pct * len(idx)))
        w[t, idx[np.argpartition(-score[idx], k - 1)[:k]]] = 1.0 / k
    return w
