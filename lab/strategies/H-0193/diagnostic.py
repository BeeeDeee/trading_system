"""H-0193 mechanism_test events: the same crossing event as strategy.py (copied, separate file)."""
import numpy as np


def events(data, params):
    X = float(params["min_buy_usd"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0193 needs the LIQ-500 point-in-time universe")
    v = np.asarray(data.extras["ins_buy_value_91d"])
    n = np.asarray(data.extras["ins_buy_n_91d"])
    mask = np.zeros(v.shape, dtype=bool)      # row 0 has no t-1: no event
    if v.shape[0] > 1:
        vt, vp, nt, np_ = v[1:], v[:-1], n[1:], n[:-1]
        with np.errstate(invalid="ignore"):
            mask[1:] = (vp < X) & (vt >= X) & (nt > np_)
        mask[1:] &= np.isfinite(vt) & np.isfinite(vp) & np.isfinite(nt) & np.isfinite(np_)
    mask &= np.asarray(uni, dtype=bool)
    mask &= np.asarray(data.tradable, dtype=bool)
    return mask, None
