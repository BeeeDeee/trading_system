"""H-0206 mechanism test: R_t (+1, Rule 201 certainly binds through the close of t+1) vs N_t (-1, near misses).

Rule-based, independent of the model; same candidate definitions as strategy.py (copied, separate file)."""
import numpy as np

THRESH = -0.10
FLOOR = -0.15


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0206 is a liq_n card: data.universe is required")
    uni = np.asarray(uni, dtype=bool)
    near_band = float(params["near_band"])
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    px = np.isfinite(close) & listed
    prev_px = np.zeros_like(px)
    prev_px[1:] = px[:-1]
    valid = px & prev_px & tradable & np.isfinite(co) & np.isfinite(oc)
    r = np.where(valid, (1.0 + co) * (1.0 + oc) - 1.0, np.nan)
    g = np.where(valid, co, np.nan)
    ok = valid & uni
    with np.errstate(invalid="ignore"):
        restricted = ok & ((r <= THRESH) | (g <= THRESH)) & (r >= FLOOR)
        near = ok & (r > THRESH) & (r <= THRESH + near_band) & (g > THRESH)
    mask = restricted | near
    side = np.where(restricted, 1.0, np.where(near, -1.0, 0.0))
    return mask, side
