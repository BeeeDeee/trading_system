"""H-0050 mechanism events: the reference vol-control rule (target 0.10, cap 1, max of 20/60-session SPY realized
variance) would mechanically raise its equity exposure by >= threshold (relative) when the oldest session leaves
its window at t+1 (new session = window mean). Same code as strategy.vol_events (copied: imports between
strategy files are not allowed). Marks SPY, IVV, VTI, DIA with side +1; IEF, TLT, SHY, AGG are unmarked peers."""
import numpy as np

EQUITY = ("SPY", "IVV", "VTI", "DIA")

TARGET_VOL = 0.10
CAP = 1.0
W_SHORT = 20
W_LONG = 60
ANN = 252.0
MIN_SHORT = 18
MIN_LONG = 55
MIN_HISTORY = 60


def _window(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    t = np.arange(len(x))
    lo = np.maximum(t - w + 1, 0)
    return cs[t + 1] - cs[lo]


def _exposure(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(ANN * v)
        e = np.minimum(CAP, TARGET_VOL / sigma)
    return np.where(sigma > 0, e, CAP)


def _spy_returns(data):
    j = data.col("SPY")
    rco = np.asarray(data.ret_co[:, j], dtype=np.float64)
    roc = np.asarray(data.ret_oc[:, j], dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    listed = np.asarray(data.listed[:, j], dtype=bool)
    tradable = np.asarray(data.tradable[:, j], dtype=bool)
    prev_listed = np.concatenate([[False], listed[:-1]])
    valid = listed & tradable & prev_listed & np.isfinite(r)
    return np.where(valid, r, 0.0), valid


def _vol_events(data, threshold):
    r, valid = _spy_returns(data)
    T = len(r)
    sq = r * r
    vf = valid.astype(np.float64)

    def current_and_next(w):
        s = _window(sq, w)
        n = _window(vf, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = s / n
        idx = np.arange(T) - (w - 1)
        ok = idx >= 0
        idxc = np.where(ok, idx, 0)
        leave_valid = ok & valid[idxc]
        leave_sq = np.where(leave_valid, sq[idxc], 0.0)
        s_next = s - leave_sq + v
        n_next = n - leave_valid.astype(np.float64) + 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            v_next = s_next / n_next
        return v, v_next, n, ok

    v20, v20n, n20, ok20 = current_and_next(W_SHORT)
    v60, v60n, n60, ok60 = current_and_next(W_LONG)
    e_now = _exposure(np.maximum(v20, v60))
    e_next = _exposure(np.maximum(v20n, v60n))
    history = np.cumsum(vf)
    eligible = ok20 & ok60 & (n20 >= MIN_SHORT) & (n60 >= MIN_LONG) & (history >= MIN_HISTORY)
    eligible &= np.isfinite(e_now) & np.isfinite(e_next)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (e_next - e_now) / e_now
    return eligible & (rel >= threshold)


def events(data, params):
    ev = _vol_events(data, float(params["threshold"]))
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    for name in EQUITY:
        j = data.col(name)
        ok = np.asarray(data.tradable[:, j], dtype=bool) & np.asarray(data.listed[:, j], dtype=bool)
        if data.universe is not None:
            ok &= np.asarray(data.universe[:, j], dtype=bool)
        mask[:, j] = ev & ok
    side = np.ones((T, N), dtype=np.float64)
    return mask, side
