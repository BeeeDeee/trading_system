"""H-0050: vol-control window drop-out flow. Tilt a 60/40 SPY/IEF book to 100 % SPY for hold_days sessions
when the reference vol-control rule (target 0.10, cap 1, max of 20/60-session SPY realized variance) would
mechanically raise its equity exposure by >= threshold (relative) when the oldest session leaves its window."""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 2, "threshold": 0.1}

# Fixed reference rule (institutional convention, not parameters; see card)
TARGET_VOL = 0.10
CAP = 1.0
W_SHORT = 20
W_LONG = 60
ANN = 252.0
MIN_SHORT = 18          # at least 18 of 20 returns present
MIN_LONG = 55           # at least 55 of 60 returns present
MIN_HISTORY = 60        # no event before 60 SPY returns exist

BASE_SPY = 0.6
BASE_IEF = 0.4


def _window(x, w):
    """Trailing sum over rows t-w+1..t (truncated at row 0)."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    t = np.arange(len(x))
    lo = np.maximum(t - w + 1, 0)
    return cs[t + 1] - cs[lo]


def _exposure(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(ANN * v)
        e = np.minimum(CAP, TARGET_VOL / sigma)
    return np.where(sigma > 0, e, CAP)


def spy_returns(data):
    """Close-to-close SPY total return and its availability mask (rows 0..t only)."""
    j = data.col("SPY")
    rco = np.asarray(data.ret_co[:, j], dtype=np.float64)
    roc = np.asarray(data.ret_oc[:, j], dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    listed = np.asarray(data.listed[:, j], dtype=bool)
    tradable = np.asarray(data.tradable[:, j], dtype=bool)
    prev_listed = np.concatenate([[False], listed[:-1]])
    # a session's return is available if SPY had a bar that day (dropped vendor-spike days are not tradable)
    # and a previous close exists
    valid = listed & tradable & prev_listed & np.isfinite(r)
    return np.where(valid, r, 0.0), valid


def vol_events(data, threshold):
    """(T,) bool: at the close of t the reference rule's exposure would rise by >= threshold (relative)
    when sessions t-19 / t-59 leave the windows and the new session equals the current window mean."""
    r, valid = spy_returns(data)
    T = len(r)
    sq = r * r
    vf = valid.astype(np.float64)

    def current_and_next(w):
        s = _window(sq, w)
        n = _window(vf, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = s / n
        idx = np.arange(T) - (w - 1)              # session leaving the window when t+1 is added
        ok = idx >= 0
        idxc = np.where(ok, idx, 0)
        leave_valid = ok & valid[idxc]
        leave_sq = np.where(leave_valid, sq[idxc], 0.0)
        # next window = available returns in t-w+2..t plus a neutral new day equal to v
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


def target_weights(data, params):
    hold = int(params["hold_days"])
    threshold = float(params["threshold"])
    T = len(data.dates)
    N = len(data.instruments)
    js = data.col("SPY")
    ji = data.col("IEF")

    ev = vol_events(data, threshold)
    # tilt on decision rows t..t+hold-1 after an event at t (held from open t+1 through close t+hold);
    # a new event inside a window extends it (no stacking)
    ce = np.concatenate([[0], np.cumsum(ev.astype(np.int64))])
    t = np.arange(T)
    tilt = (ce[t + 1] - ce[np.maximum(t - hold + 1, 0)]) > 0
    prev_tilt = np.concatenate([[False], tilt[:-1]])
    tilt_end = prev_tilt & ~tilt

    month_start = np.asarray(period_starts(data.dates, "M"), dtype=bool)
    first = np.zeros(T, dtype=bool)
    if T > 0:
        first[0] = True
    base_row = ~tilt & (month_start | tilt_end | first)

    w = np.full((T, N), np.nan)
    w[tilt, :] = 0.0
    w[tilt, js] = 1.0
    w[base_row, :] = 0.0
    w[base_row, js] = BASE_SPY
    w[base_row, ji] = BASE_IEF

    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
        dec = ~np.isnan(w)
        w = np.where(dec & ~uni, 0.0, w)
    return w
