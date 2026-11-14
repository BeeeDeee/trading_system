"""H-0152 diagnostic: EVENT (1)-(7) of the card, side -1 (same code as strategy.compute_events)."""
import numpy as np

SHARES_FIELD = "sf1_arq_sharesbas"
MAX_PREV_AGE = 150
RATIO_CAP = 3.0
SPLIT_RATIOS = (1.5, 2.0, 3.0)
SPLIT_TOL = 0.03
RET_WINDOW = 252
RET_FLOOR = -0.5
MIN_VALID = 200


def _daily_returns(data):
    r = (1.0 + np.nan_to_num(np.asarray(data.ret_co, dtype=float))) * \
        (1.0 + np.nan_to_num(np.asarray(data.ret_oc, dtype=float))) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=float))
    r = np.where(valid, r, 0.0)
    return r, valid


def _window_sum(x, n):
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[n:] = c[n:] - c[:-n]
    return out


def _events(data, params):
    hold = int(params["hold_days"])
    imin = float(params["issuance_min"])
    x = np.asarray(data.extras[SHARES_FIELD], dtype=float)
    T, N = x.shape
    if data.universe is None:
        raise ValueError("H-0152 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    rows = np.arange(T)

    prev = np.full_like(x, np.nan)
    prev[1:] = x[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cand = np.isfinite(x) & np.isfinite(prev) & (prev > 0) & (x != prev)
        ratio = np.where(cand, x / np.where(prev > 0, prev, 1.0), np.nan)
    change = np.ones_like(cand)
    change[1:] = ~(x[1:] == x[:-1])
    run_start = np.maximum.accumulate(np.where(change, rows[:, None], 0), axis=0)
    prev_start = np.zeros_like(run_start)
    prev_start[1:] = run_start[:-1]
    cand &= (rows[:, None] - prev_start) < MAX_PREV_AGE
    with np.errstate(invalid="ignore"):
        cand &= (ratio >= 1.0 + imin) & (ratio <= RATIO_CAP)
        for k in SPLIT_RATIOS:
            cand &= ~(np.abs(ratio / k - 1.0) <= SPLIT_TOL)
    r, valid = _daily_returns(data)
    logr = np.log(np.maximum(1.0 + r, 1e-12))
    n_valid = _window_sum(valid.astype(float), RET_WINDOW)
    tret = np.expm1(_window_sum(logr, RET_WINDOW))
    cand &= (n_valid >= MIN_VALID) & (tret > RET_FLOOR)
    cand &= uni & tradable
    events = np.zeros((T, N), dtype=bool)
    tt, jj = np.nonzero(cand)
    last = {}
    for t, j in zip(tt.tolist(), jj.tolist()):
        lj = last.get(j)
        if lj is None or t - lj > hold:
            events[t, j] = True
            last[j] = t
    return events


def events(data, params):
    mask = _events(data, params)
    side = np.where(mask, -1.0, 0.0)
    return mask, side
