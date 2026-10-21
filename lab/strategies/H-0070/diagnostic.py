"""H-0070 mechanism events: on an SPY stress day (z <= -k), the trust with the lowest day-t return relative
to the cross-sectional mean of the six is marked +1, the one with the highest is marked -1. Same logic as the
strategy's cohort legs (duplicated here because the diagnostic may only import numpy / lab.framework)."""
import numpy as np

TRUSTS = ("FXA", "FXB", "FXC", "FXE", "FXF", "FXY")
VOL_WINDOW = 63
VOL_MIN_VALID = 60
MIN_VALID_TRUSTS = 5


def _signal_col(data):
    # the signal-only equity instrument (SPY in the card) = the single instrument outside the six trusts
    others = [j for j, name in enumerate(data.instruments) if name not in TRUSTS]
    if len(others) != 1:
        raise ValueError("expected exactly one signal instrument besides the six trusts, got %d" % len(others))
    return others[0]


def _valid_returns(data, cols):
    ret_co = np.asarray(data.ret_co, dtype=float)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    prev_tradable = np.zeros_like(tradable)
    prev_tradable[1:] = tradable[:-1]
    valid = listed & tradable & prev_tradable & np.isfinite(r)
    return r, valid


def _stress_days(data, k):
    j = _signal_col(data)
    r, valid = _valid_returns(data, [j])
    r = r[:, 0]
    valid = valid[:, 0]
    T = r.shape[0]
    x = np.where(valid, np.log1p(np.where(valid, r, 0.0)), np.nan)
    sigma = np.full(T, np.nan)
    if T > VOL_WINDOW:
        windows = np.lib.stride_tricks.sliding_window_view(x[:-1], VOL_WINDOW)
        n = np.sum(np.isfinite(windows), axis=1)
        filled = np.where(np.isfinite(windows), windows, 0.0)
        mean = np.sum(filled, axis=1) / np.maximum(n, 1)
        dev = np.where(np.isfinite(windows), windows - mean[:, None], 0.0)
        var = np.sum(dev * dev, axis=1) / np.maximum(n - 1, 1)
        s = np.sqrt(var)
        s = np.where(n >= VOL_MIN_VALID, s, np.nan)
        sigma[VOL_WINDOW:] = s
    ok = valid & np.isfinite(sigma) & (sigma > 0)
    z = np.full(T, np.nan)
    z[ok] = r[ok] / sigma[ok]
    return ok & (z <= -k)


def events(data, params):
    k = float(params["k"])
    T = len(data.dates)
    N = len(data.instruments)
    cols = [data.col(name) for name in TRUSTS]
    r, valid = _valid_returns(data, cols)
    if data.universe is not None:
        valid = valid & np.asarray(data.universe, dtype=bool)[:, cols]
    stress = _stress_days(data, k)
    n_valid = valid.sum(axis=1)
    cohort = stress & (n_valid >= MIN_VALID_TRUSTS)
    rr = np.where(valid, r, 0.0)
    mean = rr.sum(axis=1) / np.maximum(n_valid, 1)
    u = r - mean[:, None]
    lo = np.argmin(np.where(valid, u, np.inf), axis=1)
    hi = np.argmax(np.where(valid, u, -np.inf), axis=1)
    cohort = cohort & (lo != hi)
    cols_arr = np.asarray(cols)
    rows = np.nonzero(cohort)[0]
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    mask[rows, cols_arr[lo[rows]]] = True
    side[rows, cols_arr[lo[rows]]] = 1.0
    mask[rows, cols_arr[hi[rows]]] = True
    side[rows, cols_arr[hi[rows]]] = -1.0
    return mask, side
