"""H-0033 mechanism event: positive release-session reaction of a LIQ-500 member.

Same event definition as strategy.py (code duplicated because the diagnostic may only import the
allowed modules): after the close of t, j is a LIQ-500 member tradable on t, ea_reaction_day == 1,
and a[t,j] / sigma[t,j] >= z_threshold, where a = close-to-close return minus the leave-one-out
equal-weight mean of the members tradable on t, and sigma = sample std of a over rows t-63..t-1
(at least 40 valid observations). Every such event counts, also those the strategy skips because the
tilt budget is full or the stock already has an open slot. side = +1 (long only).
"""
import numpy as np

SIGMA_WINDOW = 63
SIGMA_MIN_OBS = 40
_VAR_FLOOR = 1e-12


def _lagged_window_sum(x, window):
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) - window, 0)
    return c[:T] - c[lo]


def events(data, params):
    zthr = float(params["z_threshold"])
    if data.universe is None:
        raise ValueError("H-0033 needs the point-in-time LIQ-500 universe (data.universe is None)")
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    trad = np.asarray(data.tradable, dtype=bool)
    elig = np.asarray(data.universe, dtype=bool) & trad

    fin = np.isfinite(r)
    mem = elig & fin
    rm = np.where(mem, r, 0.0)
    S = rm.sum(axis=1, keepdims=True)
    K = mem.sum(axis=1, keepdims=True).astype(np.float64)
    k_ex = K - mem
    valid = trad & fin & (k_ex >= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(valid, r - (S - rm) / np.where(k_ex >= 1, k_ex, 1.0), 0.0)
    del rm
    n = _lagged_window_sum(valid.astype(np.float64), SIGMA_WINDOW)
    s1 = _lagged_window_sum(a, SIGMA_WINDOW)
    s2 = _lagged_window_sum(a * a, SIGMA_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    del s1, s2
    ok = valid & (n >= SIGMA_MIN_OBS) & (var > _VAR_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ok, a / np.sqrt(np.where(ok, var, 1.0)), np.nan)
        mask = elig & (ea == 1.0) & (z >= zthr)
    return mask, None
