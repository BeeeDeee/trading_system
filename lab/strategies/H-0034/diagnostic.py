"""H-0034 mechanism-test events: same event definition and side as strategy.compute_events.

Event on (i, t): LIQ-500 member at t, tradable at t, nonearn_8k_reaction_day == 1 at t, ea_reaction_day == 0
on rows t-2 .. t, |a / sigma| >= z_threshold where a = r - equal-weight mean of valid members and sigma is the
sample std of a over rows t-60 .. t-1 (>= 40 valid values). Side = sign(z).
"""
import numpy as np

WINDOW = 60
MIN_OBS = 40
EA_LOOKBACK = 2


def _events(data, params):
    thr = float(params["z_threshold"])
    if data.universe is None:
        raise ValueError("H-0034 needs the LIQ-500 point-in-time universe")
    univ = np.asarray(data.universe, dtype=bool)
    nonearn = np.asarray(data.extras["nonearn_8k_reaction_day"])
    ea = np.asarray(data.extras["ea_reaction_day"])

    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = (np.asarray(data.listed, dtype=bool)
             & np.isfinite(np.asarray(data.close, dtype=np.float64))
             & np.isfinite(r))
    T, N = r.shape

    mv = univ & valid
    cnt = mv.sum(axis=1)
    tot = np.where(mv, r, 0.0).sum(axis=1)
    mkt = np.full(T, np.nan)
    has = cnt > 0
    mkt[has] = tot[has] / cnt[has]
    a = np.where(valid, r - mkt[:, None], np.nan)
    del r, tot
    a_ok = np.isfinite(a)

    ea_zero = ea == 0
    ea_ok = ea_zero.copy()
    ea_ok[:EA_LOOKBACK] = False
    for k in range(1, EA_LOOKBACK + 1):
        ea_ok[EA_LOOKBACK:] &= ea_zero[EA_LOOKBACK - k:T - k]
    cand = univ & (nonearn == 1) & ea_ok & a_ok

    event = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    rows = np.flatnonzero(cand.any(axis=1))
    for t in rows:
        if t < MIN_OBS:
            continue
        idx = np.flatnonzero(cand[t])
        block = a[t - WINDOW if t >= WINDOW else 0:t][:, idx]
        ok = np.isfinite(block)
        n = ok.sum(axis=0)
        b0 = np.where(ok, block, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = b0.sum(axis=0) / n
            dev = np.where(ok, block - mean, 0.0)
            var = (dev * dev).sum(axis=0) / (n - 1)
            sigma = np.sqrt(var)
            z = a[t, idx] / sigma
        good = (n >= MIN_OBS) & np.isfinite(z) & (sigma > 0) & (np.abs(z) >= thr)
        sel = idx[good]
        event[t, sel] = True
        side[t, sel] = np.sign(z[good])
    return event, side


def events(data, params):
    event, side = _events(data, params)
    mask = event & np.asarray(data.universe, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    side = np.where(mask, side, 1.0)
    return mask, side
