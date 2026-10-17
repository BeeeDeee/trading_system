"""H-0046 Threshold-rebalancing pressure.

A simulated 60/40 SPY/IEF book drifts with close-to-close total returns. When its equity weight leaves the
band 0.60 +/- threshold_pp/100 after the close of t, the simulated rebalancer is reset to target and the real
book tilts against the rebalancers' trade for one session (open of t+1 to open of t+2). Otherwise the book
holds 60/40 SPY/IEF, re-set at the first session of each month. Other universe instruments are never traded.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"threshold_pp": 1.0, "tilt": 0.4}

TARGET_EQ = 0.60


def _bar_returns(data, j):
    """Close-to-close total return of column j and whether the instrument has a bar on each row."""
    r = (1.0 + np.asarray(data.ret_co[:, j], dtype=float)) * (1.0 + np.asarray(data.ret_oc[:, j], dtype=float)) - 1.0
    close = np.asarray(data.close[:, j], dtype=float)
    has_bar = np.asarray(data.listed[:, j], dtype=bool) & np.isfinite(close) & np.isfinite(r)
    r = np.where(has_bar, r, 0.0)   # no bar: return counts as 0 for the drift
    return r, has_bar


def simulate_breaches(r_s, r_b, has_bar, threshold_pp):
    """Simulated 60/40 threshold rebalancer. Returns d (T,) int: +1 equity overweight, -1 underweight, 0 none.

    Row t uses rows 0..t only. G_s = G_b = 1 at the start of the loaded period; after each close t the growth
    factors are multiplied by (1 + r_t); on a trigger day they are reset to 1 after the close.
    """
    T = len(r_s)
    band = float(threshold_pp) / 100.0
    hi = TARGET_EQ + band
    lo = TARGET_EQ - band
    d = np.zeros(T, dtype=np.int64)
    g_s = 1.0
    g_b = 1.0
    for t in range(T):
        g_s *= 1.0 + r_s[t]
        g_b *= 1.0 + r_b[t]
        w = TARGET_EQ * g_s / (TARGET_EQ * g_s + (1.0 - TARGET_EQ) * g_b)
        if not has_bar[t]:
            continue
        if w > hi:
            d[t] = 1
        elif w < lo:
            d[t] = -1
        if d[t] != 0:
            g_s = 1.0
            g_b = 1.0
    return d


def breach_signal(data, params):
    js = data.col("SPY")
    jb = data.col("IEF")
    r_s, has_s = _bar_returns(data, js)
    r_b, has_b = _bar_returns(data, jb)
    return simulate_breaches(r_s, r_b, has_s & has_b, params["threshold_pp"])


def _month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.reshape(T)
    m = np.zeros(T, dtype=bool)
    m[ps.astype(np.int64)] = True
    return m


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    js = data.col("SPY")
    jb = data.col("IEF")
    tilt = float(params["tilt"])

    d = breach_signal(data, params)

    # equity weight of the target on each row (NaN = no decision, keep positions)
    eq = np.full(T, np.nan)
    month_start = _month_start_mask(data.dates)
    base = month_start.copy()
    if T > 0:
        base[0] = True                         # establish the 60/40 book on the first row
    prev_trig = np.zeros(T, dtype=bool)
    prev_trig[1:] = d[:-1] != 0                # tilt held one session, then back to 60/40
    base |= prev_trig
    eq[base] = TARGET_EQ
    trig = d != 0
    eq[trig] = np.clip(TARGET_EQ - tilt * d[trig], 0.0, 1.0)   # a new trigger overrides the reset

    w = np.full((T, N), np.nan)
    dec = np.isfinite(eq)
    w[dec, :] = 0.0                            # the other instruments are never held
    w[dec, js] = eq[dec]
    w[dec, jb] = 1.0 - eq[dec]
    return w
