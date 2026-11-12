"""H-0131: 60/40 SPY/IEF book tilted toward IEF for 5 sessions after a large stock-over-Treasury rally.

Decision after the close of session t (rows 0..t only), fill at the open of t+1.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"k": 1.5, "tilt": 0.45}

BASE_SPY = 0.60
BASE_IEF = 0.40
SHORT_WIN = 5      # rel5: sessions t-4..t
LONG_WIN = 21      # rel21: sessions t-20..t
VOL_WIN = 252      # s: sessions t-251..t
VOL_MIN = 200      # at least 200 valid d values
STATE_WIN = 5      # rally state on any of sessions t-4..t


def _window_sum(x, n):
    """Trailing sum over rows t-n+1..t (partial at the start); uses rows <= t only."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    t = np.arange(x.shape[0])
    lo = np.maximum(t + 1 - n, 0)
    return c[t + 1] - c[lo]


def daily_diff(ret_co_s, ret_oc_s, ret_co_i, ret_oc_i, valid):
    """d(t) = r_SPY(t) - r_IEF(t), r = (1 + ret_co)(1 + ret_oc) - 1; NaN where either has no bar."""
    r_s = (1.0 + ret_co_s.astype(np.float64)) * (1.0 + ret_oc_s.astype(np.float64)) - 1.0
    r_i = (1.0 + ret_co_i.astype(np.float64)) * (1.0 + ret_oc_i.astype(np.float64)) - 1.0
    d = r_s - r_i
    ok = valid & np.isfinite(d)
    return np.where(ok, d, np.nan)


def rally_state(d, valid, k):
    """Rally state on each session: rel5 >= k * s and rel21 > 0, with s = std(d, 252 sessions) * sqrt(5)."""
    ok = np.isfinite(d)
    d0 = np.where(ok, d, 0.0)
    rel5 = _window_sum(d0, SHORT_WIN)
    rel21 = _window_sum(d0, LONG_WIN)
    n = _window_sum(ok.astype(np.float64), VOL_WIN)
    s1 = _window_sum(d0, VOL_WIN)
    s2 = _window_sum(d0 * d0, VOL_WIN)
    enough = n >= VOL_MIN
    nn = np.where(enough, n, 2.0)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)          # sample variance (ddof = 1)
    var = np.maximum(var, 0.0)
    s = np.sqrt(var) * np.sqrt(float(SHORT_WIN))
    return valid & ok & enough & (rel5 >= k * s) & (rel21 > 0.0)


def book(rally, valid, month_start, tilt):
    """Per-row (w_spy, w_ief) targets; NaN = no decision (keep / let drift).

    Decide on a session with SPY and IEF bars when the target changes or when a month started
    since the last decision; the first valid session sets the base target.
    """
    T = rally.shape[0]
    active = _window_sum(rally.astype(np.float64), STATE_WIN) > 0.5
    w_s = np.full(T, np.nan)
    w_i = np.full(T, np.nan)
    last = -1
    pending = False
    for t in range(T):
        if month_start[t]:
            pending = True
        if not valid[t]:
            continue
        a = 1 if active[t] else 0
        if pending or last < 0 or a != last:
            w_s[t] = BASE_SPY - tilt * a
            w_i[t] = BASE_IEF + tilt * a
            last = a
            pending = False
    return w_s, w_i


def target_weights(data, params):
    k = float(params["k"])
    tilt = float(params["tilt"])
    js = data.col("SPY")
    ji = data.col("IEF")
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    valid = listed[:, js] & listed[:, ji] & np.isfinite(close[:, js]) & np.isfinite(close[:, ji])
    d = daily_diff(data.ret_co[:, js], data.ret_oc[:, js], data.ret_co[:, ji], data.ret_oc[:, ji], valid)
    rally = rally_state(d, valid, k)
    ms = np.zeros(len(data.dates), dtype=bool)
    ps = np.asarray(period_starts(data.dates, "M"))
    if ps.dtype == bool:
        ms = ps.copy()
    else:
        ms[ps.astype(np.int64)] = True
    w_s, w_i = book(rally, valid, ms, tilt)
    T = len(data.dates)
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    dec = np.isfinite(w_s)
    w[dec, :] = 0.0
    w[dec, js] = w_s[dec]
    w[dec, ji] = w_i[dec]
    return w
