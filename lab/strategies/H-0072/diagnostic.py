"""H-0072 diagnostic: the same IEF event as strategy.py (state ON and |r| >= k * sd63).

IEF is marked with side = sign(r(t)); SHY, SHV and BIL are never marked (near-cash peers).
The event logic is a copy of strategy.compute_events (same constants, same rules).
"""
import numpy as np

SD_WINDOW = 63
SD_MIN_VALID = 50
YMIN_WINDOW = 252
YMIN_MIN_VALID = 200
FFILL_LIMIT = 5


def _shift(x, k):
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _ief_returns(data):
    j = data.col("IEF")
    ret_co = np.asarray(data.ret_co[:, j], dtype=float)
    ret_oc = np.asarray(data.ret_oc[:, j], dtype=float)
    close = np.asarray(data.close[:, j], dtype=float)
    bar = np.isfinite(close) & np.isfinite(ret_co) & np.isfinite(ret_oc)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    r = np.where(bar, r, np.nan)
    return r, bar


def _trailing_sd(r):
    T = len(r)
    s = np.zeros(T)
    n = np.zeros(T)
    for k in range(1, SD_WINDOW + 1):
        x = _shift(r, k)
        ok = np.isfinite(x)
        s += np.where(ok, x, 0.0)
        n += ok
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ss = np.zeros(T)
    for k in range(1, SD_WINDOW + 1):
        x = _shift(r, k)
        ok = np.isfinite(x)
        ss += np.where(ok, (x - mean) ** 2, 0.0)
    sd = np.sqrt(ss / np.maximum(n - 1, 1))
    return np.where(n >= SD_MIN_VALID, sd, np.nan)


def _ffill_limited(y, limit):
    T = len(y)
    idx = np.arange(T)
    valid = np.isfinite(y)
    last = np.where(valid, idx, -1)
    last = np.maximum.accumulate(last) if T else last
    gap = idx - last
    ok = (last >= 0) & (gap <= limit)
    out = np.full(T, np.nan)
    out[ok] = y[last[ok]]
    return out


def _convexity_state(y_raw, band_pp):
    y = _ffill_limited(np.asarray(y_raw, dtype=float), FFILL_LIMIT)
    mn = np.where(np.isfinite(y), y, np.inf)
    cnt = np.isfinite(y).astype(float)
    for k in range(1, YMIN_WINDOW):
        x = _shift(y, k)
        ok = np.isfinite(x)
        mn = np.minimum(mn, np.where(ok, x, np.inf))
        cnt += ok
    ymin = np.where(cnt >= YMIN_MIN_VALID, mn, np.nan)
    with np.errstate(invalid="ignore"):
        state = np.isfinite(y) & np.isfinite(ymin) & ((y - ymin) <= band_pp)
    return state


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    j = data.col("IEF")
    r, bar = _ief_returns(data)
    sd = _trailing_sd(r)
    state = _convexity_state(data.series["fred_macro.DGS10"], float(params["band_pp"]))
    k = float(params["k"])
    with np.errstate(invalid="ignore"):
        big = bar & np.isfinite(sd) & (sd > 0) & (np.abs(r) >= k * sd)
    ev = big & state
    sgn = np.sign(np.where(np.isfinite(r), r, 0.0))
    ev = ev & (sgn != 0)
    ev = ev & np.asarray(data.tradable[:, j], dtype=bool)
    if data.universe is not None:
        ev = ev & np.asarray(data.universe[:, j], dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    mask[:, j] = ev
    side[:, j] = np.where(ev, sgn, 0.0)
    return mask, side
