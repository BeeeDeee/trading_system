"""H-0072 Mortgage-convexity hedging continuation (IEF, long/short, event book).

After the close of session t:
  r(t)    = (1 + ret_co(t)) * (1 + ret_oc(t)) - 1  of IEF (close-to-close total return)
  sd63(t) = sample std (ddof=1) of r over rows t-63..t-1 (t excluded), >= 50 valid returns
  y(t)    = latest usable DGS10 (series as aligned by the next_morning clock), forward-filled
            over at most 5 missing rows
  ymin(t) = min of y over the last 252 rows up to and including t, >= 200 valid values
  state   = y - ymin <= band_pp
  event   = state and IEF bar on t and |r(t)| >= k * sd63(t); side = sign(r(t))
Book: weight row t = side of the most recent event e with t - e < hold_days, else 0
(entry at open e+1, exit at open e+1+hold_days; a new event resets side and exit).
SHY/SHV/BIL always 0.
"""
import numpy as np

PARAMS = {"band_pp": 0.75, "hold_days": 5, "k": 1.25}

SD_WINDOW = 63
SD_MIN_VALID = 50
YMIN_WINDOW = 252
YMIN_MIN_VALID = 200
FFILL_LIMIT = 5


def _shift(x, k):
    """x shifted down by k >= 1 rows (row t gets x[t-k]); NaN for the first k rows."""
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def ief_returns(data):
    j = data.col("IEF")
    ret_co = np.asarray(data.ret_co[:, j], dtype=float)
    ret_oc = np.asarray(data.ret_oc[:, j], dtype=float)
    close = np.asarray(data.close[:, j], dtype=float)
    bar = np.isfinite(close) & np.isfinite(ret_co) & np.isfinite(ret_oc)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    r = np.where(bar, r, np.nan)
    return r, bar


def trailing_sd(r):
    """Sample std of r over rows t-63..t-1 (t excluded); NaN with < 50 valid values."""
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


def ffill_limited(y, limit):
    """Forward fill NaNs from the last valid value if it is at most `limit` rows old."""
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


def convexity_state(y_raw, band_pp):
    y = ffill_limited(np.asarray(y_raw, dtype=float), FFILL_LIMIT)
    T = len(y)
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


def compute_events(data, params):
    """(T,) bool event on IEF and (T,) side in {+1, -1, 0}."""
    r, bar = ief_returns(data)
    sd = trailing_sd(r)
    state = convexity_state(data.series["fred_macro.DGS10"], float(params["band_pp"]))
    k = float(params["k"])
    with np.errstate(invalid="ignore"):
        big = bar & np.isfinite(sd) & (sd > 0) & (np.abs(r) >= k * sd)
    ev = big & state
    side = np.where(ev, np.sign(np.where(np.isfinite(r), r, 0.0)), 0.0)
    ev = ev & (side != 0)
    return ev, side


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    j = data.col("IEF")
    ev, side = compute_events(data, params)
    h = int(params["hold_days"])
    idx = np.arange(T)
    last = np.where(ev, idx, -1)
    last = np.maximum.accumulate(last) if T else last
    active = (last >= 0) & ((idx - last) < h)
    w = np.zeros((T, N), dtype=float)
    w[active, j] = side[last[active]]
    return w
