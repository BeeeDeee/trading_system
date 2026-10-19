"""H-0056 mechanism test: on the last XNYS session of each calendar week (same point-in-time rule as the
strategy), mark SPY when the oil-specific shock |z_t| >= 1.0, side = -sign(z_t). IEF, TLT, AGG and SHY are
never marked (unmarked peers). Signal code duplicated from strategy.py (same formulas)."""
import numpy as np

OIL_KEY = "fred_macro.DCOILWTICO"
BETA_WINDOW = 252
STD_WINDOW = 756
EVENT_THRESHOLD = 1.0     # fixed by the card, not a strategy parameter


def _rolling_sum(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    bad = ~np.isfinite(x)
    c = np.concatenate((np.zeros(1), np.cumsum(np.where(bad, 0.0, x))))
    b = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(bad.astype(np.int64))))
    out = np.full(T, np.nan)
    if T >= n:
        s = c[n:] - c[:-n]
        nb = b[n:] - b[:-n]
        out[n - 1:] = np.where(nb == 0, s, np.nan)
    return out


def _ffill_positive(p):
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(p) & (p > 0)
    idx = np.where(valid, np.arange(p.shape[0]), -1)
    idx = np.maximum.accumulate(idx) if p.shape[0] else idx
    return np.where(idx >= 0, p[np.maximum(idx, 0)], np.nan)


def _z(data, params):
    T = len(data.dates)
    L = int(params["lookback_days"])
    raw = np.asarray(data.series[OIL_KEY], dtype=np.float64).reshape(T)
    pf = _ffill_positive(raw)
    o = np.full(T, np.nan)
    j = data.col("SPY")
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)[:, j]) * \
        (1.0 + np.asarray(data.ret_oc, dtype=np.float64)[:, j]) - 1.0
    r = np.where(np.asarray(data.listed)[:, j], r, np.nan)
    s = np.full(T, np.nan)
    if T > 1:
        o[1:] = pf[1:] / pf[:-1] - 1.0
        s[1:] = r[:-1]
    ok = np.isfinite(o) & np.isfinite(s)
    x = np.where(ok, s, np.nan)
    y = np.where(ok, o, np.nan)
    n = float(BETA_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = _rolling_sum(x, BETA_WINDOW)
        sy = _rolling_sum(y, BETA_WINDOW)
        sxy = _rolling_sum(x * y, BETA_WINDOW)
        sxx = _rolling_sum(x * x, BETA_WINDOW)
        den = n * sxx - sx * sx
        beta = np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)
        r_oil = np.expm1(_rolling_sum(np.log1p(o), L))
        r_spy = np.expm1(_rolling_sum(np.log1p(s), L))
        shock = r_oil - beta * r_spy
        rms = np.sqrt(_rolling_sum(shock * shock, STD_WINDOW) / STD_WINDOW)
        z = np.where(rms > 0, shock / np.where(rms > 0, rms, 1.0), np.nan)
    return np.where(np.isfinite(z), z, np.nan)


def _weekly_decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    wd = (d + 3) % 7
    wk = (d + 3) // 7
    dec = wd == 4
    if d.shape[0] > 1:
        late = (wk[1:] > wk[:-1]) & (wd[:-1] != 4)
        dec[1:] = dec[1:] | late
    return dec


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    j = data.col("SPY")
    z = _z(data, params)
    dec = _weekly_decision_rows(data.dates)
    hit = dec & np.isfinite(z) & (np.abs(np.where(np.isfinite(z), z, 0.0)) >= EVENT_THRESHOLD)
    hit = hit & np.asarray(data.tradable)[:, j].astype(bool)
    if data.universe is not None:
        hit = hit & np.asarray(data.universe)[:, j].astype(bool)
    mask = np.zeros((T, N), dtype=bool)
    mask[:, j] = hit
    side = np.zeros((T, N), dtype=np.float64)
    side[:, j] = np.where(hit, -np.sign(np.where(np.isfinite(z), z, 0.0)), 0.0)
    side[~mask] = 1.0
    return mask, side
