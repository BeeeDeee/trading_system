"""H-0056: oil-specific shock underreaction, weekly tilt of a 60/40 SPY/IEF book.

Rules of card.signal.description and where they are implemented:
- SPY daily close-to-close total return from ret_co and ret_oc               -> _spy_returns
- WTI (fred_macro.DCOILWTICO, clock next_morning, already aligned by the
  framework: row t holds the value of session t-1); missing / non-positive
  values carried forward, return 0 on that session                          -> _oil_returns
- every window ends at session t-1 for BOTH oil and SPY                      -> both series are
  indexed so that row t holds the return of session t-1
- (1) beta_t: OLS slope (with intercept) of oil on SPY daily returns over 252 sessions ending t-1
- (2) shock_t = R_oil(L) - beta_t * R_SPY(L), compounded over L sessions ending t-1
- (3) z_t = shock_t / sqrt(mean(shock^2 over 756 rows ending t)), not demeaned, clipped to [-2, 2]
- decision after the last XNYS session of each calendar week                 -> weekly_decision_rows
- w_SPY = clip(0.60 - tilt_per_z * z, 0.20, 1.00), w_IEF = 1 - w_SPY, TLT/AGG/SHY = 0
- warm-up: first decision once IEF has a price and z is fully defined; before: flat
- z not computable on a later decision date -> z = 0 (60/40)
"""
import numpy as np

PARAMS = {"lookback_days": 21, "tilt_per_z": 0.15}

OIL_KEY = "fred_macro.DCOILWTICO"
BETA_WINDOW = 252
STD_WINDOW = 756
Z_CLIP = 2.0
BASE_SPY = 0.60
W_SPY_MIN = 0.20
W_SPY_MAX = 1.00


def _rolling_sum(x, n):
    """Trailing sum over n rows ending at t; NaN if the window is incomplete or has a non-finite value."""
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
    """Carry the last positive finite value forward; NaN before the first one."""
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(p) & (p > 0)
    idx = np.where(valid, np.arange(p.shape[0]), -1)
    idx = np.maximum.accumulate(idx) if p.shape[0] else idx
    return np.where(idx >= 0, p[np.maximum(idx, 0)], np.nan)


def _oil_returns(data):
    """Row t: simple WTI return of session t-1 (series row t already holds the value of session t-1)."""
    T = len(data.dates)
    raw = np.asarray(data.series[OIL_KEY], dtype=np.float64).reshape(T)
    pf = _ffill_positive(raw)
    o = np.full(T, np.nan)
    if T > 1:
        o[1:] = pf[1:] / pf[:-1] - 1.0
    return o


def _spy_returns(data):
    """Row t: SPY close-to-close total return of session t-1."""
    T = len(data.dates)
    j = data.col("SPY")
    rco = np.asarray(data.ret_co, dtype=np.float64)[:, j]
    roc = np.asarray(data.ret_oc, dtype=np.float64)[:, j]
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    r = np.where(np.asarray(data.listed)[:, j], r, np.nan)
    s = np.full(T, np.nan)
    if T > 1:
        s[1:] = r[:-1]
    return s


def compute_z(data, params):
    """Unclipped z_t (NaN where not fully defined), plus beta_t and shock_t for tests."""
    L = int(params["lookback_days"])
    o = _oil_returns(data)
    s = _spy_returns(data)
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
    z = np.where(np.isfinite(z), z, np.nan)
    return z, beta, shock


def weekly_decision_rows(dates):
    """Last session of each calendar week, point-in-time approximation.

    Row t is a decision row if it is a Friday (weekday arithmetic: the next weekday lies in the next week).
    If a week has no Friday session (exchange holiday), the week's last session is only known at the first
    session of the next week, so the decision is taken there (one session late).
    """
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    wd = (d + 3) % 7          # Monday = 0
    wk = (d + 3) // 7         # Monday-start week number
    dec = wd == 4
    if d.shape[0] > 1:
        late = (wk[1:] > wk[:-1]) & (wd[:-1] != 4)
        dec[1:] = dec[1:] | late
    return dec


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    tilt = float(params["tilt_per_z"])
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")

    z, _, _ = compute_z(data, params)
    zc = np.clip(z, -Z_CLIP, Z_CLIP)
    dec = weekly_decision_rows(data.dates)

    ief_priced = np.isfinite(np.asarray(data.close, dtype=np.float64)[:, j_ief])
    ready = dec & ief_priced & np.isfinite(zc)
    started = np.maximum.accumulate(ready) if T else ready

    z_use = np.where(np.isfinite(zc), zc, 0.0)
    w_spy = np.clip(BASE_SPY - tilt * z_use, W_SPY_MIN, W_SPY_MAX)

    w = np.full((T, N), np.nan)
    w[dec, :] = 0.0
    live = dec & started
    w[live, j_spy] = w_spy[live]
    w[live, j_ief] = 1.0 - w_spy[live]
    return w
