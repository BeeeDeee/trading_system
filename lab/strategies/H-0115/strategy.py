"""H-0115: ridge on 10-day intraday / overnight / volume summaries in the high-volatility half of LIQ-300.

Weekly (last session of the calendar week, approximated by Friday), dollar-neutral, long the top `top`
fraction of the ridge score and short the bottom `top` fraction, 0.5 gross per leg.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"alpha": 10, "top": 0.2}

HORIZON = 5          # card ml.horizon_days: label = open t+1 -> open t+1+5
RETRAIN_EVERY = 63   # card ml.retrain_every_days
VOL_WIN = 63         # 63-session realized volatility
MIN_NAMES = 20       # fewer scored names on a decision date -> flat book
MODEL = "ridge"


# ----------------------------------------------------------------------------- helpers (point-in-time)

def rolling_sum(x, n):
    """Trailing sum over rows t-n+1..t; NaN if the window is incomplete or holds a non-finite value."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if T < n:
        return out
    bad = ~np.isfinite(x)
    c = np.cumsum(np.where(bad, 0.0, x), axis=0)
    b = np.cumsum(bad.astype(np.int64), axis=0)
    s = c[n - 1:].copy()
    s[1:] -= c[:-n]
    k = b[n - 1:].copy()
    k[1:] -= b[:-n]
    s[k > 0] = np.nan
    out[n - 1:] = s
    return out


def lag(x, k):
    """Value of row t-k at row t (k >= 0); NaN for the first k rows."""
    x = np.asarray(x, dtype=np.float64)
    if k == 0:
        return x.copy()
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def week_end_days(dates):
    """True on rows whose date is a Friday (last weekday of the calendar week; exchange holidays ignored)."""
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7          # 1970-01-01 was a Thursday -> Monday = 0
    return weekday == 4


def high_vol_half(vol, eligible):
    """Members with a valid vol whose vol is at or above the cross-sectional median of those members."""
    elig = eligible & np.isfinite(vol)
    volm = np.where(elig, vol, np.nan)
    med = np.full(vol.shape[0], np.nan)
    rows = np.nonzero(elig.any(axis=1))[0]
    if rows.size:
        med[rows] = np.nanmedian(volm[rows], axis=1)
    with np.errstate(invalid="ignore"):
        return elig & (volm >= med[:, None])


def raw_features(ret_co, ret_oc, dollar_volume, listed):
    """The card's five features (unscaled) and the 63-session vol, all at the close of t."""
    ret_co = np.asarray(ret_co, dtype=np.float64)
    ret_oc = np.asarray(ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0                       # close-to-close total return
    # (1) intraday sum t-4..t, (2) overnight sum t-4..t
    f_oc = rolling_sum(ret_oc, 5)
    f_co = rolling_sum(ret_co, 5)
    # (3) total return over t-9..t-5
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(r)
    f_wk = lag(np.expm1(rolling_sum(lr, 5)), 5)
    # (4) log(mean dv t-4..t / mean dv t-67..t-5)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    m_recent = rolling_sum(dv, 5) / 5.0
    m_prior = lag(rolling_sum(dv, VOL_WIN) / VOL_WIN, 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_dv = np.log(m_recent / m_prior)
    f_dv[~np.isfinite(f_dv)] = np.nan
    # (5) 63-session realized vol of daily total returns; needs 63 real close-to-close returns,
    #     i.e. listed on each of the 64 rows t-63..t
    s1 = rolling_sum(r, VOL_WIN)
    s2 = rolling_sum(r * r, VOL_WIN)
    var = s2 / VOL_WIN - (s1 / VOL_WIN) ** 2
    vol = np.sqrt(np.maximum(var, 0.0))
    full_hist = rolling_sum(np.asarray(listed, dtype=np.float64), VOL_WIN + 1) >= VOL_WIN + 1
    vol[~full_hist] = np.nan
    return f_oc, f_co, f_wk, f_dv, vol


def scoring_set(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0115 needs the point-in-time LIQ-300 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    f_oc, f_co, f_wk, f_dv, vol = raw_features(data.ret_co, data.ret_oc, data.dollar_volume, data.listed)
    hv = high_vol_half(vol, uni)
    return uni, hv, (f_oc, f_co, f_wk, f_dv, vol)


def model_weights(data, params):
    """Ridge score of the high-vol half and the daily dollar-neutral book (before restricting to decision days)."""
    uni, hv, feats = scoring_set(data)
    X = stack(*[cs_zscore(f, hv) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), hv)
    score = walk_forward(X, y, horizon=HORIZON, model=MODEL, params={"alpha": float(params["alpha"])},
                         retrain_every=RETRAIN_EVERY, universe=hv)
    top = float(params["top"])
    w = np.asarray(score_weights(score, hv, top=top, bottom=top, min_names=MIN_NAMES), dtype=np.float64)
    w = np.where(np.isfinite(w) & hv, w, 0.0)                      # non-members / unscored -> 0 (sell)
    return w, hv


def target_weights(data, params):
    w, _ = model_weights(data, params)
    dec = week_end_days(data.dates)
    out = np.full(w.shape, np.nan)
    out[dec] = w[dec]
    return out
