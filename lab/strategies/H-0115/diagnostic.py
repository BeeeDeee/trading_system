"""H-0115 mechanism event: at a weekly decision date a high-vol-half stock enters the long leg (+1) or the short
leg (-1) of the ridge book when it was not in that leg at the previous weekly decision.

Same pipeline as strategy.py (copied: diagnostic.py is a separate file).
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
VOL_WIN = 63
MIN_NAMES = 20
MODEL = "ridge"


def rolling_sum(x, n):
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
    x = np.asarray(x, dtype=np.float64)
    if k == 0:
        return x.copy()
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def week_end_days(dates):
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7
    return weekday == 4


def high_vol_half(vol, eligible):
    elig = eligible & np.isfinite(vol)
    volm = np.where(elig, vol, np.nan)
    med = np.full(vol.shape[0], np.nan)
    rows = np.nonzero(elig.any(axis=1))[0]
    if rows.size:
        med[rows] = np.nanmedian(volm[rows], axis=1)
    with np.errstate(invalid="ignore"):
        return elig & (volm >= med[:, None])


def raw_features(ret_co, ret_oc, dollar_volume, listed):
    ret_co = np.asarray(ret_co, dtype=np.float64)
    ret_oc = np.asarray(ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    f_oc = rolling_sum(ret_oc, 5)
    f_co = rolling_sum(ret_co, 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(r)
    f_wk = lag(np.expm1(rolling_sum(lr, 5)), 5)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    m_recent = rolling_sum(dv, 5) / 5.0
    m_prior = lag(rolling_sum(dv, VOL_WIN) / VOL_WIN, 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_dv = np.log(m_recent / m_prior)
    f_dv[~np.isfinite(f_dv)] = np.nan
    s1 = rolling_sum(r, VOL_WIN)
    s2 = rolling_sum(r * r, VOL_WIN)
    var = s2 / VOL_WIN - (s1 / VOL_WIN) ** 2
    vol = np.sqrt(np.maximum(var, 0.0))
    full_hist = rolling_sum(np.asarray(listed, dtype=np.float64), VOL_WIN + 1) >= VOL_WIN + 1
    vol[~full_hist] = np.nan
    return f_oc, f_co, f_wk, f_dv, vol


def model_weights(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0115 needs the point-in-time LIQ-300 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    feats = raw_features(data.ret_co, data.ret_oc, data.dollar_volume, data.listed)
    hv = high_vol_half(feats[4], uni)
    X = stack(*[cs_zscore(f, hv) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), hv)
    score = walk_forward(X, y, horizon=HORIZON, model=MODEL, params={"alpha": float(params["alpha"])},
                         retrain_every=RETRAIN_EVERY, universe=hv)
    top = float(params["top"])
    w = np.asarray(score_weights(score, hv, top=top, bottom=top, min_names=MIN_NAMES), dtype=np.float64)
    w = np.where(np.isfinite(w) & hv, w, 0.0)
    return w, uni


def leg_entries(w, dec):
    """Entries into the long / short leg at decision rows relative to the previous decision row."""
    long_ = w > 0
    short = w < 0
    T, N = w.shape
    new_long = np.zeros((T, N), dtype=bool)
    new_short = np.zeros((T, N), dtype=bool)
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    for t in np.nonzero(dec)[0]:
        new_long[t] = long_[t] & ~prev_long
        new_short[t] = short[t] & ~prev_short
        prev_long = long_[t]
        prev_short = short[t]
    return new_long, new_short


def events(data, params):
    w, uni = model_weights(data, params)
    dec = week_end_days(data.dates)
    new_long, new_short = leg_entries(w, dec)
    ok = uni & np.asarray(data.tradable, dtype=bool)
    mask = (new_long | new_short) & ok
    side = np.where(new_short, -1.0, 1.0)
    return mask, side
