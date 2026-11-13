"""H-0130 mechanism event: gbm-versus-ridge disagreement on TradFi shift dates.

Copy of the strategy's feature pipeline (diagnostic.py is a separate file). On a decision date where
|dvix5| >= 0.15 or |dry5| >= 0.15 pp, a coin in the top `top` fraction of the gbm score but not of the
ridge score (same six features, label, rows and refit schedule) is an event with side +1; a coin in the
ridge's top fraction but not the gbm's is an event with side -1. Only universe members that are tradable.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

FACTOR = "BTCUSDT"
VIX_KEY = "fred_macro.VIXCLS"
RY_KEY = "fred_macro.DFII10"

HORIZON = 5
RETRAIN_EVERY = 63
BETA_WINDOW = 60
BETA_MIN_PAIRS = 45
LIQ_WINDOW = 30
SHIFT_OBS = 5
MIN_NAMES = 10
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200, "subsample": 0.8}
RIDGE_PARAMS = {"alpha": 1.0}
DVIX_SHOCK = 0.15             # |5-observation log change of VIX|
DRY_SHOCK = 0.15              # |5-observation change of DFII10|, percentage points


def daily_returns(ret_co, ret_oc, close, listed):
    ret_co = np.asarray(ret_co, dtype=float)
    ret_oc = np.asarray(ret_oc, dtype=float)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    ok = np.isfinite(np.asarray(close, dtype=float)) & np.asarray(listed, dtype=bool) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    return np.where(valid, r, np.nan)


def _rolling_sum(x, w):
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if x.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def beta_ivol(r, f, window=BETA_WINDOW, min_pairs=BETA_MIN_PAIRS):
    r = np.asarray(r, dtype=float)
    x = np.broadcast_to(np.asarray(f, dtype=float)[:, None], r.shape)
    pair = np.isfinite(r) & np.isfinite(x)
    xm = np.where(pair, x, 0.0)
    ym = np.where(pair, r, 0.0)
    n = _rolling_sum(pair.astype(float), window)
    sx = _rolling_sum(xm, window)
    sy = _rolling_sum(ym, window)
    sxx = _rolling_sum(xm * xm, window)
    sxy = _rolling_sum(xm * ym, window)
    syy = _rolling_sum(ym * ym, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / n
        sxy_c = sxy - sx * sy / n
        syy_c = syy - sy * sy / n
        beta = sxy_c / sxx_c
        ssr = np.maximum(syy_c - beta * sxy_c, 0.0)
        ivol = np.sqrt(ssr / (n - 2.0))
    ok = (n >= min_pairs) & (sxx_c > 1e-18)
    return np.where(ok, beta, np.nan), np.where(ok, ivol, np.nan)


def liq30(volume, window=LIQ_WINDOW, chunk=256):
    v = np.asarray(volume, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    T = v.shape[0]
    pad = np.full((window - 1,) + v.shape[1:], np.nan)
    vp = np.concatenate([pad, v], axis=0)
    med = np.full(v.shape, np.nan)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        win = np.lib.stride_tricks.sliding_window_view(vp[s:e + window - 1], window, axis=0)
        with np.errstate(all="ignore"):
            allnan = np.all(np.isnan(win), axis=-1)
            safe = np.where(allnan[..., None], 0.0, win)
            m = np.nanmedian(safe, axis=-1)
        med[s:e] = np.where(allnan, np.nan, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(med > 0, np.log(np.where(med > 0, med, 1.0)), np.nan)
    return out


def _new_value(s):
    fin = np.isfinite(s)
    ch = fin.copy()
    ch[1:] = fin[1:] & (~fin[:-1] | (s[1:] != s[:-1]))
    return ch


def tradfi_features(vix, ry, shift=SHIFT_OBS):
    vix = np.asarray(vix, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    T = vix.shape[0]
    pub = _new_value(vix) | _new_value(ry)
    rows = np.flatnonzero(pub)
    k = np.cumsum(pub) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lvix = np.where(vix > 0, np.log(np.where(vix > 0, vix, 1.0)), np.nan)

    def lagged(s):
        out = np.full(T, np.nan)
        has = k >= shift
        out[has] = s[rows[k[has] - shift]]
        return out

    return lvix - lagged(lvix), ry - lagged(ry), lvix


def _series(data, key, T):
    s = np.asarray(data.series[key], dtype=float).reshape(-1)
    if s.shape[0] != T:
        raise ValueError(f"series {key} has {s.shape[0]} rows, view has {T}")
    return s


def build_features(data):
    if data.universe is None:
        raise ValueError("H-0130 needs the point-in-time crypto_top_n universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    T = len(data.dates)
    r = daily_returns(data.ret_co, data.ret_oc, data.close, data.listed)
    j = data.col(FACTOR)
    beta, ivol = beta_ivol(r, r[:, j])
    btc_ok = np.isfinite(beta[:, j])
    beta[:, j] = np.where(btc_ok, 1.0, np.nan)
    ivol[:, j] = np.where(btc_ok, 0.0, np.nan)
    liq = liq30(data.dollar_volume)

    vix = _series(data, VIX_KEY, T)
    ry = _series(data, RY_KEY, T)
    dvix5, dry5, lvix = tradfi_features(vix, ry)
    date_ok = np.isfinite(dvix5) & np.isfinite(dry5) & np.isfinite(lvix)

    scored = uni & np.isfinite(beta) & np.isfinite(ivol) & np.isfinite(liq) & date_ok[:, None]
    zb = cs_zscore(np.where(scored, beta, np.nan), scored)
    zi = cs_zscore(np.where(scored, ivol, np.nan), scored)
    zl = cs_zscore(np.where(scored, liq, np.nan), scored)

    def bcast(v):
        return np.where(scored, v[:, None], np.nan)

    X = stack(zb, zi, zl, bcast(dvix5), bcast(dry5), bcast(lvix))
    return X, scored, uni, dvix5, dry5


def events(data, params):
    X, scored, uni, dvix5, dry5 = build_features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    g = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp,
                     retrain_every=RETRAIN_EVERY, universe=uni)
    rd = walk_forward(X, y, horizon=HORIZON, model="ridge", params=dict(RIDGE_PARAMS),
                      retrain_every=RETRAIN_EVERY, universe=uni)
    top = float(params["top"])
    g_top = np.nan_to_num(score_weights(g, uni, top=top, bottom=0.0, min_names=MIN_NAMES), nan=0.0) > 0
    r_top = np.nan_to_num(score_weights(rd, uni, top=top, bottom=0.0, min_names=MIN_NAMES), nan=0.0) > 0
    with np.errstate(invalid="ignore"):
        shock = (np.abs(dvix5) >= DVIX_SHOCK) | (np.abs(dry5) >= DRY_SHOCK)
    shock = np.where(np.isfinite(dvix5) | np.isfinite(dry5), shock, False)
    tradable = np.asarray(data.tradable, dtype=bool)
    gbm_only = g_top & ~r_top
    ridge_only = r_top & ~g_top
    mask = shock[:, None] & uni & tradable & (gbm_only | ridge_only)
    side = np.where(gbm_only, 1.0, np.where(ridge_only, -1.0, 0.0))
    return mask, side
