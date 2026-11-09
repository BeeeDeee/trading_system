"""H-0112 mechanism event: on a flow day, members in the top (bottom) fifth of the gbm score but not in the
top (bottom) fifth of the ridge score; both walk-forward models on the same features, rows, label and schedule.
The feature / walk_forward pipeline is a copy of strategy.py.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

Z_WINDOW = 63
DV_WINDOW = 63
DV_MIN_OBS = 32
BETA_WINDOW = 252
BETA_MIN_PAIRS = 126
PRIOR_DAYS = 5
HORIZON = 5
RETRAIN_EVERY = 126
TRAIN_SIGMA = 1.0
MIN_SCORED = 100


def _window_sum(a, w):
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if w < a.shape[0]:
        out[w:] -= c[:-w]
    return out


def _prior(a, fill):
    out = np.empty_like(a)
    out[0] = fill
    out[1:] = a[:-1]
    return out


def _shift(a, k, fill):
    out = np.full_like(a, fill)
    if k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(np.asarray(data.listed, dtype=bool), r, np.nan)


def market_series(data, r, uni):
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    ok = uni & np.isfinite(mc) & (mc > 0) & np.isfinite(r)
    num = np.where(ok, mc * np.where(ok, r, 0.0), 0.0).sum(axis=1)
    den = np.where(ok, mc, 0.0).sum(axis=1)
    m = np.full(den.shape, np.nan)
    np.divide(num, den, out=m, where=den > 0)

    fin = np.isfinite(m)
    m0 = np.where(fin, m, 0.0)
    n = _prior(_window_sum(fin.astype(np.float64), Z_WINDOW), 0.0)
    s1 = _prior(_window_sum(m0, Z_WINDOW), 0.0)
    s2 = _prior(_window_sum(m0 * m0, Z_WINDOW), 0.0)
    full = n >= Z_WINDOW
    nn = np.where(full, n, 2.0)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    z = np.full(m.shape, np.nan)
    np.divide(m, sd, out=z, where=full & (sd > 0) & fin)

    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    okv = uni & np.isfinite(dv)
    S = np.where(okv, dv, 0.0).sum(axis=1)
    has = okv.any(axis=1) & (S > 0)
    S0 = np.where(has, S, 0.0)
    ns = _prior(_window_sum(has.astype(np.float64), Z_WINDOW), 0.0)
    ss = _prior(_window_sum(S0, Z_WINDOW), 0.0)
    vshock = np.full(S.shape, np.nan)
    good = has & (ns >= Z_WINDOW) & (ss > 0)
    ratio = np.ones(S.shape)
    np.divide(S0 * np.where(ns > 0, ns, 1.0), np.where(ss > 0, ss, 1.0), out=ratio, where=good)
    vshock[good] = np.log(ratio[good])
    return m, z, vshock


def inverse_turnover(data):
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    fin = np.isfinite(dv)
    n = _window_sum(fin.astype(np.float64), DV_WINDOW)
    s = _window_sum(np.where(fin, dv, 0.0), DV_WINDOW)
    mean = np.full(dv.shape, np.nan)
    np.divide(s, n, out=mean, where=n >= DV_MIN_OBS)
    ok = np.isfinite(mc) & (mc > 0) & np.isfinite(mean) & (mean > 0)
    out = np.full(dv.shape, np.nan)
    out[ok] = np.log(mc[ok] / mean[ok])
    return out


def rolling_beta(r, m, chunk=256):
    out = np.full(r.shape, np.nan)
    for j in range(0, r.shape[1], chunk):
        out[:, j:j + chunk] = _beta_block(r[:, j:j + chunk], m)
    return out


def _beta_block(r, m):
    valid = np.isfinite(r) & np.isfinite(m)[:, None]
    x = np.where(valid, np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
    y = np.where(valid, r, 0.0)
    n = _window_sum(valid.astype(np.float64), BETA_WINDOW)
    sx = _window_sum(x, BETA_WINDOW)
    sxx = _window_sum(x * x, BETA_WINDOW)
    sxy = _window_sum(x * y, BETA_WINDOW)
    del x
    sy = _window_sum(y, BETA_WINDOW)
    del y
    enough = n >= BETA_MIN_PAIRS
    nn = np.where(enough, n, 1.0)
    cov = sxy - sx * sy / nn
    var = sxx - sx * sx / nn
    beta = np.full(r.shape, np.nan)
    np.divide(cov, var, out=beta, where=enough & (var > 0))
    return beta


def prior_return(r):
    p = np.ones_like(r)
    for k in range(1, PRIOR_DAYS + 1):
        p = p * (1.0 + _shift(r, k, np.nan))
    return p - 1.0


def build_features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0112 needs the sp500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    r = daily_returns(data)
    m, z, vshock = market_series(data, r, uni)
    T, N = r.shape
    f1 = cs_zscore(r.astype(np.float32), uni)
    f2 = cs_zscore(inverse_turnover(data).astype(np.float32), uni)
    f3 = cs_zscore(rolling_beta(r, m).astype(np.float32), uni)
    f4 = cs_zscore(prior_return(r).astype(np.float32), uni)
    f5 = np.broadcast_to(z.astype(np.float32)[:, None], (T, N)).copy()
    f6 = np.broadcast_to(vshock.astype(np.float32)[:, None], (T, N)).copy()
    X = stack(f1, f2, f3, f4, f5, f6)
    return X, z, uni


def train_rows(z):
    return np.isfinite(z) & (np.abs(np.where(np.isfinite(z), z, 0.0)) >= TRAIN_SIGMA)


def gbm_params(params):
    return {"n_estimators": int(params["n_estimators"]), "num_leaves": 15,
            "learning_rate": 0.05, "min_child_samples": 200}


def flow_days(z, flow_sigma):
    return np.isfinite(z) & (np.abs(np.where(np.isfinite(z), z, 0.0)) >= float(flow_sigma))


def disagreement(gbm_score, ridge_score, uni, flow, tradable, top):
    """+1: gbm top fraction and not ridge top fraction; -1: gbm bottom and not ridge bottom."""
    wg = np.nan_to_num(np.asarray(score_weights(gbm_score, uni, top=float(top), bottom=float(top),
                                                min_names=MIN_SCORED), dtype=np.float64), nan=0.0)
    wr = np.nan_to_num(np.asarray(score_weights(ridge_score, uni, top=float(top), bottom=float(top),
                                                min_names=MIN_SCORED), dtype=np.float64), nan=0.0)
    up = (wg > 0) & ~(wr > 0)
    dn = (wg < 0) & ~(wr < 0)
    base = flow[:, None] & uni & tradable
    mask = base & (up | dn)
    side = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side


def events(data, params):
    X, z, uni = build_features(data)
    tr = train_rows(z)
    common = dict(horizon=HORIZON, retrain_every=RETRAIN_EVERY, universe=uni)
    gbm = walk_forward(
        X, cs_rank(np.where(tr[:, None], forward_return(data, HORIZON), np.nan), uni),
        model="gbm", params=gbm_params(params), **common,
    )
    ridge = walk_forward(
        X, cs_rank(np.where(tr[:, None], forward_return(data, HORIZON), np.nan), uni),
        model="ridge", params={}, **common,
    )
    flow = flow_days(z, params["flow_sigma"])
    tradable = np.asarray(data.tradable, dtype=bool)
    return disagreement(gbm, ridge, uni, flow, tradable, params["top"])
