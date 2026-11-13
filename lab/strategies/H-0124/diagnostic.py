"""H-0124 mechanism test: a perp enters the long leg (+1) or the short leg (-1) of the gbm book on day t,
having not been in that leg on day t-1. Same features, model and leg rule as strategy.py (copied)."""
import numpy as np
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

UNIVERSE_N = 20
VOLUME_WINDOW = 30
HORIZON = 5
RETRAIN_EVERY = 63
MIN_SCORED = 10
MIN_LEG = 2
MIN_HISTORY = 60
BTC = "BTCUSDT.P"
VIX = "fred_macro.VIXCLS"
GBM_PARAMS = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 400,
              "subsample": 0.8, "colsample": 1.0}


def rolling_sum(x, n):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if n <= 0 or T < n:
        return out
    c = np.cumsum(x, axis=0)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), c], axis=0)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def top_n_universe(dollar_volume, listed, n, window):
    dv = np.asarray(dollar_volume, dtype=np.float64)
    ok = np.isfinite(dv)
    s = rolling_sum(np.where(ok, dv, 0.0), window)
    cnt = rolling_sum(ok.astype(np.float64), window)
    full = np.isfinite(cnt) & (cnt > window - 0.5) & np.asarray(listed, dtype=bool)
    mean = np.where(full, s / window, -np.inf)
    order = np.argsort(-mean, axis=1, kind="stable")
    rank = np.empty_like(order)
    T, N = mean.shape
    rank[np.arange(T)[:, None], order] = np.arange(N)[None, :]
    return full & (rank < n)


def daily_returns(ret_co, ret_oc, listed):
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    lst = np.asarray(listed, dtype=bool)
    valid = np.zeros(lst.shape, dtype=bool)
    valid[1:] = lst[1:] & lst[:-1]
    valid &= np.isfinite(r) & (r > -1.0)
    return np.where(valid, r, 0.0), valid


def coin_features(r, valid, btc_col):
    T, N = r.shape
    lr = np.log1p(r)
    v = valid.astype(np.float64)
    n60 = rolling_sum(v, MIN_HISTORY)
    elig = np.isfinite(n60) & (n60 > MIN_HISTORY - 0.5)

    ret21 = np.expm1(rolling_sum(lr, 21))
    ret5 = np.expm1(rolling_sum(lr, 5))
    s1 = rolling_sum(lr, 30)
    s2 = rolling_sum(lr * lr, 30)
    var30 = (s2 - s1 * s1 / 30.0) / 29.0
    vol30 = np.sqrt(np.maximum(var30, 0.0))

    rb = r[:, btc_col]
    btc_ok = elig[:, btc_col]
    rb2 = np.broadcast_to(rb[:, None], r.shape)
    n = float(MIN_HISTORY)
    sj = rolling_sum(r, MIN_HISTORY)
    sb = rolling_sum(rb[:, None], MIN_HISTORY)
    sbb = rolling_sum((rb * rb)[:, None], MIN_HISTORY)
    sjb = rolling_sum(r * rb2, MIN_HISTORY)
    cov = sjb - sj * sb / n
    varb = sbb - sb * sb / n
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(varb > 0.0, cov / varb, np.nan)
    beta[:, btc_col] = 1.0
    beta = np.where(btc_ok[:, None], beta, np.nan)

    btc_ret21 = np.where(btc_ok, ret21[:, btc_col], np.nan)
    elig = elig & btc_ok[:, None]
    return ret21, ret5, vol30, beta, btc_ret21, elig


def model_scores(data, params):
    T, N = len(data.dates), len(data.instruments)
    listed = np.asarray(data.listed, dtype=bool)
    if data.universe is not None:
        member = np.asarray(data.universe, dtype=bool)
    else:
        member = top_n_universe(data.dollar_volume, listed, UNIVERSE_N, VOLUME_WINDOW)
    member = member & listed

    btc = data.col(BTC)
    r, valid = daily_returns(data.ret_co, data.ret_oc, listed)
    ret21, ret5, vol30, beta, btc_ret21, elig = coin_features(r, valid, btc)

    vix = np.asarray(data.series[VIX], dtype=np.float64).reshape(-1)
    if vix.shape[0] != T:
        raise ValueError("VIXCLS series is not aligned to the view")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vix = np.where(vix > 0.0, np.log(vix), np.nan)

    use = member & elig & np.isfinite(log_vix)[:, None] & np.isfinite(btc_ret21)[:, None]

    def z(x):
        x = np.where(use & np.isfinite(x), x, np.nan)
        out = np.asarray(cs_zscore(x, member), dtype=np.float64)
        return np.where(use, out, np.nan)

    f_vix = np.where(use, log_vix[:, None], np.nan)
    f_btc = np.where(use, btc_ret21[:, None], np.nan)
    X = stack(z(ret21), z(ret5), z(vol30), z(beta), f_vix, f_btc)

    gbm = dict(GBM_PARAMS)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, forward_return(data, HORIZON), horizon=HORIZON, model="gbm",
                         params=gbm, retrain_every=RETRAIN_EVERY, universe=member)
    score = np.asarray(score, dtype=np.float64)
    score = np.where(use & np.isfinite(score), score, np.nan)
    return score, member


def legs(score, top):
    T, N = score.shape
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)
    k_arr = np.zeros(T, dtype=np.int64)
    ok = np.isfinite(score)
    m_arr = ok.sum(axis=1)
    s = np.where(ok, score, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")
    for t in range(T):
        m = int(m_arr[t])
        if m < MIN_SCORED:
            continue
        k = max(MIN_LEG, int(np.floor(top * m + 0.5 + 1e-9)))
        k = min(k, m // 2)
        long_m[t, order[t, :k]] = True
        short_m[t, order[t, m - k:m]] = True
        k_arr[t] = k
    return long_m, short_m, k_arr


def events(data, params):
    score, member = model_scores(data, params)
    long_m, short_m, _ = legs(score, float(params["top"]))
    prev_long = np.zeros_like(long_m)
    prev_short = np.zeros_like(short_m)
    prev_long[1:] = long_m[:-1]
    prev_short[1:] = short_m[:-1]
    enter_long = long_m & ~prev_long
    enter_short = short_m & ~prev_short
    ok = member & np.asarray(data.tradable, dtype=bool)
    mask = (enter_long | enter_short) & ok
    side = np.where(enter_short, -1.0, 1.0)
    return mask, side
