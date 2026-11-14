"""H-0158 mechanism_test events: at a weekly decision date with |z_t| >= 1, a
scored LIQ-500 non-bank stock placed in the top `top` fraction of the gbm score
(side +1) or the bottom `top` fraction (side -1). Same pipeline as strategy.py
(copied, this file is separate)."""
import numpy as np

from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 126
SHOCK_SESSIONS = 5
OLS_WINDOW = 252
VOL_WINDOW = 63
VOL_MIN_DAYS = 50
WINSOR = 3.0
MIN_NAMES = 100
EVENT_Z = 1.0             # mechanism_test: |z_t| >= 1
GBM_FIXED = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample": 0.8,
}

SPREAD_HI = "fred_moody_yields_1d.DBAA"
SPREAD_LO = "fred_moody_yields_1d.DAAA"


def weekday(dates):
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7


def weekly_decision_days(dates):
    return weekday(dates) == 4


def market_return(idx, listed, members, k=SHOCK_SESSIONS):
    T, N = idx.shape
    m = np.full(T, np.nan)
    if T <= k:
        return m
    now = idx[k:].astype(np.float64)
    then = idx[:-k].astype(np.float64)
    ok = (np.isfinite(now) & np.isfinite(then) & (then > 0)
          & listed[k:] & listed[:-k] & members[k:])
    r = np.where(ok, now / np.where(ok, then, 1.0) - 1.0, 0.0)
    cnt = ok.sum(axis=1)
    tot = r.sum(axis=1)
    m[k:] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return m


def spread_change(s, k=SHOCK_SESSIONS):
    ds = np.full(len(s), np.nan)
    if len(s) > k:
        ds[k:] = s[k:] - s[:-k]
    return ds


def credit_shock_z(ds, m, window=OLS_WINDOW):
    ds = np.asarray(ds, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    T = len(ds)
    z = np.full(T, np.nan)
    valid = np.isfinite(ds) & np.isfinite(m)
    run = np.concatenate([[0], np.cumsum(valid)])
    for t in range(window, T):
        if not valid[t] or run[t] - run[t - window] < window:
            continue
        x = m[t - window:t]
        y = ds[t - window:t]
        xm = x.mean()
        ym = y.mean()
        xc = x - xm
        sxx = float(xc @ xc)
        if not sxx > 0:
            continue
        b = float(xc @ (y - ym)) / sxx
        a = ym - b * xm
        sd = float(np.std(y - a - b * x))
        if not sd > 0:
            continue
        z[t] = (ds[t] - a - b * m[t]) / sd
    return z


def universe_of(data):
    if data.universe is None:
        raise ValueError("H-0158 needs the LIQ-500 point-in-time universe")
    return np.asarray(data.universe, dtype=bool)


def shock_from_data(data):
    s = (np.asarray(data.series[SPREAD_HI], dtype=np.float64).reshape(-1)
         - np.asarray(data.series[SPREAD_LO], dtype=np.float64).reshape(-1))
    uni = universe_of(data)
    idx = total_return_index(data)
    m = market_return(idx, np.asarray(data.listed, dtype=bool), uni)
    return credit_shock_z(spread_change(s), m)


def bank_mask(deposits):
    d = np.asarray(deposits, dtype=np.float64)
    return np.isfinite(d) & (d > 0)


def balance_sheet_features(assets, debt, debtc, cash):
    assets = np.asarray(assets, dtype=np.float64)
    debt = np.asarray(debt, dtype=np.float64)
    debtc = np.asarray(debtc, dtype=np.float64)
    cash = np.asarray(cash, dtype=np.float64)
    ok = (np.isfinite(assets) & np.isfinite(debt) & np.isfinite(debtc)
          & np.isfinite(cash) & (assets > 0))
    a_safe = np.where(ok, assets, 1.0)
    net = np.where(ok, (debtc - cash) / a_safe, np.nan)
    nz = ok & (debt != 0)
    share = np.where(nz, debtc / np.where(nz, debt, 1.0), 0.0)
    share = np.where(ok, share, np.nan)
    lev = np.where(ok, debt / a_safe, np.nan)
    return net, share, lev


def daily_total_return(data):
    c = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(c)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(valid & np.isfinite(r), r, np.nan)


def realized_vol(r, window=VOL_WINDOW, min_days=VOL_MIN_DAYS):
    r = np.asarray(r, dtype=np.float64)
    T = r.shape[0]
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    lo = np.maximum(np.arange(T) + 1 - window, 0)
    c = np.zeros((T + 1,) + r.shape[1:])
    np.cumsum(ok, axis=0, out=c[1:])
    n = c[1:] - c[lo]
    np.cumsum(x, axis=0, out=c[1:])
    s1 = c[1:] - c[lo]
    np.cumsum(x * x, axis=0, out=c[1:])
    s2 = c[1:] - c[lo]
    del c, x
    good = n >= min_days
    nn = np.where(good, n, 2.0)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    return np.where(good, np.sqrt(np.maximum(var, 0.0)), np.nan)


def cs_feature(x, members):
    z = cs_zscore(np.asarray(x, dtype=np.float32), members)
    return np.clip(z, -WINSOR, WINSOR)


def eligible(data):
    return universe_of(data) & ~bank_mask(data.extras["sf1_arq_deposits"])


def model_score(data, params, members, z):
    ex = data.extras
    net, share, lev = balance_sheet_features(
        ex["sf1_arq_assets"], ex["sf1_arq_debt"], ex["sf1_arq_debtc"], ex["sf1_arq_cashneq"])
    f_net = cs_feature(net, members)
    del net
    f_share = cs_feature(share, members)
    del share
    f_lev = cs_feature(lev, members)
    del lev
    f_vol = cs_feature(realized_vol(daily_total_return(data)), members)
    T, N = f_net.shape
    f_z = np.broadcast_to(np.asarray(z, dtype=np.float32)[:, None], (T, N))
    X = stack(f_z, f_net, f_share, f_lev, f_vol)
    del f_net, f_share, f_lev, f_vol
    y = cs_rank(forward_return(data, HORIZON), members)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                        retrain_every=RETRAIN_EVERY, universe=members)


def leg_book(score, members, top):
    w = score_weights(score, members, top=float(top), bottom=float(top), min_names=MIN_NAMES)
    w = np.asarray(w, dtype=np.float64)
    return np.where(np.isfinite(w), w, 0.0)


def events(data, params):
    members = eligible(data)
    z = shock_from_data(data)
    score = model_score(data, params, members, z)
    w = leg_book(score, members, params["top"])
    az = np.abs(z)
    shock = np.isfinite(az) & (az >= EVENT_Z)
    day = weekly_decision_days(data.dates) & shock
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = day[:, None] & members & tradable & (w != 0)
    side = np.where(w > 0, 1.0, np.where(w < 0, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side
