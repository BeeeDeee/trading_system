"""H-0100 mechanism test: release rows where exactly one of gbm / ridge (same six features, rows and
schedule) puts the stock in its own top_q of trailing-252-session release scores.
side +1 = only the gbm (nonlinear pick), -1 = only the ridge (linear pick).
The feature, walk-forward and entry code is a copy of strategy.py."""
import numpy as np
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
WIN = 63
RUNUP = 21
QWIN = 252
MIN_Q_ROWS = 100
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}
RIDGE_PARAMS = {"alpha": 1.0}


def _lagged_rolling_median(x, n):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T <= n:
        return out
    xf = np.asarray(x, dtype=np.float64)
    win = np.lib.stride_tricks.sliding_window_view(xf, n, axis=0)
    chunk = 32
    for s in range(0, T - n, chunk):
        e = min(s + chunk, T - n)
        out[s + n:e + n] = np.median(win[s:e], axis=2)
    return out


def _lagged_window_sum(v, ok, n):
    T, N = v.shape
    cs = np.zeros((T + 1, N))
    cs[1:] = np.cumsum(np.where(ok, v, 0.0), axis=0)
    cc = np.zeros((T + 1, N))
    cc[1:] = np.cumsum(ok.astype(np.float64), axis=0)
    s = np.full((T, N), np.nan)
    c = np.zeros((T, N))
    if T > n:
        s[n:] = cs[n:T] - cs[0:T - n]
        c[n:] = cc[n:T] - cc[0:T - n]
    return s, c


def features(data):
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = ret_co.shape

    priced = listed & np.isfinite(close)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    rvalid = np.zeros((T, N), dtype=bool)
    rvalid[1:] = priced[1:] & priced[:-1] & np.isfinite(r[1:])
    r = np.where(rvalid, r, 0.0)

    s1, c1 = _lagged_window_sum(r, rvalid, WIN)
    s2, _ = _lagged_window_sum(r * r, rvalid, WIN)
    full = c1 >= WIN
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / WIN
        var = (s2 - WIN * mean * mean) / (WIN - 1)
        sigma = np.where(full & (var > 0), np.sqrt(np.maximum(var, 0.0)), np.nan)

        lvalid = rvalid & (r > -1.0)
        lr = np.where(lvalid, np.log1p(np.where(lvalid, r, 0.0)), 0.0)
        sr, cr = _lagged_window_sum(lr, lvalid, RUNUP)
        runup = np.where(cr >= RUNUP, np.expm1(sr), np.nan)

        med = _lagged_rolling_median(dv, WIN)
        med = np.where(np.isfinite(med) & (med > 0), med, np.nan)
        dv_t = np.where(priced & np.isfinite(dv) & (dv > 0), dv, np.nan)
        co_t = np.where(priced & np.isfinite(ret_co), ret_co, np.nan)
        oc_t = np.where(priced & np.isfinite(ret_oc), ret_oc, np.nan)

        gap_z = co_t / sigma
        session_z = oc_t / sigma
        vol_shock = np.log(dv_t / med)
        runup_z = runup / (sigma * np.sqrt(RUNUP))
        log_sigma = np.log(sigma)
        log_med = np.log(med)
    out = []
    for f in (gap_z, session_z, vol_shock, runup_z, log_sigma, log_med):
        out.append(np.where(np.isfinite(f), f, np.nan))
    return out


def entries(score, top_q):
    T, N = score.shape
    ent = np.zeros((T, N), dtype=bool)
    fin = np.isfinite(score)
    counts = fin.sum(axis=1)
    cum = np.concatenate([[0], np.cumsum(counts)])
    vals = [score[t][fin[t]] for t in range(T)]
    for t in range(T):
        if counts[t] == 0:
            continue
        lo = max(0, t - QWIN + 1)
        if cum[t + 1] - cum[lo] < MIN_Q_ROWS:
            continue
        pool = np.concatenate(vals[lo:t + 1])
        thr = np.quantile(pool, 1.0 - float(top_q))
        ent[t] = fin[t] & (score[t] >= thr)
    return ent


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0100 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(data.universe, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        rel = (ea == 1.0) & uni
    X = stack(*[cs_zscore(f, uni) for f in features(data)])
    X = np.array(X, dtype=np.float32, copy=True)
    X[~rel] = np.nan
    y = forward_return(data, HORIZON)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    ents = []
    for model, mp in (("gbm", gbm_params), ("ridge", RIDGE_PARAMS)):
        s = walk_forward(X, y, horizon=HORIZON, model=model, params=mp,
                         retrain_every=RETRAIN_EVERY, universe=uni)
        s = np.asarray(s, dtype=np.float64)
        s = np.where(rel & np.isfinite(s), s, np.nan)
        ents.append(entries(s, params["top_q"]))
    g, r = ents
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (g ^ r) & rel & tradable
    side = np.where(g & ~r, 1.0, np.where(r & ~g, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side
