"""H-0100: LIQ-500 long-only core plus a 5-session tilt into top gbm-scored 8-K Item 2.02 releases.

Features (all columns, computed after the close of t from rows <= t, z-scored across the members of t):
  gap_z     = ret_co[t] / sigma63
  session_z = ret_oc[t] / sigma63
  vol_shock = log(dollar_volume[t] / median dollar_volume over t-63..t-1)
  runup_z   = total return over t-21..t-1 / (sigma63 * sqrt(21))
  log sigma63
  log median dollar volume over t-63..t-1
sigma63 = sample std of daily close-to-close total returns over t-63..t-1.
Model: walk_forward gbm, trained and scored on release rows only (features NaN elsewhere), expanding
window, refit every 63 rows, label = 5-session open-to-open forward return.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top_q": 0.2}

HORIZON = 5                 # ml.horizon_days
RETRAIN_EVERY = 63          # ml.retrain_every_days
WIN = 63                    # sigma63 / median dollar volume window (t-63..t-1)
RUNUP = 21                  # run-up window (t-21..t-1)
QWIN = 252                  # trailing sessions of release scores for the entry threshold
MIN_Q_ROWS = 100            # minimum scored release rows in that window
SLOT = 0.05                 # sleeve slot weight
SLEEVE_CAP = 0.50           # cap on the sleeve gross
HOLD = 5                    # sessions a slot is held
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}


def _lagged_rolling_median(x, n):
    """out[t] = median of x[t-n..t-1]; NaN if the window is incomplete or holds a NaN."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T <= n:
        return out
    xf = np.asarray(x, dtype=np.float64)
    win = np.lib.stride_tricks.sliding_window_view(xf, n, axis=0)   # win[i] = x[i:i+n] (N, n)
    chunk = 32
    for s in range(0, T - n, chunk):
        e = min(s + chunk, T - n)
        out[s + n:e + n] = np.median(win[s:e], axis=2)
    return out


def _lagged_window_sum(v, ok, n):
    """Sum of v and count of ok over rows t-n..t-1 (exclusive of t)."""
    T, N = v.shape
    cs = np.zeros((T + 1, N))
    cs[1:] = np.cumsum(np.where(ok, v, 0.0), axis=0)
    cc = np.zeros((T + 1, N))
    cc[1:] = np.cumsum(ok.astype(np.float64), axis=0)
    s = np.full((T, N), np.nan)
    c = np.zeros((T, N))
    if T > n:
        # rows t-n..t-1 -> cs[t] - cs[t-n]
        s[n:] = cs[n:T] - cs[0:T - n]
        c[n:] = cc[n:T] - cc[0:T - n]
    return s, c


def features(data):
    """The six raw features (T, N) each, NaN where unknown; all columns, no membership used."""
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = ret_co.shape

    priced = listed & np.isfinite(close)
    # daily close-to-close total return of row t, valid when priced on t and t-1
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
    return out, med


def release_rows(data, uni):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (ea == 1.0) & uni


def model_scores(data, uni, rel, model, model_params):
    """Walk-forward scores on release rows only (NaN elsewhere)."""
    feats, med = features(data)
    X = stack(*[cs_zscore(f, uni) for f in feats])
    X = np.array(X, dtype=np.float32, copy=True)
    X[~rel] = np.nan                        # sample mask: only release rows are trained on and scored
    y = forward_return(data, HORIZON)
    score = walk_forward(X, y, horizon=HORIZON, model=model, params=model_params,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score = np.where(rel & np.isfinite(score), score, np.nan)
    return score, med


def entries(score, top_q):
    """Release row is an entry when its score >= the (1 - top_q) quantile of all scored release rows in
    rows t-251..t (needs >= 100 scored rows in that window)."""
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


def member_run_length(uni):
    T, N = uni.shape
    L = np.zeros((T, N), dtype=np.int64)
    prev = np.zeros(N, dtype=np.int64)
    for t in range(T):
        prev = np.where(uni[t], prev + 1, 0)
        L[t] = prev
    return L


def month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.reshape(T)
    m = np.zeros(T, dtype=bool)
    m[ps.astype(np.int64)] = True
    return m


def core_weights(uni, med, starts, L):
    """Members of the month-start row weighted by their 63-day median dollar volume, held as fixed
    fractions until the next month start; a stock that leaves the universe is dropped (no re-entry
    before the next refresh) and the remaining fixed fractions are rescaled to sum to 1 (gross 1.0)."""
    T, N = uni.shape
    core = np.zeros((T, N))
    c = None
    m = -1
    for t in range(T):
        if starts[t]:
            ok = uni[t] & np.isfinite(med[t])
            if ok.any():
                c = np.where(ok, med[t], 0.0)
                c = c / c.sum()
                m = t
            else:
                c = None
        if c is not None:
            x = c * (L[t] >= (t - m + 1))
            tot = x.sum()
            if tot > 0:
                core[t] = x / tot
    return core


def book(ent, core, L):
    T, N = ent.shape
    slots = np.zeros((T, N))
    for k in range(HOLD):
        ek = np.zeros((T, N), dtype=bool)
        if k == 0:
            ek[:] = ent
        elif T > k:
            ek[k:] = ent[:-k]
        slots += (ek & (L >= k + 1)).astype(np.float64)    # closed at the first non-member row
    s_raw = SLOT * slots.sum(axis=1)
    scale = np.where(s_raw > SLEEVE_CAP, SLEEVE_CAP / np.where(s_raw > 0, s_raw, 1.0), 1.0)
    sleeve = SLOT * slots * scale[:, None]
    S = s_raw * scale
    w = (1.0 - S)[:, None] * core + sleeve
    gross = np.abs(w).sum(axis=1)
    w = w / np.maximum(gross, 1.0)[:, None]
    return w


def target_weights(data, params):
    if data.universe is None:
        raise ValueError("H-0100 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(data.universe, dtype=bool)
    rel = release_rows(data, uni)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    score, med = model_scores(data, uni, rel, "gbm", gbm_params)
    ent = entries(score, params["top_q"])
    L = member_run_length(uni)
    starts = month_start_mask(data.dates)
    core = core_weights(uni, med, starts, L)
    return book(ent, core, L)
