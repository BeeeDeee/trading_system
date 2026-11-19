"""H-0178: macro-trend pass-through into LIQ-500 stock exposures, learned (gbm), two monthly 2-month sleeves.

Pipeline (all point-in-time, decided after the close of row t, traded at the open of t+1):
  1. FRED DGS10, DCOILWTICO, DEXUSEU, each value carried forward at most 5 rows (older -> NaN).
     Factors: Y = DGS10 (pp), O = log(max(WTI, 5)), D = -log(DEXUSEU).
  2. Decision rows = first trading row of each calendar month (period_starts "M").
  3. Stock features on decision rows only: OLS over the last 104 non-overlapping 5-row weeks ending at t of
     stock 5-row total return on [1, EW member market, dY5, dO5, dD5]; >= 78 valid weeks; features b_y, b_o, b_d,
     winsorized at the 1st/99th cross-sectional percentile of the date's members, then cs_zscore.
  4. Date-level features on decision rows: trend z_k = (F[t] - F[t-L]) / (std(dF over last 252 rows) * sqrt(L)), raw.
  5. Label = cs_rank of forward_return(42); walk_forward gbm, expanding window, refit every 63 rows.
  6. Book: two overlapping sleeves (this month's top fifth + last month's), 0.5 / 0.5; fallback EW members.
"""
import numpy as np
from lab.framework.api import only_on, period_starts, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "trend_lookback": 63}

# Fixed by the card (not in signal.params).
HORIZON = 42
RETRAIN_EVERY = 63
N_WEEKS = 104
MIN_WEEKS = 78
WEEK = 5
STALE_MAX = 5
VOL_WINDOW = 252
WTI_FLOOR = 5.0
TOP = 0.2
MIN_SCORED = 50
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 500, "subsample": 0.8,
             "colsample": 1.0}


def decision_rows(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        rows = np.flatnonzero(ps)
    else:
        rows = np.sort(ps.astype(np.int64))
    return ps, rows


def cap_stale(v, max_rows=STALE_MAX):
    """Set a value to NaN once it has been carried forward more than max_rows rows unchanged."""
    v = np.asarray(v, dtype=np.float64).reshape(-1).copy()
    T = v.shape[0]
    if T == 0:
        return v
    idx = np.arange(T)
    prev = np.empty(T)
    prev[0] = np.nan
    prev[1:] = v[:-1]
    changed = ~(v == prev)          # NaN comparisons count as a change
    changed[0] = True
    last = np.maximum.accumulate(np.where(changed, idx, 0))
    v[(idx - last) > max_rows] = np.nan
    return v


def factor_levels(data):
    T = len(data.dates)
    y = cap_stale(data.series["fred_macro.DGS10"])
    o = cap_stale(data.series["fred_macro.DCOILWTICO"])
    e = cap_stale(data.series["fred_macro.DEXUSEU"])
    for s in (y, o, e):
        if s.shape[0] != T:
            raise ValueError("FRED series length does not match the calendar")
    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.log(np.maximum(o, WTI_FLOOR))     # np.maximum keeps NaN
        d = -np.log(e)
    return y, o, d


def trend_z(F, rows, L, win=VOL_WINDOW):
    """(F[t] - F[t-L]) / (std of daily changes over the last win rows * sqrt(L)) on the given rows."""
    T = F.shape[0]
    out = np.full(T, np.nan)
    dF = np.full(T, np.nan)
    dF[1:] = F[1:] - F[:-1]
    for t in rows:
        if t < win or t < L:
            continue
        num = F[t] - F[t - L]
        if not np.isfinite(num):
            continue
        w = dF[t - win + 1:t + 1]
        w = w[np.isfinite(w)]
        if w.shape[0] < 2:
            continue
        sd = float(np.std(w, ddof=1))
        if sd > 0:
            out[t] = num / (sd * np.sqrt(L))
    return out


def stock_betas(idx, listed, uni, factors, rows, n_weeks=N_WEEKS, min_weeks=MIN_WEEKS, week=WEEK):
    """Partial betas (b_y, b_o, b_d) on the given rows for the members of that row; NaN elsewhere."""
    T, N = idx.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        r5 = np.full((T, N), np.nan)
        r5[week:] = idx[week:] / idx[:-week] - 1.0
    ok = np.zeros((T, N), dtype=bool)
    ok[week:] = listed[week:] & listed[:-week]
    r5[~(ok & np.isfinite(r5))] = np.nan
    mem_ok = uni & np.isfinite(r5)
    cnt = mem_ok.sum(axis=1)
    mkt = np.full(T, np.nan)
    tot = np.where(mem_ok, r5, 0.0).sum(axis=1)
    has = cnt > 0
    mkt[has] = tot[has] / cnt[has]
    f5 = []
    for F in factors:
        x = np.full(T, np.nan)
        x[week:] = F[week:] - F[:-week]
        f5.append(x)
    X_all = np.column_stack([np.ones(T), mkt] + f5)          # (T, 5)
    gvalid = np.all(np.isfinite(X_all), axis=1)
    out = [np.full((T, N), np.nan, dtype=np.float32) for _ in factors]
    for t in rows:
        cols = np.flatnonzero(uni[t])
        if cols.shape[0] == 0:
            continue
        s = t - week * np.arange(n_weeks)
        s = s[s >= week]
        if s.shape[0] < min_weeks:
            continue
        X = X_all[s]
        R = r5[s][:, cols]
        V = np.isfinite(R) & gvalid[s][:, None]
        n_ok = V.sum(axis=0)
        good = n_ok >= min_weeks
        if not good.any():
            continue
        X0 = np.where(gvalid[s][:, None], X, 0.0)
        Vf = V.astype(np.float64)
        R0 = np.where(V, R, 0.0)
        A = np.einsum("wn,wi,wj->nij", Vf, X0, X0)
        b = np.einsum("wn,wi->ni", R0, X0)
        coef = np.matmul(np.linalg.pinv(A[good]), b[good][:, :, None])[:, :, 0]
        gc = cols[good]
        for k in range(len(factors)):
            out[k][t, gc] = coef[:, 2 + k]
    return out


def winsorize_rows(x, rows, lo=1.0, hi=99.0):
    x = x.copy()
    for t in rows:
        v = x[t]
        f = np.isfinite(v)
        if f.sum() == 0:
            continue
        a, b = np.percentile(v[f].astype(np.float64), [lo, hi])
        x[t, f] = np.clip(v[f], a, b)
    return x


def features(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("card needs a point-in-time universe (liq_n)")
    uni = np.asarray(uni, dtype=bool)
    T, N = uni.shape
    _, rows = decision_rows(data.dates)
    L = int(params["trend_lookback"])
    facs = factor_levels(data)
    idx = np.asarray(total_return_index(data), dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    betas = stock_betas(idx, listed, uni, facs, rows)
    feats = []
    for bk in betas:
        bk = winsorize_rows(bk, rows)
        bk = np.where(uni, bk, np.nan)
        feats.append(cs_zscore(bk, uni))
    for F in facs:
        z = trend_z(F, rows, L)
        zz = np.where(uni, z[:, None].astype(np.float32), np.float32(np.nan))
        feats.append(zz)                                   # date-level, fed raw
    return feats, rows


def model_scores(data, params):
    feats, rows = features(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    X = stack(*feats)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp, retrain_every=RETRAIN_EVERY,
                         universe=uni, train_days=None, seed=0)
    return np.asarray(score, dtype=np.float64), rows


def sleeve_book(score, uni, keep, rows, top=TOP, min_scored=MIN_SCORED):
    """Two overlapping monthly sleeves. Returns (weights with NaN off decision rows, model-sleeve flag per row,
    top-fifth membership per row)."""
    T, N = uni.shape
    sw = np.asarray(score_weights(score, uni, top=top, bottom=0.0, min_names=min_scored), dtype=np.float64)
    W = np.full((T, N), np.nan)
    is_model = np.zeros(T, dtype=bool)
    in_top = np.zeros((T, N), dtype=bool)
    prev = None
    for t in rows:
        nsc = int((np.isfinite(score[t]) & uni[t]).sum())
        cur = None
        if nsc >= min_scored:
            c = np.where(np.isfinite(sw[t]) & uni[t], np.maximum(sw[t], 0.0), 0.0)
            if c.sum() > 0:
                cur = c / c.sum()
                is_model[t] = True
                in_top[t] = cur > 0
        if cur is None:
            m = uni[t].astype(np.float64)
            cur = m / m.sum() if m.sum() > 0 else m
        if prev is None:
            book = cur
        else:
            p = prev * keep[t]
            ps = p.sum()
            book = 0.5 * cur + 0.5 * (p / ps) if ps > 0 else cur
        W[t] = book
        prev = cur
    return W, is_model, in_top


def target_weights(data, params):
    score, rows = model_scores(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    keep = uni & np.asarray(data.listed, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    W, _, _ = sleeve_book(score, uni, keep, rows)
    ps, _ = decision_rows(data.dates)
    return only_on(ps, W)
