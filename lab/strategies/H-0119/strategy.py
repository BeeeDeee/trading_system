"""H-0119: delayed stock response to non-oil commodity news (GLD, UNG, DBA, DBB), learned.

Weekly decision on the last session of the calendar week (approximated by Friday, see
`last_session_of_week`). On each decision date t:
  (1) each commodity ETF's daily return is regressed on SPY and USO over the 252 sessions
      ending at t (with intercept); residuals e_k, R_k = sum of the last 10 residuals,
      z_k = R_k / (std(e_k) * sqrt(10)); shocked set S_t = {k : |z_k| >= shock_z}.
  (2) on shock dates, every universe member with >= 200 valid daily returns in the window is
      regressed on [1, SPY, USO, e_GLD, e_UNG, e_DBA, e_DBB]; coefficients, t-statistics and
      residuals.
  (3) features F1 implied move, F2 max |t| over S, F3 residual std over the last 63 sessions,
      F4 log 21-session mean dollar volume (F1-F4 z-scored across scored members), F5 shock
      size max |z| over S (raw).
  (4) framework walk_forward gbm, label = cross-sectional rank of the 5-session forward return
      among the scored members, training rows = scored rows of past shock dates, expanding
      window, refit every 63 rows.
  (5) long top `top` / short bottom `top` of the score, 0.5 gross per side, at least 100
      scored members; flat on non-shock decision dates.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"shock_z": 1.5, "top": 0.1}

SIGNAL_ETFS = ("SPY", "USO", "GLD", "UNG", "DBA", "DBB")   # market, oil control, 4 commodities
WINDOW = 252          # sessions of the regressions
MOVE_N = 10           # sessions of the commodity move R_k
MIN_OBS = 200         # valid daily returns a stock needs in the window
IVOL_N = 63           # sessions of the idiosyncratic volatility
DV_N = 21             # sessions of the mean dollar volume
HORIZON = 5           # label horizon (open t+1 -> open t+6)
RETRAIN = 63          # refit every 63 trading rows
MIN_NAMES = 100       # scored members needed for a book
GBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample": 1.0,
}


def last_session_of_week(dates):
    """True on Fridays (weekday arithmetic on the row's own date; exchange holidays ignored)."""
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (d + 3) % 7          # epoch day 0 was a Thursday -> Monday = 0
    return weekday == 4


def daily_total_return(data):
    """Close-to-close total return of each row: (1 + ret_co) (1 + ret_oc) - 1."""
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    return (1.0 + rco) * (1.0 + roc) - 1.0


def commodity_moves(r_spy, r_uso, r_com, n_move=MOVE_N):
    """Step 1 on one window. r_spy, r_uso (L,), r_com (L, K).

    Returns residuals e (L, K), R (K,) = sum of the last n_move residuals and
    z (K,) = R / (std(e) * sqrt(n_move))."""
    L = r_spy.shape[0]
    X = np.column_stack([np.ones(L), r_spy, r_uso])
    beta = np.linalg.lstsq(X, r_com, rcond=None)[0]
    e = r_com - X @ beta
    R = e[-n_move:].sum(axis=0)
    sd = e.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = R / (sd * np.sqrt(n_move))
    return e, R, z


def exposure_regression(Y, M, F, min_obs=MIN_OBS, ivol_n=IVOL_N):
    """Step 2 on one window, all columns at once (per-column masks).

    Y (L, n) daily returns, M (L, n) bool valid, F (L, p-1) regressors (an intercept is added).
    Returns b (n, p), t-statistics (n, p), idiosyncratic vol (n,) over the last ivol_n rows of
    the residuals and ok (n,) = at least min_obs valid rows. Columns that are not ok are NaN."""
    L, n = Y.shape
    X = np.column_stack([np.ones(L), F])
    p = X.shape[1]
    Mf = M.astype(np.float64)
    Yz = np.where(M, Y, 0.0)
    nobs = Mf.sum(axis=0)
    ok = nobs >= min_obs
    XX = (X[:, :, None] * X[:, None, :]).reshape(L, p * p)
    XtX = (Mf.T @ XX).reshape(n, p, p)
    XtX[~ok] = np.eye(p)
    Xty = Yz.T @ X
    inv = np.linalg.pinv(XtX)
    b = np.einsum("nij,nj->ni", inv, Xty)
    resid = np.where(M, Yz - X @ b.T, np.nan)
    sse = np.nansum(resid * resid, axis=0)
    dof = np.maximum(nobs - p, 1.0)
    s2 = sse / dof
    diag = np.einsum("nii->ni", inv)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.maximum(s2[:, None] * diag, 0.0))
        tstat = b / se
    tail = resid[-ivol_n:]
    cnt = np.isfinite(tail).sum(axis=0)
    mu = np.nansum(tail, axis=0) / np.maximum(cnt, 1)
    dev = np.where(np.isfinite(tail), tail - mu, 0.0)
    ivol = np.sqrt((dev * dev).sum(axis=0) / np.maximum(cnt - 1, 1))
    ivol = np.where(cnt >= 2, ivol, np.nan)
    b[~ok] = np.nan
    tstat[~ok] = np.nan
    ivol[~ok] = np.nan
    return b, tstat, ivol, ok


def build_features(data, shock_z):
    """Steps 1-3. Returns raw features (5, T, N), scored (T, N), decision (T,), shock (T,),
    implied move IM (T, N) raw (NaN where not scored)."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0119 needs the liq_n universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    T, N = uni.shape
    cols = [data.col(s) for s in SIGNAL_ETFS]
    r = daily_total_return(data)
    r6 = r[:, cols]
    listed = np.asarray(data.listed, dtype=bool)
    etf_ok = np.all(listed[:, cols] & np.isfinite(r6), axis=1)
    close = np.asarray(data.close, dtype=np.float64)
    valid = listed & np.isfinite(close) & np.isfinite(r)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    decision = last_session_of_week(data.dates)

    feats = np.full((5, T, N), np.nan, dtype=np.float32)
    scored = np.zeros((T, N), dtype=bool)
    shock = np.zeros(T, dtype=bool)
    for t in np.flatnonzero(decision):
        if t < WINDOW - 1:
            continue
        lo = t - WINDOW + 1
        if not etf_ok[lo:t + 1].all():
            continue
        e, R, z = commodity_moves(r6[lo:t + 1, 0], r6[lo:t + 1, 1], r6[lo:t + 1, 2:6])
        S = np.isfinite(z) & (np.abs(z) >= shock_z)
        if not S.any():
            continue
        shock[t] = True
        members = np.flatnonzero(uni[t])
        if members.size == 0:
            continue
        Y = np.where(valid[lo:t + 1][:, members], r[lo:t + 1][:, members], 0.0)
        M = valid[lo:t + 1][:, members]
        F = np.column_stack([r6[lo:t + 1, 0], r6[lo:t + 1, 1], e])
        b, tst, ivol, ok = exposure_regression(Y, M, F)
        k = np.flatnonzero(S)
        im = b[:, 3 + k] @ R[k]
        rel = np.max(np.abs(tst[:, 3 + k]), axis=1)
        dvw = dv[t - DV_N + 1:t + 1][:, members]
        dcnt = np.isfinite(dvw).sum(axis=0)
        dmean = np.where(dcnt > 0, np.nansum(dvw, axis=0) / np.maximum(dcnt, 1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ldv = np.where(dmean > 0, np.log(np.where(dmean > 0, dmean, 1.0)), np.nan)
        size = float(np.max(np.abs(z[k])))
        f = np.vstack([im, rel, ivol, ldv, np.full(members.size, size)])
        good = ok & np.all(np.isfinite(f), axis=0)
        cols_ok = members[good]
        feats[:, t, cols_ok] = f[:, good]
        scored[t, cols_ok] = True
    im_raw = feats[0].copy()          # unscored entries are NaN already
    return feats, scored, decision, shock, im_raw


def model_book(data, params):
    """Steps 1-5. Returns the decision-row book (T, N) (0 = flat), decision (T,), shock (T,),
    raw implied move (T, N) and the universe (T, N)."""
    shock_z = float(params["shock_z"])
    top = float(params["top"])
    feats, scored, decision, shock, im_raw = build_features(data, shock_z)
    uni = np.asarray(data.universe, dtype=bool)
    for i in range(4):                # F1-F4 z-scored across the scored members of each date
        zi = np.asarray(cs_zscore(feats[i], scored))
        feats[i] = np.where(scored, zi, np.nan)
        del zi
    final = scored & np.all(np.isfinite(feats), axis=0)
    feats[:, ~final] = np.nan
    X = stack(*[feats[i] for i in range(5)])
    del feats
    y = cs_rank(forward_return(data, HORIZON), final)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=dict(GBM_PARAMS),
                         retrain_every=RETRAIN, universe=data.universe)
    score = np.where(final, np.asarray(score, dtype=np.float64), np.nan)
    sw = np.asarray(score_weights(score, data.universe, top=top, bottom=top, min_names=MIN_NAMES),
                    dtype=np.float64)
    sw = np.where(np.isfinite(sw), sw, 0.0)
    book = np.where((decision & shock)[:, None], sw, 0.0)
    return book, decision, shock, im_raw, uni


def target_weights(data, params):
    book, decision, shock, im_raw, uni = model_book(data, params)
    T, N = book.shape
    w = np.full((T, N), np.nan)
    w[decision] = book[decision]
    # a held stock that leaves the universe between two decision dates is closed on the first
    # row outside it (the engine accepts no weight on non-members)
    rows = np.arange(T)
    last = np.maximum.accumulate(np.where(decision, rows, -1))
    held = np.where((last >= 0)[:, None], book[np.maximum(last, 0)], 0.0)
    exit_rows = (held != 0.0) & ~uni & ~decision[:, None]
    w[exit_rows] = 0.0
    return w
