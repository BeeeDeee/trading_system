"""H-0119 mechanism event (copy of the strategy pipeline; diagnostic.py is a separate file).

Event: on a shock date (a weekly decision date where at least one of GLD, UNG, DBA, DBB has
|z| >= shock_z), a universe member that is tradable and enters the model's long leg (top `top`
fraction of the score) with a positive raw implied move (side +1) or the short leg (bottom
`top` fraction) with a negative raw implied move (side -1).
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

SIGNAL_ETFS = ("SPY", "USO", "GLD", "UNG", "DBA", "DBB")
WINDOW = 252
MOVE_N = 10
MIN_OBS = 200
IVOL_N = 63
DV_N = 21
HORIZON = 5
RETRAIN = 63
MIN_NAMES = 100
GBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample": 1.0,
}


def last_session_of_week(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (d + 3) % 7
    return weekday == 4


def daily_total_return(data):
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    return (1.0 + rco) * (1.0 + roc) - 1.0


def commodity_moves(r_spy, r_uso, r_com, n_move=MOVE_N):
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
    im_raw = feats[0].copy()
    return feats, scored, decision, shock, im_raw


def model_book(data, params):
    shock_z = float(params["shock_z"])
    top = float(params["top"])
    feats, scored, decision, shock, im_raw = build_features(data, shock_z)
    uni = np.asarray(data.universe, dtype=bool)
    for i in range(4):
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


def events(data, params):
    book, decision, shock, im_raw, uni = model_book(data, params)
    tradable = np.asarray(data.tradable, dtype=bool)
    on = (decision & shock)[:, None] & uni & tradable
    im = np.where(np.isfinite(im_raw), im_raw, 0.0)
    long_ev = on & (book > 0) & (im > 0)
    short_ev = on & (book < 0) & (im < 0)
    mask = long_ev | short_ev
    side = np.where(short_ev, -1.0, 1.0)
    return mask, side
