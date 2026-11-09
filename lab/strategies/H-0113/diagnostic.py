"""H-0113 mechanism event: at each quarterly decision date D, a scored S&P 500 member in the top `top` fraction of
the model score (+1, predicted net index buying) or in the bottom `top` fraction (-1, predicted net selling).
Calendar, features and the walk_forward call are copied verbatim from strategy.py (same pipeline)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

SHARES = "sf1_arq_sharesbas"
HORIZON = 5
RETRAIN_EVERY = 63
MIN_SCORED = 50
LOOKBACK_SPLIT = 252
LOOKBACK_VOL = 63
AGE_CAP = 252
SPLIT_TOL = 0.10
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 200, "subsample": 0.8,
             "colsample": 1.0}
FEATURES = ("dq", "dq_prev", "demand_days", "filing_age", "vol63", "ret5")


def quarter_calendar(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    mi = dates.astype("datetime64[M]").astype(np.int64)
    qmi = mi + (2 - mi % 3)

    def third_friday(month_idx):
        first = month_idx.astype("datetime64[M]").astype("datetime64[D]")
        return np.busday_offset(first, 2, roll="forward", weekmask="Fri")

    tf = third_friday(qmi)
    tf = np.where(dates > tf, third_friday(qmi + 3), tf)
    d_approx = np.busday_offset(tf, -5, roll="forward")
    is_dec = np.zeros(T, dtype=bool)
    if T > 1:
        is_dec[1:] = (dates[1:] >= d_approx[1:]) & (dates[:-1] < d_approx[1:])
    return is_dec, tf


def compute_features(data, is_dec):
    T, N = data.ret_co.shape
    S = data.extras[SHARES]
    close = data.close
    dv = data.dollar_volume
    rco, roc = data.ret_co, data.ret_oc
    uni = data.universe
    if uni is None:
        raise ValueError("card universe sp500 requires data.universe")

    def f64(x, a, b):
        return np.asarray(x[a:b], dtype=np.float64)

    rows = np.flatnonzero(is_dec)
    K = rows.size
    feats = np.full((len(FEATURES), K, N), np.nan, dtype=np.float32)
    scored = np.zeros((K, N), dtype=bool)
    n_ret = np.zeros(N, dtype=np.int64)
    pos = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, d in enumerate(rows):
            d = int(d)
            if d >= pos:
                c = f64(close, pos - 1, d + 1)
                vc = np.isfinite(c) & (c > 0)
                n_ret += (vc[1:] & vc[:-1]).sum(axis=0)
                pos = d + 1
            if k < 2 or d < HORIZON:
                continue
            dp, dp2 = int(rows[k - 1]), int(rows[k - 2])
            s_d, s_p, s_p2 = (np.asarray(S[i], dtype=np.float64) for i in (d, dp, dp2))
            okS = (np.isfinite(s_d) & np.isfinite(s_p) & np.isfinite(s_p2) & (s_d > 0) & (s_p > 0) & (s_p2 > 0))
            dq = np.log(s_d / s_p)
            dq_prev = np.log(s_p / s_p2)

            a = max(1, d - LOOKBACK_SPLIT + 1)
            c = f64(close, a - 1, d + 1)
            r = (1.0 + f64(rco, a, d + 1)) * (1.0 + f64(roc, a, d + 1)) - 1.0
            gap = np.abs(np.log(c[1:] / c[:-1]) - np.log1p(r))
            split = np.any(np.isfinite(gap) & (gap > SPLIT_TOL), axis=0)

            b = max(1, d - LOOKBACK_VOL + 1)
            vc = np.isfinite(c) & (c > 0)
            hr = (vc[1:] & vc[:-1])[b - a:]
            lr = np.where(hr, np.log1p(r[b - a:]), np.nan)
            cnt = hr.sum(axis=0)
            mean = np.nansum(lr, axis=0) / np.maximum(cnt, 1)
            var = np.nansum((lr - mean) ** 2, axis=0) / np.maximum(cnt - 1, 1)
            vol63 = np.where(cnt >= 2, np.sqrt(var), np.nan)
            dvw = f64(dv, max(0, d - LOOKBACK_VOL + 1), d + 1)
            dv_ok = np.isfinite(dvw).any(axis=0)
            med_dv = np.full(N, np.nan)
            if dv_ok.any():
                med_dv[dv_ok] = np.nanmedian(dvw[:, dv_ok], axis=0)
            demand = dq * s_d * c[-1] / med_dv
            demand = np.where(np.isfinite(med_dv) & (med_dv > 0), demand, np.nan)

            e = max(1, d - AGE_CAP + 1)
            seg = f64(S, e - 1, d + 1)
            changed = np.isfinite(seg[1:]) & ~(seg[1:] == seg[:-1])
            any_ch = changed.any(axis=0)
            last = (changed.shape[0] - 1) - np.argmax(changed[::-1], axis=0)
            age = np.minimum(np.where(any_ch, (d - e) - last, AGE_CAP), AGE_CAP).astype(np.float64)

            ret5 = np.prod(1.0 + r[-HORIZON:], axis=0) - 1.0

            ok = (np.asarray(uni[d], dtype=bool) & okS & ~split & (n_ret >= LOOKBACK_VOL) & vc[-1])
            vals = (dq, dq_prev, demand, age, vol63, ret5)
            for v in vals:
                ok &= np.isfinite(v)
            scored[k] = ok
            for i, v in enumerate(vals):
                feats[i, k] = np.where(ok, v, np.nan)
    return rows, feats, scored


def zscore_scored(f, scored):
    z = np.asarray(cs_zscore(f, scored), dtype=np.float32)
    fm = np.where(scored, f, np.nan)
    has = scored.any(axis=1)
    hi = np.full(f.shape[0], np.nan)
    lo = np.full(f.shape[0], np.nan)
    if has.any():
        hi[has] = np.nanmax(fm[has], axis=1)
        lo[has] = np.nanmin(fm[has], axis=1)
    flat = has & (hi == lo)
    fix = flat[:, None] & scored & np.isfinite(f)
    return np.where(fix, np.float32(0.0), z)


def model_scores(data, params, is_dec):
    T, N = data.ret_co.shape
    rows, feats, scored = compute_features(data, is_dec)
    X = np.full((T, N, len(FEATURES)), np.nan, dtype=np.float32)
    y = np.full((T, N), np.nan, dtype=np.float32)
    if rows.size:
        X[rows] = stack(*[zscore_scored(feats[i], scored) for i in range(len(FEATURES))])
        fr = forward_return(data, HORIZON)
        y[rows] = cs_rank(np.asarray(fr[rows]), np.asarray(data.universe[rows], dtype=bool))
        del fr
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY,
                         universe=data.universe)
    del X
    out = np.full((rows.size, N), np.nan)
    if rows.size:
        out = np.where(scored, np.asarray(score[rows], dtype=np.float64), np.nan)
    return rows, out


def events(data, params):
    T, N = data.ret_co.shape
    is_dec, _ = quarter_calendar(data.dates)
    rows, score = model_scores(data, params, is_dec)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    if rows.size == 0:
        return mask, side
    u_rows = np.asarray(data.universe[rows], dtype=bool)
    sc = np.where(u_rows, score, np.nan)
    sel = np.isfinite(sc).sum(axis=1) >= MIN_SCORED      # same 50-member rule as the strategy's window
    if not sel.any():
        return mask, side
    top = float(params["top"])
    w = np.asarray(score_weights(sc[sel], u_rows[sel], top=top, bottom=top, min_names=MIN_SCORED), dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    r_sel = rows[sel]
    tradable = np.asarray(data.tradable[r_sel], dtype=bool)
    m = (w != 0) & u_rows[sel] & tradable
    mask[r_sel] = m
    side[r_sel] = np.where(w > 0, 1.0, np.where(w < 0, -1.0, 0.0))
    return mask, side
