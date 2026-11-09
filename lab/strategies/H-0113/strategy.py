"""H-0113 Quarterly index share-update pressure, learned.

S&P 500 cap-weighted core (weight ~ S_t * close_u_t, reset weekly and on switch-back days); for 5 sessions
before each quarterly rebalance (R = third Friday of Mar/Jun/Sep/Dec) the book switches into the top `top`
fraction of LightGBM-scored members (equal weight).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.1}

SHARES = "sf1_arq_sharesbas"
HORIZON = 5            # card: label horizon_days 5 = sessions from the D+1 open to the open after R
RETRAIN_EVERY = 63     # card: ml.retrain_every_days
MIN_SCORED = 50        # card: at least 50 scored members, else stay in the core
LOOKBACK_SPLIT = 252   # split screen: sessions up to D
LOOKBACK_VOL = 63      # vol63, median dollar volume, minimum sessions of returns
AGE_CAP = 252          # filing_age cap
SPLIT_TOL = 0.10
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 200, "subsample": 0.8,
             "colsample": 1.0}
FEATURES = ("dq", "dq_prev", "demand_days", "filing_age", "vol63", "ret5")


# ----------------------------------------------------------------------------------------------- calendar
def quarter_calendar(dates):
    """Decision rows and the third-Friday date of their quarter, point-in-time.

    From the date of row t alone: R_approx = third Friday of the current or next quarter-end month (the first one
    on or after the date); D_approx = R_approx minus 5 weekdays (np.busday_offset; exchange holidays ignored).
    Row t is a decision row when it is the first row on or after D_approx (its previous row is before D_approx).
    Returns (is_decision bool (T,), third_friday datetime64[D] (T,)).
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    mi = dates.astype("datetime64[M]").astype(np.int64)          # months since 1970-01
    qmi = mi + (2 - mi % 3)                                       # quarter-end month (Mar/Jun/Sep/Dec)

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


def window_end_rows(dates, is_dec, tf):
    """For each decision row d, the switch-back row R: the first row after d whose date is on or after the third
    Friday (-1 if it is not in the data yet)."""
    dates = np.asarray(dates).astype("datetime64[D]")
    out = {}
    for d in np.flatnonzero(is_dec):
        later = np.flatnonzero(dates[d + 1:] >= tf[d])
        out[int(d)] = int(d + 1 + later[0]) if later.size else -1
    return out


# ----------------------------------------------------------------------------------------------- features
def compute_features(data, is_dec):
    """Raw features at the decision rows.

    Returns (rows (K,), feats (6, K, N) float32 NaN where not scored, scored (K, N) bool)."""
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
    n_ret = np.zeros(N, dtype=np.int64)          # sessions with a return (both closes known) up to the row
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

            # split screen over the 252 sessions up to D
            a = max(1, d - LOOKBACK_SPLIT + 1)
            c = f64(close, a - 1, d + 1)
            r = (1.0 + f64(rco, a, d + 1)) * (1.0 + f64(roc, a, d + 1)) - 1.0
            gap = np.abs(np.log(c[1:] / c[:-1]) - np.log1p(r))
            split = np.any(np.isfinite(gap) & (gap > SPLIT_TOL), axis=0)

            # 63-session window to D: vol63 (log total returns on sessions with a return), median dollar volume
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

            # filing_age: sessions since S last changed value (NaN -> value counts as a change), capped at 252
            e = max(1, d - AGE_CAP + 1)
            seg = f64(S, e - 1, d + 1)
            changed = np.isfinite(seg[1:]) & ~(seg[1:] == seg[:-1])
            any_ch = changed.any(axis=0)
            last = (changed.shape[0] - 1) - np.argmax(changed[::-1], axis=0)      # row offset of the last change
            age = np.minimum(np.where(any_ch, (d - e) - last, AGE_CAP), AGE_CAP).astype(np.float64)

            # ret5: total return over the 5 sessions to D
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
    """Framework cs_zscore across the scored members of each date. A feature with zero dispersion across them
    on a date (all equal) carries no cross-sectional information: z = 0 there instead of NaN."""
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
    """LightGBM walk-forward score (T, N); finite only on decision rows for scored members.

    Training rows are the decision-date rows (features NaN elsewhere); label = 5-session forward return
    (framework forward_return), ranked across the members of each decision date."""
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


# ----------------------------------------------------------------------------------------------- book
def core_weights(data, t):
    """Cap weights S_t * close_u_t over members with a valid S and close at row t (sum 1)."""
    S = np.asarray(data.extras[SHARES][t], dtype=np.float64)
    c = np.asarray(data.close[t], dtype=np.float64)
    mv = S * c
    ok = np.asarray(data.universe[t], dtype=bool) & np.isfinite(mv) & (mv > 0)
    w = np.zeros(S.shape[0])
    tot = mv[ok].sum()
    if tot > 0:
        w[ok] = mv[ok] / tot
    return w


def build_book(data, rows, score, top, is_dec, tf):
    """Target weights. `score` (K, N) is the model score on the decision rows `rows` (NaN = not scored)."""
    dates = data.dates
    T = dates.shape[0]
    N = len(data.instruments)
    uni = data.universe
    W = np.full((T, N), np.nan)

    week = np.zeros(T, dtype=bool)
    week[period_starts(dates, "W")] = True
    week[0] = True                                      # start invested in the core

    ends = window_end_rows(dates, is_dec, tf)
    in_window = np.zeros(T, dtype=bool)
    switch_back = np.zeros(T, dtype=bool)
    starts = {}
    if rows.size:
        u_rows = np.asarray(uni[rows], dtype=bool)
        sc = np.where(u_rows, score, np.nan)
        n_scored = np.isfinite(sc).sum(axis=1)
        sel = n_scored >= MIN_SCORED
        if sel.any():
            sw = np.asarray(score_weights(sc[sel], u_rows[sel], top=float(top), bottom=0.0, min_names=MIN_SCORED),
                            dtype=np.float64)
            for i, k in enumerate(np.flatnonzero(sel)):
                d = int(rows[k])
                starts[d] = np.where(np.isfinite(sw[i]), sw[i], 0.0)
                end = ends[d]
                in_window[d:(end if end >= 0 else T)] = True
                if end >= 0:
                    switch_back[end] = True

    for t in range(T):
        if t in starts:
            W[t] = starts[t]
        elif switch_back[t] or (week[t] and not in_window[t]):
            W[t] = core_weights(data, t)

    # a held name that is not a member at row t is sold at the next open (its weight goes to cash)
    held = np.zeros(N)
    for t in range(T):
        row = W[t]
        held = np.where(np.isfinite(row), row, held)
        bad = (held != 0) & ~np.asarray(uni[t], dtype=bool)
        if bad.any():
            W[t, bad] = 0.0
            held[bad] = 0.0
    return W


def target_weights(data, params):
    is_dec, tf = quarter_calendar(data.dates)
    rows, score = model_scores(data, params, is_dec)
    return build_book(data, rows, score, params["top"], is_dec, tf)
