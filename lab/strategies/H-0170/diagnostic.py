"""H-0170 mechanism events: on a shock-week decision date (|z_t| >= shock_z), a scored LIQ-500
member in the top fifth of the walk-forward gbm score (side +1) or in the bottom fifth (side -1).
The feature / walk_forward pipeline is a copy of strategy.py (separate file by contract)."""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 126
SHOCK_LAG = 5
MKT_LAG = 5
SD_WINDOW = 252
SD_MIN = 200
GROWTH_LAG = 252
TOP = 0.2
MIN_SCORED = 100
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 200}


def _decision_days(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (d + 3) % 7
    return weekday == 4


def _ffill(x):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(T), -1)
    idx = np.maximum.accumulate(idx) if T else idx
    out = np.full(T, np.nan)
    has = idx >= 0
    out[has] = x[idx[has]]
    return out


def _lag_diff(x, k):
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[k:] - x[:-k]
    return out


def _trailing_sd(x, window, min_n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        w = x[max(0, t - window + 1):t + 1]
        w = w[np.isfinite(w)]
        if w.size >= min_n and w.size >= 2:
            out[t] = np.std(w, ddof=1)
    return out


def _shock_z(r_raw):
    r = _ffill(r_raw)
    s = _lag_diff(r, SHOCK_LAG)
    sd = _trailing_sd(s, SD_WINDOW, SD_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(sd) & (sd > 0), s / sd, np.nan)
    return z


def _market_move(idx, members, listed):
    T, N = idx.shape
    mean5 = np.full(T, np.nan)
    if T > MKT_LAG:
        with np.errstate(invalid="ignore", divide="ignore"):
            r5 = idx[MKT_LAG:] / idx[:-MKT_LAG] - 1.0
        ok = (members[MKT_LAG:] & listed[MKT_LAG:] & listed[:-MKT_LAG] & np.isfinite(r5))
        n = ok.sum(axis=1)
        tot = np.where(ok, r5, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean5[MKT_LAG:] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    sd = _trailing_sd(mean5, SD_WINDOW, SD_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(sd) & (sd > 0), mean5 / sd, np.nan)


def _book_to_market(pb):
    pb = np.asarray(pb, dtype=np.float64)
    ok = np.isfinite(pb) & (pb != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, 1.0 / np.where(ok, pb, 1.0), np.nan)


def _revenue_growth(rev):
    rev = np.asarray(rev, dtype=np.float64)
    prev = np.full(rev.shape, np.nan)
    if rev.shape[0] > GROWTH_LAG:
        prev[GROWTH_LAG:] = rev[:-GROWTH_LAG]
    ok = np.isfinite(rev) & np.isfinite(prev) & (prev > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, rev / np.where(ok, prev, 1.0) - 1.0, np.nan)


def _winsorize_rows(x, members, rows):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    for t in rows:
        v = x[t]
        m = members[t] & np.isfinite(v)
        if not m.any():
            continue
        lo, hi = np.percentile(v[m], [1.0, 99.0])
        out[t, m] = np.clip(v[m], lo, hi)
    return out


def _scores(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0170 needs the point-in-time LIQ-500 universe (data.universe is None)")
    members = np.asarray(uni, dtype=bool)
    T, N = members.shape
    listed = np.asarray(data.listed, dtype=bool)
    dec = _decision_days(data.dates)

    z = _shock_z(np.asarray(data.series["fred_macro.DFII10"], dtype=np.float64))
    idx = np.asarray(total_return_index(data), dtype=np.float64)
    m = _market_move(idx, members, listed)
    divy = np.asarray(data.extras["sf1_art_divyield"], dtype=np.float64)
    btm = _book_to_market(data.extras["sf1_art_pb"])
    roe = np.asarray(data.extras["sf1_art_roe"], dtype=np.float64)
    growth = _revenue_growth(data.extras["sf1_art_revenue"])

    shock_z = float(params["shock_z"])
    with np.errstate(invalid="ignore"):
        shock = dec & np.isfinite(z) & (np.abs(z) >= shock_z)
    rows = np.flatnonzero(shock)

    feats = []
    for raw in (divy, btm, roe, growth):
        w = _winsorize_rows(raw, members, rows)
        feats.append(np.asarray(cs_zscore(w, members), dtype=np.float64))
    zf = np.full((T, N), np.nan)
    mf = np.full((T, N), np.nan)
    if rows.size:
        zf[rows] = np.where(members[rows], z[rows][:, None], np.nan)
        mf[rows] = np.where(members[rows], m[rows][:, None], np.nan)
    for f in feats:
        f[~shock] = np.nan
        f[~members] = np.nan

    h = HORIZON
    if rows.size == 0:
        return np.full((T, N), np.nan), shock, members
    X = stack(zf, mf, feats[0], feats[1], feats[2], feats[3])
    y = cs_rank(forward_return(data, h), members)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=h, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=members)
    score = np.asarray(score, dtype=np.float64)
    score[~shock] = np.nan
    score[~members] = np.nan
    return score, shock, members


def events(data, params):
    score, shock, members = _scores(data, params)
    T, N = members.shape
    scored = np.isfinite(score) & members
    n_scored = scored.sum(axis=1)
    use = shock & (n_scored >= MIN_SCORED)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    if use.any():
        legs = np.asarray(score_weights(score, members, top=TOP, bottom=TOP, min_names=MIN_SCORED),
                          dtype=np.float64)
        legs = np.nan_to_num(legs, nan=0.0)
        legs[~use] = 0.0
        tradable = np.asarray(data.tradable, dtype=bool)
        mask = (legs != 0.0) & members & tradable
        side = np.where(legs > 0, 1.0, np.where(legs < 0, -1.0, 0.0))
        side[~mask] = 0.0
    return mask, side
