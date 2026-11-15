"""H-0170: real-yield (DFII10) shock x cash-flow duration, learned (gbm), LIQ-500 long-only.

Weekly decisions (last weekday of the calendar week, i.e. Friday; see _decision_days).
On a shock-week decision date with >= 100 scored members: equal-weight top fifth of the
walk-forward gbm score. On every other decision date: equal-weight core of all current
members with a valid price.
"""
import numpy as np
from lab.framework.api import only_on, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "shock_z": 1.0}

HORIZON = 5            # ml.horizon_days
RETRAIN_EVERY = 126    # ml.retrain_every_days
SHOCK_LAG = 5          # 5-session change of DFII10
MKT_LAG = 5            # 5-session member return
SD_WINDOW = 252        # trailing window of the standard deviations
SD_MIN = 200           # minimum valid values in that window
GROWTH_LAG = 252       # revenue growth lag in sessions
TOP = 0.2              # top fifth
MIN_SCORED = 100       # minimum scored members for the tilt
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 200}


# ----------------------------------------------------------------------------- helpers

def _decision_days(dates):
    """Last trading session of the calendar week, approximated by weekday arithmetic on the
    date of row t alone: row t is a decision day when it is a Friday (the last Mon-Fri
    weekday of its week). Exchange holidays are not knowable point-in-time from the data, so a
    week whose Friday is a holiday has no decision (positions are held into the next week)."""
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (d + 3) % 7          # 1970-01-01 was a Thursday; Monday = 0
    return weekday == 4


def _ffill(x):
    """Carry the last earlier valid value forward (point-in-time)."""
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
    """x[t] - x[t-k] (NaN for the first k rows)."""
    out = np.full(x.shape, np.nan)
    if x.shape[0] > k:
        out[k:] = x[k:] - x[:-k]
    return out


def _trailing_sd(x, window, min_n):
    """Sample sd (ddof=1) of the finite values of x over rows t-window+1..t; NaN if fewer
    than min_n finite values."""
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
    """z_t = (r_t - r_{t-5}) / trailing-252 sd of the same 5-session changes (>= 200 valid)."""
    r = _ffill(r_raw)
    s = _lag_diff(r, SHOCK_LAG)
    sd = _trailing_sd(s, SD_WINDOW, SD_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(sd) & (sd > 0), s / sd, np.nan)
    return z


def _market_move(idx, members, listed):
    """m_t = EW mean over members of the 5-session total return ending at t, divided by the
    trailing-252 sd of that mean (>= 200 valid)."""
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
    """On the given rows only: clip member values at the cross-sectional 1st/99th percentile
    of the finite member values. All other rows and non-members are NaN."""
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


def _features(data):
    """(X, shock-week rows info) built point-in-time. Returns X (T, N, 6), the shock z series,
    the decision-day mask and the member mask."""
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
    return z, m, divy, btm, roe, growth, dec, members


def _scores(data, params):
    """Walk-forward gbm score on shock-week decision rows (NaN elsewhere)."""
    z, m, divy, btm, roe, growth, dec, members = _features(data)
    T, N = members.shape
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
    # every feature is NaN outside shock-week decision rows and outside the members
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


# ----------------------------------------------------------------------------- strategy

def target_weights(data, params):
    score, shock, members = _scores(data, params)
    T, N = members.shape
    dec = _decision_days(data.dates)

    # core: all current members with a valid price, equal weight, gross 1
    close = np.asarray(data.close, dtype=np.float64)
    core_m = members & np.isfinite(close)
    n_core = core_m.sum(axis=1)
    core = np.where(core_m, 1.0 / np.maximum(n_core, 1)[:, None], 0.0)

    # tilt: top fifth of the scored members, equal weight, on shock rows with >= 100 scored
    scored = np.isfinite(score) & members
    n_scored = scored.sum(axis=1)
    use_tilt = shock & (n_scored >= MIN_SCORED)
    w = core.copy()
    if use_tilt.any():
        tilt = np.asarray(score_weights(score, members, top=TOP, bottom=0.0, min_names=MIN_SCORED),
                          dtype=np.float64)
        tilt = np.nan_to_num(tilt, nan=0.0)
        w[use_tilt] = tilt[use_tilt]
    w[~members] = 0.0
    return only_on(dec, w)
