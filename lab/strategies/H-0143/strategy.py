"""H-0143: frog-in-the-pan momentum in asset-class ETFs, learned (gbm), fixed 60/40 sleeves, monthly cohorts.

Pipeline (card signal.description):
  r = log((1+ret_co)(1+ret_oc)) on rows with a bar; six features per ETF and date from rows <= t;
  eligibility (>= 230 valid returns and >= 200 valid dollar volumes in the last 252 rows);
  cross-sectional z-score of every feature over the scored ETFs (framework cs_zscore);
  label = framework forward_return(horizon), ranked across the universe (framework cs_rank);
  framework walk_forward, model gbm, expanding window, refit every 63 rows;
  month-end decisions, K = horizon / 21 overlapping cohorts, inside a cohort 0.60 equally over the top fifth
  of the scored RISK members and 0.40 over the top fifth of the scored DEFENSIVE members;
  sleeve benchmark (SPY / IEF) when a sleeve has fewer than 2 scored members or before the first fit.
"""
import math

import numpy as np
from lab.framework.api import only_on
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"horizon": 42, "n_estimators": 100}

RISK = ("SPY", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "EFA", "EEM",
        "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWN", "EWP", "EWQ", "EWS", "EWU", "EWW",
        "EWZ", "EWT", "EWY", "FXI", "DBC")
DEFENSIVE = ("SHY", "IEF", "TLT", "TIP", "AGG", "LQD", "HYG", "EMB", "MUB", "GLD")
RISK_BENCH = "SPY"
DEF_BENCH = "IEF"
RISK_WEIGHT = 0.60
DEF_WEIGHT = 0.40

LONG = 252          # rows t-251..t
SKIP = 21           # formation window skips the latest 21 rows
FORM = 231          # formation window F = rows t-251..t-21
RECENT = 21         # rows t-20..t
MID = 63            # rows t-62..t
TOPJ = 5            # largest squared returns in F
MIN_RET = 230
MIN_DV = 200
RETRAIN_EVERY = 63
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 0.8}


# ----------------------------------------------------------------------------------------------- helpers
def _shift(x, k):
    """x shifted down by k rows (row t holds x[t-k]); NaN on the first k rows. Point-in-time."""
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _rsum(x, w):
    """Trailing sum over rows t-w+1..t of a NaN-free array; NaN where fewer than w rows exist."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T >= w:
        c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
        out[w - 1:] = c[w:] - c[: T - w + 1]
    return out


def _rstd(x0, v, w):
    """Trailing sample std over rows t-w+1..t of the valid values (x0 = x with invalid set to 0, v = 0/1)."""
    n = _rsum(v, w)
    s1 = _rsum(x0, w)
    s2 = _rsum(x0 * x0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    var = np.where(n >= 2, np.maximum(var, 0.0), np.nan)
    return np.sqrt(var)


def _jump_share(r2):
    """Sum of the TOPJ largest r^2 in F (rows t-251..t-21) / sum of all r^2 in F. r2 NaN-free (T, N)."""
    T, N = r2.shape
    out = np.full((T, N), np.nan)
    if T < LONG:
        return out
    rows = np.arange(LONG - 1, T)
    gidx = (rows - (LONG - 1))[:, None] + np.arange(FORM)[None, :]       # rows t-251 .. t-21
    for j in range(N):
        g = r2[gidx, j]
        top = np.partition(g, FORM - TOPJ, axis=1)[:, FORM - TOPJ:].sum(axis=1)
        tot = g.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows, j] = np.where(tot > 0, top / tot, np.nan)
    return out


def _features(r, valid, dv):
    """The card's six raw features and the eligibility mask, from rows <= t only.

    r: (T, N) daily log total return, valid: (T, N) bool a bar exists, dv: (T, N) dollar volume (NaN = none).
    Returns (list of six (T, N) arrays, NaN where not scored; eligible (T, N) bool).
    """
    valid = np.asarray(valid, dtype=bool) & np.isfinite(r)
    r0 = np.where(valid, r, 0.0)
    v = valid.astype(float)
    dvv = np.isfinite(dv)
    dv0 = np.where(dvv, dv, 0.0)

    n_ret = _rsum(v, LONG)
    n_dv = _rsum(dvv.astype(float), LONG)
    n_dv21 = _rsum(dvv.astype(float), RECENT)
    sd_long = _rstd(r0, v, LONG)
    sd_mid = _rstd(r0, v, MID)

    with np.errstate(invalid="ignore", divide="ignore"):
        s_form = _shift(_rsum(r0, FORM), SKIP)
        mom = s_form / (sd_long * math.sqrt(FORM))
        n_pos = _shift(_rsum((r0 > 0).astype(float), FORM), SKIP)
        n_neg = _shift(_rsum((r0 < 0).astype(float), FORM), SKIP)
        path = (n_pos - n_neg) / FORM
        jump = _jump_share(r0 * r0)
        recent = _rsum(r0, RECENT) / (sd_mid * math.sqrt(RECENT))
        volratio = sd_mid / sd_long
        m21 = _rsum(dv0, RECENT) / n_dv21
        m252 = _rsum(dv0, LONG) / n_dv
        attention = np.log(np.where((m21 > 0) & (m252 > 0), m21 / m252, np.nan))

    feats = [mom, path, jump, recent, volratio, attention]
    elig = (n_ret >= MIN_RET) & (n_dv >= MIN_DV)
    for f in feats:
        elig &= np.isfinite(f)
    feats = [np.where(elig, f, np.nan) for f in feats]
    return feats, elig


def _log_returns(data):
    r = np.log((1.0 + data.ret_co) * (1.0 + data.ret_oc))
    valid = np.isfinite(data.close) & np.asarray(data.listed, dtype=bool) & np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def _gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def _scores(data, params):
    """Point-in-time gbm score (T, N); NaN where an ETF is not scored or no model is fitted yet."""
    uni = data.universe
    r, valid = _log_returns(data)
    feats, elig = _features(r, valid, np.asarray(data.dollar_volume, dtype=float))
    if uni is not None:
        elig = elig & np.asarray(uni, dtype=bool)
        feats = [np.where(elig, f, np.nan) for f in feats]
    X = stack(*[cs_zscore(f, uni) for f in feats])
    h = int(params["horizon"])
    y = cs_rank(forward_return(data, h), uni)
    score = walk_forward(X, y, horizon=h, model="gbm", params=_gbm_params(params),
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=float)
    return np.where(elig & np.isfinite(score), score, np.nan)


def _decision_rows(dates):
    """Month-end decision rows from the date of each row alone.

    Row t is a decision if the next weekday after its date (np.busday_offset, ignores exchange holidays) lies in
    another calendar month. If a month has no such row in the view (its last weekday was a holiday), the first
    row of the next month is the decision instead (one day late). Row 0 never gets the fallback.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    dec = np.zeros(T, dtype=bool)
    if T == 0:
        return dec
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    dec[:] = nxt.astype("datetime64[M]") != month
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    for a in range(1, starts.shape[0]):
        s0, s1 = starts[a - 1], starts[a]
        if not dec[s0:s1].any():
            dec[s1] = True
    return dec


def _top_k(n):
    """ceil(0.2 n), at least 2 (integer arithmetic)."""
    return max(2, -(-n // 5))


def _cohort(srow, sleeves, n_cols):
    """One cohort's weights from one row of scores. sleeves: [(cols array, weight, bench col), ...]."""
    w = np.zeros(n_cols)
    for cols, sw, bench in sleeves:
        s = srow[cols]
        ok = np.isfinite(s)
        n = int(ok.sum())
        if n < 2:
            w[bench] += sw
            continue
        k = min(_top_k(n), n)
        cand = cols[ok]
        order = np.argsort(-s[ok], kind="stable")
        w[cand[order[:k]]] += sw / k
    return w


def _book(score, dec, sleeves, K):
    """(T, N) target weights: NaN except on decision rows, where the book is the mean of the live cohorts."""
    T, N = score.shape
    W = np.full((T, N), np.nan)
    cohorts = []
    for t in np.flatnonzero(dec):
        cohorts.append(_cohort(score[t], sleeves, N))
        live = cohorts[-K:]
        W[t] = np.mean(np.stack(live), axis=0)
    return W


def _sleeves(data):
    risk = np.array([data.col(x) for x in RISK], dtype=int)
    dfn = np.array([data.col(x) for x in DEFENSIVE], dtype=int)
    return [(risk, RISK_WEIGHT, data.col(RISK_BENCH)), (dfn, DEF_WEIGHT, data.col(DEF_BENCH))]


def _n_cohorts(params):
    return max(1, int(params["horizon"]) // 21)


# ----------------------------------------------------------------------------------------------- strategy
def target_weights(data, params):
    sleeves = _sleeves(data)
    score = _scores(data, params)
    dec = _decision_rows(data.dates)
    W = _book(score, dec, sleeves, _n_cohorts(params))
    return only_on(dec, W)
