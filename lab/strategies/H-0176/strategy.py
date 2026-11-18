"""H-0176: volatility-target de-risking cycle in asset-class ETFs, learned (gbm), two monthly 2-month sleeves.

Features (per ETF and day, rows <= t only): dd, vr, vlev, r21, r12_1, cross-sectionally z-scored over the
ETFs scored on day t (all five finite and a bar on t). Label: 42-session forward open-to-open return, ranked
across the scored ETFs (passed to walk_forward only). Book: at the last weekday of each month a new sleeve with
half the capital (top fraction by score, inverse 63-session volatility weights, or 60/40 SPY/IEF as fallback);
the sleeve formed two decisions earlier is closed; the surviving sleeve keeps its drifted internal weights.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

HORIZON = 42            # card ml.horizon_days
RETRAIN_EVERY = 63      # card ml.retrain_every_days
MIN_SCORED = 10         # fewer scored ETFs -> benchmark-mix sleeve
MIN_TOP = 3             # at least 3 names in a sleeve
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 400,
             "subsample": 0.8, "colsample": 1.0}


# ----------------------------------------------------------------------------------------------- features
def shift_down(x, k):
    """x[t - k] at row t (NaN for t < k); k >= 0."""
    out = np.full(x.shape, np.nan)
    if k == 0:
        return x.astype(float).copy()
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def rolling_std_strict(lr, valid, n):
    """Sample std of lr over rows t-n+1..t; NaN unless all n returns in the window are valid."""
    x = np.where(valid, lr, 0.0)
    v = valid.astype(float)
    T = x.shape[0]
    z = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(v, axis=0)])
    out = np.full(x.shape, np.nan)
    if T >= n:
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        cnt = cn[n:] - cn[:-n]
        var = (s2 - s1 * s1 / n) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        out[n - 1:] = np.where(cnt >= n - 0.5, sd, np.nan)
    return out


def rolling_std_min(lr, valid, n, min_count):
    """Sample std over the valid returns among rows t-n+1..t; NaN with fewer than min_count valid returns."""
    x = np.where(valid, lr, 0.0)
    v = valid.astype(float)
    T = x.shape[0]
    z = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(v, axis=0)])
    out = np.full(x.shape, np.nan)
    if T >= n:
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        cnt = cn[n:] - cn[:-n]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / np.maximum(cnt, 1.0)) / np.maximum(cnt - 1.0, 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        out[n - 1:] = np.where(cnt >= min_count - 0.5, sd, np.nan)
    return out


def rolling_max_strict(x, n):
    """max of x over rows t-n+1..t; NaN if any element of the window is NaN or the window is incomplete."""
    out = x.astype(float).copy()
    for k in range(1, n):
        out = np.maximum(out, shift_down(x, k))      # np.maximum propagates NaN
    return out


def compute_features(ret_co, ret_oc, close):
    """Returns (tr_raw, feats dict, bar, vol63). tr_raw: compounded index over all rows (for drift);
    the feature index is NaN before an ETF's first bar."""
    rc = np.nan_to_num(np.asarray(ret_co, dtype=float), nan=0.0)
    ro = np.nan_to_num(np.asarray(ret_oc, dtype=float), nan=0.0)
    daily = (1.0 + rc) * (1.0 + ro)
    tr_raw = np.cumprod(daily, axis=0)
    bar = np.isfinite(np.asarray(close, dtype=float))
    started = np.cumsum(bar, axis=0) > 0                     # on or after the first bar (point-in-time)
    tr = np.where(started, tr_raw, np.nan)

    prev_bar = np.zeros_like(bar)
    prev_bar[1:] = bar[:-1]
    valid = bar & prev_bar & (daily > 0)                    # a return between two consecutive bars
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(valid, np.log(np.where(daily > 0, daily, 1.0)), 0.0)

    sd21 = rolling_std_strict(lr, valid, 21)
    sd126 = rolling_std_strict(lr, valid, 126)
    sd63 = rolling_std_strict(lr, valid, 63)
    sd504 = rolling_std_min(lr, valid, 504, 378)

    with np.errstate(invalid="ignore", divide="ignore"):
        dd = tr / rolling_max_strict(tr, 252) - 1.0
        vr = np.log(sd21 / sd126)
        vlev = np.log(sd63 / sd504)
        r21 = tr / shift_down(tr, 21) - 1.0
        r12_1 = shift_down(tr, 21) / shift_down(tr, 252) - 1.0
    feats = {"dd": dd, "vr": vr, "vlev": vlev, "r21": r21, "r12_1": r12_1}
    return tr_raw, feats, bar, sd63


def scored_mask(feats, bar, universe=None):
    m = bar.copy()
    for v in feats.values():
        m &= np.isfinite(v)
    if universe is not None:
        m &= np.asarray(universe, dtype=bool)
    return m


# ----------------------------------------------------------------------------------------------- calendar
def decision_rows(dates):
    """Rows decided after the close of the month's last trading day, approximated point-in-time as the last
    weekday of the month (np.busday_offset, exchange holidays ignored). If that weekday has no row (holiday),
    the decision is made at the first row of the next month instead (catch-up)."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    mon = d.astype("datetime64[M]")
    is_last = nxt.astype("datetime64[M]") != mon
    dec = is_last.copy()
    starts = np.asarray(period_starts(d, "M"), dtype=bool)
    for t in np.flatnonzero(starts):
        if t == 0:
            continue
        pm = mon[t - 1]
        j = t - 1
        had = False
        while j >= 0 and mon[j] == pm:
            if is_last[j]:
                had = True
                break
            j -= 1
        if not had:                     # the previous month had no decision row (holiday on its last weekday)
            dec[t] = True
    return dec


# ----------------------------------------------------------------------------------------------- book
def select_sleeve(score_row, mask_row, vol_row, top, col_spy, col_ief, min_scored=MIN_SCORED):
    """Internal weights (sum 1) of a new sleeve."""
    N = score_row.shape[0]
    w = np.zeros(N)
    ok = mask_row & np.isfinite(score_row)
    n = int(ok.sum())
    if n < min_scored:
        w[col_spy] = 0.6
        w[col_ief] = 0.4
        return w, False
    k = max(MIN_TOP, int(np.floor(top * n + 0.5)))
    k = min(k, n)
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(-score_row[cand], kind="stable")]
    pick = order[:k]
    inv = 1.0 / vol_row[pick]
    w[pick] = inv / inv.sum()
    return w, True


def build_book(score, mask, vol63, tr_raw, ret_co, dec, top, col_spy, col_ief, min_scored=MIN_SCORED):
    T, N = score.shape
    out = np.full((T, N), np.nan)
    rc = np.nan_to_num(np.asarray(ret_co, dtype=float), nan=0.0)
    prev_w = None
    prev_d = -1
    for d in np.flatnonzero(dec):
        w_new, _ = select_sleeve(score[d], mask[d], vol63[d], top, col_spy, col_ief, min_scored)
        if prev_w is None:
            p = w_new                       # first decision: both halves formed from the same sleeve
        else:
            base = tr_raw[prev_d] * (1.0 + rc[prev_d + 1])      # value at the open the sleeve was bought
            g = tr_raw[d] / base
            val = prev_w * g
            s = val.sum()
            p = val / s if s > 0 else prev_w
        w = 0.5 * p + 0.5 * w_new
        gross = np.abs(w).sum()
        if gross > 1.0:
            w = w / gross
        out[d] = w
        prev_w, prev_d = w_new, d
    return out


# ----------------------------------------------------------------------------------------------- model
def model_scores(data, params):
    tr_raw, feats, bar, vol63 = compute_features(data.ret_co, data.ret_oc, data.close)
    mask = scored_mask(feats, bar, data.universe)
    X = stack(*[cs_zscore(feats[k], mask) for k in ("dd", "vr", "vlev", "r21", "r12_1")])
    y = cs_rank(forward_return(data, HORIZON), mask)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=mask)
    return np.asarray(score, dtype=float), mask, vol63, tr_raw


def target_weights(data, params):
    col_spy = data.col("SPY")
    col_ief = data.col("IEF")
    score, mask, vol63, tr_raw = model_scores(data, params)
    dec = decision_rows(data.dates)
    return build_book(score, mask, vol63, tr_raw, data.ret_co, dec, float(params["top"]), col_spy, col_ief)
