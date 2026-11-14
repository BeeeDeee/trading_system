"""H-0154 mechanism events: a member enters the top fifth of the model score at a monthly decision.

Same calendar, features and walk_forward call as strategy.py (copied: this file is separate). An event on
stock j at decision row t: the model has been fitted, at least 50 members are scored (the book is then
score-driven), j is in the top fifth of the scored members at t, j was not in the top fifth at the previous
monthly decision, j is a member and tradable at t. Side +1. Warm-up decisions (no fitted model yet) and
decisions with fewer than 50 scored members have an empty top fifth and mark nothing.
"""
import numpy as np
from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

EVENT_TOP = 0.2              # "top fifth" in the card's event (fixed, not the book's `top`)
HORIZON = 60
RETRAIN_EVERY = 126
BETA_WINDOW = 252
BETA_MIN_PAIRS = 180
RET_WINDOW = 126
DY_RECENT_DAYS = 400
MIN_SCORED = 50
GBM_FIXED = {"num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.05, "subsample": 0.8}
DGS10 = "fred_macro.DGS10"
DGS2 = "fred_macro.DGS2"
DY_FIELD = "sf1_art_divyield"


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    cur_m = d.astype("datetime64[M]")
    is_end = cur_m != nxt.astype("datetime64[M]")
    late = np.zeros(T, dtype=bool)
    late[1:] = (cur_m[1:] != cur_m[:-1]) & ~is_end[:-1]
    return is_end | late


def _rolling_sum(x, w):
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if c.shape[0] > w:
        out[w:] -= c[:-w]
    return out


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    ok = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    return np.where(valid & np.isfinite(r), r, np.nan)


def member_mean(r, uni):
    m = uni & np.isfinite(r)
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def cum_return(m, w):
    m = np.asarray(m, dtype=np.float64)
    ok = np.isfinite(m) & (m > -1.0)
    lg = np.where(ok, np.log1p(np.where(ok, m, 0.0)), 0.0)
    s = _rolling_sum(lg, w)
    n = _rolling_sum(ok.astype(np.float64), w)
    return np.where(n > w - 0.5, np.expm1(s), np.nan)


def level_change(level, k):
    level = np.asarray(level, dtype=np.float64).reshape(-1)
    out = np.full(level.shape[0], np.nan)
    if level.shape[0] > k:
        out[k:] = level[k:] - level[:-k]
    return out


def fill_dividend_yield(dy, dates, recent_days=DY_RECENT_DAYS):
    dy = np.asarray(dy, dtype=np.float64)
    T = dy.shape[0]
    fin = np.isfinite(dy)
    changed = fin.copy()
    if T > 1:
        changed[1:] = fin[1:] & (~fin[:-1] | (dy[1:] != dy[:-1]))
    rows = np.arange(T, dtype=np.int32)[:, None]
    last = np.maximum.accumulate(np.where(changed, rows, -1), axis=0)
    day = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    age = day[:, None] - day[np.maximum(last, 0)]
    recent = (last >= 0) & (age < recent_days)
    return np.where(fin, dy, np.where(recent, 0.0, np.nan))


def rate_beta(r, mkt, level, window=BETA_WINDOW, min_pairs=BETA_MIN_PAIRS, chunk=256):
    r = np.asarray(r, dtype=np.float64)
    T, N = r.shape
    x = level_change(level, 1)
    rel = r - np.asarray(mkt, dtype=np.float64).reshape(-1)[:, None]
    y = np.full((T, N), np.nan)
    if T > 1:
        y[1:] = rel[:-1]
    del rel
    xf = np.isfinite(x)
    out = np.full((T, N), np.nan, dtype=np.float32)
    for c0 in range(0, N, chunk):
        yy = y[:, c0:c0 + chunk]
        ok = np.isfinite(yy) & xf[:, None]
        xb = np.where(ok, np.where(xf, x, 0.0)[:, None], 0.0)
        yb = np.where(ok, yy, 0.0)
        n = _rolling_sum(ok.astype(np.float64), window)
        sx = _rolling_sum(xb, window)
        sy = _rolling_sum(yb, window)
        sxx = _rolling_sum(xb * xb, window)
        sxy = _rolling_sum(xb * yb, window)
        den = n * sxx - sx * sx
        num = n * sxy - sx * sy
        good = (n > min_pairs - 0.5) & (den > 1e-10 * np.maximum(n * sxx, 1e-300))
        out[:, c0:c0 + chunk] = np.where(good, num / np.where(good, den, 1.0), np.nan)
    return out


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0154 needs the point-in-time liq_n universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def build_features(data):
    uni = _universe(data)
    T, N = uni.shape
    lvl10 = np.asarray(data.series[DGS10], dtype=np.float64).reshape(-1)
    lvl2 = np.asarray(data.series[DGS2], dtype=np.float64).reshape(-1)
    dy_raw = data.extras[DY_FIELD]

    r = daily_returns(data)
    mkt = member_mean(r, uni)
    f_rb = cs_zscore(rate_beta(r, mkt, lvl10), uni).astype(np.float32)
    del r
    f_dy = cs_zscore(fill_dividend_yield(dy_raw, data.dates), uni).astype(np.float32)
    tri = total_return_index(data)
    f_r126 = cs_zscore(trailing_return(tri, RET_WINDOW), uni).astype(np.float32)
    del tri

    def date_level(v):
        return np.broadcast_to(np.asarray(v, dtype=np.float32)[:, None], (T, N))

    d10 = level_change(lvl10, RET_WINDOW)
    y2 = lvl2
    mkt126 = cum_return(mkt, RET_WINDOW)
    X = stack(f_dy, f_rb, f_r126, date_level(d10), date_level(y2), date_level(mkt126))
    return X, uni


def model_scores(data, params):
    X, uni = build_features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    del X, y
    score = np.asarray(score, dtype=np.float64)
    return np.where(uni & np.isfinite(score), score, np.nan), uni


def top_names(score_row, member_row, frac):
    ok = member_row & np.isfinite(score_row)
    n = int(ok.sum())
    mask = np.zeros(score_row.shape[0], dtype=bool)
    if n == 0:
        return mask, 0
    k = max(1, int(np.floor(frac * n + 0.5)))
    idx = np.flatnonzero(ok)
    order = np.argsort(-score_row[idx], kind="stable")
    mask[idx[order[:k]]] = True
    return mask, n


def entry_events(score, uni, tradable, decision, frac=EVENT_TOP, min_scored=MIN_SCORED):
    """(T, N) bool: new entries to the top `frac` of scored members at decision rows."""
    score = np.asarray(score, dtype=np.float64)
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    prev_top = np.zeros(N, dtype=bool)
    for t in range(T):
        if not decision[t]:
            continue
        top, n_scored = top_names(score[t], uni[t], frac)
        if n_scored < min_scored:
            top = np.zeros(N, dtype=bool)
        mask[t] = top & ~prev_top & uni[t] & tradable[t]
        prev_top = top
    return mask


def events(data, params):
    score, uni = model_scores(data, params)
    decision = decision_rows(data.dates)
    mask = entry_events(score, uni, data.tradable, decision)
    return mask, None
