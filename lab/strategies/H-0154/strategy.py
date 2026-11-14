"""H-0154 Reaching for income across the rate cycle, learned.

LIQ-500 long-only, three monthly sleeves of 1/3 (each lives about three months), LightGBM (framework gbm)
on dividend yield, rate beta and 6-month return (z-scored across members) plus three raw date-level
inputs: the 126-session change of DGS10, the DGS2 level and the 126-session equal-weight universe return.
"""
import numpy as np
from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

# fixed by the card (not in signal.params)
HORIZON = 60                 # label: open t+1 -> open 60 sessions later, ranked across members
RETRAIN_EVERY = 126          # ml.retrain_every_days, expanding window
BETA_WINDOW = 252            # rate-beta regression window (sessions)
BETA_MIN_PAIRS = 180         # minimum valid pairs for the rate beta
RET_WINDOW = 126             # r126, d10, mkt126 look-back (sessions)
DY_RECENT_DAYS = 400         # a missing dividend yield counts as 0 if an SF1 filing is this recent
MIN_SCORED = 50              # fewer scored members -> sleeve rebuilt as the equal-weight universe
N_SLEEVES = 3
GBM_FIXED = {"num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.05, "subsample": 0.8}
DGS10 = "fred_macro.DGS10"
DGS2 = "fred_macro.DGS2"
DY_FIELD = "sf1_art_divyield"


# ----------------------------------------------------------------------------------------------- calendar
def decision_rows(dates):
    """Last trading day of each month, from the date of row t alone.

    Row t is a month end when the next weekday (np.busday_offset, exchange holidays ignored) falls in
    another month. If that rule misses a month because its last weekday is an exchange holiday (e.g.
    Memorial Day on May 31, Good Friday on March 30), the first row of the next month is the decision
    (one session late); this needs only rows <= t.
    """
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


# ----------------------------------------------------------------------------------------------- helpers
def _rolling_sum(x, w):
    """Sum over the last w rows (rows t-w+1..t) along axis 0; x must have no NaN."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if c.shape[0] > w:
        out[w:] -= c[:-w]
    return out


def daily_returns(data):
    """Close-to-close total return of each stock; NaN unless the stock is listed with a price on both days."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    ok = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    return np.where(valid & np.isfinite(r), r, np.nan)


def member_mean(r, uni):
    """Equal-weight mean daily return of the universe members of each row (NaN if none)."""
    m = uni & np.isfinite(r)
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def cum_return(m, w):
    """Compounded return of the daily series m over the last w rows up to t; NaN unless all w are present."""
    m = np.asarray(m, dtype=np.float64)
    ok = np.isfinite(m) & (m > -1.0)
    lg = np.where(ok, np.log1p(np.where(ok, m, 0.0)), 0.0)
    s = _rolling_sum(lg, w)
    n = _rolling_sum(ok.astype(np.float64), w)
    return np.where(n > w - 0.5, np.expm1(s), np.nan)


def level_change(level, k):
    """level[t] - level[t-k] (NaN for the first k rows)."""
    level = np.asarray(level, dtype=np.float64).reshape(-1)
    out = np.full(level.shape[0], np.nan)
    if level.shape[0] > k:
        out[k:] = level[k:] - level[:-k]
    return out


def fill_dividend_yield(dy, dates, recent_days=DY_RECENT_DAYS):
    """Missing dividend yield -> 0 if the stock has a visible SF1 filing within recent_days, else NaN.

    Only sf1_art_divyield is attached, so a filing is visible as a row where the attached value appears or
    changes (the first session after the filing). A NaN at t is set to 0 when such a row lies less than
    recent_days calendar days before t; otherwise the stock stays unscored.
    """
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
    """OLS slope (with intercept) of (r - mkt) on the daily change of `level`, sessions t-window..t-1.

    `level` is the aligned FRED series: row t holds the last value dated <= t-1, so level[t] - level[t-1]
    is the change of day t-1 and is paired with the stock's relative return of day t-1. At row t the
    window covers rows t-window+1..t, i.e. days t-window..t-1; at least min_pairs valid pairs are needed.
    """
    r = np.asarray(r, dtype=np.float64)
    T, N = r.shape
    x = level_change(level, 1)                       # change of day t-1, stored on row t
    rel = r - np.asarray(mkt, dtype=np.float64).reshape(-1)[:, None]
    y = np.full((T, N), np.nan)
    if T > 1:
        y[1:] = rel[:-1]                             # relative return of day t-1, stored on row t
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


# ----------------------------------------------------------------------------------------------- features
def build_features(data):
    """(T, N, 6) features in card order: dy, rbeta, r126 (z-scored), d10, y2, mkt126 (raw, date-level)."""
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

    d10 = level_change(lvl10, RET_WINDOW)            # DGS10 (last value <= t-1) minus 126 sessions earlier
    y2 = lvl2                                        # DGS2 level, last value <= t-1
    mkt126 = cum_return(mkt, RET_WINDOW)             # 126-session EW member return up to t
    X = stack(f_dy, f_rb, f_r126, date_level(d10), date_level(y2), date_level(mkt126))
    return X, uni


def model_scores(data, params):
    """Point-in-time LightGBM scores (NaN for non-members and before the first fit)."""
    X, uni = build_features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    del X, y
    score = np.asarray(score, dtype=np.float64)
    return np.where(uni & np.isfinite(score), score, np.nan), uni


# ----------------------------------------------------------------------------------------------- book
def top_names(score_row, member_row, frac):
    """Top `frac` (rounded half up, at least 1) of the scored members; ties broken by column order."""
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


def sleeve_book(score, uni, decision, top, min_scored=MIN_SCORED, n_sleeves=N_SLEEVES):
    """Three-sleeve book. All-NaN rows = no decision (keep); decision and exit rows carry the full target
    vector (sum of the sleeves' nominal weights), so an unchanged element keeps its target (the engine
    trades only where targets change).

    - decision row: sleeve (decision count mod 3) is rebuilt with weight 1/3: equal weight in the top `top`
      fraction of scored members, or the equal-weight universe if fewer than min_scored are scored or no
      model has been fitted yet (warm-up). During warm-up, sleeves that hold nothing are also filled with the
      equal-weight universe so the placeholder book is fully invested.
    - any row: a held name that is no longer a member is sold (target 0); its weight stays in cash in its
      sleeve until that sleeve is rebuilt.
    """
    score = np.asarray(score, dtype=np.float64)
    uni = np.asarray(uni, dtype=bool)
    T, N = score.shape
    model_exists = np.logical_or.accumulate(np.isfinite(score).any(axis=1)) if T else np.zeros(0, bool)
    out = np.full((T, N), np.nan)
    sleeves = np.zeros((n_sleeves, N))
    held = np.zeros(N, dtype=bool)
    k_dec = 0
    for t in range(T):
        mem = uni[t]
        if decision[t]:
            s = k_dec % n_sleeves
            k_dec += 1
            n_mem = int(mem.sum())
            ew = mem / (n_sleeves * n_mem) if n_mem > 0 else np.zeros(N)
            if model_exists[t]:
                sel, n_scored = top_names(score[t], mem, top)
                if n_scored >= min_scored:
                    sleeves[s] = sel / (n_sleeves * int(sel.sum()))
                else:
                    sleeves[s] = ew
            else:
                sleeves[s] = ew
                empty = sleeves.sum(axis=1) <= 0.0
                sleeves[empty] = ew
            sleeves[:, ~mem] = 0.0
            tot = sleeves.sum(axis=0)
            out[t] = tot
            held = tot > 0.0
        else:
            gone = held & ~mem
            if gone.any():
                sleeves[:, gone] = 0.0
                out[t] = sleeves.sum(axis=0)
                held = held & ~gone
    return out


def target_weights(data, params):
    score, uni = model_scores(data, params)
    decision = decision_rows(data.dates)
    return sleeve_book(score, uni, decision, float(params["top"]))
