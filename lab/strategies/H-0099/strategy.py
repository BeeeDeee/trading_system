"""H-0099 State-dependent liquidity provision, learned.

LIQ-200 long-only. Weekly (first trading session of each calendar week) top fraction of a walk-forward
LightGBM score on six features: res5, res21x, ivol63, beta126 (cross-sectionally z-scored) and the
date-level states vix_state and mkt21 (not z-scored). Turnover buffer: held names inside the top
2 * top * N ranks are kept (at most K), the rest of the K slots are filled by the best-ranked new names.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

# fixed by the card (not in the G2 grid)
HORIZON = 5                 # label: open t+1 -> open t+6
RETRAIN_EVERY = 126         # ml.retrain_every_days
MIN_NAMES = 50           # fewer scored members -> flat
BETA_WIN = 126
BETA_MIN_OBS = 100
IVOL_WIN = 63
STATE_WIN = 252
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500,
             "subsample": 0.8, "colsample": 1.0}


# ----------------------------------------------------------------------------------------------
# point-in-time helpers (all causal: row t uses rows <= t only)
# ----------------------------------------------------------------------------------------------
def _shift(x, k):
    """Causal shift by k >= 0 rows (row t gets row t-k), NaN padded."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rsum(x, n):
    """Trailing sum over rows t-n+1..t; NaN while fewer than n rows exist. x must be finite."""
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(x, axis=0)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] >= n:
        out[n - 1] = c[n - 1]
        out[n:] = c[n:] - c[:-n]
    return out


def _compound(r, n):
    """Compounded return over rows t-n+1..t; NaN if any of them is NaN or the window is incomplete."""
    g = np.ones(r.shape, dtype=np.float64)
    for k in range(n):
        g = g * (1.0 + _shift(r, k))
    return g - 1.0


def _rolling_median_full(v, n):
    """Median of v over rows t-n+1..t; NaN if the window is incomplete or holds a NaN."""
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if v.shape[0] >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = np.median(win, axis=1)
    return out


def _rolling_std_full(v, n):
    """Sample std (ddof=1) of v over rows t-n+1..t; NaN if the window is incomplete or holds a NaN."""
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if v.shape[0] >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = np.std(win, axis=1, ddof=1)
    return out


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------
def daily_returns(data):
    """Daily total return (1 + ret_co)(1 + ret_oc) - 1, NaN where the stock is not listed."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    return np.where(listed & np.isfinite(r), r, np.nan)


def market_return(r, uni):
    """Equal-weight mean of the members' daily total returns; NaN on a day without a valid member."""
    mem = uni & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def stock_features(r, m):
    """res5, res21x, ivol63, beta126 as (T, N) float64 arrays (NaN where undefined)."""
    T, N = r.shape
    mm = np.broadcast_to(m[:, None], (T, N))
    pv = np.isfinite(r) & np.isfinite(mm)                  # valid (stock, market) pairs
    x = np.where(pv, mm, 0.0)
    y = np.where(pv, r, 0.0)
    w = pv.astype(np.float64)

    # beta126: OLS slope of r_i on m over the 126 sessions to t, >= 100 valid days
    n = _rsum(w, BETA_WIN)
    sx, sy = _rsum(x, BETA_WIN), _rsum(y, BETA_WIN)
    sxy, sxx = _rsum(x * y, BETA_WIN), _rsum(x * x, BETA_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        varx = sxx - sx * sx / n
        beta = np.where((n >= BETA_MIN_OBS) & (varx > 0), cov / varx, np.nan)

    # residual returns with beta126 of date t
    rs5 = _compound(r, 5)
    rm5 = _compound(m, 5)
    res5 = rs5 - beta * rm5[:, None]
    rs21x = _shift(_compound(r, 21), 5)                    # sessions t-25..t-5
    rm21x = _shift(_compound(m, 21), 5)
    res21x = rs21x - beta * rm21x[:, None]

    # ivol63: std (ddof=1) of r_i - beta126_t * m over the valid days of the 63 sessions to t
    n63 = _rsum(w, IVOL_WIN)
    sx63, sy63 = _rsum(x, IVOL_WIN), _rsum(y, IVOL_WIN)
    sxy63, sxx63, syy63 = _rsum(x * y, IVOL_WIN), _rsum(x * x, IVOL_WIN), _rsum(y * y, IVOL_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        sz = sy63 - beta * sx63
        szz = syy63 - 2.0 * beta * sxy63 + beta * beta * sxx63
        var = (szz - sz * sz / n63) / (n63 - 1.0)
        ivol = np.where(n63 >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    ivol = np.where(np.isfinite(beta), ivol, np.nan)
    return res5, res21x, ivol, beta


def vix_state(vix):
    """log(VIX_t / median VIX over the trailing 252 sessions); VIX as aligned by the framework."""
    v = np.asarray(vix, dtype=np.float64).reshape(-1)
    v = np.where(v > 0, v, np.nan)
    med = _rolling_median_full(v, STATE_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(v / med)


def mkt21_state(m):
    """21-session compounded market return / std of 21-session market returns over 252 sessions."""
    m21 = _compound(m, 21)
    sd = _rolling_std_full(m21, STATE_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, m21 / sd, np.nan)


def build_features(data):
    """(T, N, 6) feature stack and the universe mask."""
    if data.universe is None:                              # liq_n card: fail loudly without a universe
        raise ValueError("H-0099 needs the point-in-time LIQ-200 universe")
    uni = np.asarray(data.universe, dtype=bool)
    T, N = uni.shape
    r = daily_returns(data)
    m = market_return(r, uni)
    res5, res21x, ivol, beta = stock_features(r, m)
    vs = vix_state(data.series["fred_macro.VIXCLS"])
    if vs.shape[0] != T:
        raise ValueError("VIXCLS series does not match the calendar")
    ms = mkt21_state(m)
    complete = (uni & np.isfinite(res5) & np.isfinite(res21x) & np.isfinite(ivol) & np.isfinite(beta)
                & np.isfinite(vs)[:, None] & np.isfinite(ms)[:, None])
    feats = [cs_zscore(np.where(complete, f, np.nan), uni) for f in (res5, res21x, ivol, beta)]
    vs_b = np.where(complete, vs[:, None], np.nan)
    ms_b = np.where(complete, ms[:, None], np.nan)
    X = stack(*feats, vs_b, ms_b)
    return X, uni


def model_scores(data, params):
    X, uni = build_features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)       # label ranked across the universe per date
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), uni


# ----------------------------------------------------------------------------------------------
# book
# ----------------------------------------------------------------------------------------------
def round_half_up(x):
    return int(np.floor(x + 0.5))


def build_book(score, uni, decision, top, min_names=MIN_NAMES):
    """Weekly buffered top-K equal-weight book.

    Decision rows: full rows (1/K on the picks, 0 elsewhere). Between decisions: NaN (keep), except a
    held stock that is not a universe member on a row gets 0 there (closed, cash until the next decision).
    """
    T, N = score.shape
    w = np.full((T, N), np.nan, dtype=np.float64)
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        if not decision[t]:
            out = held & ~uni[t]
            if out.any():
                w[t, out] = 0.0
                held = held & ~out
            continue
        row = np.zeros(N, dtype=np.float64)
        scored = uni[t] & np.isfinite(score[t])
        n_sc = int(scored.sum())
        k = round_half_up(top * n_sc)
        if n_sc < min_names or k <= 0:
            w[t] = row
            held = np.zeros(N, dtype=bool)
            continue
        cols = np.flatnonzero(scored)
        order = cols[np.argsort(-score[t, cols], kind="stable")]   # best first, ties by column
        rank = np.empty(N, dtype=np.int64)
        rank[:] = N + 1
        rank[order] = np.arange(1, n_sc + 1)
        keep = [j for j in order if held[j] and rank[j] <= 2.0 * top * n_sc][:k]
        keep_set = set(keep)
        fill = [j for j in order if not held[j] and j not in keep_set][:k - len(keep)]
        picks = np.array(keep + fill, dtype=np.int64)
        row[picks] = 1.0 / k
        w[t] = row
        held = row > 0
    return w


def target_weights(data, params):
    score, uni = model_scores(data, params)
    decision = np.asarray(period_starts(data.dates, "W"), dtype=bool)   # first session of each week
    return build_book(score, uni, decision, float(params["top"]))
