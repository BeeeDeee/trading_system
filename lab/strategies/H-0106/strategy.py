"""H-0106: ETF-arbitrage flow pressure in S&P 500 stocks, learned (LightGBM), dollar-neutral 5-day cohorts.

Decision after the close of t with rows 0..t only, executed at the open of t+1 by the engine.
SPY and the nine original sector SPDRs are signal columns only (never in the universe, never weighted).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.1}

MARKET_ETF = "SPY"
SECTOR_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
CORR_WINDOW = 252        # sessions in the sector-assignment window (ending at the month's first close)
MIN_PAIRS = 189          # minimum valid (stock, ETF) pairs in that window
FLOW_WINDOW = 63         # median dollar volume over t-63..t-1
SUM_WINDOW = 5           # t-4..t
HORIZON = 5              # label: open t+1 -> open t+6
REFIT = 126              # ml.retrain_every_days
COHORTS = 5              # live cohorts (entered t+1, closed at the open of t+6)
LEG_GROSS = 0.1          # gross per leg per cohort
MIN_NAMES = 100          # minimum scored members for a new cohort
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}


# ----------------------------------------------------------------------------------------------
# helpers (pure numpy, point-in-time)
# ----------------------------------------------------------------------------------------------
def _cc_return(data):
    """Close-to-close total return (1 + ret_co) * (1 + ret_oc) - 1; NaN where the instrument has no price."""
    r = (1.0 + np.asarray(data.ret_co, np.float64)) * (1.0 + np.asarray(data.ret_oc, np.float64)) - 1.0
    valid = np.asarray(data.listed, bool) & np.isfinite(np.asarray(data.close, np.float64))
    return np.where(valid, r, np.nan)


def _trailing_sum(x, n):
    """Sum over rows t-n+1..t; NaN if any value is missing or fewer than n rows exist."""
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if T >= n:
        acc = x[n - 1:].copy()
        for j in range(1, n):
            acc = acc + x[n - 1 - j:T - j]
        out[n - 1:] = acc
    return out


def _lagged_median(v, need, window=FLOW_WINDOW, chunk=128):
    """Median of the finite values of v over rows t-window..t-1 (row t excluded).

    NaN before `window` rows of history exist, where the window has no finite value, and where need[t, j]
    is False (computed only where needed, then masked, so the result never depends on later rows)."""
    v = np.asarray(v, np.float64)
    need = np.asarray(need, bool)
    T, M = v.shape
    out = np.full((T, M), np.nan)
    for start in range(window, T, chunk):
        stop = min(T, start + chunk)
        cols = np.nonzero(need[start:stop].any(axis=0))[0]
        if cols.size == 0:
            continue
        sub = v[start - window:stop][:, cols]
        rows = stop - start
        win = np.stack([sub[j:j + rows] for j in range(window)], axis=-1)   # rows t-window+j
        n = np.isfinite(win).sum(axis=-1)
        s = np.sort(win, axis=-1)                                           # NaN sorted last
        lo = np.clip((n - 1) // 2, 0, window - 1)[..., None]
        hi = np.clip(n // 2, 0, window - 1)[..., None]
        med = 0.5 * (np.take_along_axis(s, lo, axis=-1)[..., 0] + np.take_along_axis(s, hi, axis=-1)[..., 0])
        med[n == 0] = np.nan
        out[start:stop, cols] = med
    return np.where(need, out, np.nan)


def _sector_assignment(e, x, starts, window=CORR_WINDOW, min_pairs=MIN_PAIRS):
    """Monthly sector assignment.

    At each month-start row t0 (first session of the month), over rows max(0, t0-window+1)..t0, correlate each
    stock's excess return e[:, i] with each ETF excess return x[:, k] on the rows where both are valid.
    k* = argmax correlation among ETFs with >= min_pairs valid pairs; rho = that correlation; beta = OLS slope
    of e on x_k*. Held from t0 through the row before the next month start. Returns per-row arrays
    kstar (T, N) int (-1 = unassigned), beta (T, N), rho (T, N)."""
    e = np.asarray(e, np.float64)
    x = np.asarray(x, np.float64)
    starts = np.asarray(starts, bool)
    T, N = e.shape
    idx = np.nonzero(starts)[0]
    km = np.full((len(idx) + 1, N), -1, np.int64)            # last slot = "before any month start"
    bm = np.full((len(idx) + 1, N), np.nan)
    rm = np.full((len(idx) + 1, N), np.nan)
    for m, t0 in enumerate(idx):
        lo = max(0, t0 - window + 1)
        E = e[lo:t0 + 1]
        X = x[lo:t0 + 1]
        ve = np.isfinite(E).astype(np.float64)
        E0 = np.where(ve > 0, E, 0.0)
        vx = np.isfinite(X).astype(np.float64)
        X0 = np.where(vx > 0, X, 0.0)
        n = vx.T @ ve
        sx = X0.T @ ve
        sxx = (X0 * X0).T @ ve
        se = vx.T @ E0
        see = vx.T @ (E0 * E0)
        sxe = X0.T @ E0
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = sx / n
            me = se / n
            cov = sxe / n - mx * me
            varx = sxx / n - mx * mx
            vare = see / n - me * me
            rho = cov / np.sqrt(varx * vare)
            beta = cov / varx
        ok = (n >= min_pairs) & (varx > 0) & (vare > 0) & np.isfinite(rho) & np.isfinite(beta)
        score = np.where(ok, rho, -np.inf)
        k = np.argmax(score, axis=0)
        cols = np.arange(N)
        assigned = ok.any(axis=0)
        km[m] = np.where(assigned, k, -1)
        bm[m] = np.where(assigned, beta[k, cols], np.nan)
        rm[m] = np.where(assigned, rho[k, cols], np.nan)
    month = np.cumsum(starts) - 1                             # -1 before the first month start
    month = np.where(month < 0, len(idx), month)
    return km[month], bm[month], rm[month]


def _gather(a, kstar):
    """a: (T, K) per-ETF series; kstar: (T, N) assigned ETF index (-1 none) -> (T, N) a[t, kstar[t, i]]."""
    a = np.asarray(a, np.float64)
    g = np.take_along_axis(a, np.clip(kstar, 0, None), axis=1)
    return np.where(kstar >= 0, g, np.nan)


def _news_flag(ea, ne, n=SUM_WINDOW):
    """1 if ea_reaction_day or nonearn_8k_reaction_day equals 1 on any session t-n+1..t; 0 if both flags are
    known (finite) on all those sessions and none equals 1; NaN otherwise (unknown, not scored)."""
    ea = np.asarray(ea, np.float64)
    ne = np.asarray(ne, np.float64)
    hit = ((ea == 1.0) | (ne == 1.0)).astype(np.float64)
    known = (np.isfinite(ea) & np.isfinite(ne)).astype(np.float64)
    any_hit = _trailing_sum(hit, n)
    all_known = _trailing_sum(known, n)
    return np.where(any_hit > 0, 1.0, np.where(all_known == n, 0.0, np.nan))


def _log_ratio(v, med):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(v / med)
    return np.where(np.isfinite(out), out, np.nan)


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0106 needs the point-in-time S&P 500 universe (data.universe is None)")
    return np.asarray(uni, bool)


def _inputs(data):
    """Raw (unscaled) features and the sector assignment, all from rows <= t."""
    uni = _universe(data)
    r = _cc_return(data)
    r_spy = r[:, data.col(MARKET_ETF)]
    etf_cols = [data.col(k) for k in SECTOR_ETFS]
    x = r[:, etf_cols] - r_spy[:, None]                         # (T, 9) sector excess returns
    e = r - r_spy[:, None]                                      # (T, N) stock excess returns
    del r
    dv = np.asarray(data.dollar_volume, np.float64)
    dv = np.where(dv > 0, dv, np.nan)

    starts = period_starts(data.dates, "M")
    kstar, beta, rho = _sector_assignment(e, x, starts)

    x5 = _trailing_sum(x, SUM_WINDOW)                           # (T, 9)
    xk = _gather(x, kstar)
    xk5 = _gather(x5, kstar)

    dv_etf = dv[:, etf_cols]
    med_etf = _lagged_median(dv_etf, np.ones(dv_etf.shape, bool))
    flow_etf = _log_ratio(dv_etf, med_etf)                      # (T, 9)

    push_1d = beta * xk
    push_5d = beta * xk5
    etf_flow = _gather(flow_etf, kstar)
    idio_5d = _trailing_sum(e, SUM_WINDOW) - beta * xk5
    del e
    own_flow = _log_ratio(dv, _lagged_median(dv, uni))
    own_news = _news_flag(data.extras["ea_reaction_day"], data.extras["nonearn_8k_reaction_day"])
    loading = rho
    feats = {
        "push_1d": push_1d, "push_5d": push_5d, "etf_flow": etf_flow, "idio_5d": idio_5d,
        "own_flow": own_flow, "own_news": own_news, "loading": loading,
    }
    feats = {k: np.where(uni, v, np.nan).astype(np.float32) for k, v in feats.items()}
    return {"uni": uni, "kstar": kstar, "x": x, "dv_etf": dv_etf, "med_etf": med_etf, "feats": feats}


def _feature_matrix(inp):
    """z-score every continuous feature across the members of date t; own_news enters raw as 0/1."""
    uni = inp["uni"]
    f = inp["feats"]
    z = lambda name: cs_zscore(f[name], uni)
    return stack(z("push_1d"), z("push_5d"), z("etf_flow"), z("idio_5d"), z("own_flow"),
                 f["own_news"], z("loading"))


def _model_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def _scores(data, params, inp):
    X = _feature_matrix(inp)
    uni = inp["uni"]
    y = cs_rank(forward_return(data, HORIZON), uni)
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=_model_params(params),
                        retrain_every=REFIT, universe=uni)


def _cohort_book(sw, uni, n_cohorts=COHORTS, leg=LEG_GROSS):
    """Each row's score_weights legs -> a cohort with `leg` gross per leg (equal weight within the leg);
    the book on row t is the sum of the cohorts of rows t-n_cohorts+1..t; non-members at t are closed."""
    sw = np.nan_to_num(np.asarray(sw, np.float64))
    uni = np.asarray(uni, bool)
    lng = sw > 0
    sht = sw < 0
    nl = np.maximum(lng.sum(axis=1, keepdims=True), 1)
    ns = np.maximum(sht.sum(axis=1, keepdims=True), 1)
    cohort = leg * lng / nl - leg * sht / ns
    T = cohort.shape[0]
    book = np.zeros_like(cohort)
    for j in range(n_cohorts):
        if j < T:
            book[j:] += cohort[:T - j]
    return np.where(uni, book, 0.0)


def target_weights(data, params):
    inp = _inputs(data)
    uni = inp["uni"]
    score = _scores(data, params, inp)
    top = float(params["top"])
    sw = score_weights(score, uni, top=top, bottom=top, min_names=MIN_NAMES)
    return _cohort_book(sw, uni)
