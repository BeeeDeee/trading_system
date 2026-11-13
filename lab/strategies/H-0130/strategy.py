"""H-0130 - TradFi risk shifts re-order the crypto risk ladder, learned.

Binance spot top-20 (point-in-time crypto_top_n universe), long-only daily 5-day cohorts of the top
`top` fraction of a walk-forward LightGBM score on six features:

  coin level (cross-sectionally z-scored over the scored universe members of the date)
    beta60  OLS slope of the coin's daily total return on BTCUSDT's over the last 60 rows (>= 45 pairs)
    ivol60  standard deviation of the residuals of that regression
    liq30   log of the median daily quote volume over the last 30 rows
  date level (identical for every coin, fed raw)
    dvix5   log VIX (latest usable observation) - log VIX five published observations earlier
    dry5    DFII10 (latest usable observation) - DFII10 five published observations earlier (pp)
    lvix    log VIX (latest usable observation)

Label: framework forward_return over 5 rows, ranked across universe members per date.
Book: each date's cohort = equal weight in the top `top` fraction of scored members (cash with fewer
than 10 scored members), 1/5 of the book, held 5 rows; book = sum of the 5 live cohorts; a coin that
leaves the universe is closed at the first decision row outside it.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

FACTOR = "BTCUSDT"            # the beta factor, a universe member, scored like any other coin
VIX_KEY = "fred_macro.VIXCLS"
RY_KEY = "fred_macro.DFII10"

HORIZON = 5                   # label horizon (card ml.horizon_days)
HOLD = 5                      # cohort hold in rows
RETRAIN_EVERY = 63            # card ml.retrain_every_days
BETA_WINDOW = 60
BETA_MIN_PAIRS = 45
LIQ_WINDOW = 30
SHIFT_OBS = 5                 # "five published observations earlier"
MIN_NAMES = 10
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200, "subsample": 0.8}


# ----------------------------------------------------------------------------------------------
# coin-level features
# ----------------------------------------------------------------------------------------------
def daily_returns(ret_co, ret_oc, close, listed):
    """Close-to-close total return of row t; NaN unless the coin has a price on rows t-1 and t."""
    ret_co = np.asarray(ret_co, dtype=float)
    ret_oc = np.asarray(ret_oc, dtype=float)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    ok = np.isfinite(np.asarray(close, dtype=float)) & np.asarray(listed, dtype=bool) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    return np.where(valid, r, np.nan)


def _rolling_sum(x, w):
    """Trailing sum over rows t-w+1..t (fewer rows at the start); x must be NaN-free."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if x.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def beta_ivol(r, f, window=BETA_WINDOW, min_pairs=BETA_MIN_PAIRS):
    """Rolling OLS of r[:, j] on the factor f over the last `window` rows, only rows where both exist.

    Returns (beta, ivol): slope and residual standard deviation (n - 2 degrees of freedom);
    NaN with fewer than `min_pairs` valid pairs or no factor variance.
    """
    r = np.asarray(r, dtype=float)
    x = np.broadcast_to(np.asarray(f, dtype=float)[:, None], r.shape)
    pair = np.isfinite(r) & np.isfinite(x)
    xm = np.where(pair, x, 0.0)
    ym = np.where(pair, r, 0.0)
    n = _rolling_sum(pair.astype(float), window)
    sx = _rolling_sum(xm, window)
    sy = _rolling_sum(ym, window)
    sxx = _rolling_sum(xm * xm, window)
    sxy = _rolling_sum(xm * ym, window)
    syy = _rolling_sum(ym * ym, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / n
        sxy_c = sxy - sx * sy / n
        syy_c = syy - sy * sy / n
        beta = sxy_c / sxx_c
        ssr = np.maximum(syy_c - beta * sxy_c, 0.0)
        ivol = np.sqrt(ssr / (n - 2.0))
    ok = (n >= min_pairs) & (sxx_c > 1e-18)
    return np.where(ok, beta, np.nan), np.where(ok, ivol, np.nan)


def liq30(volume, window=LIQ_WINDOW, chunk=256):
    """log of the median daily quote volume over the last `window` rows (NaN days skipped)."""
    v = np.asarray(volume, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    T = v.shape[0]
    pad = np.full((window - 1,) + v.shape[1:], np.nan)
    vp = np.concatenate([pad, v], axis=0)
    med = np.full(v.shape, np.nan)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        win = np.lib.stride_tricks.sliding_window_view(vp[s:e + window - 1], window, axis=0)
        with np.errstate(all="ignore"):
            allnan = np.all(np.isnan(win), axis=-1)
            safe = np.where(allnan[..., None], 0.0, win)
            m = np.nanmedian(safe, axis=-1)
        med[s:e] = np.where(allnan, np.nan, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(med > 0, np.log(np.where(med > 0, med, 1.0)), np.nan)
    return out


# ----------------------------------------------------------------------------------------------
# date-level TradFi features
# ----------------------------------------------------------------------------------------------
def _new_value(s):
    fin = np.isfinite(s)
    ch = fin.copy()
    ch[1:] = fin[1:] & (~fin[:-1] | (s[1:] != s[:-1]))
    return ch


def publication_rows(vix, ry):
    """Rows on which a new FRED observation arrived: either series took a new value.

    The series are carried over weekends and US holidays, so a row whose values equal the previous
    row's is a carry. Using the union of both series catches days on which one series repeats its
    value (DFII10 often does at two decimals).
    """
    return _new_value(vix) | _new_value(ry)


def tradfi_features(vix, ry, shift=SHIFT_OBS):
    """(dvix5, dry5, lvix) per row from the carried FRED series, point-in-time."""
    vix = np.asarray(vix, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    T = vix.shape[0]
    pub = publication_rows(vix, ry)
    rows = np.flatnonzero(pub)
    k = np.cumsum(pub) - 1                       # index of the latest observation (-1 before the first)
    with np.errstate(divide="ignore", invalid="ignore"):
        lvix = np.where(vix > 0, np.log(np.where(vix > 0, vix, 1.0)), np.nan)

    def lagged(s):
        out = np.full(T, np.nan)
        has = k >= shift
        out[has] = s[rows[k[has] - shift]]
        return out

    dvix5 = lvix - lagged(lvix)
    dry5 = ry - lagged(ry)
    return dvix5, dry5, lvix


# ----------------------------------------------------------------------------------------------
# features, model, book
# ----------------------------------------------------------------------------------------------
def _series(data, key, T):
    s = np.asarray(data.series[key], dtype=float).reshape(-1)
    if s.shape[0] != T:
        raise ValueError(f"series {key} has {s.shape[0]} rows, view has {T}")
    return s


def build_features(data):
    """Returns X (T, N, 6), the scored mask, the universe and the date-level (dvix5, dry5)."""
    if data.universe is None:
        raise ValueError("H-0130 needs the point-in-time crypto_top_n universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    T = len(data.dates)
    r = daily_returns(data.ret_co, data.ret_oc, data.close, data.listed)
    j = data.col(FACTOR)
    beta, ivol = beta_ivol(r, r[:, j])
    btc_ok = np.isfinite(beta[:, j])
    beta[:, j] = np.where(btc_ok, 1.0, np.nan)    # BTC's own beta is 1, its residual 0
    ivol[:, j] = np.where(btc_ok, 0.0, np.nan)
    liq = liq30(data.dollar_volume)

    vix = _series(data, VIX_KEY, T)
    ry = _series(data, RY_KEY, T)
    dvix5, dry5, lvix = tradfi_features(vix, ry)
    date_ok = np.isfinite(dvix5) & np.isfinite(dry5) & np.isfinite(lvix)

    scored = uni & np.isfinite(beta) & np.isfinite(ivol) & np.isfinite(liq) & date_ok[:, None]
    zb = cs_zscore(np.where(scored, beta, np.nan), scored)
    zi = cs_zscore(np.where(scored, ivol, np.nan), scored)
    zl = cs_zscore(np.where(scored, liq, np.nan), scored)

    def bcast(v):
        return np.where(scored, v[:, None], np.nan)

    X = stack(zb, zi, zl, bcast(dvix5), bcast(dry5), bcast(lvix))
    return X, scored, uni, dvix5, dry5


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def model_score(data, X, uni, model, model_params):
    y = cs_rank(forward_return(data, HORIZON), uni)
    return walk_forward(X, y, horizon=HORIZON, model=model, params=model_params,
                        retrain_every=RETRAIN_EVERY, universe=uni)


def cohort_book(cohort, hold=HOLD):
    """Book = (1/hold) x sum of the cohorts formed on the last `hold` rows (rows before 0 are cash)."""
    c = np.nan_to_num(np.asarray(cohort, dtype=float), nan=0.0)
    book = np.zeros_like(c)
    for k in range(hold):
        if k == 0:
            book += c
        elif k < c.shape[0]:
            book[k:] += c[:-k]
    return book / float(hold)


def book_from_scores(score, uni, top, hold=HOLD):
    cohort = score_weights(score, uni, top=float(top), bottom=0.0, min_names=MIN_NAMES)
    w = cohort_book(cohort, hold)
    w[~uni] = 0.0                                 # closed at the first decision row outside the universe
    return w


def target_weights(data, params):
    X, scored, uni, _, _ = build_features(data)
    score = model_score(data, X, uni, "gbm", gbm_params(params))
    return book_from_scores(score, uni, params["top"])
