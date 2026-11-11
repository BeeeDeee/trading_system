"""H-0123 Macro-cleaned liquidity concession, learned.

LIQ-500, weekly (first trading session of each ISO week), ridge on five per-stock features built from an
OLS regression of each stock's daily total return on five signal-only ETFs (SPY, IEF, GLD, DBC, UUP),
label = 5-day forward return, beta-neutral long/short book, held until the next weekly decision.
"""
import math

import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

PARAMS = {"beta_days": 252, "top": 0.2}

FACTORS = ("SPY", "IEF", "GLD", "DBC", "UUP")   # signal-only inputs (card data_requirements), never held
HORIZON = 5            # card ml.horizon_days
RETRAIN_EVERY = 63     # card ml.retrain_every_days
MIN_VALID_SHARE = 0.8  # regression window coverage rule
RESID_VOL_DAYS = 63    # idiosyncratic volatility window
SHORT_DAYS = 5         # 5-session move
LONG_DAYS = 21         # 21-session residual move
VOL_RECENT = 5         # volume shock: sessions t-4..t
VOL_PRIOR = 63         # volume shock: sessions t-67..t-5
MIN_NAMES = 20         # fewer scored members: flat
BETA_FLOOR = 0.1       # leg mean SPY beta below this: 0.5 / 0.5
A_MIN, A_MAX = 0.3, 0.7


def daily_returns(data):
    """Daily total return (1 + ret_co)(1 + ret_oc) - 1 and its validity (listed, priced, finite)."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64)) & np.isfinite(r)
    return r, valid


def decision_rows(dates):
    """First trading session of each ISO week (point-in-time: depends only on dates up to the row)."""
    ps = np.asarray(period_starts(dates, "W"))
    if ps.dtype == bool:
        return ps.copy()
    mask = np.zeros(len(dates), dtype=bool)
    mask[ps.astype(np.int64)] = True
    return mask


def regression_features(r_stock, valid_stock, r_etf, valid_etf, dec, beta_days):
    """Per decision row t and stock: OLS with intercept of daily returns on the five factor returns over the
    beta_days sessions ending at t (rows used: stock and all factors valid; >= 80 % of the window required).

    Returns dict of (T, N) float64 arrays (NaN where not scored): beta_spy, xasset5 (feature a),
    resid5 (b), resid21 (c), ivol (d). Factor column order: FACTORS (SPY first).
    """
    T, N = r_stock.shape
    K = r_etf.shape[1]
    out = {k: np.full((T, N), np.nan) for k in ("beta_spy", "xasset5", "resid5", "resid21", "ivol")}
    beta_days = int(beta_days)
    min_obs = int(math.ceil(MIN_VALID_SHARE * beta_days - 1e-9))
    ysafe = np.where(valid_stock, r_stock, 0.0)
    rsum = np.where(np.isfinite(r_stock), r_stock, 0.0)          # summed stock move (0 = no trading)
    xsafe = np.where(valid_etf, r_etf, 0.0)
    etf_ok = valid_etf.all(axis=1)
    for t in np.flatnonzero(dec):
        if t + 1 < beta_days or t + 1 < max(LONG_DAYS, RESID_VOL_DAYS):
            continue
        s = t - beta_days + 1
        Xa = np.empty((beta_days, K + 1))
        Xa[:, 0] = 1.0
        Xa[:, 1:] = xsafe[s:t + 1]
        M = valid_stock[s:t + 1] & etf_ok[s:t + 1, None]
        n_obs = M.sum(axis=0)
        ok = n_obs >= min_obs
        if not ok.any():
            continue
        Mo = M[:, ok].astype(np.float64)
        P = (Xa[:, :, None] * Xa[:, None, :]).reshape(beta_days, (K + 1) * (K + 1))
        A = (Mo.T @ P).reshape(-1, K + 1, K + 1)
        Y = ysafe[s:t + 1][:, ok] * Mo
        b = Y.T @ Xa
        coef = np.einsum("nij,nj->ni", np.linalg.pinv(A), b)     # (n_ok, K + 1): intercept, SPY, IEF, GLD, DBC, UUP
        f5 = xsafe[t - SHORT_DAYS + 1:t + 1].sum(axis=0)          # factor 5-session summed returns
        f21 = xsafe[t - LONG_DAYS + 1:t + 1].sum(axis=0)
        s5 = rsum[t - SHORT_DAYS + 1:t + 1, ok].sum(axis=0)
        s21 = rsum[t - LONG_DAYS + 1:t + 1, ok].sum(axis=0)
        out["beta_spy"][t, ok] = coef[:, 1]
        out["xasset5"][t, ok] = coef[:, 2:] @ f5[1:]             # IEF, GLD, DBC, UUP only
        out["resid5"][t, ok] = s5 - coef[:, 1:] @ f5             # intercept excluded
        out["resid21"][t, ok] = s21 - coef[:, 1:] @ f21
        E = Y[-RESID_VOL_DAYS:] - Xa[-RESID_VOL_DAYS:] @ coef.T  # regression residuals, last 63 sessions
        Mr = Mo[-RESID_VOL_DAYS:]
        nr = Mr.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (E * Mr).sum(axis=0) / nr
            var = (((E - mu) ** 2) * Mr).sum(axis=0) / (nr - 1.0)
        sd = np.where(nr >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
        out["ivol"][t, ok] = sd
    return out


def volume_shock(dollar_volume, dec):
    """log(mean dollar volume over t-4..t / mean over t-67..t-5), on decision rows; means over finite > 0 values."""
    dv = np.asarray(dollar_volume, dtype=np.float64)
    T, N = dv.shape
    good = np.isfinite(dv) & (dv > 0)
    cs = np.zeros((T + 1, N))
    cn = np.zeros((T + 1, N))
    np.cumsum(np.where(good, dv, 0.0), axis=0, out=cs[1:])
    np.cumsum(good.astype(np.float64), axis=0, out=cn[1:])
    out = np.full((T, N), np.nan)
    span = VOL_RECENT + VOL_PRIOR
    for t in np.flatnonzero(dec):
        if t + 1 < span:
            continue
        e = t + 1
        m = e - VOL_RECENT
        b = e - span
        n1 = cn[e] - cn[m]
        n0 = cn[m] - cn[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean1 = (cs[e] - cs[m]) / n1
            mean0 = (cs[m] - cs[b]) / n0
            v = np.log(mean1 / mean0)
        v[(n1 < 1) | (n0 < 1) | ~np.isfinite(v)] = np.nan
        out[t] = v
    return out


def features(data, params):
    """Raw (unscaled) features, decision mask, SPY betas and the universe."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0123 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    r, valid = daily_returns(data)
    cols = [data.col(name) for name in FACTORS]                    # raises if a factor ETF is missing
    dec = decision_rows(data.dates)
    reg = regression_features(r, valid, r[:, cols], valid[:, cols], dec, params["beta_days"])
    vshock = volume_shock(data.dollar_volume, dec)
    scored = np.isfinite(reg["resid5"])
    vshock = np.where(scored, vshock, np.nan)
    raw = [reg["xasset5"], reg["resid5"], reg["resid21"], reg["ivol"], vshock]
    return raw, reg["beta_spy"], dec, uni


def model_scores(data, params):
    raw, beta_spy, dec, uni = features(data, params)
    X = stack(*[cs_zscore(f, uni) for f in raw])
    score = walk_forward(X, forward_return(data, HORIZON), horizon=HORIZON, model="ridge", params={},
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), beta_spy, dec, uni


def legs(score_row, member_row, beta_row, top):
    """Long / short column indices on one decision row (empty arrays = flat)."""
    ok = np.isfinite(score_row) & member_row & np.isfinite(beta_row)
    idx = np.flatnonzero(ok)
    n = idx.size
    empty = np.zeros(0, dtype=np.int64)
    if n < MIN_NAMES:
        return empty, empty
    k = int(math.floor(float(top) * n + 1e-9))
    if k < 1:
        return empty, empty
    order = idx[np.argsort(-score_row[idx], kind="stable")]
    return order[:k], order[n - k:]


def leg_sizes(b_long, b_short):
    """Beta-neutral leg gross: long a, short 1 - a."""
    if not (b_long >= BETA_FLOOR and b_short >= BETA_FLOOR):
        return 0.5, 0.5
    a = min(max(b_short / (b_long + b_short), A_MIN), A_MAX)
    return a, 1.0 - a


def book(score, beta_spy, dec, uni, top):
    T, N = score.shape
    W = np.full((T, N), np.nan)
    held = np.zeros(N)
    for t in range(T):
        if dec[t]:
            row = np.zeros(N)
            lo, sh = legs(score[t], uni[t], beta_spy[t], top)
            if lo.size and sh.size:
                a, s = leg_sizes(float(beta_spy[t, lo].mean()), float(beta_spy[t, sh].mean()))
                row[lo] = a / lo.size
                row[sh] = -s / sh.size
            W[t] = row
            held = row.copy()
        else:
            leave = (held != 0.0) & ~uni[t]                        # left the universe mid-week: close it
            if leave.any():
                W[t, leave] = 0.0
                held[leave] = 0.0
    return W


def target_weights(data, params):
    score, beta_spy, dec, uni = model_scores(data, params)
    return book(score, beta_spy, dec, uni, params["top"])
