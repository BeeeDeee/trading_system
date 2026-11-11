"""H-0123 mechanism event: a stock enters the long (short) leg of the ridge score at a weekly decision date
and was not in that same leg at the previous weekly decision date. Side +1 long-leg entries, -1 short-leg.

The feature, walk_forward and leg code is a copy of strategy.py (diagnostic.py is a separate file).
"""
import math

import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

FACTORS = ("SPY", "IEF", "GLD", "DBC", "UUP")
HORIZON = 5
RETRAIN_EVERY = 63
MIN_VALID_SHARE = 0.8
RESID_VOL_DAYS = 63
SHORT_DAYS = 5
LONG_DAYS = 21
VOL_RECENT = 5
VOL_PRIOR = 63
MIN_NAMES = 20


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64)) & np.isfinite(r)
    return r, valid


def decision_rows(dates):
    ps = np.asarray(period_starts(dates, "W"))
    if ps.dtype == bool:
        return ps.copy()
    mask = np.zeros(len(dates), dtype=bool)
    mask[ps.astype(np.int64)] = True
    return mask


def regression_features(r_stock, valid_stock, r_etf, valid_etf, dec, beta_days):
    T, N = r_stock.shape
    K = r_etf.shape[1]
    out = {k: np.full((T, N), np.nan) for k in ("beta_spy", "xasset5", "resid5", "resid21", "ivol")}
    beta_days = int(beta_days)
    min_obs = int(math.ceil(MIN_VALID_SHARE * beta_days - 1e-9))
    ysafe = np.where(valid_stock, r_stock, 0.0)
    rsum = np.where(np.isfinite(r_stock), r_stock, 0.0)
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
        coef = np.einsum("nij,nj->ni", np.linalg.pinv(A), b)
        f5 = xsafe[t - SHORT_DAYS + 1:t + 1].sum(axis=0)
        f21 = xsafe[t - LONG_DAYS + 1:t + 1].sum(axis=0)
        s5 = rsum[t - SHORT_DAYS + 1:t + 1, ok].sum(axis=0)
        s21 = rsum[t - LONG_DAYS + 1:t + 1, ok].sum(axis=0)
        out["beta_spy"][t, ok] = coef[:, 1]
        out["xasset5"][t, ok] = coef[:, 2:] @ f5[1:]
        out["resid5"][t, ok] = s5 - coef[:, 1:] @ f5
        out["resid21"][t, ok] = s21 - coef[:, 1:] @ f21
        E = Y[-RESID_VOL_DAYS:] - Xa[-RESID_VOL_DAYS:] @ coef.T
        Mr = Mo[-RESID_VOL_DAYS:]
        nr = Mr.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (E * Mr).sum(axis=0) / nr
            var = (((E - mu) ** 2) * Mr).sum(axis=0) / (nr - 1.0)
        sd = np.where(nr >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
        out["ivol"][t, ok] = sd
    return out


def volume_shock(dollar_volume, dec):
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
    uni = data.universe
    if uni is None:
        raise ValueError("H-0123 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    r, valid = daily_returns(data)
    cols = [data.col(name) for name in FACTORS]
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


def entry_events(score, beta_spy, dec, uni, tradable, top):
    """New entries into each leg relative to the previous weekly decision date's leg."""
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    for t in np.flatnonzero(dec):
        lo, sh = legs(score[t], uni[t], beta_spy[t], top)
        cur_long = np.zeros(N, dtype=bool)
        cur_short = np.zeros(N, dtype=bool)
        cur_long[lo] = True
        cur_short[sh] = True
        new_long = cur_long & ~prev_long
        new_short = cur_short & ~prev_short
        ok = uni[t] & tradable[t]
        mask[t] = (new_long | new_short) & ok
        side[t, new_long & ok] = 1.0
        side[t, new_short & ok] = -1.0
        prev_long, prev_short = cur_long, cur_short
    return mask, side


def events(data, params):
    score, beta_spy, dec, uni = model_scores(data, params)
    tradable = np.asarray(data.tradable, dtype=bool)
    return entry_events(score, beta_spy, dec, uni, tradable, params["top"])
