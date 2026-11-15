"""H-0162 mechanism event: on a shock-week decision date, a scored LIQ-500 member in the top 10 % of the
walk-forward gbm score (side +1) or the bottom 10 % (side -1). Same pipeline as strategy.py (copied)."""
import numpy as np
from lab.framework.api import total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
CHG_K = 5
Z_WIN = 252
RET_K = 5
TOP = 0.10
MIN_NAMES = 50
WINSOR = 4.0
RETRAIN = 126
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}


def week_end_mask(dates):
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7 == 4


def rate_z(x, k=CHG_K, win=Z_WIN):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.full(x.shape, np.nan)
    pos = np.flatnonzero(np.isfinite(x))
    v = x[pos]
    if v.size < k + win:
        return out
    chg = v[k:] - v[:-k]
    c1 = np.concatenate([[0.0], np.cumsum(chg)])
    c2 = np.concatenate([[0.0], np.cumsum(chg * chg)])
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    var = (s2 - s1 * s1 / win) / (win - 1)
    var = np.where(var > 1e-14, var, np.nan)
    z = chg[win - 1:] / np.sqrt(var)
    out[pos[k + win - 1:]] = z
    return out


def shock_rows(zc, z2, week_end, shock_z):
    ok = week_end & np.isfinite(zc) & np.isfinite(z2)
    with np.errstate(invalid="ignore"):
        big = (np.abs(zc) >= shock_z) | (np.abs(z2) >= shock_z)
    return ok & big


def balance_features(deposits, assets, debtc, debt, intexp, revenue):
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(np.isfinite(assets) & (assets > 0), assets, np.nan)
        dep = np.where(np.isfinite(deposits), deposits, 0.0) / a
        sdebt = debtc / a
        tdebt = debt / a
        ib = np.where(np.isfinite(revenue) & (revenue > 0), np.clip(intexp / revenue, 0.0, 2.0), np.nan)
        ib = np.where(np.isfinite(a), ib, np.nan)
    return dep, sdebt, tdebt, ib


def relative_return(r5, uni):
    valid = uni & np.isfinite(r5)
    s = np.where(valid, r5, 0.0).sum(axis=1)
    n = valid.sum(axis=1)
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return r5 - mean[:, None]


def _cs(x, uni):
    z = np.asarray(cs_zscore(x, uni), dtype=np.float32)
    return np.clip(z, -WINSOR, WINSOR)


def build_features(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    T, N = uni.shape
    zc = rate_z(data.series["fred_macro.T10Y2Y"])
    z2 = rate_z(data.series["fred_macro.DGS2"])
    if zc.shape[0] != T or z2.shape[0] != T:
        raise ValueError("rate series not aligned with the view")
    shock = shock_rows(zc, z2, week_end_mask(data.dates), float(params["shock_z"]))

    ex = data.extras
    dep, sdebt, tdebt, ib = balance_features(
        np.asarray(ex["sf1_arq_deposits"], dtype=np.float64),
        np.asarray(ex["sf1_arq_assets"], dtype=np.float64),
        np.asarray(ex["sf1_arq_debtc"], dtype=np.float64),
        np.asarray(ex["sf1_arq_debt"], dtype=np.float64),
        np.asarray(ex["sf1_art_intexp"], dtype=np.float64),
        np.asarray(ex["sf1_art_revenue"], dtype=np.float64),
    )
    f_dep = _cs(dep, uni); del dep
    f_sd = _cs(sdebt, uni); del sdebt
    f_td = _cs(tdebt, uni); del tdebt
    f_ib = _cs(ib, uni); del ib

    idx = total_return_index(data)
    r5 = np.asarray(trailing_return(idx, RET_K), dtype=np.float64)
    del idx
    f_r5 = _cs(relative_return(r5, uni), uni)
    del r5

    zc_s = np.where(shock, zc, np.nan).astype(np.float32)
    z2_s = np.where(shock, z2, np.nan).astype(np.float32)
    f_zc = np.where(uni, zc_s[:, None], np.float32(np.nan)).astype(np.float32)
    f_z2 = np.where(uni, z2_s[:, None], np.float32(np.nan)).astype(np.float32)
    X = stack(f_dep, f_sd, f_td, f_ib, f_r5, f_zc, f_z2)
    return X, shock


def model_scores(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    X, shock = build_features(data, params)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN, universe=uni)
    return np.asarray(score, dtype=np.float64), shock


def leg_books(score, uni, shock):
    base = np.asarray(score_weights(score, uni, top=TOP, bottom=TOP, min_names=MIN_NAMES), dtype=np.float64)
    base = np.nan_to_num(base, nan=0.0)
    scored = (np.isfinite(score) & uni).sum(axis=1)
    sides = np.sign(base) * uni
    sides[~shock] = 0.0
    sides[scored < MIN_NAMES] = 0.0
    return sides


def events(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    score, shock = model_scores(data, params)
    sides = leg_books(score, uni, shock)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (sides != 0) & uni & tradable
    side = np.where(mask, sides, 0.0)
    return mask, side
