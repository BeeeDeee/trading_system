"""H-0094 mechanism event: after the close of t the stock is in the top (side +1) or bottom (side -1)
leg_fraction of the walk-forward model score among scored LIQ-500 members, i.e. it enters that day's long or
short cohort. Same features, label and walk_forward call as strategy.py (copied: separate file)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
VOL_WIN = 63
DV_SHORT = 5
DV_LONG = 63
EA_WIN = 5
RETRAIN_EVERY = 63
MIN_NAMES = 50
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 1000, "subsample": 0.5,
             "colsample": 1.0}


def _shift(x, k):
    out = np.full_like(x, np.nan if x.dtype.kind == "f" else False)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _trailing_sum(x, w):
    acc = x.astype(np.float64).copy()
    for k in range(1, w):
        if k >= x.shape[0]:
            break
        acc[k:] += x[:-k]
    acc[: min(w - 1, x.shape[0])] = np.nan
    return acc


def _trailing_std(x, w):
    m = _trailing_sum(x, w) / w
    ss = np.zeros_like(m)
    for k in range(0, w):
        if k >= x.shape[0]:
            break
        if k == 0:
            ss += (x - m) ** 2
        else:
            ss[k:] += (x[:-k] - m[k:]) ** 2
    return np.sqrt(ss / (w - 1))


def _features(data, uni):
    r = (1.0 + data.ret_co.astype(np.float64)) * (1.0 + data.ret_oc.astype(np.float64)) - 1.0
    r[~(np.asarray(data.listed, dtype=bool) & np.isfinite(data.close))] = np.nan
    ok = uni & np.isfinite(r)
    cnt = ok.sum(axis=1)
    s = np.where(ok, r, 0.0).sum(axis=1)
    mean = np.full(r.shape[0], np.nan)
    np.divide(s, cnt, out=mean, where=cnt > 0)
    e = r - mean[:, None]
    vol = _trailing_std(r, VOL_WIN)
    vol[~(vol > 0)] = np.nan
    z1 = e / vol
    z5 = _trailing_sum(e, 5) / (vol * np.sqrt(5.0))
    hit = data.extras["ea_reaction_day"] == 1.0
    found = np.zeros(r.shape, dtype=bool)
    ea_z = np.zeros(r.shape, dtype=np.float64)
    for k in range(EA_WIN):
        f = _shift(hit, k)
        sel = f & ~found
        ea_z[sel] = _shift(z1, k)[sel]
        found |= f
    ea5 = np.where(uni, found.astype(np.float64), np.nan)
    dvf = data.dollar_volume.astype(np.float64)
    ratio = (_trailing_sum(dvf, DV_SHORT) / DV_SHORT) / _shift(_trailing_sum(dvf, DV_LONG) / DV_LONG, DV_SHORT)
    dv_ab = np.full(r.shape, np.nan)
    np.log(ratio, out=dv_ab, where=ratio > 0)
    lvol = np.log(vol)
    return stack(cs_zscore(z1, uni), cs_zscore(z5, uni), ea5, cs_zscore(ea_z, uni),
                 cs_zscore(dv_ab, uni), cs_zscore(lvol, uni))


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0094 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    X = _features(data, uni)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY,
                         universe=uni)
    lf = float(params["leg_fraction"])
    cohort = np.nan_to_num(np.asarray(score_weights(score, uni, top=lf, bottom=lf, min_names=MIN_NAMES),
                                      dtype=np.float64), nan=0.0)
    ok = uni & np.asarray(data.tradable, dtype=bool)
    mask = (cohort != 0) & ok
    side = np.where(cohort < 0, -1.0, 1.0)
    return mask, side
