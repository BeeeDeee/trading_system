"""H-0143 mechanism event: at a month-end decision date with a fitted model, an ETF is in the top fifth (+1) or
the bottom fifth (-1) of the gbm score among the scored members of its own sleeve (ceil(0.2 n), at least 2 per side).

Same feature, label and walk_forward pipeline as strategy.py (copied: diagnostic.py is a separate file).
"""
import math

import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

RISK = ("SPY", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR", "EFA", "EEM",
        "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWN", "EWP", "EWQ", "EWS", "EWU", "EWW",
        "EWZ", "EWT", "EWY", "FXI", "DBC")
DEFENSIVE = ("SHY", "IEF", "TLT", "TIP", "AGG", "LQD", "HYG", "EMB", "MUB", "GLD")

LONG = 252
SKIP = 21
FORM = 231
RECENT = 21
MID = 63
TOPJ = 5
MIN_RET = 230
MIN_DV = 200
RETRAIN_EVERY = 63
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 0.8}


def _shift(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _rsum(x, w):
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T >= w:
        c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
        out[w - 1:] = c[w:] - c[: T - w + 1]
    return out


def _rstd(x0, v, w):
    n = _rsum(v, w)
    s1 = _rsum(x0, w)
    s2 = _rsum(x0 * x0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    var = np.where(n >= 2, np.maximum(var, 0.0), np.nan)
    return np.sqrt(var)


def _jump_share(r2):
    T, N = r2.shape
    out = np.full((T, N), np.nan)
    if T < LONG:
        return out
    rows = np.arange(LONG - 1, T)
    gidx = (rows - (LONG - 1))[:, None] + np.arange(FORM)[None, :]
    for j in range(N):
        g = r2[gidx, j]
        top = np.partition(g, FORM - TOPJ, axis=1)[:, FORM - TOPJ:].sum(axis=1)
        tot = g.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows, j] = np.where(tot > 0, top / tot, np.nan)
    return out


def _features(r, valid, dv):
    valid = np.asarray(valid, dtype=bool) & np.isfinite(r)
    r0 = np.where(valid, r, 0.0)
    v = valid.astype(float)
    dvv = np.isfinite(dv)
    dv0 = np.where(dvv, dv, 0.0)

    n_ret = _rsum(v, LONG)
    n_dv = _rsum(dvv.astype(float), LONG)
    n_dv21 = _rsum(dvv.astype(float), RECENT)
    sd_long = _rstd(r0, v, LONG)
    sd_mid = _rstd(r0, v, MID)

    with np.errstate(invalid="ignore", divide="ignore"):
        s_form = _shift(_rsum(r0, FORM), SKIP)
        mom = s_form / (sd_long * math.sqrt(FORM))
        n_pos = _shift(_rsum((r0 > 0).astype(float), FORM), SKIP)
        n_neg = _shift(_rsum((r0 < 0).astype(float), FORM), SKIP)
        path = (n_pos - n_neg) / FORM
        jump = _jump_share(r0 * r0)
        recent = _rsum(r0, RECENT) / (sd_mid * math.sqrt(RECENT))
        volratio = sd_mid / sd_long
        m21 = _rsum(dv0, RECENT) / n_dv21
        m252 = _rsum(dv0, LONG) / n_dv
        attention = np.log(np.where((m21 > 0) & (m252 > 0), m21 / m252, np.nan))

    feats = [mom, path, jump, recent, volratio, attention]
    elig = (n_ret >= MIN_RET) & (n_dv >= MIN_DV)
    for f in feats:
        elig &= np.isfinite(f)
    feats = [np.where(elig, f, np.nan) for f in feats]
    return feats, elig


def _log_returns(data):
    r = np.log((1.0 + data.ret_co) * (1.0 + data.ret_oc))
    valid = np.isfinite(data.close) & np.asarray(data.listed, dtype=bool) & np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def _gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def _scores(data, params):
    uni = data.universe
    r, valid = _log_returns(data)
    feats, elig = _features(r, valid, np.asarray(data.dollar_volume, dtype=float))
    if uni is not None:
        elig = elig & np.asarray(uni, dtype=bool)
        feats = [np.where(elig, f, np.nan) for f in feats]
    X = stack(*[cs_zscore(f, uni) for f in feats])
    h = int(params["horizon"])
    y = cs_rank(forward_return(data, h), uni)
    score = walk_forward(X, y, horizon=h, model="gbm", params=_gbm_params(params),
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=float)
    return np.where(elig & np.isfinite(score), score, np.nan)


def _decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    dec = np.zeros(T, dtype=bool)
    if T == 0:
        return dec
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    dec[:] = nxt.astype("datetime64[M]") != month
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    for a in range(1, starts.shape[0]):
        s0, s1 = starts[a - 1], starts[a]
        if not dec[s0:s1].any():
            dec[s1] = True
    return dec


def _top_k(n):
    return max(2, -(-n // 5))


def _sleeve_events(score, dec, sleeves):
    """mask/side for top-fifth (+1) and bottom-fifth (-1) members of each sleeve on decision rows.

    A sleeve with fewer than 4 scored members is skipped (top and bottom sets of at least 2 would overlap).
    """
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for t in np.flatnonzero(dec):
        srow = score[t]
        for cols in sleeves:
            s = srow[cols]
            ok = np.isfinite(s)
            n = int(ok.sum())
            if n < 4:
                continue
            k = _top_k(n)
            cand = cols[ok]
            order = np.argsort(-s[ok], kind="stable")
            top = cand[order[:k]]
            bot = cand[order[n - k:]]
            mask[t, top] = True
            side[t, top] = 1.0
            mask[t, bot] = True
            side[t, bot] = -1.0
    return mask, side


def events(data, params):
    risk = np.array([data.col(x) for x in RISK], dtype=int)
    dfn = np.array([data.col(x) for x in DEFENSIVE], dtype=int)
    score = _scores(data, params)
    dec = _decision_rows(data.dates)
    mask, side = _sleeve_events(score, dec, [risk, dfn])
    ok = np.asarray(data.tradable, dtype=bool)
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)
    mask &= ok
    side = np.where(mask, side, 0.0)
    return mask, side
