"""H-0116 mechanism diagnostic. Same feature, gate, model and leg code as strategy.py (copied, this file is
separate). Event: on a weekly decision date with the gate on, a stock enters the long (short) leg of the model
score (not in the same leg at the previous weekly decision) and implied_rate + implied_dollar + implied_credit
(unstandardized) is positive (negative); side +1 for long-leg entries, -1 for short-leg entries. Only members of
the universe that are tradable on the event row are marked."""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
GBM_PARAMS = {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}
REG_DAYS = 252
MIN_STOCK_OBS = 200
MIN_FACTOR_OBS = 200
SHOCK_DAYS = 10
REALIZED_DAYS = 10
IVOL_DAYS = 63
VIS_DAYS = 63
MIN_NAMES = 50
N_FACTORS = 3


def _as_mask(x, T):
    a = np.asarray(x)
    if a.dtype == bool and a.shape == (T,):
        return a.copy()
    m = np.zeros(T, dtype=bool)
    if a.size:
        m[a.astype(np.int64)] = True
    return m


def _ffill(x):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    ok = np.isfinite(x)
    idx = np.maximum.accumulate(np.where(ok, np.arange(T), 0))
    out = x[idx]
    out[~np.maximum.accumulate(ok)] = np.nan
    return out


def _day_changes(level):
    lv = _ffill(level)
    d = np.full(lv.shape[0], np.nan)
    if lv.shape[0] > 1:
        d[:-1] = lv[1:] - lv[:-1]
    return d


def _excl_cumsum(x):
    x = np.asarray(x, dtype=np.float64)
    C = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    if x.shape[0]:
        np.cumsum(x, axis=0, out=C[1:])
    return C


def _win_prev(C, w):
    T = C.shape[0] - 1
    lo = np.maximum(np.arange(T) - w, 0)
    return C[:T] - C[lo]


def _win_incl(C, w):
    T = C.shape[0] - 1
    lo = np.maximum(np.arange(1, T + 1) - w, 0)
    return C[1:] - C[lo]


def _stock_returns(ret_co, ret_oc, listed, close):
    co = np.asarray(ret_co, dtype=np.float64)
    oc = np.asarray(ret_oc, dtype=np.float64)
    r = (1.0 + co) * (1.0 + oc) - 1.0
    lst = np.asarray(listed, dtype=bool)
    px = np.isfinite(np.asarray(close, dtype=np.float64))
    valid = np.zeros(r.shape, dtype=bool)
    if r.shape[0] > 1:
        valid[1:] = lst[1:] & lst[:-1] & px[1:] & px[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)
    return r, valid


def _market(r, valid, uni):
    m = valid & uni
    n = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def _factor_shock(dF, mkt):
    dF = np.asarray(dF, dtype=np.float64)
    mkt = np.asarray(mkt, dtype=np.float64)
    p = np.isfinite(dF) & np.isfinite(mkt)
    x = np.where(p, mkt, 0.0)
    y = np.where(p, dF, 0.0)
    pf = p.astype(np.float64)
    n = _win_prev(_excl_cumsum(pf), REG_DAYS)
    sx = _win_prev(_excl_cumsum(x), REG_DAYS)
    sy = _win_prev(_excl_cumsum(y), REG_DAYS)
    sxx = _win_prev(_excl_cumsum(x * x), REG_DAYS)
    sxy = _win_prev(_excl_cumsum(x * y), REG_DAYS)
    syy = _win_prev(_excl_cumsum(y * y), REG_DAYS)
    n10 = _win_prev(_excl_cumsum(pf), SHOCK_DAYS)
    sx10 = _win_prev(_excl_cumsum(x), SHOCK_DAYS)
    sy10 = _win_prev(_excl_cumsum(y), SHOCK_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1.0)
        cxx = sxx - sx * sx / nn
        cxy = sxy - sx * sy / nn
        cyy = syy - sy * sy / nn
        ok = (n >= MIN_FACTOR_OBS) & (cxx > 0)
        b = np.where(ok, cxy / np.where(cxx > 0, cxx, 1.0), np.nan)
        a = np.where(ok, (sy - b * sx) / nn, np.nan)
        ssr = np.maximum(cyy - b * cxy, 0.0)
        sd = np.sqrt(ssr / np.maximum(n - 1.0, 1.0))
        shock = np.where(ok, sy10 - a * n10 - b * sx10, np.nan)
        z = np.where(ok & (sd > 0), shock / (sd * np.sqrt(SHOCK_DAYS)), np.nan)
    return a, b, shock, z


def _exposures(r, valid, mkt, dF, a, b, month_mask):
    T, N = r.shape
    betas = [np.full((T, N), np.nan, dtype=np.float32) for _ in range(1 + N_FACTORS)]
    ivol = np.full((T, N), np.nan, dtype=np.float32)
    starts = np.flatnonzero(month_mask)
    ends = np.append(starts[1:], T)
    K = 2 + N_FACTORS
    for m, e in zip(starts, ends):
        m = int(m)
        e = int(e)
        if m < REG_DAYS:
            continue
        am = a[:, m]
        bm = b[:, m]
        if not (np.all(np.isfinite(am)) and np.all(np.isfinite(bm))):
            continue
        s0 = m - REG_DAYS
        mk = mkt[s0:m]
        X = np.empty((REG_DAYS, K))
        X[:, 0] = 1.0
        X[:, 1] = mk
        for f in range(N_FACTORS):
            X[:, 2 + f] = dF[f, s0:m] - am[f] - bm[f] * mk
        rowok = np.isfinite(X).all(axis=1)
        X0 = np.where(rowok[:, None], X, 0.0)
        Wm = valid[s0:m] & rowok[:, None]
        cnt = Wm.sum(axis=0)
        ok = np.flatnonzero(cnt >= MIN_STOCK_OBS)
        if ok.size == 0:
            continue
        Wo = Wm[:, ok]
        XX = (X0[:, :, None] * X0[:, None, :]).reshape(REG_DAYS, K * K)
        XtX = (Wo.T.astype(np.float64) @ XX).reshape(ok.size, K, K)
        R = np.where(Wo, r[s0:m][:, ok], 0.0)
        Xty = R.T @ X0
        coef = np.einsum("kij,kj->ki", np.linalg.pinv(XtX), Xty)
        for i in range(1 + N_FACTORS):
            betas[i][m:e, ok] = coef[:, 1 + i][None, :].astype(np.float32)
        d0 = m - IVOL_DAYS
        d1 = e - 1
        if d1 <= d0:
            continue
        mkd = mkt[d0:d1]
        fit = mkd[:, None] * coef[:, 1][None, :]
        rowok_d = np.isfinite(mkd)
        for f in range(N_FACTORS):
            ef = dF[f, d0:d1] - am[f] - bm[f] * mkd
            rowok_d &= np.isfinite(ef)
            fit = fit + ef[:, None] * coef[:, 2 + f][None, :]
        Vd = valid[d0:d1][:, ok] & rowok_d[:, None]
        res = np.where(Vd, r[d0:d1][:, ok] - fit, 0.0)
        C1 = _excl_cumsum(res)
        C2 = _excl_cumsum(res * res)
        Cn = _excl_cumsum(Vd.astype(np.float64))
        hi = np.arange(m, e) - d0
        lo = hi - IVOL_DAYS
        n = Cn[hi] - Cn[lo]
        s1 = C1[hi] - C1[lo]
        s2 = C2[hi] - C2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
            sd = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
        ivol[m:e, ok] = sd.astype(np.float32)
    return betas, ivol


def _compute(r, valid, uni, dv, levels, month_mask):
    T, N = r.shape
    uni = np.asarray(uni, dtype=bool)
    mkt = _market(r, valid, uni)
    dF = np.vstack([_day_changes(lv) for lv in levels])
    a = np.full((N_FACTORS, T), np.nan)
    b = np.full((N_FACTORS, T), np.nan)
    shock = np.full((N_FACTORS, T), np.nan)
    z = np.full((N_FACTORS, T), np.nan)
    for f in range(N_FACTORS):
        a[f], b[f], shock[f], z[f] = _factor_shock(dF[f], mkt)
    betas, ivol = _exposures(r, valid, mkt, dF, a, b, month_mask)
    beta_mkt = betas[0]
    implied = [(betas[1 + f] * shock[f][:, None].astype(np.float32)) for f in range(N_FACTORS)]
    rr = np.where(valid, r, 0.0)
    okr = valid & (rr > -1.0)
    lr = np.where(okr, np.log1p(np.where(okr, rr, 0.0)), 0.0)
    del rr
    s10 = _win_incl(_excl_cumsum(lr), REALIZED_DAYS)
    n10 = _win_incl(_excl_cumsum(okr.astype(np.float64)), REALIZED_DAYS)
    stock10 = np.where(n10 >= REALIZED_DAYS, np.expm1(s10), np.nan)
    del lr, okr, s10, n10
    mok = np.isfinite(mkt) & (mkt > -1.0)
    ml = np.where(mok, np.log1p(np.where(mok, mkt, 0.0)), 0.0)
    ms = _win_incl(_excl_cumsum(ml), REALIZED_DAYS)
    mn = _win_incl(_excl_cumsum(mok.astype(np.float64)), REALIZED_DAYS)
    mkt10 = np.where(mn >= REALIZED_DAYS, np.expm1(ms), np.nan)
    realized = (stock10 - beta_mkt.astype(np.float64) * mkt10[:, None]).astype(np.float32)
    del stock10
    dvv = np.asarray(dv, dtype=np.float64)
    okd = np.isfinite(dvv)
    sd_ = _win_incl(_excl_cumsum(np.where(okd, dvv, 0.0)), VIS_DAYS)
    nd_ = _win_incl(_excl_cumsum(okd.astype(np.float64)), VIS_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dv = np.where(nd_ > 0, sd_ / np.maximum(nd_, 1.0), np.nan)
        visibility = np.where(mean_dv > 0, np.log(np.where(mean_dv > 0, mean_dv, 1.0)), np.nan).astype(np.float32)
    del sd_, nd_, mean_dv, dvv, okd
    implied_sum = implied[0] + implied[1] + implied[2]
    feats = [implied[0], implied[1], implied[2], realized, ivol, visibility]
    return {"feats": feats, "z": z, "shock": shock, "beta_mkt": beta_mkt, "betas": betas, "implied_sum": implied_sum,
            "mkt": mkt, "dF": dF}


def _gate(z, shock_z):
    az = np.where(np.isfinite(z), np.abs(z), -np.inf)
    return az.max(axis=0) >= float(shock_z)


def _legs(score, uni, beta_mkt, gate, week_rows, top, min_names=MIN_NAMES):
    out = {}
    empty = np.zeros(0, dtype=np.int64)
    for d in week_rows:
        d = int(d)
        if not gate[d]:
            out[d] = (empty, empty)
            continue
        cand = np.flatnonzero(uni[d] & np.isfinite(score[d]) & np.isfinite(beta_mkt[d]))
        n = cand.size
        k = min(int(np.floor(float(top) * n + 1e-9)), n // 2)
        if n < min_names or k < 1:
            out[d] = (empty, empty)
            continue
        order = np.argsort(-np.asarray(score[d, cand], dtype=np.float64), kind="stable")
        out[d] = (cand[order[:k]], cand[order[n - k:]])
    return out


def _inputs(data):
    T = len(data.dates)
    if data.universe is None:
        raise ValueError("H-0116 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    r, valid = _stock_returns(data.ret_co, data.ret_oc, data.listed, data.close)
    s = data.series
    dgs10 = np.asarray(s["fred_macro.DGS10"], dtype=np.float64)
    eur = np.asarray(s["fred_macro.DEXUSEU"], dtype=np.float64)
    baa = np.asarray(s["fred_moody_yields_1d.DBAA"], dtype=np.float64)
    aaa = np.asarray(s["fred_moody_yields_1d.DAAA"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        dollar = -np.log(np.where(eur > 0, eur, np.nan))
    credit = _ffill(baa) - _ffill(aaa)
    levels = (dgs10, dollar, credit)
    month_mask = _as_mask(period_starts(data.dates, "M"), T)
    week_mask = _as_mask(period_starts(data.dates, "W"), T)
    return uni, r, valid, levels, month_mask, week_mask


def _scores(data, feats, gate, uni):
    zs = []
    for i in range(len(feats)):
        zs.append(cs_zscore(feats[i], uni))
        feats[i] = None
    X = stack(*zs)
    del zs
    y = np.array(cs_rank(forward_return(data, HORIZON), uni), dtype=np.float32, copy=True)
    y[~gate] = np.nan
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=dict(GBM_PARAMS),
                         retrain_every=RETRAIN_EVERY, universe=uni, train_days=None, seed=0)
    return np.asarray(score, dtype=np.float64)


def _events_from_legs(legs, week_rows, implied_sum, uni, tradable, T, N):
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev_long = np.zeros(0, dtype=np.int64)
    prev_short = np.zeros(0, dtype=np.int64)
    for d in week_rows:
        d = int(d)
        lo, sh = legs[d]
        new_lo = np.setdiff1d(lo, prev_long)
        new_sh = np.setdiff1d(sh, prev_short)
        el = new_lo[implied_sum[d, new_lo] > 0]
        es = new_sh[implied_sum[d, new_sh] < 0]
        mask[d, el] = True
        side[d, el] = 1.0
        mask[d, es] = True
        side[d, es] = -1.0
        prev_long, prev_short = lo, sh
    mask &= uni & tradable
    side = np.where(mask, side, 0.0)
    return mask, side


def events(data, params):
    uni, r, valid, levels, month_mask, week_mask = _inputs(data)
    T, N = r.shape
    c = _compute(r, valid, uni, data.dollar_volume, levels, month_mask)
    del r, valid
    beta_mkt = c["beta_mkt"]
    implied_sum = c["implied_sum"]
    gate = _gate(c["z"], params["shock_z"])
    feats = c["feats"]
    c = None
    score = _scores(data, feats, gate, uni)
    week_rows = np.flatnonzero(week_mask)
    legs = _legs(score, uni, beta_mkt, gate, week_rows, params["top"])
    tradable = np.asarray(data.tradable, dtype=bool)
    return _events_from_legs(legs, week_rows, implied_sum, uni, tradable, T, N)
