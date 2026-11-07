"""H-0108 mechanism_test events: on date t, the candidates (bottom tail_q of the 5-session
beta-residual return among LIQ-500 members, no 8-K earnings reaction in t-4..t, current sf1 row,
equity > 0) in the top third of the walk-forward gbm score are marked +1, those in the bottom
third -1 (floor(n/3) per side, at least 2, else no events that date). Same pipeline as
strategy.py (copied verbatim: the diagnostic is a separate file). Events only where the stock is a
universe member and tradable.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
BETA_WINDOW = 252
BETA_MIN_OBS = 200
RES_DAYS = 5
VOL_BASE = 63
TREND_END = 252
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}
MIN_PER_SIDE = 2


def _rsum(x, n):
    c = np.cumsum(x, axis=0, dtype=np.float64)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] -= c[:-n]
    return out


def _total_return(data):
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close))
    rco = np.where(valid & np.isfinite(rco), rco, 0.0)
    roc = np.where(valid & np.isfinite(roc), roc, 0.0)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    return r, rco, valid


def _market(x, mv):
    cnt = mv.sum(axis=1)
    s = np.where(mv, x, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def _beta(r, m, valid):
    v = valid.astype(np.float64)
    n = _rsum(v, BETA_WINDOW)
    mv = m[:, None] * v
    del v
    sx = _rsum(mv, BETA_WINDOW)
    mv *= m[:, None]
    den = n * _rsum(mv, BETA_WINDOW)
    del mv
    den -= sx * sx
    num = n * _rsum(r * m[:, None], BETA_WINDOW)
    num -= sx * _rsum(r, BETA_WINDOW)
    del sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num / den
    ok = (n >= BETA_MIN_OBS) & (den > 1e-18) & np.isfinite(b)
    del num, den, n
    b[~ok] = np.nan
    return b


def _current_mcap(mc, r):
    T, N = mc.shape
    tri = np.cumprod(1.0 + r, axis=0)
    tri_prev = np.empty_like(tri)
    tri_prev[0] = 1.0
    tri_prev[1:] = tri[:-1]
    change = np.ones((T, N), dtype=bool)
    change[1:] = ~(mc[1:] == mc[:-1])
    idx = np.where(change, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    base = np.take_along_axis(tri_prev, idx, axis=0)
    del tri_prev, idx, change
    with np.errstate(invalid="ignore", divide="ignore"):
        cap = mc * (tri / base)
    cap[~np.isfinite(cap) | (cap <= 0)] = np.nan
    return cap


def _row_rank(key):
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    return rank


def _bottom_fraction(x, valid, q):
    key = np.where(valid, x, np.inf)
    rank = _row_rank(key)
    k = np.floor(q * valid.sum(axis=1) + 1e-9).astype(np.int64)
    return valid & (rank < k[:, None])


def compute_inputs(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0108 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    ex = data.extras
    netinc = np.asarray(ex["sf1_art_netinc"])
    equity = np.asarray(ex["sf1_art_equity"])
    mcap_sf1 = np.asarray(ex["sf1_art_marketcap"])
    ea = np.asarray(ex["ea_reaction_day"])
    dvol = np.asarray(data.dollar_volume)
    T = uni.shape[0]
    f32 = np.float32

    r, rco, valid = _total_return(data)
    mem = uni & valid
    m = _market(r, mem)
    m_co = _market(rco, mem)
    del mem
    beta = _beta(r, m, valid)
    v = valid.astype(np.float64)

    r5 = _rsum(r, RES_DAYS)
    mv5 = _rsum(m[:, None] * v, RES_DAYS)
    res5 = r5 - beta * mv5
    res_ok = np.isfinite(res5) & valid
    res5[~res_ok] = np.nan

    on5 = _rsum(rco, RES_DAYS)
    del rco
    on5 -= beta * _rsum(m_co[:, None] * v, RES_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        on5 /= res5
    share = np.clip(on5, -1.0, 2.0).astype(f32)
    del on5

    trend = _rsum(r, TREND_END)
    trend -= r5
    del r5
    tm = _rsum(m[:, None] * v, TREND_END)
    tm -= mv5
    del mv5, v
    trend -= beta * tm
    del tm
    trend[~np.isfinite(trend)] = np.nan
    if T > 0:
        trend[np.arange(T) < TREND_END - 1] = np.nan
    trend = trend.astype(f32)

    cap = _current_mcap(mcap_sf1, r)
    del r
    with np.errstate(invalid="ignore", divide="ignore"):
        ey = (netinc / cap).astype(f32)
        bm = (equity / cap).astype(f32)
        roe = np.where(equity > 0, netinc / equity, np.nan).astype(f32)
    ey[~np.isfinite(ey)] = np.nan
    bm[~np.isfinite(bm)] = np.nan
    roe[~np.isfinite(roe)] = np.nan
    sf1_ok = np.isfinite(netinc) & np.isfinite(equity) & np.isfinite(mcap_sf1) & np.isfinite(cap)
    del cap

    dv_ok = np.isfinite(dvol)
    dv = np.where(dv_ok, dvol, 0.0)
    c = dv_ok.astype(np.float64)
    del dv_ok
    s5, n5 = _rsum(dv, RES_DAYS), _rsum(c, RES_DAYS)
    sL, nL = _rsum(dv, RES_DAYS + VOL_BASE), _rsum(c, RES_DAYS + VOL_BASE)
    del dv, c
    sL -= s5
    nL -= n5
    with np.errstate(invalid="ignore", divide="ignore"):
        vshock = (s5 / n5) / (sL / nL)
    ok = (n5 > 0) & (nL > 0) & (sL > 0) & np.isfinite(vshock)
    del s5, n5, sL, nL
    vshock[~ok] = np.nan
    if T > 0:
        vshock[np.arange(T) < RES_DAYS + VOL_BASE - 1] = np.nan
    vshock = vshock.astype(f32)

    tail = _bottom_fraction(res5, uni & res_ok, float(params["tail_q"]))
    earn = _rsum((ea == 1.0).astype(np.float64), RES_DAYS) > 0.5
    cand = tail & ~earn & sf1_ok & np.isfinite(roe)

    feats = [res5.astype(f32), share, vshock, ey, bm, roe, trend]
    return feats, cand


def model_scores(data, params, feats, cand):
    uni = np.asarray(data.universe, dtype=bool)
    X = stack(*[cs_zscore(f, uni) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), cand)
    hp = dict(GBM_FIXED)
    hp["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=hp,
                         retrain_every=RETRAIN_EVERY, universe=cand)
    score = np.asarray(score, dtype=np.float64)
    return np.where(cand, score, np.nan)


def cohort_legs(score):
    ok = np.isfinite(score)
    n = ok.sum(axis=1)
    k = n // 3
    rank_lo = _row_rank(np.where(ok, score, np.inf))
    rank_hi = _row_rank(np.where(ok, -score, np.inf))
    active = (k >= MIN_PER_SIDE)[:, None]
    long_ = ok & active & (rank_hi < k[:, None])
    short = ok & active & (rank_lo < k[:, None])
    return long_, short


def events(data, params):
    feats, cand = compute_inputs(data, params)
    score = model_scores(data, params, feats, cand)
    del feats
    long_, short = cohort_legs(score)
    ok = np.asarray(data.universe, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    mask = (long_ | short) & ok
    side = np.where(long_, 1.0, np.where(short, -1.0, 0.0))
    return mask, side
