"""H-0108: value-anchored rebound vs value-trap continuation after extreme non-earnings drops.

LIQ-500, learned (LightGBM via the framework walk_forward), five overlapping 5-session cohorts.

Pipeline (decision after the close of row t, fill at the open of t+1):
  r      = daily total return (1 + ret_co)(1 + ret_oc) - 1, 0 on sessions without a price
  m      = equal-weight mean r of the universe members with a price that day (m_co: same for ret_co)
  beta   = 252-session OLS beta of r on m over the stock's valid sessions (>= 200 valid, else NaN)
  res5   = sum_{t-4..t} (r - beta_t * m)
  candidates = members, bottom tail_q of res5 among members with a valid res5, no ea_reaction_day = 1
               in t-4..t, current sf1 row (netinc, equity, marketcap finite), equity > 0
  features (7, cs_zscore over members): res5, overnight share, volume shock, E/P now, B/M now, ROE,
               12-month residual trend
  label  = forward_return(data, 5) ranked across the candidates of each date (cs_rank)
  model  = walk_forward(gbm, expanding, refit every 63 rows, rows = candidate mask of each date)
  book   = per date: top floor(n/3) scored candidates +0.1 split equally, bottom floor(n/3) -0.1,
           only if floor(n/3) >= 2; cohort held for rows t..t+4 (exit at the open of t+6);
           the target of row t is the sum of the cohorts formed on rows t-4..t, zeroed for non-members.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "tail_q": 0.05}

HORIZON = 5                 # ml.horizon_days: label horizon, purge, and cohort hold (sessions)
RETRAIN_EVERY = 63          # ml.retrain_every_days
BETA_WINDOW = 252
BETA_MIN_OBS = 200
RES_DAYS = 5                # residual return window t-4..t
VOL_BASE = 63               # volume base window t-67..t-5
TREND_END = 252             # 12-month residual trend window t-251..t-5
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}
COHORT_SIDE_GROSS = 0.1     # +0.1 longs, -0.1 shorts per cohort
MIN_PER_SIDE = 2


# ----------------------------------------------------------------------------- helpers
def _rsum(x, n):
    """Trailing sum over rows t-n+1..t (partial at the start), float64, point-in-time."""
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
    """Equal-weight mean of x over the mask mv per row (0 where no member has a price)."""
    cnt = mv.sum(axis=1)
    s = np.where(mv, x, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def _beta(r, m, valid):
    """Rolling 252-session OLS beta of r on m over the stock's valid sessions; NaN with < 200."""
    v = valid.astype(np.float64)
    n = _rsum(v, BETA_WINDOW)
    mv = m[:, None] * v
    del v
    sx = _rsum(mv, BETA_WINDOW)
    mv *= m[:, None]
    den = n * _rsum(mv, BETA_WINDOW)              # n * sum(m^2)
    del mv
    den -= sx * sx
    num = n * _rsum(r * m[:, None], BETA_WINDOW)  # n * sum(r m); r is 0 on invalid sessions
    num -= sx * _rsum(r, BETA_WINDOW)
    del sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num / den
    ok = (n >= BETA_MIN_OBS) & (den > 1e-18) & np.isfinite(b)
    del num, den, n
    b[~ok] = np.nan
    return b


def _current_mcap(mc, r):
    """sf1 marketcap times the stock's cumulative total return from the first session the
    current sf1 row is usable (detected as a change of the marketcap value) through t."""
    T, N = mc.shape
    tri = np.cumprod(1.0 + r, axis=0)
    tri_prev = np.empty_like(tri)
    tri_prev[0] = 1.0
    tri_prev[1:] = tri[:-1]
    change = np.ones((T, N), dtype=bool)
    change[1:] = ~(mc[1:] == mc[:-1])            # NaN never equals: NaN rows give NaN anyway
    idx = np.where(change, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    base = np.take_along_axis(tri_prev, idx, axis=0)
    del tri_prev, idx, change
    with np.errstate(invalid="ignore", divide="ignore"):
        cap = mc * (tri / base)
    cap[~np.isfinite(cap) | (cap <= 0)] = np.nan
    return cap


def _row_rank(key):
    """0-based ascending rank per row (stable: ties keep column order)."""
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    return rank


def _bottom_fraction(x, valid, q):
    """Mask of entries in the bottom q fraction (floor(q * n_valid) names) of x among valid per row."""
    key = np.where(valid, x, np.inf)
    rank = _row_rank(key)
    k = np.floor(q * valid.sum(axis=1) + 1e-9).astype(np.int64)
    return valid & (rank < k[:, None])


# ----------------------------------------------------------------------------- signals
def compute_inputs(data, params):
    """Returns (features list of 7 raw (T, N) arrays, candidate mask (T, N) bool)."""
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

    # (1) residual 5-day return, beta of t applied to t-4..t
    r5 = _rsum(r, RES_DAYS)
    mv5 = _rsum(m[:, None] * v, RES_DAYS)
    res5 = r5 - beta * mv5
    res_ok = np.isfinite(res5) & valid
    res5[~res_ok] = np.nan

    # (2) overnight share of the residual drop, clipped to [-1, 2]
    on5 = _rsum(rco, RES_DAYS)
    del rco
    on5 -= beta * _rsum(m_co[:, None] * v, RES_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        on5 /= res5
    share = np.clip(on5, -1.0, 2.0).astype(f32)   # NaN stays NaN
    del on5

    # (7) 12-month residual trend t-251..t-5 (computed here to free r early)
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
        trend[np.arange(T) < TREND_END - 1] = np.nan          # window not complete in the view
    trend = trend.astype(f32)

    # (4)-(6) fundamentals on the current market cap
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

    # (3) volume shock: mean dollar volume t-4..t / mean over t-67..t-5 (sessions with a value)
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
        vshock[np.arange(T) < RES_DAYS + VOL_BASE - 1] = np.nan  # base window not complete in the view
    vshock = vshock.astype(f32)

    # candidates
    tail = _bottom_fraction(res5, uni & res_ok, float(params["tail_q"]))
    earn = _rsum((ea == 1.0).astype(np.float64), RES_DAYS) > 0.5   # NaN (no EDGAR history) is not a flag
    cand = tail & ~earn & sf1_ok & np.isfinite(roe)

    feats = [res5.astype(f32), share, vshock, ey, bm, roe, trend]
    return feats, cand


def model_scores(data, params, feats, cand):
    """Walk-forward gbm scores on candidate rows (NaN elsewhere)."""
    uni = np.asarray(data.universe, dtype=bool)
    X = stack(*[cs_zscore(f, uni) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), cand)       # label ranked across each date's training rows
    hp = dict(GBM_FIXED)
    hp["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=hp,
                         retrain_every=RETRAIN_EVERY, universe=cand)
    score = np.asarray(score, dtype=np.float64)
    return np.where(cand, score, np.nan)


def cohort_legs(score):
    """Per date: long the top floor(n/3), short the bottom floor(n/3) scored candidates
    (n = scored candidates), only if floor(n/3) >= 2. Returns (long, short) bool masks."""
    ok = np.isfinite(score)
    n = ok.sum(axis=1)
    k = n // 3
    rank_lo = _row_rank(np.where(ok, score, np.inf))           # 0 = lowest score
    rank_hi = _row_rank(np.where(ok, -score, np.inf))          # 0 = highest score
    active = (k >= MIN_PER_SIDE)[:, None]
    long_ = ok & active & (rank_hi < k[:, None])
    short = ok & active & (rank_lo < k[:, None])
    return long_, short


def book(long_, short, universe):
    """Sum of the cohorts formed on rows t-4..t, each +0.1 / -0.1 split equally; non-members zeroed."""
    nl = long_.sum(axis=1).astype(np.float64)
    ns = short.sum(axis=1).astype(np.float64)
    wl = np.where(nl > 0, COHORT_SIDE_GROSS / np.maximum(nl, 1.0), 0.0)
    ws = np.where(ns > 0, COHORT_SIDE_GROSS / np.maximum(ns, 1.0), 0.0)
    coh = long_ * wl[:, None] - short * ws[:, None]
    w = coh.copy()
    for j in range(1, HORIZON):                                # cohorts of rows t-1..t-4 (exact sums)
        if j < coh.shape[0]:
            w[j:] += coh[:-j]
    w = np.where(np.asarray(universe, dtype=bool), w, 0.0)
    return w


def target_weights(data, params):
    feats, cand = compute_inputs(data, params)
    score = model_scores(data, params, feats, cand)
    del feats
    long_, short = cohort_legs(score)
    return book(long_, short, data.universe)
