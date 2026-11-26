"""H-0206 Rule 201 short-sale circuit breaker: ridge-ranked restricted droppers vs near misses, 2-session cohorts.

Rules of card.signal.description and where they are implemented:
- r_t = (1+ret_co)(1+ret_oc)-1, g_t = ret_co; members with missing returns are not candidates -> day_returns()
- R_t: (r<=-0.10 or g<=-0.10) and r>=-0.15; N_t: -0.10<r<=-0.10+near_band and g>-0.10 -> candidate_sets()
- six features z-scored across the PIT universe (cs_zscore) -> features()
- label: open t+1 -> open t+3 total return (forward_return, horizon 2), ranked across the universe (cs_rank)
- ridge(alpha), walk_forward, expanding window, refit every 63 rows -> target_weights()
- book: >= 2 scored candidates, long floor(k/2) best, short floor(k/2) worst, 0.15 each, held 2 sessions,
  overlapping cohorts add, gross > 1 scaled down, non-members closed -> book_from_scores()
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"alpha": 10.0, "near_band": 0.015}

THRESH = -0.10        # Rule 201 trigger: 10 % below the previous close
FLOOR = -0.15         # drops beyond -15 % are a different event class (excluded from R)
LOOKBACK = 63         # sessions t-63..t-1 for the dollar-volume median and the realized volatility
HORIZON = 2           # card ml.horizon_days: label and holding period
HOLD = 2              # each cohort is held two sessions (open t+1 -> open t+3)
RETRAIN = 63          # card ml.retrain_every_days
LEG_WEIGHT = 0.15     # weight per name
MAX_GROSS = 1.0


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0206 is a liq_n card: data.universe is required")
    return np.asarray(uni, dtype=bool)


def day_returns(data):
    """r_t, g_t and the validity mask (price on t and t-1, finite returns, tradable on t)."""
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    px = np.isfinite(close) & listed
    prev_px = np.zeros_like(px)
    prev_px[1:] = px[:-1]
    valid = px & prev_px & tradable & np.isfinite(co) & np.isfinite(oc)
    r = np.where(valid, (1.0 + co) * (1.0 + oc) - 1.0, np.nan)
    g = np.where(valid, co, np.nan)
    return r, g, valid


def candidate_sets(r, g, valid, uni, near_band):
    """Restricted set R_t and near-miss set N_t (disjoint), members with valid returns only."""
    ok = valid & uni
    with np.errstate(invalid="ignore"):
        restricted = ok & ((r <= THRESH) | (g <= THRESH)) & (r >= FLOOR)
        near = ok & (r > THRESH) & (r <= THRESH + near_band) & (g > THRESH)
    return restricted, near


def trailing_window_stat(x, ok, mask, n, stat):
    """stat ('median' or 'std') of x over rows t-n..t-1 for cells where mask[t, j] and all n rows are ok.

    Only rows before t are used; cells without a full window are NaN."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T <= n:
        return out
    c = np.zeros((T + 1, N), dtype=np.int64)
    c[1:] = np.cumsum(ok, axis=0)
    cnt = np.zeros((T, N), dtype=np.int64)
    cnt[n:] = c[n:T] - c[0:T - n]           # number of ok rows in t-n..t-1
    cells = mask & (cnt == n)
    cells[:n] = False
    ti, tj = np.nonzero(cells)
    offs = np.arange(-n, 0)
    step = 100000
    for s in range(0, ti.size, step):
        a = ti[s:s + step]
        b = tj[s:s + step]
        win = x[a[:, None] + offs[None, :], b[:, None]]
        if stat == "median":
            v = np.median(win, axis=1)
        else:
            v = np.std(win, axis=1, ddof=1)
        out[a, b] = v
    return out


def features(data, uni):
    """The card's six features (raw, NaN where unknown), in card order."""
    r, g, valid = day_returns(data)
    flag = np.where(valid, ((r <= THRESH) | (g <= THRESH)).astype(np.float64), np.nan)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    dv_ok = np.isfinite(dv) & (dv > 0)
    med = trailing_window_stat(np.where(dv_ok, dv, np.nan), dv_ok, uni, LOOKBACK, "median")
    with np.errstate(divide="ignore", invalid="ignore"):
        logdv = np.where(dv_ok & np.isfinite(med) & (med > 0), np.log(dv / med), np.nan)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    ea = np.where(np.isfinite(ea), ea, 0.0)          # card: NaN treated as 0
    ea = np.where(valid, ea, np.nan)
    tr = np.where(valid, r, np.nan)
    vol = trailing_window_stat(np.where(valid, tr, 0.0), valid, uni, LOOKBACK, "std")
    return [flag, r, g, logdv, ea, vol]


def zscore_feature(f, uni):
    """cs_zscore over the PIT universe; a date on which the feature is constant across the members that have it
    (e.g. no restricted stock, no earnings release) gets z = 0 for those members instead of NaN."""
    z = np.array(cs_zscore(f, uni), dtype=np.float64)
    have = uni & np.isfinite(f)
    hi = np.max(np.where(have, f, -np.inf), axis=1)
    lo = np.min(np.where(have, f, np.inf), axis=1)
    const_row = np.any(have, axis=1) & (hi == lo)
    fix = have & const_row[:, None] & ~np.isfinite(z)
    z[fix] = 0.0
    return z


def book_from_scores(score, cand, uni):
    """Cohort per row: k scored candidates, long floor(k/2) highest, short floor(k/2) lowest, LEG_WEIGHT each;
    held HOLD rows (row t and t+1), overlapping cohorts add, non-members zero, gross scaled to <= MAX_GROSS."""
    T, N = score.shape
    sc = np.where(cand & np.isfinite(score), score, np.nan)
    scored = np.isfinite(sc)
    k = scored.sum(axis=1)
    half = k // 2
    key = np.where(scored, sc, np.inf)                       # unscored last
    order = np.argsort(key, axis=1, kind="stable")           # ascending score
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    open_ = (k >= 2)[:, None] & scored
    short = open_ & (rank < half[:, None])
    long_ = open_ & (rank >= (k - half)[:, None]) & (rank < k[:, None])
    cohort = LEG_WEIGHT * long_.astype(np.float64) - LEG_WEIGHT * short.astype(np.float64)
    w = np.zeros((T, N))
    for lag in range(HOLD):
        w[lag:] += cohort[:T - lag] if lag else cohort
    w = np.where(uni, w, 0.0)
    gross = np.abs(w).sum(axis=1)
    scale = np.where(gross > MAX_GROSS, MAX_GROSS / np.maximum(gross, 1e-300), 1.0)
    return w * scale[:, None]


def model_scores(data, params, uni):
    X = stack(*[zscore_feature(f, uni) for f in features(data, uni)])
    y = cs_rank(forward_return(data, HORIZON), uni)
    return np.asarray(walk_forward(X, y, horizon=HORIZON, model="ridge",
                                   params={"alpha": float(params["alpha"])},
                                   retrain_every=RETRAIN, universe=uni), dtype=np.float64)


def target_weights(data, params):
    uni = _universe(data)
    r, g, valid = day_returns(data)
    restricted, near = candidate_sets(r, g, valid, uni, float(params["near_band"]))
    score = model_scores(data, params, uni)
    return book_from_scores(score, restricted | near, uni)
