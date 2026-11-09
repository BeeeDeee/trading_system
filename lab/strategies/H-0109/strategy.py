"""H-0109 Disagreement resolution at earnings releases (Miller overpricing), learned.

LIQ-500 long/short 5-session slots: short predicted-release window entrants that a LightGBM ranks low,
long those it ranks high. Decisions after the close of t, executed at the open of t+1.
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

PARAMS = {"num_leaves": 8, "tail": 0.2}

# fixed by the card (not in the G2 grid)
HORIZON = 5            # label: open t+1 -> open t+6
RETRAIN_EVERY = 63     # card ml.retrain_every_days
N_ESTIMATORS = 100
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 200
ANNIV_DAYS = 364       # A(s) = calendar date of s + 364 days
WINDOW_DAYS = 5        # entrant: first session with date >= A(s) - 5, still before A(s)
NO_REACTION = 40       # no reaction session in the 40 sessions up to and including t
TRAIL = 63             # trailing sessions of entrant scores for the quantile thresholds
MIN_POOL = 50          # scored entrants required in the trailing set
SLOT_W = 0.04          # weight of one slot
HOLD = 5               # sessions held: open t+1 .. open t+6
MAX_GROSS = 1.0


# ----------------------------------------------------------------------------- rolling helpers
def _roll_sum_full(x, n):
    """Trailing sum over n rows ending at t; NaN unless all n values are finite (and NaN for t < n-1)."""
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T < n:
        return out
    ok = np.isfinite(x)
    k = np.zeros((T + 1,) + x.shape[1:], dtype=np.int32)
    np.cumsum(ok, axis=0, out=k[1:])
    full = (k[n:] - k[:-n]) == n
    del k
    c = np.zeros((T + 1,) + x.shape[1:])
    np.cumsum(np.where(ok, x, 0.0), axis=0, out=c[1:])
    del ok
    s = c[n:] - c[:-n]
    del c
    s[~full] = np.nan
    out[n - 1:] = s
    return out


def _roll_mean_full(x, n):
    return _roll_sum_full(x, n) / n


def _roll_std_full(x, n):
    """Sample std (ddof 1) over n rows; NaN unless the full window is finite."""
    s1 = _roll_sum_full(x, n)
    s2 = _roll_sum_full(x * x, n)
    var = (s2 - s1 * s1 / n) / (n - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _roll_median_members(x, n, mask, chunk=100000):
    """Trailing median over n rows, computed only where mask[t, j] (NaN elsewhere or if the window has a NaN)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    ti, tj = np.nonzero(mask)
    keep = ti >= n - 1
    ti, tj = ti[keep], tj[keep]
    offs = np.arange(n - 1, -1, -1)
    for a in range(0, len(ti), chunk):
        r = ti[a:a + chunk]
        j = tj[a:a + chunk]
        win = x[r[:, None] - offs[None, :], j[:, None]]
        med = np.median(win, axis=1)
        med[~np.all(np.isfinite(win), axis=1)] = np.nan
        out[r, j] = med
    return out


def _member_mean(x, uni):
    """Equal-weight mean over members with a finite value, per row (NaN if none)."""
    ok = uni & np.isfinite(x)
    cnt = ok.sum(axis=1)
    s = np.where(ok, x, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


# ----------------------------------------------------------------------------- events
def _reaction(ea):
    """ea_reaction_day == 1; NaN counts as no release."""
    return np.nan_to_num(np.asarray(ea, dtype=float), nan=0.0) == 1.0


def _window_candidates(days, react):
    """(T, N) bool: t is the first session with date >= A(s) - 5 and date(t) < A(s) for a past reaction s."""
    T, N = react.shape
    cand = np.zeros((T, N), dtype=bool)
    s, j = np.nonzero(react)
    if len(s) == 0:
        return cand
    anniv = days[s] + ANNIV_DAYS
    r = np.searchsorted(days, anniv - WINDOW_DAYS, side="left")
    ok = r < T
    s, j, anniv, r = s[ok], j[ok], anniv[ok], r[ok]
    ok = (days[r] < anniv) & (r > s)
    cand[r[ok], j[ok]] = True
    return cand


def _recent_reaction(react, n):
    """(T, N) bool: a reaction session in rows t-n+1 .. t."""
    T = react.shape[0]
    c = np.zeros((T + 1,) + react.shape[1:], dtype=np.int64)
    np.cumsum(react, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) + 1 - n, 0)
    return (c[1:] - c[lo]) > 0


def _announcement_risk(react, ex_abs):
    """Mean of ex_abs over the stock's last 4 reaction sessions up to t (>= 2 required), forward-filled."""
    T, N = react.shape
    out = np.full((T, N), np.nan)
    s, j = np.nonzero(react & np.isfinite(ex_abs))
    if len(s) == 0:
        return out
    order = np.lexsort((s, j))
    s, j = s[order], j[order]
    v = ex_abs[s, j]
    tot = v.copy()
    cnt = np.ones(len(v))
    for lag in (1, 2, 3):
        same = np.zeros(len(v), dtype=bool)
        same[lag:] = j[lag:] == j[:-lag]
        add = np.zeros(len(v))
        add[lag:] = v[:-lag]
        tot = tot + np.where(same, add, 0.0)
        cnt = cnt + same
    val = np.where(cnt >= 2, tot / cnt, np.nan)
    has = np.zeros((T, N), dtype=bool)
    grid = np.full((T, N), np.nan)
    has[s, j] = True
    grid[s, j] = val
    last = np.maximum.accumulate(np.where(has, np.arange(T)[:, None], -1), axis=0)
    cols = np.broadcast_to(np.arange(N)[None, :], (T, N))
    filled = grid[np.maximum(last, 0), cols]
    out[last >= 0] = filled[last >= 0]
    return out


# ----------------------------------------------------------------------------- features
def _features(data, uni, react):
    """Six raw features (T, N) for all columns, point-in-time."""
    T, N = len(data.dates), len(data.instruments)
    idx = total_return_index(data)
    prev = np.vstack([np.ones((1, N)), idx[:-1]])
    r = idx / prev - 1.0                               # close-to-close total return of session t
    r[~(np.asarray(data.listed, dtype=bool) & np.isfinite(data.close))] = np.nan
    del idx, prev
    mkt = _member_mean(r, uni)

    # (a) idiosyncratic volatility: std over 63 sessions of r minus the members' EW mean
    ex = r - mkt[:, None]
    f_ivol = _roll_std_full(ex, 63).astype(np.float32)

    # (e) announcement risk: mean |r - mkt| over the last 4 reaction sessions (>= 2)
    f_arisk = _announcement_risk(react, np.abs(ex)).astype(np.float32)
    del ex

    # (c) 21-session run-up minus members' EW mean; (f) 12-1 month return (t-252 .. t-21)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log1p(np.where(r > -1.0, r, np.nan))
    del r
    run21 = np.expm1(_roll_sum_full(lr, 21))
    f_runup = (run21 - _member_mean(run21, uni)[:, None]).astype(np.float32)
    del run21
    m231 = _roll_sum_full(lr, 231)
    del lr
    f_mom = np.full((T, N), np.nan, dtype=np.float32)
    if T > 21:
        f_mom[21:] = np.expm1(m231[:-21])
    del m231

    # (b) abnormal turnover, (d) log 63-session median dollar volume
    dv = np.asarray(data.dollar_volume, dtype=float)
    dv = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
    m21 = _roll_mean_full(dv, 21)
    m252 = _roll_mean_full(dv, 252)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_turn = np.log(m21 / m252).astype(np.float32)
    del m21, m252
    dv = dv.astype(np.float32)
    med = _roll_median_members(dv, 63, uni)
    del dv
    with np.errstate(divide="ignore", invalid="ignore"):
        f_liq = np.log(med).astype(np.float32)
    del med
    return [f_ivol, f_turn, f_runup, f_liq, f_arisk, f_mom]


def _entrants_and_scores(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("liq_n card needs data.universe")
    uni = np.asarray(uni, dtype=bool)
    react = _reaction(data.extras["ea_reaction_day"])
    days = np.asarray(data.dates).astype(np.int64)

    feats = _features(data, uni, react)
    for i in range(len(feats)):                        # z-score each feature across that date's members
        feats[i] = np.asarray(cs_zscore(feats[i], uni), dtype=np.float32)
    X = stack(*feats)
    del feats
    allfin = np.isfinite(X).all(axis=2)

    ent = _window_candidates(days, react) & ~_recent_reaction(react, NO_REACTION) & uni & allfin
    X[~ent] = np.nan                                   # train and score on window-entrant rows only
    y = forward_return(data, HORIZON)
    gbm = {"n_estimators": N_ESTIMATORS, "learning_rate": LEARNING_RATE,
           "min_child_samples": MIN_CHILD_SAMPLES, "num_leaves": int(params["num_leaves"])}
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.where(ent, score, np.nan)
    return uni, ent, score


def _sets(score, tail):
    """(T, N) side: -1 LOW (score <= tail quantile), +1 HIGH (>= 1-tail quantile) of the entrant scores issued
    in the trailing 63 sessions including t; nothing on t unless that set holds >= 50 scores."""
    T, N = score.shape
    side = np.zeros((T, N))
    scored = np.isfinite(score)
    st, sj = np.nonzero(scored)                        # row-major: sorted by t
    sv = score[st, sj]
    off = np.zeros(T + 1, dtype=np.int64)
    np.cumsum(np.bincount(st, minlength=T), out=off[1:])
    for t in np.unique(st):
        pool = sv[off[max(0, t - TRAIL + 1)]:off[t + 1]]
        if len(pool) < MIN_POOL:
            continue
        lo = np.quantile(pool, tail)
        hi = np.quantile(pool, 1.0 - tail)
        js = sj[off[t]:off[t + 1]]
        v = sv[off[t]:off[t + 1]]
        low = v <= lo
        high = v >= hi
        both = low & high
        side[t, js[low & ~both]] = -1.0
        side[t, js[high & ~both]] = 1.0
    return side


def _book(side, uni):
    """Fixed 0.04 slots held 5 sessions; new slots of a day scaled down together to keep gross <= 1;
    a slot closes at the first decision row outside the universe; one slot per stock."""
    T, N = side.shape
    w = np.zeros((T, N))
    aj = np.zeros(0, dtype=np.int64)
    at = np.zeros(0, dtype=np.int64)
    aw = np.zeros(0)
    for t in range(T):
        if len(aj):
            keep = ((t - at) < HOLD) & uni[t, aj]
            aj, at, aw = aj[keep], at[keep], aw[keep]
            w[t, aj] = aw
        gross = float(np.abs(aw).sum())
        new = np.nonzero(side[t] != 0)[0]
        if len(new) and len(aj):
            new = new[~np.isin(new, aj)]
        if len(new) == 0:
            continue
        cap = MAX_GROSS - gross
        size = SLOT_W
        if size * len(new) > cap:
            size = max(cap, 0.0) / len(new) * (1.0 - 1e-9)
        if size <= 0.0:
            continue
        nw = side[t, new] * size
        w[t, new] = nw
        aj = np.concatenate([aj, new])
        at = np.concatenate([at, np.full(len(new), t, dtype=np.int64)])
        aw = np.concatenate([aw, nw])
    return w


def target_weights(data, params):
    uni, ent, score = _entrants_and_scores(data, params)
    side = _sets(score, float(params["tail"]))
    return _book(side, uni)
