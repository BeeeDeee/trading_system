"""H-0105 Overnight news without a filing, drift or fade, learned.

LIQ-500 market-neutral 5-session slots on peer-residual overnight gaps with no 8-K and no earnings window;
the side of each slot is the LightGBM score against its trailing 252-session median on candidate rows.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"gap_z": 2.5, "n_estimators": 100}

# fixed constants of the card (not in signal.params)
N_PEERS = 10            # peers per stock
PEER_WINDOW = 252       # trailing sessions for the peer correlation
PEER_MIN_COMMON = 200   # common valid sessions per pair
Z_WINDOW = 63           # std window of g and i, t-63..t-1
Z_MIN = 50              # minimum valid values in the std window
PRIOR_WINDOW = 21       # prior peer-residual return, t-21..t-1
RV_WINDOW = 63          # realized volatility window
DV_WINDOW = 63          # median dollar volume window, t-63..t-1
SINCE_CAP = 63          # cap of sessions since the last earnings release
EA_LOOKBACK = 5         # ea_reaction_day == 0 on t-5..t
K8_LOOKBACK = 1         # nonearn_8k_reaction_day == 0 on t-1..t
HORIZON = 5             # label horizon and slot hold
RETRAIN = 63            # refit every 63 rows
MEDIAN_WINDOW = 252     # side median over scored candidates of t-251..t
SLOT_W = 0.03           # weight per slot
HOLD = 5                # sessions a slot is held
HEDGE_N = 100           # most liquid non-slot members in the hedge
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 0.8}
RIDGE_PARAMS = {"alpha": 1.0}


# ----------------------------------------------------------------------------------------------- helpers
def _shift(x, k, fill):
    """x shifted down by k >= 0 rows (row t gets row t-k)."""
    if k == 0:
        return x.copy()
    out = np.full(x.shape, fill, dtype=x.dtype)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _prefix(x):
    """Cumulative sums along rows with a leading zero row."""
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=float)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def _window_stats(x, lo, hi):
    """Count, sum and sum of squares of the finite values of x over rows [lo, hi) (row vectors lo, hi)."""
    v = np.isfinite(x)
    x0 = np.where(v, x, 0.0)
    c = _prefix(v.astype(float))
    s1 = _prefix(x0)
    s2 = _prefix(x0 * x0)
    return c[hi] - c[lo], s1[hi] - s1[lo], s2[hi] - s2[lo]


def _std_from(n, s1, s2, min_n):
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
    return np.where((n >= min_n) & (sd > 0), sd, np.nan)


def _returns(data):
    co = np.asarray(data.ret_co, dtype=float)
    oc = np.asarray(data.ret_oc, dtype=float)
    close = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    fin = np.isfinite(close)
    # a close-to-close return needs a price on t and on t-1
    valid_cc = listed & fin & _shift(fin, 1, False) & np.isfinite(co) & np.isfinite(oc)
    # overnight and intraday components also need an open on t (tradable at the open of t)
    valid_io = valid_cc & tradable
    cc = (1.0 + co) * (1.0 + oc) - 1.0
    return co, oc, cc, valid_io, valid_cc


def _peers(cc, valid_cc, mem, f):
    """Peers formed after the close of row f among the members `mem`: the N_PEERS other members with the
    highest correlation of close-to-close returns over rows f-251..f (pairs need >= 200 common valid rows).
    Returns (stocks with peers, (M', N_PEERS) peer columns)."""
    lo = max(0, f - PEER_WINDOW + 1)
    vb = valid_cc[lo:f + 1][:, mem]
    V = vb.astype(float)
    R = np.where(vb, cc[lo:f + 1][:, mem], 0.0)
    n = V.T @ V
    sx = R.T @ V                    # [i, j] sum of x_i over the rows where i and j are both valid
    sxx = (R * R).T @ V
    sxy = R.T @ R
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx * sx / n
        vy = vx.T
        corr = cov / np.sqrt(vx * vy)
    ok = (n >= PEER_MIN_COMMON) & (vx > 0) & (vy > 0) & np.isfinite(corr)
    np.fill_diagonal(ok, False)
    key = np.where(ok, corr, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")[:, :N_PEERS]
    has = ok.sum(axis=1) >= N_PEERS
    return mem[has], mem[order[has]]


def _residual(x, v, rows, js, P):
    """Own value minus the mean over the valid peers on each row; NaN if own invalid or no valid peer."""
    xp = x[rows][:, P]
    vp = v[rows][:, P]
    cnt = vp.sum(axis=2)
    s = np.where(vp, xp, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = s / cnt
    own = x[rows][:, js]
    ok = v[rows][:, js] & (cnt > 0)
    return np.where(ok, own - pm, np.nan)


def _peer_features(data, co, oc, cc, valid_io, valid_cc):
    """zg, zi (own trailing std) and the prior 21-session peer-residual return, with the peers of the month."""
    T, N = co.shape
    uni = np.asarray(data.universe, dtype=bool)
    starts = np.flatnonzero(np.asarray(period_starts(data.dates, "M"), dtype=bool))
    zg = np.full((T, N), np.nan)
    zi = np.full((T, N), np.nan)
    prior = np.full((T, N), np.nan)
    for m, f in enumerate(starts):
        b = int(starts[m + 1]) if m + 1 < len(starts) else T
        f = int(f)
        mem = np.flatnonzero(uni[f])
        if mem.size < N_PEERS + 1:
            continue
        js, P = _peers(cc, valid_cc, mem, f)
        if js.size == 0:
            continue
        a = max(0, f - Z_WINDOW)
        rows = slice(a, b)
        G = _residual(co, valid_io, rows, js, P)
        I = _residual(oc, valid_io, rows, js, P)
        C = _residual(cc, valid_cc, rows, js, P)
        t = np.arange(f - a, b - a)                       # block-local rows of the month
        lo = np.maximum(t - Z_WINDOW, 0)
        n, s1, s2 = _window_stats(G, lo, t)
        zg[f:b, js] = G[t] / _std_from(n, s1, s2, Z_MIN)
        n, s1, s2 = _window_stats(I, lo, t)
        zi[f:b, js] = I[t] / _std_from(n, s1, s2, Z_MIN)
        lo21 = t - PRIOR_WINDOW
        okp = lo21 >= 0
        n, s1, _ = _window_stats(C, np.maximum(lo21, 0), t)
        prior[f:b, js] = np.where(okp[:, None] & (n == PRIOR_WINDOW), s1, np.nan)
    return zg, zi, prior


def _abnormal_dollar_volume(data, uni):
    """log(dollar_volume_t / median dollar_volume of t-63..t-1) on member rows (>= 50 valid values)."""
    dv = np.asarray(data.dollar_volume, dtype=float)
    T, N = dv.shape
    out = np.full((T, N), np.nan)
    tt, jj = np.nonzero(uni)
    lags = np.arange(DV_WINDOW, 0, -1)
    chunk = 100000
    for c0 in range(0, tt.size, chunk):
        t = tt[c0:c0 + chunk]
        j = jj[c0:c0 + chunk]
        r = t[:, None] - lags[None, :]
        W = dv[np.maximum(r, 0), j[:, None]]
        W = np.where((r >= 0) & np.isfinite(W) & (W >= 0), W, np.nan)
        n = np.isfinite(W).sum(axis=1)
        Ws = np.sort(W, axis=1)                           # NaN last
        k1 = np.maximum((n - 1) // 2, 0)
        k2 = np.maximum(n // 2, 0)
        med = 0.5 * (Ws[np.arange(t.size), k1] + Ws[np.arange(t.size), k2])
        cur = dv[t, j]
        ok = (n >= Z_MIN) & (med > 0) & np.isfinite(cur) & (cur > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t, j] = np.where(ok, np.log(cur / med), np.nan)
    return out


def _realized_vol(cc, valid_cc):
    """Std of daily total returns over the 63 sessions t-62..t (>= 50 valid values)."""
    T, N = cc.shape
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - RV_WINDOW, 0)
    out = np.full((T, N), np.nan, dtype=np.float32)
    for c0 in range(0, N, 256):                          # column chunks keep the memory small
        x = np.where(valid_cc[:, c0:c0 + 256], cc[:, c0:c0 + 256], np.nan)
        n, s1, s2 = _window_stats(x, lo, hi)
        out[:, c0:c0 + 256] = _std_from(n, s1, s2, Z_MIN)
    return out


def _since_release(ea, start_row):
    """Sessions since the last ea_reaction_day == 1, capped at 63; NaN where it is not knowable."""
    T, N = ea.shape
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(ea == 1, rows, -1), axis=0)
    finite = np.isfinite(ea)
    first = np.where(finite.any(axis=0), np.argmax(finite, axis=0), T)[None, :]
    known_from = np.maximum(first, start_row)           # no 8-K data before the dataset starts
    since = np.where(last >= 0, rows - last, np.where(rows - known_from >= SINCE_CAP, SINCE_CAP, np.nan))
    since = np.minimum(since, SINCE_CAP)
    return np.where(finite, since, np.nan)


def _flags_ok(flag, lookback):
    """flag == 0 (not NaN) on every row t-lookback..t, and the dataset (first flag == 1 anywhere) has started
    by t-lookback."""
    T = flag.shape[0]
    bad = ~(flag == 0)
    nb = _prefix(bad.astype(float))
    hi = np.arange(1, T + 1)
    lo = hi - (lookback + 1)
    okwin = (lo >= 0)[:, None] & ((nb[hi] - nb[np.maximum(lo, 0)]) == 0)
    started = np.maximum.accumulate((flag == 1).any(axis=1))
    started_lb = _shift(started, lookback, False)
    first = np.flatnonzero(started)
    start_row = int(first[0]) if first.size else T
    return okwin & started_lb[:, None], start_row


def _zscore(x, uni):
    """Framework cross-sectional z-score over the members; a known value on a date where all members share it
    (zero dispersion, e.g. sp500_member all equal) is 0 instead of NaN: it is known, not missing."""
    z = cs_zscore(x, uni)
    fin = uni & np.isfinite(x)
    hi = np.where(fin, x, -np.inf).max(axis=1)
    lo = np.where(fin, x, np.inf).min(axis=1)
    flat = (hi == lo)[:, None]
    return np.where(fin & flat & ~np.isfinite(z), 0.0, z).astype(np.float32)


def build_inputs(data, params):
    """Candidate mask, features (NaN off candidates) and the ranked label, all as the card defines them."""
    uni = np.asarray(data.universe, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=float)
    k8 = np.asarray(data.extras["nonearn_8k_reaction_day"], dtype=float)
    sp = np.asarray(data.extras["sp500_member"], dtype=float)
    co, oc, cc, valid_io, valid_cc = _returns(data)
    zg, zi, prior = _peer_features(data, co, oc, cc, valid_io, valid_cc)
    with np.errstate(invalid="ignore"):
        big = np.abs(zg) >= float(params["gap_z"])
    # feature order: zg, zi, abnormal dollar volume, prior 21-session residual, rv63, sp500_member, since release
    f_zg = _zscore(zg, uni)
    del zg
    f_zi = _zscore(zi, uni)
    del zi
    f_prior = _zscore(prior, uni)
    del prior
    del co, oc
    f_rv = _zscore(_realized_vol(cc, valid_cc), uni)
    del cc, valid_cc, valid_io
    f_adv = _zscore(_abnormal_dollar_volume(data, uni), uni)
    f_sp = _zscore(sp, uni)
    ea_ok, ea_start = _flags_ok(ea, EA_LOOKBACK)
    k8_ok, _ = _flags_ok(k8, K8_LOOKBACK)
    f_since = _zscore(_since_release(ea, ea_start), uni)
    feats = [f_zg, f_zi, f_adv, f_prior, f_rv, f_sp, f_since]
    complete = np.ones(uni.shape, dtype=bool)
    for z in feats:
        complete &= np.isfinite(z)
    cand = uni & big & ea_ok & k8_ok & complete
    X = stack(*feats)
    X[~cand] = np.nan
    y = cs_rank(forward_return(data, HORIZON), uni)
    return cand, X, y


def model_scores(data, X, y, model, model_params):
    return walk_forward(X, y, horizon=HORIZON, model=model, params=model_params, retrain_every=RETRAIN,
                        universe=np.asarray(data.universe, dtype=bool))


def gbm_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def sides(score, cand):
    """+1 if the score is above the median of the model's scores on all scored candidate rows of t-251..t,
    else -1; NaN off scored candidates."""
    sc = cand & np.isfinite(score)
    out = np.full(score.shape, np.nan)
    rows, cols = np.nonzero(sc)                         # row-major: sorted by row
    if rows.size == 0:
        return out, sc
    vals = np.asarray(score, dtype=float)[rows, cols]
    urows = np.unique(rows)
    lo = np.searchsorted(rows, urows - (MEDIAN_WINDOW - 1), side="left")
    st = np.searchsorted(rows, urows, side="left")
    hi = np.searchsorted(rows, urows, side="right")
    for a, s, h in zip(lo, st, hi):
        med = np.median(vals[a:h])
        out[rows[s:h], cols[s:h]] = np.where(vals[s:h] > med, 1.0, -1.0)
    return out, sc


def slot_book(side, sc, uni, liq_rank):
    """5-session slots of 0.03 x side, closed early when the stock leaves the universe; net hedged by the 100
    most liquid members without a slot; everything scaled down when the gross exceeds 1."""
    T, N = uni.shape
    s0 = np.where(sc, side, 0.0)
    rows = np.arange(T)[:, None]
    last_out = np.maximum.accumulate(np.where(~uni, rows, -1), axis=0)
    run = rows - last_out                                # consecutive member rows ending at t
    units = np.zeros((T, N), dtype=np.int16)             # sum of the sides of the open slots
    nslot = np.zeros((T, N), dtype=np.int16)             # number of open slots
    for k in range(HOLD):
        sk = _shift(s0, k, 0.0).astype(np.int16)
        alive = run >= k + 1
        units += sk * alive
        nslot += ((sk != 0) & alive).astype(np.int16)
    w = SLOT_W * units.astype(float)
    net = SLOT_W * units.sum(axis=1, dtype=np.int64).astype(float)   # exact: no residue when slots cancel
    lr = np.asarray(liq_rank, dtype=float)
    for t in np.flatnonzero(net != 0):
        key = np.where(uni[t] & (nslot[t] == 0) & np.isfinite(lr[t]), lr[t], np.inf)
        order = np.argsort(key, kind="stable")[:HEDGE_N]
        order = order[np.isfinite(key[order])]
        if order.size:
            w[t, order] = -net[t] / order.size
    gross = np.abs(w).sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / np.where(gross > 0, gross, 1.0), 1.0)
    return w * scale[:, None]


def target_weights(data, params):
    cand, X, y = build_inputs(data, params)
    score = model_scores(data, X, y, "gbm", gbm_params(params))
    del X, y
    side, sc = sides(score, cand)
    del score
    return slot_book(side, sc, np.asarray(data.universe, dtype=bool), data.extras["liq_rank"])
