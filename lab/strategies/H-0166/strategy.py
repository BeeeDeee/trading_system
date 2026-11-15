"""H-0166: VIX shocks unwind crowded crypto leverage (learned, gbm).

Top-20 Binance USD-M perps, dollar-neutral 5-day books rebuilt only on VIX-shock dates from walk-forward
LightGBM scores on funding_7d, basis_rel, flush_5d, rv_30d (cross-sectionally z-scored over the scored coins)
and vix_shock, vix_state (date-level, raw).
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "shock_abs": 0.1}

# Fixed by the card (not in signal.params, so not varied by G2).
HORIZON = 5            # label horizon and holding period (crypto rows)
HOLD = 5               # rows the book is held after a rebuild
RETRAIN = 63           # ml.retrain_every_days
NUM_LEAVES = 8
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 100
TOP_N = 20             # crypto_top_n n (only used if the gate passes no universe)
VOL_WINDOW = 30        # trailing quote-volume window of the universe rule
MIN_HISTORY = 30       # rows of price, funding and basis history
MIN_SCORED = 10        # scored coins needed on a shock date
FUND_W = 7
BASIS_W = 30
FLUSH_W = 5
RV_W = 30
VIX_MED_N = 252        # previous usable US sessions in the vix_state median
BTC = "BTCUSDT.P"


# ----------------------------------------------------------------------------------------------- helpers
def _shift(x, k):
    """Rows moved down by k >= 0 (row t holds x[t-k]); NaN where t < k."""
    out = np.full(x.shape, np.nan)
    if k == 0:
        out[:] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _min_count(w):
    return int(np.ceil(2.0 * w / 3.0))


def _roll_mean_std(x, w):
    """Trailing mean and sample std over rows t-w+1..t using the finite values; NaN below 2/3 of the window."""
    T = x.shape[0]
    s1 = np.zeros(x.shape)
    s2 = np.zeros(x.shape)
    n = np.zeros(x.shape)
    for k in range(min(w, T)):
        xs = _shift(x, k)
        f = np.isfinite(xs)
        v = np.where(f, xs, 0.0)
        s1 += v
        s2 += v * v
        n += f
    ok = n >= _min_count(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, s1 / np.maximum(n, 1.0), np.nan)
        var = (s2 - s1 * s1 / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
    std = np.where(ok & (n >= 2), np.sqrt(np.maximum(var, 0.0)), np.nan)
    return mean, std


def _median_of_stack(st, min_count):
    """Median along axis 0 of the finite values (NaN sorted last); NaN below min_count values."""
    cnt = np.isfinite(st).sum(axis=0)
    s = np.sort(st, axis=0)
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.maximum(cnt // 2, 0)
    a = np.take_along_axis(s, lo[None], axis=0)[0]
    b = np.take_along_axis(s, hi[None], axis=0)[0]
    med = 0.5 * (a + b)
    return np.where(cnt >= max(min_count, 1), med, np.nan)


def _roll_median_prev(x, w):
    """Median of rows t-w..t-1 (the previous w rows, today excluded), column blocks to bound memory."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    mc = _min_count(w)
    for c0 in range(0, N, 32):
        xs = x[:, c0:c0 + 32]
        st = np.full((w,) + xs.shape, np.nan)
        for k in range(1, w + 1):
            st[k - 1] = _shift(xs, k)
        out[:, c0:c0 + 32] = _median_of_stack(st, mc)
    return out


def vix_features(v):
    """From the aligned VIX series (row t = latest value usable after the close of t, forward filled):
    new[t]   a NEW observation became usable on row t (value differs from row t-1);
    shock[t] log(latest / previous observation);
    state[t] log(latest / median of the previous 252 observations)."""
    v = np.asarray(v, dtype=float)
    T = v.shape[0]
    fin = np.isfinite(v)
    prev = _shift(v[:, None], 1)[:, 0]
    new = fin & (~np.isfinite(prev) | (v != prev))   # the first usable value is an observation without a shock
    obs = v[new]
    K = obs.shape[0]
    shock_obs = np.full(K, np.nan)
    state_obs = np.full(K, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):   # non-positive values (never in real VIX) give NaN
        if K > 1:
            shock_obs[1:] = np.log(obs[1:] / obs[:-1])
        for k in range(VIX_MED_N, K):
            state_obs[k] = np.log(obs[k] / np.median(obs[k - VIX_MED_N:k]))
    idx = np.cumsum(new) - 1
    has = idx >= 0
    shock = np.full(T, np.nan)
    state = np.full(T, np.nan)
    shock[has] = shock_obs[idx[has]]
    state[has] = state_obs[idx[has]]
    return new, shock, state


def shock_dates(new, shock, shock_abs):
    with np.errstate(invalid="ignore"):
        return new & np.isfinite(shock) & (np.abs(shock) >= float(shock_abs))


def top_n_universe(dollar_volume, listed, n, window):
    """PIT top-n by trailing `window`-row quote volume among listed perps (fallback if data.universe is None)."""
    dv = np.where(np.isfinite(dollar_volume), dollar_volume, 0.0)
    T, N = dv.shape
    tot = np.zeros((T, N))
    for k in range(min(window, T)):
        tot += np.nan_to_num(_shift(dv, k), nan=0.0)
    tot = np.where(listed & (tot > 0), tot, -np.inf)
    order = np.argsort(-tot, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return (rank < n) & np.isfinite(tot)


def build_books(score, rebuild, member, hold):
    """Rebuild rows: long top fifth, short bottom fifth of the finite scores (+0.5/k, -0.5/k, k = round(0.2 c),
    at least 2). Held `hold` rows (the rebuild row included), then flat; a later rebuild replaces the book.
    A held coin that is not a member on a row is closed there for the rest of that book."""
    T, N = score.shape
    w = np.zeros((T, N))
    entries = np.zeros((T, N))
    cur = None
    age = 0
    for t in range(T):
        if rebuild[t]:
            s = score[t]
            cand = np.flatnonzero(np.isfinite(s))
            c = cand.size
            k = max(2, int(np.floor(0.2 * c + 0.5)))
            order = cand[np.argsort(-s[cand], kind="stable")]
            book = np.zeros(N)
            book[order[:k]] = 0.5 / k
            book[order[c - k:]] = -0.5 / k
            cur = book
            age = 0
            entries[t] = np.sign(book)
        elif cur is not None:
            age += 1
            if age >= hold:
                cur = None
        if cur is not None:
            cur[~member[t]] = 0.0
            w[t] = cur
    return w, entries


# ----------------------------------------------------------------------------------------------- pipeline
def _universe(data):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    return top_n_universe(np.asarray(data.dollar_volume, dtype=float), listed, TOP_N, VOL_WINDOW)


def coin_features(data):
    listed = np.asarray(data.listed, dtype=bool)
    funding = np.asarray(data.extras["funding"], dtype=float)
    basis = np.asarray(data.extras["basis"], dtype=float)
    close = np.asarray(data.close, dtype=float)
    tri = np.asarray(total_return_index(data), dtype=float)
    btc = data.col(BTC)

    funding = np.where(listed, funding, np.nan)
    basis = np.where(listed, basis, np.nan)

    # history: rows with a price, funding and basis, counted up to t
    valid = listed & np.isfinite(close) & np.isfinite(funding) & np.isfinite(basis)
    hist = np.cumsum(valid, axis=0)

    # (1) funding_7d
    funding_7d, _ = _roll_mean_std(funding, FUND_W)
    # (2) basis_rel: today minus the median of the previous 30 days
    basis_rel = basis - _roll_median_prev(basis, BASIS_W)
    # daily log returns from the total-return index, only between listed rows
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(tri / _shift(tri, 1))
        lr5 = np.log(tri / _shift(tri, FLUSH_W))
    lr = np.where(listed & (_shift(listed.astype(float), 1) == 1.0), lr, np.nan)
    lr5 = np.where(listed & (_shift(listed.astype(float), FLUSH_W) == 1.0), lr5, np.nan)
    # (3) flush_5d: coin minus BTC (BTC itself 0)
    flush_5d = lr5 - lr5[:, [btc]]
    flush_5d[:, btc] = np.where(np.isfinite(lr5[:, btc]), 0.0, np.nan)
    # (4) rv_30d
    _, rv_30d = _roll_mean_std(lr, RV_W)
    return funding_7d, basis_rel, flush_5d, rv_30d, hist


def model_inputs(data):
    uni = _universe(data)
    T, N = uni.shape
    funding_7d, basis_rel, flush_5d, rv_30d, hist = coin_features(data)
    new, vshock, vstate = vix_features(data.series["fred_macro.VIXCLS"])
    coin = (funding_7d, basis_rel, flush_5d, rv_30d)
    scored = uni & (hist >= MIN_HISTORY) & np.isfinite(vshock)[:, None] & np.isfinite(vstate)[:, None]
    for f in coin:
        scored &= np.isfinite(f)
    feats = []
    for f in coin:
        z = np.asarray(cs_zscore(np.where(scored, f, np.nan), scored), dtype=float)
        feats.append(np.where(scored, z, np.nan))
    for d in (vshock, vstate):                       # date-level, raw
        feats.append(np.where(scored, np.broadcast_to(d[:, None], (T, N)), np.nan))
    return uni, scored, new, vshock, feats


def model_scores(data, params, uni, scored, feats):
    X = stack(*feats)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm",
                         params={"n_estimators": int(params["n_estimators"]), "num_leaves": NUM_LEAVES,
                                 "learning_rate": LEARNING_RATE, "min_child_samples": MIN_CHILD_SAMPLES},
                         retrain_every=RETRAIN, universe=uni)
    score = np.asarray(score, dtype=float)
    return np.where(scored & np.isfinite(score), score, np.nan)


def books(data, params):
    uni, scored, new, vshock, feats = model_inputs(data)
    score = model_scores(data, params, uni, scored, feats)
    n_scored = np.isfinite(score).sum(axis=1)
    rebuild = shock_dates(new, vshock, params["shock_abs"]) & (n_scored >= MIN_SCORED)
    w, entries = build_books(score, rebuild, uni, HOLD)
    return w, entries, uni


def target_weights(data, params):
    w, _, _ = books(data, params)
    return w
