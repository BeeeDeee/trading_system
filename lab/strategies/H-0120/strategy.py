"""H-0120: session origin of coin-specific moves, learned (Binance top-20 perps, dollar-neutral gbm book).

Features per coin and day t (rows 0..t only), each cross-sectionally z-scored over the universe members of t:
  1-3  sum over t-2..t of (coin session return - BTCUSDT session return), sessions asia / europe / us
  4    mean vshare_us over t-2..t minus mean over t-31..t-3
  5    mean rv over t-6..t divided by mean rv over t-59..t-7
  6    sum of the daily funding extra over t-2..t
Label: 5-day forward return (framework), ranked across the universe. Model gbm through walk_forward,
refit every 63 rows. Book every fifth calendar day since the panel start: long top `top`, short bottom `top`.
"""
import numpy as np
from lab.framework.api import only_on
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

HORIZON = 5            # ml.horizon_days
RETRAIN_EVERY = 63     # ml.retrain_every_days
NUM_LEAVES = 15        # card: num_leaves 15
LEARNING_RATE = 0.05   # card: learning_rate 0.05
REBALANCE_EVERY = 5    # decision rows: day count since panel start divisible by 5
MIN_NAMES = 10         # fewer scored members on a decision date -> flat
MIN_PER_LEG = 2        # at least two names per leg
MIN_HISTORY = 60       # a coin with fewer than 60 days of history is not scored
UNIVERSE_N = 20        # crypto_top_n, n = 20 (fallback when the gate passes no universe)
VOLUME_WINDOW = 30     # 30-day quote volume (fallback universe)
FEATURE_SET = "binance_1h_features"
BTC_PERP = "BTCUSDT.P"


def window_sum(x, lo, hi):
    """Sum of x over rows t-hi..t-lo (0 <= lo <= hi); NaN unless all hi-lo+1 rows exist and are finite.

    Uses prefix sums over rows <= t only; views shorter than the window give NaN (no crash)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T <= hi:
        return out
    good = np.isfinite(x)
    v = np.where(good, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    cs = np.concatenate([pad, np.cumsum(v, axis=0)], axis=0)
    cb = np.concatenate([pad, np.cumsum((~good).astype(np.float64), axis=0)], axis=0)
    t = np.arange(hi, T)
    s = cs[t - lo + 1] - cs[t - hi]
    b = cb[t - lo + 1] - cb[t - hi]
    out[t] = np.where(b == 0, s, np.nan)
    return out


def window_mean(x, lo, hi):
    return window_sum(x, lo, hi) / float(hi - lo + 1)


def spot_key(name):
    """Perp id XXXUSDT.P -> spot pair key XXXUSDT."""
    return name[:-2] if name.endswith(".P") else name


def effective_universe(data):
    """data.universe if the gate passes one; otherwise top UNIVERSE_N listed perps by trailing 30-day
    dollar (quote) volume, computed from rows <= t (shorter views use the rows available)."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    dv = np.where(np.isfinite(dv), dv, 0.0)
    T, N = dv.shape
    cs = np.concatenate([np.zeros((1, N)), np.cumsum(dv, axis=0)], axis=0)
    t = np.arange(T)
    start = np.maximum(t + 1 - VOLUME_WINDOW, 0)
    vol = cs[t + 1] - cs[start]
    listed = np.asarray(data.listed, dtype=bool)
    eligible = listed & (vol > 0)
    key = np.where(eligible, -vol, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return eligible & (rank < UNIVERSE_N)


def build_features(data, uni):
    """(T, N, 6) features, cross-sectionally z-scored over the universe members of each row."""
    T = len(data.dates)
    N = len(data.instruments)
    fields = ("ret_asia", "ret_europe", "ret_us", "vshare_us", "rv")
    btc_key = spot_key(BTC_PERP)
    # BTCUSDT spot is the residual reference: index directly so a missing series raises.
    btc = {f: np.asarray(data.series[FEATURE_SET + "." + btc_key + "." + f], dtype=np.float64) for f in fields[:3]}
    raw = {f: np.full((T, N), np.nan) for f in fields}
    found = 0
    for j, name in enumerate(data.instruments):
        prefix = FEATURE_SET + "." + spot_key(name) + "."
        if (prefix + "rv") not in data.series:
            continue                      # perp without a spot key: not scored (card)
        found += 1
        for f in fields:
            raw[f][:, j] = np.asarray(data.series[prefix + f], dtype=np.float64)
    if found == 0:
        raise KeyError("no binance_1h_features series match any instrument")

    f_sess = [window_sum(raw[f] - btc[f][:, None], 0, 2) for f in fields[:3]]
    f_vsh = window_mean(raw["vshare_us"], 0, 2) - window_mean(raw["vshare_us"], 3, 31)
    rv_long = window_mean(raw["rv"], 7, 59)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_rv = np.where(rv_long > 0, window_mean(raw["rv"], 0, 6) / rv_long, np.nan)
    f_fund = window_sum(np.asarray(data.extras["funding"], dtype=np.float64), 0, 2)

    # fewer than 60 days of perp history: not scored
    history = np.cumsum(np.asarray(data.listed, dtype=np.int64), axis=0)
    young = history < MIN_HISTORY
    feats = []
    for x in f_sess + [f_vsh, f_rv, f_fund]:
        x = np.where(young, np.nan, x)
        feats.append(cs_zscore(x, uni))
    return stack(*feats)


def model_scores(data, params, uni):
    X = build_features(data, uni)
    score = walk_forward(
        X,
        cs_rank(forward_return(data, HORIZON), uni),
        horizon=HORIZON,
        model="gbm",
        params={"n_estimators": int(params["n_estimators"]), "num_leaves": NUM_LEAVES,
                "learning_rate": LEARNING_RATE},
        retrain_every=RETRAIN_EVERY,
        universe=uni,
    )
    return np.where(uni, score, np.nan)


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    days = (d - d[0]).astype(np.int64)
    return days % REBALANCE_EVERY == 0


def book(score, uni, top):
    """Equal-weight dollar-neutral book on one score matrix: long the top `top` fraction, short the bottom
    `top` fraction (rounded down, at least MIN_PER_LEG names per leg), 0.5 per leg; fewer than MIN_NAMES
    scored members -> flat (zeros)."""
    T, N = score.shape
    w = np.zeros((T, N))
    valid = np.isfinite(score) & uni
    n = valid.sum(axis=1)
    k = np.maximum(np.floor(n * float(top) + 1e-9).astype(np.int64), MIN_PER_LEG)
    key_hi = np.where(valid, -score, np.inf)
    order = np.argsort(key_hi, axis=1, kind="stable")       # best first, invalid last
    rank_hi = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank_hi[rows, order] = np.arange(N)[None, :]
    key_lo = np.where(valid, score, np.inf)
    order_lo = np.argsort(key_lo, axis=1, kind="stable")    # worst first, invalid last
    rank_lo = np.empty_like(order_lo)
    rank_lo[rows, order_lo] = np.arange(N)[None, :]
    ok = (n >= MIN_NAMES)[:, None]
    kk = k[:, None]
    long_ = valid & ok & (rank_hi < kk)
    short = valid & ok & (rank_lo < kk) & ~long_
    w[long_] = 0.5 / np.broadcast_to(kk, (T, N))[long_]
    w[short] = -0.5 / np.broadcast_to(kk, (T, N))[short]
    return w


def target_weights(data, params):
    uni = effective_universe(data)
    score = model_scores(data, params, uni)
    dec = decision_rows(data.dates)
    w = only_on(dec, book(score, uni, params["top"]))
    return close_universe_exits(w, dec, uni)


def close_universe_exits(w, dec, uni):
    """Engine rule: a held coin that leaves the universe is closed (0) at the first row where it is outside;
    other entries of non-decision rows stay NaN (keep)."""
    w = np.array(w, dtype=np.float64)
    T, N = w.shape
    held = np.zeros(N)
    for t in range(T):
        if dec[t]:
            held = np.where(np.isfinite(w[t]), w[t], held)
        else:
            out = (held != 0) & ~uni[t]
            if out.any():
                w[t, out] = 0.0
                held = np.where(out, 0.0, held)
    return w
