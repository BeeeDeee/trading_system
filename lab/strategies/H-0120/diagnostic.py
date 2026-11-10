"""H-0120 mechanism event: a coin enters the long leg (top `top` fraction of the gbm score, side +1) or the
short leg (bottom `top` fraction, side -1) at a decision date, i.e. it was not in that leg at the previous
decision date. Same features, walk_forward and book as strategy.py (copied: separate file)."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
NUM_LEAVES = 15
LEARNING_RATE = 0.05
REBALANCE_EVERY = 5
MIN_NAMES = 10
MIN_PER_LEG = 2
MIN_HISTORY = 60
UNIVERSE_N = 20
VOLUME_WINDOW = 30
FEATURE_SET = "binance_1h_features"
BTC_PERP = "BTCUSDT.P"


def window_sum(x, lo, hi):
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
    return name[:-2] if name.endswith(".P") else name


def effective_universe(data):
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
    T = len(data.dates)
    N = len(data.instruments)
    fields = ("ret_asia", "ret_europe", "ret_us", "vshare_us", "rv")
    btc_key = spot_key(BTC_PERP)
    btc = {f: np.asarray(data.series[FEATURE_SET + "." + btc_key + "." + f], dtype=np.float64) for f in fields[:3]}
    raw = {f: np.full((T, N), np.nan) for f in fields}
    found = 0
    for j, name in enumerate(data.instruments):
        prefix = FEATURE_SET + "." + spot_key(name) + "."
        if (prefix + "rv") not in data.series:
            continue
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


def legs(score, uni, top):
    """(long, short) bool masks of the book's legs on every row (same rule as strategy.book)."""
    T, N = score.shape
    valid = np.isfinite(score) & uni
    n = valid.sum(axis=1)
    k = np.maximum(np.floor(n * float(top) + 1e-9).astype(np.int64), MIN_PER_LEG)
    rows = np.arange(T)[:, None]
    order = np.argsort(np.where(valid, -score, np.inf), axis=1, kind="stable")
    rank_hi = np.empty_like(order)
    rank_hi[rows, order] = np.arange(N)[None, :]
    order_lo = np.argsort(np.where(valid, score, np.inf), axis=1, kind="stable")
    rank_lo = np.empty_like(order_lo)
    rank_lo[rows, order_lo] = np.arange(N)[None, :]
    ok = (n >= MIN_NAMES)[:, None]
    kk = k[:, None]
    long_ = valid & ok & (rank_hi < kk)
    short = valid & ok & (rank_lo < kk) & ~long_
    return long_, short


def events(data, params):
    uni = effective_universe(data)
    score = model_scores(data, params, uni)
    long_, short = legs(score, uni, params["top"])
    dec = decision_rows(data.dates)
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    for t in np.flatnonzero(dec):
        new_long = long_[t] & ~prev_long
        new_short = short[t] & ~prev_short
        ok = uni[t] & tradable[t]
        mask[t] = (new_long | new_short) & ok
        side[t] = np.where(new_long, 1.0, np.where(new_short, -1.0, 0.0))
        prev_long = long_[t]
        prev_short = short[t]
    side = np.where(mask, side, 0.0)
    return mask, side
