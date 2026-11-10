"""H-0122 mechanism event: gbm leg entries the ridge control does not make on the same day."""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 1
RETRAIN_EVERY = 63
VOL_WINDOW = 21
SPLIT_TOL = 0.01
MIN_NAMES = 100
GBM_FIXED = {"num_leaves": 15, "min_child_samples": 200}
RIDGE_ALPHA = 1.0


def _features(data):
    p = np.asarray(data.close, dtype=np.float64)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    prev = np.full_like(p, np.nan)
    prev[1:] = p[:-1]
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = np.where(np.isfinite(p) & np.isfinite(prev), r, np.nan)

    level = np.floor(p + 0.5)
    cents = p - level
    roundness = np.where(np.mod(level, 50) == 0, 3.0,
                         np.where(np.mod(level, 10) == 0, 2.0,
                                  np.where(np.mod(level, 5) == 0, 1.0, 0.0)))
    roundness = np.where(np.isfinite(level), roundness, np.nan)
    logp = np.log(p)
    up = (prev < level) & (level <= p)
    down = (prev >= level) & (level > p)
    crossing = np.where(up, 1.0, np.where(down, -1.0, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mismatch = np.abs((p / prev) / (1.0 + r) - 1.0) > SPLIT_TOL
    crossing = np.where(mismatch, 0.0, crossing)
    crossing = np.where(np.isfinite(p) & np.isfinite(prev), crossing, np.nan)

    T, N = r.shape
    valid = np.isfinite(r)
    x = np.where(valid, r, 0.0)
    z = np.zeros((1, N))
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(valid.astype(np.float64), axis=0)])
    vol = np.full((T, N), np.nan)
    w = VOL_WINDOW
    if T >= w:
        s1 = c1[w:] - c1[:-w]
        s2 = c2[w:] - c2[:-w]
        n = cn[w:] - cn[:-w]
        var = (s2 - s1 * s1 / w) / (w - 1)
        vol[w - 1:] = np.where(n == w, np.sqrt(np.maximum(var, 0.0)), np.nan)

    uni = data.universe
    return stack(cents, roundness, cs_zscore(logp, uni), crossing, cs_zscore(r, uni), cs_zscore(vol, uni))


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0122 needs the LIQ-500 point-in-time universe")
    X = _features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gp = dict(GBM_FIXED)
    gp["n_estimators"] = int(params["n_estimators"])
    g = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp, retrain_every=RETRAIN_EVERY, universe=uni)
    rd = walk_forward(X, y, horizon=HORIZON, model="ridge", params={"alpha": RIDGE_ALPHA},
                      retrain_every=RETRAIN_EVERY, universe=uni)
    top = float(params["top"])
    wg = np.nan_to_num(np.asarray(score_weights(g, uni, top=top, bottom=top, min_names=MIN_NAMES), dtype=np.float64))
    wr = np.nan_to_num(np.asarray(score_weights(rd, uni, top=top, bottom=top, min_names=MIN_NAMES), dtype=np.float64))

    g_long, g_short = wg > 0, wg < 0
    r_long, r_short = wr > 0, wr < 0
    prev_long = np.zeros_like(g_long)
    prev_short = np.zeros_like(g_short)
    prev_long[1:] = g_long[:-1]
    prev_short[1:] = g_short[:-1]

    new_long = g_long & ~prev_long & ~r_long
    new_short = g_short & ~prev_short & ~r_short
    ok = np.asarray(uni, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    mask = (new_long | new_short) & ok
    side = np.where(new_long, 1.0, np.where(new_short, -1.0, 0.0))
    return mask, side
