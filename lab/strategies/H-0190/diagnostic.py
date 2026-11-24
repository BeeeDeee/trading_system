"""H-0190 mechanism test: on a weekly decision date (Friday close) whose style-unwind state is <= -1.5, members in
the top fifth of the model score (+1) and in the bottom fifth (-1). Same features, label and walk_forward as
strategy.py (copied: diagnostic.py is a separate file)."""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 126
GBM_FIXED = {"num_leaves": 15, "min_child_samples": 200, "learning_rate": 0.05}
MOM_LONG, MOM_SKIP = 252, 21
VOL_WINDOW, VOL_MIN = 63, 50
STATE_K = 5
SD_WINDOW, SD_MIN = 252, 200
QUINTILE = 0.2
UNWIND_THRESHOLD = -1.5      # card: mechanism_test.event
EVENT_FRACTION = 0.2         # card: top fifth / bottom fifth of the score


def weekly_decision_mask(dates):
    day = dates.astype("datetime64[D]").astype(np.int64)
    weekday = (day + 3) % 7
    return weekday == 4


def lag_rows(x, k):
    out = np.zeros_like(x) if x.dtype == bool else np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    has = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = has.copy()
    valid[0] = False
    valid[1:] &= has[:-1]
    valid &= np.isfinite(r)
    return r, valid


def low_vol_exposure(r, valid, window=VOL_WINDOW, min_valid=VOL_MIN):
    T = r.shape[0]
    x = np.where(valid, r, 0.0)
    shape = (T + 1,) + r.shape[1:]
    c1 = np.zeros(shape)
    c2 = np.zeros(shape)
    cn = np.zeros(shape)
    np.cumsum(x, axis=0, out=c1[1:])
    np.cumsum(x * x, axis=0, out=c2[1:])
    np.cumsum(valid.astype(np.float64), axis=0, out=cn[1:])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - window, 0)
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    n = cn[hi] - cn[lo]
    del c1, c2, cn
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    var = np.maximum(var, 0.0)
    out = -np.sqrt(var)
    out[~(n >= min_valid)] = np.nan
    return out


def momentum_exposure(idx, listed, long=MOM_LONG, skip=MOM_SKIP):
    idx = np.asarray(idx, dtype=np.float64)
    T = idx.shape[0]
    out = np.full(idx.shape, np.nan)
    if T > long:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[long:] = idx[long - skip:T - skip] / idx[:T - long] - 1.0
        ok = np.zeros(idx.shape, dtype=bool)
        ok[long:] = listed[long:] & listed[:T - long]
        out[~ok] = np.nan
    return out


def value_exposure(pb, idx):
    pbf = np.asarray(pb, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.float64)
    T, N = pbf.shape
    ok = np.isfinite(pbf)
    prev = lag_rows(pbf, 1)
    new = ok & ~(np.isfinite(prev) & (prev == pbf))
    anchor = np.where(new, np.arange(T)[:, None], -1)
    anchor = np.maximum.accumulate(anchor, axis=0)
    filing_row = np.maximum(anchor - 1, 0)
    idx_f = idx[filing_row, np.arange(N)[None, :]]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (1.0 / pbf) * idx_f / idx
    out[(anchor < 0) | ~ok | ~(pbf > 0)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def quintile_masks(comp, uni, frac=QUINTILE):
    T, N = comp.shape
    top = np.zeros((T, N), dtype=bool)
    bot = np.zeros((T, N), dtype=bool)
    for t in range(T):
        j = np.flatnonzero(np.isfinite(comp[t]) & uni[t])
        n = j.size
        q = int(np.floor(n * frac))
        if q < 1:
            continue
        order = j[np.argsort(comp[t, j], kind="stable")]
        bot[t, order[:q]] = True
        top[t, order[n - q:]] = True
    return top, bot


def _masked_mean(x, mask):
    m = mask & np.isfinite(x)
    cnt = m.sum(axis=1)
    s = np.where(m, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    pos = cnt > 0
    out[pos] = s[pos] / cnt[pos]
    return out


def scale_by_trailing_sd(s, window=SD_WINDOW, min_valid=SD_MIN):
    T = s.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        if not np.isfinite(s[t]):
            continue
        w = s[max(0, t - window):t]
        w = w[np.isfinite(w)]
        if w.size < min_valid:
            continue
        sd = float(np.std(w, ddof=1))
        if sd > 0:
            out[t] = s[t] / sd
    return out


def state_inputs(comp, idx, uni, k=STATE_K):
    idx = np.asarray(idx, dtype=np.float64)
    r_k = np.full(idx.shape, np.nan)
    if idx.shape[0] > k:
        with np.errstate(invalid="ignore", divide="ignore"):
            r_k[k:] = idx[k:] / idx[:-k] - 1.0
    top, bot = quintile_masks(comp, uni)
    spread = _masked_mean(r_k, lag_rows(top, k)) - _masked_mean(r_k, lag_rows(bot, k))
    market = _masked_mean(r_k, lag_rows(uni, k))
    return scale_by_trailing_sd(spread), scale_by_trailing_sd(market)


def build_features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0190 needs the point-in-time liq_n universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    idx = np.asarray(total_return_index(data), dtype=np.float64)
    pb = data.extras["sf1_arq_pb"]
    roe = np.asarray(data.extras["sf1_art_roe"], dtype=np.float64)

    r, valid = daily_returns(data)
    raw = [value_exposure(pb, idx), momentum_exposure(idx, listed), roe.copy(), low_vol_exposure(r, valid)]
    for x in raw:
        x[~listed] = np.nan
    z = [np.asarray(cs_zscore(x, uni), dtype=np.float64) for x in raw]
    comp = (z[0] + z[1] + z[2] + z[3]) / 4.0
    comp[~uni] = np.nan
    unwind, market = state_inputs(comp, idx, uni)
    N = uni.shape[1]
    unwind_b = np.repeat(unwind[:, None], N, axis=1)
    market_b = np.repeat(market[:, None], N, axis=1)
    X = stack(z[0], z[1], z[2], z[3], unwind_b, market_b)
    return X, unwind, uni


def model_score(data, params, X, uni):
    h = HORIZON
    y = cs_rank(forward_return(data, h), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=h, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score[~uni] = np.nan
    return score


def events(data, params):
    X, unwind, uni = build_features(data)
    score = model_score(data, params, X, uni)
    T, N = score.shape
    tradable = np.asarray(data.tradable, dtype=bool)
    state_day = weekly_decision_mask(data.dates) & np.isfinite(unwind) & (unwind <= UNWIND_THRESHOLD)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for t in np.flatnonzero(state_day):
        j = np.flatnonzero(np.isfinite(score[t]) & uni[t])
        n = j.size
        q = int(np.floor(n * EVENT_FRACTION))
        if q < 1:
            continue
        order = j[np.argsort(-score[t, j], kind="stable")]
        hi, lo = order[:q], order[n - q:]
        side[t, hi] = 1.0
        side[t, lo] = -1.0
        mask[t, hi] = True
        mask[t, lo] = True
    mask &= uni & tradable
    return mask, side
