"""H-0103 mechanism event: TCN-only picks against a ridge on the same five session-t features.
Same pipeline as strategy.py (copied: diagnostic.py is a separate file)."""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN = 252
HIDDEN = 16
VOL_WIN = 21
RET_WIN = 5
DV_WIN = 63
MIN_NAMES = 20


def lag(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def rolling_all(ok, n):
    T = ok.shape[0]
    c = np.cumsum(ok.astype(np.int64), axis=0)
    prev = np.zeros_like(c)
    if n < T:
        prev[n:] = c[: T - n]
    full = np.zeros(ok.shape, dtype=bool)
    full[n - 1:] = True
    return full & ((c - prev) == n)


def rolling_std_strict(r, n):
    x = np.where(np.isfinite(r), r, 0.0)
    ok = np.isfinite(r)
    T = r.shape[0]
    s1 = np.cumsum(x, axis=0)
    s2 = np.cumsum(x * x, axis=0)
    p1 = np.zeros_like(s1)
    p2 = np.zeros_like(s2)
    if n < T:
        p1[n:] = s1[: T - n]
        p2[n:] = s2[: T - n]
    a = s1 - p1
    b = s2 - p2
    var = (b - a * a / n) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(rolling_all(ok, n), sd, np.nan)


def lagged_median_strict(x, n):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T <= n:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, n, axis=0)
    m = T - n
    step = max(1, int(20_000_000 // max(1, N * n)))
    mid = n // 2
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        w = win[i0:i1]
        bad = np.isnan(w).any(axis=2)
        if n % 2 == 1:
            med = np.partition(w, mid, axis=2)[:, :, mid]
        else:
            p = np.partition(w, [mid - 1, mid], axis=2)
            med = 0.5 * (p[:, :, mid - 1] + p[:, :, mid])
        out[n + i0:n + i1] = np.where(bad, np.nan, med)
    return out


def raw_features(data):
    listed = np.asarray(data.listed, dtype=bool)
    ret_co = np.where(listed, np.asarray(data.ret_co, dtype=np.float64), np.nan)
    ret_oc = np.where(listed, np.asarray(data.ret_oc, dtype=np.float64), np.nan)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    dv = np.where(listed & np.isfinite(dv) & (dv > 0), dv, np.nan)
    med = lagged_median_strict(dv, DV_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        f3 = np.where(med > 0, np.log(dv / med), np.nan)
    g = 1.0 + r
    p = g.copy()
    for k in range(1, RET_WIN):
        p = p * lag(g, k)
    f4 = p - 1.0
    f5 = rolling_std_strict(r, VOL_WIN)
    return ret_oc, ret_co, f3, f4, f5


def model_inputs(data, seq_len):
    uni = data.universe
    feats = raw_features(data)
    z = [np.asarray(cs_zscore(f, uni), dtype=np.float64) for f in feats]
    X = stack(*z)
    complete = np.all(np.isfinite(X), axis=2)
    seq_ok = rolling_all(complete, seq_len)
    return X, feats[4], seq_ok


def eligible(score, vol, uni, seq_ok):
    scored = np.isfinite(score) & seq_ok & np.isfinite(vol)
    if uni is not None:
        scored &= np.asarray(uni, dtype=bool)
    v = np.where(scored, vol, np.nan)
    cnt = scored.sum(axis=1)
    med = np.full(v.shape[0], np.nan)
    rows = np.nonzero(cnt > 0)[0]
    if rows.size:
        med[rows] = np.nanmedian(v[rows], axis=1)
    with np.errstate(invalid="ignore"):
        return scored & (v >= med[:, None])


def picks(score, elig, top):
    s = np.where(elig, score, np.nan)
    w = np.asarray(score_weights(s, elig, top=top, bottom=top, min_names=MIN_NAMES), dtype=np.float64)
    return np.where(np.isfinite(w), w, 0.0)


def tcn_ridge_events(score_t, score_r, vol, uni, seq_ok, dec, tradable, top):
    """Pure event logic: +1 TCN top but not ridge top, -1 TCN bottom but not ridge bottom (same eligible set)."""
    elig = eligible(score_t, vol, uni, seq_ok)
    wt = picks(score_t, elig, top)
    elig_r = elig & np.isfinite(score_r)
    wr = picks(score_r, elig_r, top)
    base = dec[:, None] & elig_r & np.asarray(tradable, dtype=bool)
    if uni is not None:
        base &= np.asarray(uni, dtype=bool)
    long_ev = base & (wt > 0) & ~(wr > 0)
    short_ev = base & (wt < 0) & ~(wr < 0)
    mask = long_ev | short_ev
    side = np.where(short_ev, -1.0, 1.0)
    return mask, side


def events(data, params):
    seq_len = int(params["seq_len"])
    top = float(params["top"])
    uni = data.universe
    X, vol, seq_ok = model_inputs(data, seq_len)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score_t = walk_forward(X, y, horizon=HORIZON, model="tcn",
                           params={"seq_len": seq_len, "hidden": HIDDEN},
                           retrain_every=RETRAIN, universe=uni)
    score_t = np.where(seq_ok, np.asarray(score_t, dtype=np.float64), np.nan)
    # linear control: session-t features only, on the same rows as the TCN (full sequence available)
    Xr = np.where(seq_ok[:, :, None], X, np.nan).astype(np.float32)
    score_r = walk_forward(Xr, y, horizon=HORIZON, model="ridge", params={},
                           retrain_every=RETRAIN, universe=uni)
    score_r = np.where(seq_ok, np.asarray(score_r, dtype=np.float64), np.nan)
    dec = np.zeros(len(data.dates), dtype=bool)
    dec[np.asarray(period_starts(data.dates, "W"))] = True
    return tcn_ridge_events(score_t, score_r, vol, uni, seq_ok, dec, data.tradable, top)
