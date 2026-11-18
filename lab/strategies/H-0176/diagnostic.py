"""H-0176 mechanism test: an ETF enters the top fifth of the model score at a monthly decision date (scored, and
not in the top selection at the previous monthly decision); side +1. Same features, walk_forward and sleeve
selection as strategy.py (copied: diagnostic.py is a separate file)."""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 42
RETRAIN_EVERY = 63
MIN_SCORED = 10
MIN_TOP = 3
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 400,
             "subsample": 0.8, "colsample": 1.0}


def shift_down(x, k):
    out = np.full(x.shape, np.nan)
    if k == 0:
        return x.astype(float).copy()
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out


def rolling_std_strict(lr, valid, n):
    x = np.where(valid, lr, 0.0)
    v = valid.astype(float)
    T = x.shape[0]
    z = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(v, axis=0)])
    out = np.full(x.shape, np.nan)
    if T >= n:
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        cnt = cn[n:] - cn[:-n]
        var = (s2 - s1 * s1 / n) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        out[n - 1:] = np.where(cnt >= n - 0.5, sd, np.nan)
    return out


def rolling_std_min(lr, valid, n, min_count):
    x = np.where(valid, lr, 0.0)
    v = valid.astype(float)
    T = x.shape[0]
    z = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cn = np.concatenate([z, np.cumsum(v, axis=0)])
    out = np.full(x.shape, np.nan)
    if T >= n:
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        cnt = cn[n:] - cn[:-n]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / np.maximum(cnt, 1.0)) / np.maximum(cnt - 1.0, 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        out[n - 1:] = np.where(cnt >= min_count - 0.5, sd, np.nan)
    return out


def rolling_max_strict(x, n):
    out = x.astype(float).copy()
    for k in range(1, n):
        out = np.maximum(out, shift_down(x, k))
    return out


def compute_features(ret_co, ret_oc, close):
    rc = np.nan_to_num(np.asarray(ret_co, dtype=float), nan=0.0)
    ro = np.nan_to_num(np.asarray(ret_oc, dtype=float), nan=0.0)
    daily = (1.0 + rc) * (1.0 + ro)
    tr_raw = np.cumprod(daily, axis=0)
    bar = np.isfinite(np.asarray(close, dtype=float))
    started = np.cumsum(bar, axis=0) > 0
    tr = np.where(started, tr_raw, np.nan)

    prev_bar = np.zeros_like(bar)
    prev_bar[1:] = bar[:-1]
    valid = bar & prev_bar & (daily > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(valid, np.log(np.where(daily > 0, daily, 1.0)), 0.0)

    sd21 = rolling_std_strict(lr, valid, 21)
    sd126 = rolling_std_strict(lr, valid, 126)
    sd63 = rolling_std_strict(lr, valid, 63)
    sd504 = rolling_std_min(lr, valid, 504, 378)

    with np.errstate(invalid="ignore", divide="ignore"):
        dd = tr / rolling_max_strict(tr, 252) - 1.0
        vr = np.log(sd21 / sd126)
        vlev = np.log(sd63 / sd504)
        r21 = tr / shift_down(tr, 21) - 1.0
        r12_1 = shift_down(tr, 21) / shift_down(tr, 252) - 1.0
    feats = {"dd": dd, "vr": vr, "vlev": vlev, "r21": r21, "r12_1": r12_1}
    return tr_raw, feats, bar, sd63


def scored_mask(feats, bar, universe=None):
    m = bar.copy()
    for v in feats.values():
        m &= np.isfinite(v)
    if universe is not None:
        m &= np.asarray(universe, dtype=bool)
    return m


def decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    mon = d.astype("datetime64[M]")
    is_last = nxt.astype("datetime64[M]") != mon
    dec = is_last.copy()
    starts = np.asarray(period_starts(d, "M"), dtype=bool)
    for t in np.flatnonzero(starts):
        if t == 0:
            continue
        pm = mon[t - 1]
        j = t - 1
        had = False
        while j >= 0 and mon[j] == pm:
            if is_last[j]:
                had = True
                break
            j -= 1
        if not had:
            dec[t] = True
    return dec


def top_set(score_row, mask_row, top):
    """Column indices of the top selection (same rule as the strategy's sleeve); empty when the strategy would
    hold the benchmark mix (fewer than MIN_SCORED scored ETFs or no model yet)."""
    ok = mask_row & np.isfinite(score_row)
    n = int(ok.sum())
    if n < MIN_SCORED:
        return np.zeros(0, dtype=int)
    k = min(n, max(MIN_TOP, int(np.floor(top * n + 0.5))))
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(-score_row[cand], kind="stable")]
    return order[:k]


def entry_events(score, mask, tradable, dec, top):
    T, N = score.shape
    ev = np.zeros((T, N), dtype=bool)
    prev = np.zeros(N, dtype=bool)
    for d in np.flatnonzero(dec):
        cur = np.zeros(N, dtype=bool)
        cur[top_set(score[d], mask[d], top)] = True
        ev[d] = cur & ~prev & mask[d] & tradable[d]
        prev = cur
    return ev


def events(data, params):
    tr_raw, feats, bar, vol63 = compute_features(data.ret_co, data.ret_oc, data.close)
    mask = scored_mask(feats, bar, data.universe)
    X = stack(*[cs_zscore(feats[k], mask) for k in ("dd", "vr", "vlev", "r21", "r12_1")])
    y = cs_rank(forward_return(data, HORIZON), mask)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = np.asarray(walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                                    retrain_every=RETRAIN_EVERY, universe=mask), dtype=float)
    dec = decision_rows(data.dates)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask_ev = entry_events(score, mask, tradable, dec, float(params["top"]))
    return mask_ev, None
