"""H-0104: event-time decay of earnings-reaction drift, learned (LightGBM), LIQ-500 long-only weekly top fraction.

Rules of the card and where they live:
- r = (1+ret_co)(1+ret_oc)-1, NaN where the stock is not listed           -> daily_returns
- u = r minus the equal-weight mean of r over the members on t             -> universe_relative
- s = std of u over the 63 sessions ending the session before (>= 40 valid) -> prior_scale
- features react_z, react_z_prev, sessions_since, post_car_z, pre_car_z    -> event_features
- each feature z-scored across the members on t (framework cs_zscore)      -> build_features
- label: open t+1 -> open t+6 total return, ranked across members           -> target_weights
- gbm walk_forward, expanding, refit every 63 rows, fixed hyperparameters   -> gbm_score
- weekly decision after the last session of the week (Friday by weekday arithmetic) -> decision_rows
- equal-weight top `top` fraction of scored members; < 50 scored -> all members equal weight -> book
- a held stock that leaves the universe is set to 0 on that row             -> book
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"num_leaves": 15, "top": 0.2}

# fixed by the card (not grid parameters)
HORIZON = 5
RETRAIN_EVERY = 63
SCALE_WINDOW = 63
SCALE_MIN_VALID = 40
EVENT_WINDOW = 126          # R1 within the last 126 sessions (t - R1 <= 125); R0 within 126 sessions before R1
PRE_WINDOW = 21
MIN_SCORED = 50
GBM_FIXED = {"n_estimators": 100, "learning_rate": 0.05, "min_child_samples": 200,
             "subsample": 0.8, "colsample": 1.0}


def daily_returns(ret_co, ret_oc, listed):
    """Close-to-close total return; NaN where the instrument is not listed."""
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    r[~np.asarray(listed, dtype=bool)] = np.nan
    return r


def universe_relative(r, uni):
    """u[t, j] = r[t, j] - equal-weight mean of r over the members on t (members with a return)."""
    uni = np.asarray(uni, dtype=bool)
    ok = uni & np.isfinite(r)
    n = ok.sum(axis=1)
    s = np.where(ok, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return r - m[:, None]


def _padded_cumsum(x):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def prior_scale(u, window=SCALE_WINDOW, min_valid=SCALE_MIN_VALID):
    """S[t, j] = sample std (ddof 1) of u over rows t-window .. t-1 (the sessions ending the session
    before t); NaN with fewer than min_valid valid values. Accepts views shorter than the window."""
    T = u.shape[0]
    valid = np.isfinite(u)
    x = np.where(valid, u, 0.0)
    cn = _padded_cumsum(valid.astype(np.float64))
    cx = _padded_cumsum(x)
    cxx = _padded_cumsum(x * x)
    b = np.arange(T)                      # window end (exclusive) = t
    a = np.maximum(b - window, 0)
    n = cn[b] - cn[a]
    sx = cx[b] - cx[a]
    sxx = cxx[b] - cxx[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sxx - sx * sx / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd[n < min_valid] = np.nan
    sd[sd <= 0.0] = np.nan
    return sd


def _last_event_index(ev):
    """last[t, j] = largest row index <= t with an event, -1 if none."""
    T = ev.shape[0]
    idx = np.where(ev, np.arange(T)[:, None], -1)
    return np.maximum.accumulate(idx, axis=0)


def event_features(u, ev, window=EVENT_WINDOW, pre_window=PRE_WINDOW):
    """The five raw features (T, N) each, NaN where undefined.

    R1 = most recent event row <= t with t - R1 <= window - 1; R0 = event row before R1 with
    R1 - R0 <= window. s(R) = prior_scale at row R (std of u over the 63 rows before R)."""
    T, N = u.shape
    ev = np.asarray(ev, dtype=bool)
    S = prior_scale(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_ev = u / S                                           # react_z if row t were the reaction row
    valid = np.isfinite(u)
    x = np.where(valid, u, 0.0)
    cu = _padded_cumsum(x)                                     # cu[k] = sum of u over rows < k
    cn = _padded_cumsum(valid.astype(np.float64))
    b = np.arange(T)
    a = np.maximum(b - pre_window, 0)
    pre_sum = cu[b] - cu[a]                                    # sum of u over rows t-21 .. t-1
    pre_n = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_z = pre_sum / (S * np.sqrt(float(pre_window)))
    pre_z[pre_n < pre_window] = np.nan                         # all 21 sessions must have a return
    del pre_sum, pre_n, cn

    last = _last_event_index(ev)
    rows = np.arange(T)[:, None]
    r1 = last
    have1 = (r1 >= 0) & ((rows - r1) <= window - 1)
    r1c = np.where(have1, r1, 0)

    react_z = np.take_along_axis(z_ev, r1c, axis=0)
    since = (rows - r1c).astype(np.float64)
    s_r1 = np.take_along_axis(S, r1c, axis=0)
    post_sum = cu[1:] - np.take_along_axis(cu, r1c + 1, axis=0)   # sum of u over rows R1+1 .. t
    with np.errstate(invalid="ignore", divide="ignore"):
        post_z = post_sum / s_r1
    pre_r1 = np.take_along_axis(pre_z, r1c, axis=0)

    # previous event strictly before row R: last[R-1]
    prev_before = np.full((T, N), -1, dtype=last.dtype)
    if T > 1:
        prev_before[1:] = last[:-1]
    r0 = np.take_along_axis(prev_before, r1c, axis=0)
    have0 = have1 & (r0 >= 0) & ((r1c - r0) <= window)
    r0c = np.where(have0, r0, 0)
    react_prev = np.take_along_axis(z_ev, r0c, axis=0)

    for f in (react_z, since, post_z, pre_r1):
        f[~have1] = np.nan
    react_prev[~have0] = np.nan
    return react_z, react_prev, since, post_z, pre_r1


def reaction_flag(data):
    """1 on the reaction session of an Item 2.02 release (0/1 flag, NaN before the first filing)."""
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    return np.nan_to_num(flag, nan=0.0) > 0.5


def build_features(data):
    """(T, N, 5) features, cross-sectionally z-scored over the members whose five raw features are all
    defined (a member with any NaN feature is not scored); non-members NaN."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0104 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    r = daily_returns(data.ret_co, data.ret_oc, data.listed)
    u = universe_relative(r, uni)
    del r
    raw = event_features(u, reaction_flag(data))
    del u
    complete = uni.copy()
    for f in raw:
        complete &= np.isfinite(f)
    feats = []
    for f in raw:
        f = np.where(complete, f, np.nan)
        feats.append(cs_zscore(f, uni))
    del raw
    return stack(*feats)


def label(data):
    uni = data.universe
    return cs_rank(forward_return(data, HORIZON), uni)


def gbm_score(data, X, y, params):
    gp = dict(GBM_FIXED)
    gp["num_leaves"] = int(params["num_leaves"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gp, retrain_every=RETRAIN_EVERY,
                        universe=data.universe)


def decision_rows(dates):
    """Last session of the week, approximated point-in-time as rows dated on a Friday (weekday arithmetic;
    a week whose Friday is an exchange holiday has no decision and keeps its positions)."""
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return ((d + 3) % 7) == 4          # 1970-01-01 was a Thursday; Monday = 0


def book(score, uni, dec, top, min_scored=MIN_SCORED):
    """Weights on decision rows: equal-weight top fraction of the scored members; with fewer than
    min_scored scored members all members equal weight. Other rows NaN, except 0 for a held stock
    that is not a member on that row."""
    uni = np.asarray(uni, dtype=bool)
    score = np.where(uni, score, np.nan)
    n_scored = np.isfinite(score).sum(axis=1)
    w = np.asarray(score_weights(score, uni, top=float(top), bottom=0.0, min_names=min_scored),
                   dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0)
    n_mem = uni.sum(axis=1)
    ew = np.where(uni, 1.0 / np.maximum(n_mem, 1)[:, None], 0.0)
    fallback = n_scored < min_scored
    w[fallback] = ew[fallback]
    w[~uni] = 0.0
    dec = np.asarray(dec, dtype=bool)
    out = np.where(dec[:, None], w, np.nan)          # NaN rows: no decision, keep positions
    # positions held from the last decision row that left the universe: sell (0) on that row
    T = w.shape[0]
    last_dec = np.maximum.accumulate(np.where(dec, np.arange(T), -1))
    held = np.where((last_dec >= 0)[:, None], w[np.maximum(last_dec, 0)], 0.0)
    leave = (held > 0) & ~uni & ~dec[:, None]
    # only the first such row after the decision (the position is 0 afterwards)
    c = np.cumsum(leave, axis=0)
    c_dec = np.where((last_dec >= 0)[:, None], c[np.maximum(last_dec, 0)], 0)
    first = leave & ((c - c_dec) == 1)
    out[first] = 0.0
    return out


# ---- mechanism-event helpers: verbatim copy of diagnostic.py (tests may import only `strategy`);
# ---- not used by the book.

def top_sets(score, uni, top, min_scored=MIN_SCORED):
    """(T, N) bool in the top fraction of the scored members, and (T,) bool ranking available."""
    uni = np.asarray(uni, dtype=bool)
    score = np.where(uni, np.asarray(score, dtype=np.float64), np.nan)
    ok = np.isfinite(score).sum(axis=1) >= min_scored
    w = np.asarray(score_weights(score, uni, top=float(top), bottom=0.0, min_names=min_scored),
                   dtype=np.float64)
    in_top = (np.nan_to_num(w, nan=0.0) > 0) & ok[:, None] & uni
    return in_top, ok


def gbm_only_entries(gbm, ridge, uni, tradable, dec, top, min_scored=MIN_SCORED):
    uni = np.asarray(uni, dtype=bool)
    g_top, g_ok = top_sets(gbm, uni, top, min_scored)
    r_top, r_ok = top_sets(ridge, uni, top, min_scored)
    dec_idx = np.flatnonzero(np.asarray(dec, dtype=bool))
    mask = np.zeros(uni.shape, dtype=bool)
    if dec_idx.size < 2:
        return mask
    cur = dec_idx[1:]
    prev = dec_idx[:-1]
    ok = (g_ok[cur] & r_ok[cur] & g_ok[prev])[:, None]
    ev = ok & g_top[cur] & ~g_top[prev] & ~r_top[cur]
    mask[cur] = ev
    mask &= uni & np.asarray(tradable, dtype=bool)
    return mask


def target_weights(data, params):
    X = build_features(data)
    y = label(data)
    score = gbm_score(data, X, y, params)
    del X, y
    return book(np.asarray(score, dtype=np.float64), data.universe, decision_rows(data.dates), params["top"])
