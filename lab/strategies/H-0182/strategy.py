"""H-0182: overextension x trend in asset-class ETFs, learned (gbm), long-only 60/40 sleeves,
three overlapping 60-session tranches of the top fraction of the model score within each sleeve.

Rules of the card and where they live:
  features (5, card order of ml.features)          -> _features
  per-date cross-sectional z-score                 -> cs_zscore in model_score
  label: 60-session open-to-open return, ranked    -> cs_rank(forward_return(data, 60)) in model_score
  gbm walk-forward, refit every 63 rows            -> walk_forward in model_score
  decisions on rows with index % 20 == 0           -> _book
  RISK 60 % / DEFENSIVE 40 %, top round(top*n)>=2  -> _book
  fallback (< 5 scored members): EW tradable with >= 63 priced sessions, else cash -> _book
  book = mean of the last three tranches           -> _book
"""
import math

import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

RISK = ("SPY", "QQQ", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR",
        "EFA", "EEM", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY", "EWZ", "EWW",
        "FXI", "DBC", "USO", "SLV", "HYG", "EMB")
DEFENSIVE = ("SHY", "IEF", "TLT", "LQD", "AGG", "TIP", "MUB", "GLD", "FXE", "FXY", "FXF", "FXB",
             "FXA", "FXC", "UUP")
SLEEVE_SHARES = (0.6, 0.4)

DECISION_EVERY = 20
HORIZON = 60
RETRAIN_EVERY = 63
N_TRANCHES = 3
MIN_MODEL_MEMBERS = 5
MIN_PICKS = 2
MIN_HISTORY = 63
COVERAGE = 0.9
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 1000, "subsample": 0.5}


# ----------------------------------------------------------------------------- rolling helpers
def _lag(x, k):
    """x shifted down by k rows (row t holds x[t-k]); NaN / False where t-k < 0."""
    out = np.full(x.shape, np.nan) if x.dtype != bool else np.zeros(x.shape, dtype=bool)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _wsum(x, n):
    """Trailing sum over rows t-n+1..t (partial at the start), x without NaN."""
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:])
    c[1:] = np.cumsum(x, axis=0)
    idx = np.arange(T)
    lo = np.maximum(idx + 1 - n, 0)
    return c[idx + 1] - c[lo]


def _need(n):
    """Minimum count of non-missing days in an n-day window (90 %)."""
    return math.ceil(COVERAGE * n - 1e-9)


def _rolling_mean(v, ok, n):
    """Mean of v over the valid days of the trailing n-day window; NaN below 90 % coverage."""
    x = np.where(ok, v, 0.0)
    cnt = _wsum(ok.astype(np.float64), n)
    s1 = _wsum(x, n)
    good = cnt >= _need(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / cnt
    return np.where(good, m, np.nan)


def _rolling_std(v, ok, n):
    """Sample stdev (ddof 1) over the valid days of the trailing n-day window; NaN below 90 %
    coverage or when the stdev is not positive."""
    x = np.where(ok, v, 0.0)
    cnt = _wsum(ok.astype(np.float64), n)
    s1 = _wsum(x, n)
    s2 = _wsum(x * x, n)
    good = (cnt >= _need(n)) & (cnt >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1.0)
    var = np.where(good, var, np.nan)
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    return sd


# ----------------------------------------------------------------------------- features
def _features(data):
    """The card's five features, (T, N) float64 each, in ml.features order:
    overextension, trend, recent, trading surge, vol regime. An ETF-date with any feature missing
    has all five set to NaN (it is not scored)."""
    tr = np.asarray(total_return_index(data), dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    px = np.asarray(data.listed, dtype=bool) & np.isfinite(close) & np.isfinite(tr) & (tr > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(px, np.log(np.where(px, tr, 1.0)), np.nan)
    rok = px & _lag(px, 1)
    r = np.where(rok, L - _lag(L, 1), np.nan)

    sd63 = _rolling_std(r, rok, 63)
    sd252 = _rolling_std(r, rok, 252)
    sd756 = _rolling_std(r, rok, 756)
    mean_L756 = _rolling_mean(L, px, 756)

    L21 = _lag(L, 21)
    L252 = _lag(L, 252)
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = (L21 - L252) / (sd252 * np.sqrt(231.0))
        overext = (L - mean_L756) / (sd756 * np.sqrt(252.0))
        recent = (L - L21) / (sd252 * np.sqrt(21.0))

        dv = np.asarray(data.dollar_volume, dtype=np.float64)
        dok = px & np.isfinite(dv)
        dv63 = _rolling_mean(dv, dok, 63)
        dv252 = _rolling_mean(dv, dok, 252)
        surge = np.where((dv63 > 0) & (dv252 > 0), np.log(np.where(dv63 > 0, dv63, 1.0) / np.where(dv252 > 0, dv252, 1.0)), np.nan)
        surge = np.where(np.isfinite(dv63) & np.isfinite(dv252), surge, np.nan)

        volreg = np.log(sd63 / sd756)

    feats = [overext, trend, recent, surge, volreg]
    valid = np.ones(L.shape, dtype=bool)
    for f in feats:
        valid &= np.isfinite(f)
    return [np.where(valid, f, np.nan) for f in feats]


def model_score(data, params):
    """Point-in-time walk-forward gbm score, (T, N); NaN where an ETF is not scored."""
    uni = data.universe
    feats = _features(data)
    X = stack(*[cs_zscore(f, uni) for f in feats])
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                        retrain_every=RETRAIN_EVERY, universe=uni)


# ----------------------------------------------------------------------------- book
def _n_picks(top, n):
    """round(top * n), half rounded up, at least MIN_PICKS."""
    return max(MIN_PICKS, int(math.floor(top * n + 0.5 + 1e-9)))


def _sleeve_cols(data):
    return [np.array([data.col(name) for name in RISK], dtype=int),
            np.array([data.col(name) for name in DEFENSIVE], dtype=int)]


def _book(score, data, params):
    """Target weights from a (T, N) score: NaN rows except decision rows (index % 20 == 0)."""
    score = np.asarray(score, dtype=np.float64)
    T, N = score.shape
    top = float(params["top"])
    close = np.asarray(data.close, dtype=np.float64)
    px = np.asarray(data.listed, dtype=bool) & np.isfinite(close)
    hist = np.cumsum(px, axis=0)                      # priced sessions up to and including row t
    tradable = np.asarray(data.tradable, dtype=bool)
    member = np.ones((T, N), dtype=bool) if data.universe is None else np.asarray(data.universe, dtype=bool)
    sleeves = _sleeve_cols(data)

    out = np.full((T, N), np.nan)
    tranches = []
    for t in range(0, T, DECISION_EVERY):
        w = np.zeros(N)
        for cols, share in zip(sleeves, SLEEVE_SHARES):
            s = score[t, cols]
            scored = np.isfinite(s) & member[t, cols]
            n = int(scored.sum())
            if n >= MIN_MODEL_MEMBERS:
                cand = cols[scored]
                order = np.argsort(-s[scored], kind="stable")
                picks = cand[order[: _n_picks(top, n)]]
                picks = picks[tradable[t, picks]]      # not tradable: its share goes to the other picks
            else:
                ok = member[t, cols] & tradable[t, cols] & (hist[t, cols] >= MIN_HISTORY)
                picks = cols[ok]
            if picks.size:
                w[picks] += share / picks.size        # else the sleeve's share stays in cash
        tranches.append(w)
        out[t] = np.mean(np.array(tranches[-N_TRANCHES:]), axis=0)
    return out


def _events_from_score(score, data, params):
    """Identical copy of diagnostic._events_from_score (kept here so the tests can import it;
    the test file may import only `strategy`). Not used by target_weights."""
    score = np.asarray(score, dtype=np.float64)
    T, N = score.shape
    frac = float(params["top"])
    tradable = np.asarray(data.tradable, dtype=bool)
    member = np.ones((T, N), dtype=bool) if data.universe is None else np.asarray(data.universe, dtype=bool)
    sleeves = [np.array([data.col(name) for name in RISK], dtype=int),
               np.array([data.col(name) for name in DEFENSIVE], dtype=int)]

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev_top = np.zeros(N, dtype=bool)
    prev_bot = np.zeros(N, dtype=bool)
    for t in range(0, T, DECISION_EVERY):
        cur_top = np.zeros(N, dtype=bool)
        cur_bot = np.zeros(N, dtype=bool)
        for cols in sleeves:
            s = score[t, cols]
            scored = np.isfinite(s) & member[t, cols]
            n = int(scored.sum())
            if n < MIN_MODEL_MEMBERS:
                continue
            k = _n_picks(frac, n)
            cand = cols[scored]
            vals = s[scored]
            cur_top[cand[np.argsort(-vals, kind="stable")[:k]]] = True
            cur_bot[cand[np.argsort(vals, kind="stable")[:k]]] = True
        ok = member[t] & tradable[t]
        new_top = cur_top & ~prev_top & ok
        new_bot = cur_bot & ~prev_bot & ok & ~new_top
        mask[t] = new_top | new_bot
        side[t, new_top] = 1.0
        side[t, new_bot] = -1.0
        prev_top, prev_bot = cur_top, cur_bot
    side = np.where(mask, side, 1.0)
    return mask, side


def target_weights(data, params):
    return _book(model_score(data, params), data, params)
