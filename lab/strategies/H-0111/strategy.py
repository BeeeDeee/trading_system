"""H-0111 Realized-skewness lottery premium, learned (LIQ-500, dollar-neutral, weekly, gbm).

Rules of the card -> code
- Decision dates: first trading day of each week (period_starts "W"), decided after its close, held to
  the next weekly decision (only_on).
- Half-day log returns log(1+ret_co), log(1+ret_oc) of the last 21 sessions; a session with either half
  missing is dropped; >= 36 valid half-day returns needed, else the stock is not scored at t.
- Features: rskew21 (sample skewness), rvol21 (std), onshare21 (overnight share of sum of squares),
  ret5, ret21 (total return), ea21 (any ea_reaction_day == 1 in last 21 sessions, NaN counts as 0);
  each z-scored across universe members with cs_zscore.
- Model: walk_forward gbm, label = cs_rank of forward_return(5), horizon 5, expanding window, refit
  every 126 rows, num_leaves 15, n_estimators from params, on all columns with the universe mask.
- Book: score_weights(top=tail, bottom=tail, min_names=100) only on weekly decision rows.
- A held stock that leaves the universe between decisions is closed (weight 0) at the first row where
  it is no longer a member (the engine refuses non-member weights).
"""
import numpy as np
from lab.framework.api import only_on, period_starts, total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "tail": 0.1}

HORIZON = 5          # ml.horizon_days
WINDOW = 21          # sessions per feature window
MIN_HALF = 36        # minimum valid half-day returns
RETRAIN = 126        # ml.retrain_every_days
NUM_LEAVES = 15      # card: num_leaves 15
MIN_SCORED = 100     # fewer scored members on a decision date = flat


def _lag(x, k, fill):
    """x shifted down by k rows (row t holds x[t-k]); first k rows = fill."""
    if k == 0:
        return x
    out = np.full_like(x, fill)
    out[k:] = x[:-k]
    return out


def valid_sessions(ret_co, ret_oc, close, listed):
    """A session counts when the stock has a price today and yesterday (so the overnight half exists),
    both halves are finite and both log returns are defined."""
    has_px = np.isfinite(close)
    prev_px = _lag(has_px, 1, False)
    with np.errstate(invalid="ignore"):
        ok = (listed.astype(bool) & has_px & prev_px & np.isfinite(ret_co) & np.isfinite(ret_oc)
              & (ret_co > -1.0) & (ret_oc > -1.0))
    return ok


def halfday_features(ret_co, ret_oc, close, listed, window=WINDOW, min_half=MIN_HALF):
    """Returns rskew, rvol, onshare, n_half (each (T, N)); NaN where n_half < min_half."""
    ok = valid_sessions(ret_co, ret_oc, close, listed)
    with np.errstate(invalid="ignore", divide="ignore"):
        co = np.where(ok, np.log1p(np.where(ok, ret_co, 0.0)), 0.0).astype(np.float64)
        oc = np.where(ok, np.log1p(np.where(ok, ret_oc, 0.0)), 0.0).astype(np.float64)
    okf = ok.astype(np.float64)

    n = np.zeros(co.shape, dtype=np.float64)
    s1 = np.zeros(co.shape, dtype=np.float64)
    sq_co = np.zeros(co.shape, dtype=np.float64)
    sq_all = np.zeros(co.shape, dtype=np.float64)
    for k in range(window):
        a = _lag(co, k, 0.0)
        b = _lag(oc, k, 0.0)
        n += 2.0 * _lag(okf, k, 0.0)
        s1 += a + b
        a2 = a * a
        sq_co += a2
        sq_all += a2 + b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
    del s1
    d2 = np.zeros(co.shape, dtype=np.float64)
    d3 = np.zeros(co.shape, dtype=np.float64)
    for k in range(window):
        v = _lag(okf, k, 0.0)
        a = (_lag(co, k, 0.0) - mean) * v
        b = (_lag(oc, k, 0.0) - mean) * v
        d2 += a * a + b * b
        d3 += a * a * a + b * b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = d2 / n
        m3 = d3 / n
        g1 = m3 / m2 ** 1.5
        # adjusted Fisher-Pearson sample skewness G1
        rskew = g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)
        rvol = np.sqrt(d2 / (n - 1.0))
        onshare = sq_co / sq_all
    good = (n >= min_half) & (m2 > 0) & (sq_all > 0)
    rskew = np.where(good, rskew, np.nan)
    rvol = np.where(good, rvol, np.nan)
    onshare = np.where(good, onshare, np.nan)
    return rskew, rvol, onshare, n


def ea_window(ea, window=WINDOW):
    """1 if ea == 1 on any of the last `window` rows (NaN counts as 0)."""
    hit = (np.nan_to_num(np.asarray(ea, dtype=np.float64), nan=0.0) == 1.0).astype(np.int64)
    c = np.cumsum(hit, axis=0)
    prev = _lag(c, window, 0)
    return ((c - prev) > 0).astype(np.float64)


def features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0111 needs the liq_n point-in-time universe (data.universe is None)")
    ea = data.extras["ea_reaction_day"]
    rskew, rvol, onshare, n_half = halfday_features(data.ret_co, data.ret_oc, data.close, data.listed)
    eligible = n_half >= MIN_HALF
    idx = total_return_index(data)
    ret5 = trailing_return(idx, 5)
    ret21 = trailing_return(idx, WINDOW)
    ea21 = ea_window(ea, WINDOW)
    raw = [rskew, rvol, onshare, ret5, ret21, ea21]
    # unscored stocks (fewer than 36 valid half-days) are excluded from every feature and the z-scores
    raw = [np.where(eligible, f, np.nan) for f in raw]
    return [cs_zscore(f, uni) for f in raw], ret5


def scores(data, params):
    feats, ret5 = features(data)
    X = stack(*feats)
    uni = data.universe
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(X, y, horizon=HORIZON, model="gbm",
                         params={"n_estimators": int(params["n_estimators"]), "num_leaves": NUM_LEAVES},
                         retrain_every=RETRAIN, universe=uni)
    return score, ret5


def book(score, data, params):
    """Full-row book from scores (every row) and the weekly decision mask."""
    tail = float(params["tail"])
    w_all = score_weights(score, data.universe, top=tail, bottom=tail, min_names=MIN_SCORED)
    days = period_starts(data.dates, "W")
    return w_all, days


def weekly_with_exits(w_all, days, universe):
    """Weights only on decision rows (NaN = keep elsewhere); a stock held since the last decision that is
    not a universe member at row t gets weight 0 at t (the other elements of that row stay NaN)."""
    days = np.asarray(days, dtype=bool)
    w = np.array(only_on(days, w_all), dtype=np.float64)
    T = w.shape[0]
    rows = np.arange(T)
    last = np.maximum.accumulate(np.where(days, rows, -1))
    held = np.where((last >= 0)[:, None], np.nan_to_num(w_all[np.maximum(last, 0)], nan=0.0), 0.0)
    uni = np.asarray(universe, dtype=bool)
    exit_mask = np.isnan(w) & (held != 0.0) & ~uni
    w[exit_mask] = 0.0
    return w


P_LO = 20.0
P_HI = 80.0


def tail_events(w_all, score, ret5, days, uni, tradable):
    """Mechanism-event logic. Not used by the book: diagnostic.py carries an identical copy (the static
    rules forbid importing diagnostic.py from the tests), kept here so the tests exercise it."""
    T, N = w_all.shape
    w = np.nan_to_num(np.asarray(w_all, dtype=np.float64), nan=0.0)
    uni = np.asarray(uni, dtype=bool)
    long_now = w > 0
    short_now = w < 0
    rows = np.arange(T)
    dec = np.asarray(days, dtype=bool)
    # previous weekly decision row for each row (strictly earlier)
    last_incl = np.maximum.accumulate(np.where(dec, rows, -1))
    prev_dec = np.full(T, -1)
    prev_dec[1:] = last_incl[:-1]
    has_prev = prev_dec >= 0
    pl = np.where(has_prev[:, None], long_now[np.maximum(prev_dec, 0)], False)
    ps = np.where(has_prev[:, None], short_now[np.maximum(prev_dec, 0)], False)

    scored = uni & np.isfinite(score) & np.isfinite(ret5)
    r = np.where(scored, ret5, np.nan)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    any_scored = scored.any(axis=1) & dec
    if any_scored.any():
        sub = r[any_scored]
        lo[any_scored] = np.nanpercentile(sub, P_LO, axis=1)
        hi[any_scored] = np.nanpercentile(sub, P_HI, axis=1)
    with np.errstate(invalid="ignore"):
        mid = scored & (r >= lo[:, None]) & (r <= hi[:, None])
    base = dec[:, None] & uni & np.asarray(tradable, dtype=bool) & mid
    enter_long = base & long_now & ~pl
    enter_short = base & short_now & ~ps
    mask = enter_long | enter_short
    side = np.where(enter_short, -1.0, 1.0)
    return mask, side


def target_weights(data, params):
    score, _ = scores(data, params)
    w_all, days = book(score, data, params)
    return weekly_with_exits(w_all, days, data.universe)
