"""H-0121 Hedging vs informed volume moves, learned (LIQ-500, dollar-neutral, weekly, gbm, hold 5 days).

Pipeline (all point-in-time, row t uses rows <= t only):
  r = (1+ret_co)(1+ret_oc)-1; e = r - equal-weight mean r of the LIQ-500 members of t;
  sigma = std of e over the last 63 rows (>= 40 valid);
  F1 = sum e[t-4..t] / (sigma[t-5] sqrt 5)
  F2 = log(mean dv[t-4..t] / median dv[t-67..t-5])
  F3 = sessions since the last ea_reaction_day == 1 session, capped at 126 (none in 400 calendar days: NaN)
  F4 = mean of |e[rel]| / sigma[rel-1] over the last 4 releases within the last 504 rows (>= 2)
  F5 = sf1_art_rnd / sf1_art_revenue (rnd missing & revenue > 0 -> 0; cap 1; revenue missing/<=0 -> date median)
  F6 = log sigma[t]
  eligible = member & all finite & |F1| >= 1 & no release in t-4..t
Features z-scored over the eligible rows of each date, gbm walk-forward (expanding, refit every 63 rows),
label = 5-day open-to-open forward return. Weekly book on the last weekday of each calendar week.
"""
import math

import numpy as np
from lab.framework.ml import cs_zscore, forward_return, stack, walk_forward

PARAMS = {"num_leaves": 15, "top": 0.1}

# fixed rules of the card (not grid parameters)
SIGMA_WIN = 63
SIGMA_MIN = 40
MOVE_WIN = 5
DV_LAG_HI = 67          # median window t-67 .. t-5 (63 rows)
DV_LAG_LO = 5
DV_MED_MIN = 40         # NaN handling of the median window (same minimum as sigma)
F3_CAP = 126
F3_MAX_DAYS = 400
F4_WIN = 504
F4_N = 4
F4_MIN = 2
MIN_MOVE = 1.0
MIN_ROWS = 20
LEG_GROSS = 0.5
HORIZON = 5
RETRAIN = 63
GBM_FIXED = {"n_estimators": 100, "learning_rate": 0.05, "min_child_samples": 200}


# ----------------------------------------------------------------------------- helpers
def window_sum(x, n):
    """Trailing sum and count of finite values of x over rows t-n+1..t (prefix sums, point-in-time)."""
    T = x.shape[0]
    valid = np.isfinite(x)
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cs[1:] = np.cumsum(np.where(valid, x, 0.0), axis=0)
    cc = np.zeros((T + 1,) + x.shape[1:], dtype=np.int32)
    cc[1:] = np.cumsum(valid, axis=0)
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    return cs[hi] - cs[lo], cc[hi] - cc[lo]


def shift_down(x, k):
    """x[t-k] at row t (NaN for the first k rows)."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def residual_returns(ret_co, ret_oc, listed, close, uni):
    r = (1.0 + ret_co.astype(np.float64)) * (1.0 + ret_oc.astype(np.float64)) - 1.0
    valid = listed.astype(bool) & np.isfinite(r) & np.isfinite(close)
    mem = valid & uni
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return np.where(valid, r - m[:, None], np.nan)


def rolling_sigma(e):
    s1, c = window_sum(e, SIGMA_WIN)
    s2, _ = window_sum(e * e, SIGMA_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.maximum(c, 1)) / np.maximum(c - 1, 1)
    return np.where(c >= SIGMA_MIN, np.sqrt(np.maximum(var, 0.0)), np.nan)


def release_cycle(rel, day_num):
    """F3 (sessions since last release, capped; NaN if none within 400 calendar days) and the
    'no release in t-4..t' flag. rel: (T, N) bool release sessions, day_num: (T,) int days."""
    T, N = rel.shape
    rows = np.arange(T, dtype=np.int64)
    last = np.maximum.accumulate(np.where(rel, rows[:, None], -1), axis=0)
    has = last >= 0
    since = rows[:, None] - last
    cal = day_num[:, None] - day_num[np.maximum(last, 0)]
    f3 = np.where(has & (cal <= F3_MAX_DAYS), np.minimum(since, F3_CAP), np.nan).astype(np.float64)
    quiet_week = ~has | (since >= MOVE_WIN)
    return f3, quiet_week


def informativeness(rel, e, sigma):
    """F4: mean of |e[rel]| / sigma[rel-1] over the last 4 releases (with a computable value) whose
    session lies in rows t-503..t; at least 2 needed."""
    T, N = rel.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    sig_prev = shift_down(sigma, 1)
    rows = np.arange(T)
    for j in np.nonzero(rel.any(axis=0))[0]:
        rr = np.nonzero(rel[:, j])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.abs(e[rr, j]) / sig_prev[rr, j]
        ok = np.isfinite(v)
        rr = rr[ok]
        v = v[ok]
        if rr.size < F4_MIN:
            continue
        cs = np.concatenate(([0.0], np.cumsum(v)))
        hi = np.searchsorted(rr, rows, side="right")               # releases at rows <= t
        lo = np.searchsorted(rr, rows - F4_WIN, side="right")      # releases at rows <= t-504 (outside)
        start = np.maximum(lo, hi - F4_N)
        n = hi - start
        val = (cs[hi] - cs[start]) / np.maximum(n, 1)
        out[:, j] = np.where(n >= F4_MIN, val, np.nan)
    return out


def rd_intensity(rnd, rev, uni):
    rnd = rnd.astype(np.float64)
    rev = rev.astype(np.float64)
    pos = np.isfinite(rev) & (rev > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(pos, np.where(np.isfinite(rnd), rnd, 0.0) / np.where(pos, rev, 1.0), np.nan)
    ratio = np.minimum(ratio, 1.0)
    pool = np.where(uni & np.isfinite(ratio), ratio, np.nan)
    med = np.full(ratio.shape[0], np.nan)
    has = np.isfinite(pool).any(axis=1)
    if has.any():
        med[has] = np.nanmedian(pool[has], axis=1)
    return np.where(pos, ratio, med[:, None])


def abnormal_dollar_volume(dv, cand):
    """F2 only on the candidate cells (all other eligibility rules already hold)."""
    T, N = dv.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    tt, jj = np.nonzero(cand)
    offs = np.arange(-DV_LAG_HI, 1)                     # t-67 .. t  (68 rows)
    n_long = DV_LAG_HI - DV_LAG_LO + 1                  # 63 rows t-67..t-5
    step = 50000
    for a in range(0, tt.size, step):
        t = tt[a:a + step]
        j = jj[a:a + step]
        rows = t[:, None] + offs[None, :]
        vals = dv[np.maximum(rows, 0), j[:, None]].astype(np.float64)
        vals[rows < 0] = np.nan
        short = vals[:, n_long:]                        # t-4 .. t
        longw = np.sort(vals[:, :n_long], axis=1)       # NaN sorted to the end
        c = np.isfinite(longw).sum(axis=1)
        i = np.arange(t.size)
        lo_i = np.maximum((c - 1) // 2, 0)
        hi_i = np.minimum(c // 2, n_long - 1)
        med = 0.5 * (longw[i, lo_i] + longw[i, hi_i])
        med = np.where(c >= DV_MED_MIN, med, np.nan)
        smean = short.mean(axis=1)                      # NaN unless all 5 valid
        with np.errstate(invalid="ignore", divide="ignore"):
            f2 = np.log(smean / med)
        out[t, j] = f2
    return out


def build_features(ret_co, ret_oc, listed, close, dv, uni, ea, rnd, rev, day_num):
    """Raw features, eligibility and the 5-day residual move. All arrays (T, N), day_num (T,)."""
    uni = uni.astype(bool)
    e = residual_returns(ret_co, ret_oc, listed, close, uni)
    sigma = rolling_sigma(e)
    s5, c5 = window_sum(e, MOVE_WIN)
    move = np.where(c5 == MOVE_WIN, s5, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = move / (shift_down(sigma, MOVE_WIN) * math.sqrt(MOVE_WIN))
        f6 = np.log(sigma)
    rel = np.asarray(ea, dtype=np.float64) == 1.0
    f3, quiet_week = release_cycle(rel, day_num)
    f4 = informativeness(rel, e, sigma)
    f5 = rd_intensity(rnd, rev, uni)
    with np.errstate(invalid="ignore"):
        big = np.abs(f1) >= MIN_MOVE
    cand = (uni & quiet_week & np.isfinite(f1) & big & np.isfinite(f3) & np.isfinite(f4)
            & np.isfinite(f5) & np.isfinite(f6))
    f2 = abnormal_dollar_volume(dv, cand)
    elig = cand & np.isfinite(f2)
    feats = [f1, f2, f3, f4, f5, f6]
    return feats, elig, move


def decision_days(dates):
    """Last weekday of each calendar week: the next business day (weekday arithmetic, exchange
    holidays ignored) falls in a later Monday-based week."""
    nxt = np.busday_offset(dates, 1, roll="forward")
    wk = (dates.astype(np.int64) + 3) // 7
    wn = (nxt.astype(np.int64) + 3) // 7
    return wn != wk


def leg_masks(score, decision, top):
    """On decision rows: top `top` fraction (long) and bottom `top` fraction (short) of the scored
    eligible rows, k = max(1, floor(top * n)) names per leg; no legs with fewer than 20 rows."""
    T, N = score.shape
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)
    for t in np.nonzero(decision)[0]:
        ok = np.nonzero(np.isfinite(score[t]))[0]
        n = ok.size
        if n >= MIN_ROWS:
            k = max(1, int(math.floor(top * n)))
            k = min(k, n // 2)
            order = ok[np.argsort(score[t, ok], kind="stable")]
            long_m[t, order[n - k:]] = True
            short_m[t, order[:k]] = True
    return long_m, short_m


def make_book(score, decision, top):
    """Weekly dollar-neutral book: equal weight, 0.5 gross per leg, flat (zeros) on a decision row
    with fewer than 20 scored rows, NaN (hold) on all other rows."""
    long_m, short_m = leg_masks(score, decision, top)
    nl = long_m.sum(axis=1, keepdims=True)
    ns = short_m.sum(axis=1, keepdims=True)
    w = np.where(long_m, LEG_GROSS / np.maximum(nl, 1), 0.0) - np.where(short_m, LEG_GROSS / np.maximum(ns, 1), 0.0)
    w[~decision] = np.nan
    return w


def continuation_events(long_m, short_m, decision, move, member):
    """Mechanism event (used by diagnostic.py, which holds an identical copy): new leg entries vs the
    previous decision row whose 5-day residual move has the leg's sign (long & up +1, short & down -1)."""
    T, N = long_m.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev = -1
    for t in np.nonzero(decision)[0]:
        if prev >= 0:
            new_l = long_m[t] & ~long_m[prev]
            new_s = short_m[t] & ~short_m[prev]
        else:
            new_l = long_m[t].copy()
            new_s = short_m[t].copy()
        with np.errstate(invalid="ignore"):
            up = move[t] > 0
            dn = move[t] < 0
        ev_l = new_l & up & member[t]
        ev_s = new_s & dn & member[t]
        mask[t] = ev_l | ev_s
        side[t] = np.where(ev_l, 1.0, np.where(ev_s, -1.0, 0.0))
        prev = t
    return mask, side


def model_scores(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0121 needs the LIQ-500 point-in-time universe")
    uni = np.asarray(uni, dtype=bool)
    ex = data.extras
    feats, elig, move = build_features(
        data.ret_co, data.ret_oc, data.listed, data.close, data.dollar_volume, uni,
        ex["ea_reaction_day"], ex["sf1_art_rnd"], ex["sf1_art_revenue"],
        data.dates.astype(np.int64))
    zs = []
    for f in feats:
        z = np.asarray(cs_zscore(np.where(elig, f, np.nan).astype(np.float32), elig), dtype=np.float32)
        zs.append(np.where(elig, z, np.nan).astype(np.float32))
    del feats
    X = stack(*zs)
    del zs
    y = forward_return(data, HORIZON)
    gbm = dict(GBM_FIXED)
    gbm["num_leaves"] = int(params["num_leaves"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN, universe=uni)
    score = np.where(elig & uni, np.asarray(score, dtype=np.float64), np.nan)
    return score, elig, move


def target_weights(data, params):
    score, _, _ = model_scores(data, params)
    return make_book(score, decision_days(data.dates), float(params["top"]))
