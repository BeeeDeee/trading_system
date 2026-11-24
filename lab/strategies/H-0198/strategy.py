"""H-0198: reporting-cycle reversal concession, learned linearly.

LIQ-500 long-only, weekly top `top` fraction of a ridge 5-day score on
rev5, pre, rev5 x pre, rev5 x post, rev5 x vol, rev5 x vlm, with a 2 x top hold buffer.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"pre_window": 10, "top": 0.1}

HORIZON = 5            # label: open t+1 -> open t+6
RIDGE_ALPHA = 100.0    # card: ridge alpha 100
RETRAIN_EVERY = 63     # card ml.retrain_every_days
MIN_PRED_MEMBERS = 100  # rows train/score only once >= 100 members have a predicted release date
MIN_SCORED = 100       # fewer scored members on a decision date: flat
PRED_DAYS = 364        # last year's reaction session + 52 weeks
REV_N = 5              # rev5: sessions t-4..t
POST_N = 5             # post: actual reaction session in t-4..t
VOL_N = 63             # vol: 63-session std of daily total returns
VLM_SHORT = 5          # vlm numerator: mean dollar volume t-4..t
VLM_LONG = 63          # vlm denominator: mean dollar volume t-67..t-5


# ----------------------------------------------------------------------------- helpers

def _rsum(x, n):
    """Trailing sum over rows t-n+1..t (fewer rows at the start). x must be finite."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) + 1 - int(n), 0)
    return c[1:] - c[lo]


def _event_flags(ea, dates, uni, pre_window, min_pred_members=MIN_PRED_MEMBERS):
    """pre, post (T, N) bool and active (T,) bool from the 8-K Item 2.02 reaction flag.

    Predicted release date of a reaction session s: first weekday on or after s + 364 days.
    pre[t] = a predicted date lies 1..W weekdays after t, and no actual reaction session in t-W+1..t.
    post[t] = an actual reaction session in t-4..t.
    active[t] = on some row <= t at least `min_pred_members` members had a predicted release date
    on or before that row (first predicted date reached).
    """
    W = int(pre_window)
    ea = np.asarray(ea)
    T, N = ea.shape
    ev = ea == 1.0                      # NaN (no EDGAR history) -> no event -> pre = post = 0
    evf = ev.astype(np.float64)
    post = _rsum(evf, POST_N) > 0.5
    no_recent = _rsum(evf, W) < 0.5

    s_rows, s_cols = np.nonzero(ev)
    inwin = np.zeros((T, N), dtype=bool)
    has_pred = np.zeros((T, N), dtype=bool)
    if s_rows.size:
        sd = dates[s_rows]
        d = (sd.astype(np.int64) + PRED_DAYS).astype(sd.dtype)   # datetime64[D]: + 364 calendar days
        p =np.busday_offset(d, 0, roll="forward")          # weekday approximation of XNYS sessions
        w_start = np.busday_offset(p, -W, roll="forward")   # t = p - W weekdays
        w_end = np.busday_offset(p, -1, roll="forward")     # t = p - 1 weekday
        r0 = np.searchsorted(dates, w_start, side="left")
        r1 = np.searchsorted(dates, w_end, side="right")
        diff = np.zeros((T + 1, N), dtype=np.int32)
        np.add.at(diff, (r0, s_cols), 1)
        np.add.at(diff, (r1, s_cols), -1)
        inwin = np.cumsum(diff[:T], axis=0) > 0
        rp = np.searchsorted(dates, p, side="left")
        diff2 = np.zeros((T + 1, N), dtype=np.int32)
        np.add.at(diff2, (rp, s_cols), 1)
        has_pred = np.cumsum(diff2[:T], axis=0) > 0
    pre = inwin & no_recent
    n_pred = (has_pred & uni).sum(axis=1)
    active = np.maximum.accumulate(n_pred >= min_pred_members) if T else np.zeros(0, bool)
    return pre, post, active


def _raw_inputs(ret_co, ret_oc, listed, dollar_volume, uni):
    """rev5, vol, vlm (T, N) float64, NaN where not computable or not a member."""
    with np.errstate(all="ignore"):
        rco = np.asarray(ret_co, dtype=np.float64)
        roc = np.asarray(ret_oc, dtype=np.float64)
        lst = np.asarray(listed, dtype=bool)
        lst_prev = np.zeros_like(lst)
        lst_prev[1:] = lst[:-1]
        r = (1.0 + rco) * (1.0 + roc) - 1.0                  # daily total return close->close
        g = np.log1p(r)
        rvalid = lst & lst_prev & np.isfinite(g)              # a genuine return of a listed stock
        g0 = np.where(rvalid, g, 0.0)
        r0 = np.where(rvalid, r, 0.0)
        rv = rvalid.astype(np.float64)

        # (1) rev5: minus 5-session return t-4..t, minus the member equal-weighted mean
        n5 = _rsum(rv, REV_N)
        r5 = np.expm1(_rsum(g0, REV_N))
        ok5 = (n5 > REV_N - 0.5) & uni
        cnt = ok5.sum(axis=1)
        mkt = np.where(ok5, r5, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        rev5 = np.where(ok5, -(r5 - mkt[:, None]), np.nan)
        del g, g0, r5, n5

        # (5) vol: 63-session std of daily total returns
        nv = _rsum(rv, VOL_N)
        s1 = _rsum(r0, VOL_N)
        s2 = _rsum(r0 * r0, VOL_N)
        var = (s2 - s1 * s1 / np.maximum(nv, 1.0)) / np.maximum(nv - 1.0, 1.0)
        vol = np.where((nv > VOL_N - 0.5) & uni, np.sqrt(np.maximum(var, 0.0)), np.nan)
        del nv, s1, s2, var, r0, rv, r

        # (6) vlm: log(mean dv t-4..t / mean dv t-67..t-5)
        dv = np.asarray(dollar_volume, dtype=np.float64)
        dvalid = np.isfinite(dv) & (dv >= 0)
        dv0 = np.where(dvalid, dv, 0.0)
        dn = dvalid.astype(np.float64)
        tot = VLM_SHORT + VLM_LONG
        ss = _rsum(dv0, VLM_SHORT)
        sn = _rsum(dn, VLM_SHORT)
        ls = _rsum(dv0, tot) - ss
        ln_ = _rsum(dn, tot) - sn
        ok = (sn > VLM_SHORT - 0.5) & (ln_ > VLM_LONG - 0.5) & (ss > 0) & (ls > 0) & uni
        vlm = np.where(ok, np.log((ss / VLM_SHORT) / (np.where(ls > 0, ls, 1.0) / VLM_LONG)), np.nan)
    return rev5, vol, vlm


def _cs(x, uni, valid):
    """Framework cross-sectional z-score on the valid cells; a date without dispersion -> 0."""
    xv = np.where(valid, x, np.nan).astype(np.float32)
    z = np.asarray(cs_zscore(xv, uni), dtype=np.float32)
    x64 = np.where(valid, xv, 0.0).astype(np.float64)
    cnt = valid.sum(axis=1)
    m = x64.sum(axis=1) / np.maximum(cnt, 1)
    ss = np.where(valid, (x64 - m[:, None]) ** 2, 0.0).sum(axis=1)
    flat = (cnt < 2) | ~(ss > 0)
    z = np.where(flat[:, None], np.float32(0.0), z)
    return np.where(valid, z, np.float32(np.nan)).astype(np.float32)


def _columns(data, params, min_pred_members=MIN_PRED_MEMBERS, with_plain=False):
    """The six interaction columns, the five plain columns (or None), universe, scored mask."""
    if data.universe is None:
        raise ValueError("H-0198 needs the point-in-time LIQ-500 universe (data.universe)")
    uni = np.asarray(data.universe, dtype=bool)
    ea = data.extras["ea_reaction_day"]
    pre, post, active = _event_flags(ea, data.dates, uni, params["pre_window"], min_pred_members)
    rev5, vol, vlm = _raw_inputs(data.ret_co, data.ret_oc, data.listed, data.dollar_volume, uni)
    sc = uni & np.isfinite(rev5) & np.isfinite(vol) & np.isfinite(vlm) & active[:, None]
    rz = _cs(rev5, uni, sc)
    vz = _cs(vol, uni, sc)
    mz = _cs(vlm, uni, sc)
    del rev5, vol, vlm
    pre_f = np.where(sc, pre, np.nan).astype(np.float32)
    post_f = np.where(sc, post, np.nan).astype(np.float32)
    inter = [_cs(c, uni, sc) for c in (rz, pre_f, rz * pre_f, rz * post_f, rz * vz, rz * mz)]
    plain = [_cs(c, uni, sc) for c in (rz, pre_f, post_f, vz, mz)] if with_plain else None
    return inter, plain, uni, sc


def _score(cols, data, uni):
    X = stack(*cols)
    y = cs_rank(forward_return(data, HORIZON), uni)
    return np.asarray(
        walk_forward(X, y, horizon=HORIZON, model="ridge", params={"alpha": RIDGE_ALPHA},
                     retrain_every=RETRAIN_EVERY, universe=uni, train_days=None),
        dtype=np.float64)


def _decision_rows(dates):
    ps = np.asarray(period_starts(dates, "W"))
    if ps.dtype == bool:
        return np.nonzero(ps)[0]
    return np.sort(ps.astype(np.int64))


def _ranked(s, ok):
    """Column ids of the scored members, best score first (ties by column id)."""
    idx = np.nonzero(ok)[0]
    return idx[np.lexsort((idx, -s[idx]))]


def _round(x):
    return int(np.floor(x + 0.5))


def _book(score, uni, rows, top, min_scored=MIN_SCORED):
    """Weekly long-only book with a 2 x top hold buffer; NaN on non-decision rows."""
    T, N = score.shape
    w = np.full((T, N), np.nan)
    held = np.zeros(N, dtype=bool)
    for t in rows:
        s = score[t]
        ok = uni[t] & np.isfinite(s)
        S = int(ok.sum())
        row = np.zeros(N)
        if S < min_scored:
            held[:] = False
            w[t] = row
            continue
        K = max(_round(top * S), 1)
        K2 = max(_round(2.0 * top * S), K)
        order = _ranked(s, ok)
        buf = order[:K2]
        keep = buf[held[buf]][:K]                      # held names still in the top 2 x top
        new = order[~held[order]][: K - keep.size]     # best names not held
        book = np.concatenate([keep, new])
        row[book] = 1.0 / K
        held[:] = False
        held[book] = True
        w[t] = row
    return w


# Mechanism-event logic: a verbatim copy lives in diagnostic.py (which cannot import this file);
# kept here so test_strategy.py can test it on hand-made scores. Not used by target_weights.

def _top_set(s, ok, top, min_scored=MIN_SCORED):
    """Bool (N,) top `top` fraction (K = round(top x scored)); empty below min_scored scored members."""
    out = np.zeros(s.shape[0], dtype=bool)
    S = int(ok.sum())
    if S < min_scored:
        return out
    K = max(_round(top * S), 1)
    out[_ranked(s, ok)[:K]] = True
    return out


def _entry_events(s_int, s_pl, uni, tradable, rows, top, min_scored=MIN_SCORED):
    T, N = s_int.shape
    mask = np.zeros((T, N), dtype=bool)
    prev = np.zeros(N, dtype=bool)
    for t in rows:
        top_i = _top_set(s_int[t], uni[t] & np.isfinite(s_int[t]), top, min_scored)
        top_p = _top_set(s_pl[t], uni[t] & np.isfinite(s_pl[t]), top, min_scored)
        mask[t] = top_i & ~prev & ~top_p & uni[t] & tradable[t]
        prev = top_i
    return mask


def target_weights(data, params):
    inter, _, uni, _ = _columns(data, params)
    score = _score(inter, data, uni)
    return _book(score, uni, _decision_rows(data.dates), float(params["top"]))
