"""H-0198 mechanism event: interaction-only picks.

At a weekly decision (first session of the week) a member ranks in the top `top` fraction of the
interaction ridge's score and did not at the previous weekly decision, while the plain ridge (same
label, schedule, alpha and members, fitted on the five raw inputs rev5, pre, post, vol, vlm without
product terms) does not rank it in its own top `top` fraction. Side +1.

The feature and walk-forward code is a copy of strategy.py (separate file by contract).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

HORIZON = 5
RIDGE_ALPHA = 100.0
RETRAIN_EVERY = 63
MIN_PRED_MEMBERS = 100
MIN_SCORED = 100
PRED_DAYS = 364
REV_N = 5
POST_N = 5
VOL_N = 63
VLM_SHORT = 5
VLM_LONG = 63


def _rsum(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) + 1 - int(n), 0)
    return c[1:] - c[lo]


def _event_flags(ea, dates, uni, pre_window, min_pred_members=MIN_PRED_MEMBERS):
    W = int(pre_window)
    ea = np.asarray(ea)
    T, N = ea.shape
    ev = ea == 1.0
    evf = ev.astype(np.float64)
    post = _rsum(evf, POST_N) > 0.5
    no_recent = _rsum(evf, W) < 0.5

    s_rows, s_cols = np.nonzero(ev)
    inwin = np.zeros((T, N), dtype=bool)
    has_pred = np.zeros((T, N), dtype=bool)
    if s_rows.size:
        sd = dates[s_rows]
        d = (sd.astype(np.int64) + PRED_DAYS).astype(sd.dtype)   # datetime64[D]: + 364 calendar days
        p =np.busday_offset(d, 0, roll="forward")
        w_start = np.busday_offset(p, -W, roll="forward")
        w_end = np.busday_offset(p, -1, roll="forward")
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
    with np.errstate(all="ignore"):
        rco = np.asarray(ret_co, dtype=np.float64)
        roc = np.asarray(ret_oc, dtype=np.float64)
        lst = np.asarray(listed, dtype=bool)
        lst_prev = np.zeros_like(lst)
        lst_prev[1:] = lst[:-1]
        r = (1.0 + rco) * (1.0 + roc) - 1.0
        g = np.log1p(r)
        rvalid = lst & lst_prev & np.isfinite(g)
        g0 = np.where(rvalid, g, 0.0)
        r0 = np.where(rvalid, r, 0.0)
        rv = rvalid.astype(np.float64)

        n5 = _rsum(rv, REV_N)
        r5 = np.expm1(_rsum(g0, REV_N))
        ok5 = (n5 > REV_N - 0.5) & uni
        cnt = ok5.sum(axis=1)
        mkt = np.where(ok5, r5, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        rev5 = np.where(ok5, -(r5 - mkt[:, None]), np.nan)
        del g, g0, r5, n5

        nv = _rsum(rv, VOL_N)
        s1 = _rsum(r0, VOL_N)
        s2 = _rsum(r0 * r0, VOL_N)
        var = (s2 - s1 * s1 / np.maximum(nv, 1.0)) / np.maximum(nv - 1.0, 1.0)
        vol = np.where((nv > VOL_N - 0.5) & uni, np.sqrt(np.maximum(var, 0.0)), np.nan)
        del nv, s1, s2, var, r0, rv, r

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
    xv = np.where(valid, x, np.nan).astype(np.float32)
    z = np.asarray(cs_zscore(xv, uni), dtype=np.float32)
    x64 = np.where(valid, xv, 0.0).astype(np.float64)
    cnt = valid.sum(axis=1)
    m = x64.sum(axis=1) / np.maximum(cnt, 1)
    ss = np.where(valid, (x64 - m[:, None]) ** 2, 0.0).sum(axis=1)
    flat = (cnt < 2) | ~(ss > 0)
    z = np.where(flat[:, None], np.float32(0.0), z)
    return np.where(valid, z, np.float32(np.nan)).astype(np.float32)


def _columns(data, params, min_pred_members=MIN_PRED_MEMBERS):
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
    plain = [_cs(c, uni, sc) for c in (rz, pre_f, post_f, vz, mz)]
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
    idx = np.nonzero(ok)[0]
    return idx[np.lexsort((idx, -s[idx]))]


def _round(x):
    return int(np.floor(x + 0.5))


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


def events(data, params):
    inter, plain, uni, _ = _columns(data, params)
    s_int = _score(inter, data, uni)
    del inter
    s_pl = _score(plain, data, uni)
    del plain
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = _entry_events(s_int, s_pl, uni, tradable, _decision_rows(data.dates), float(params["top"]))
    return mask, None
