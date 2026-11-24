"""H-0197 mechanism_test events: the same event rule as strategy.py (copied, separate file).

Event on row t for stock j: predicted next release (last year's release date + 364 days, weekday
calendar) is the second session after t, no release in the last 30 sessions, the previous release
40-120 sessions back had a reaction >= hi_t (side +1) or <= lo_t (side -1) of the trailing-252-session
LIQ-500 reaction distribution. Only members that are tradable at t with a valid return at t.
"""
import numpy as np

DEDUP_SESSIONS = 20
VOL_WINDOW = 60
VOL_MIN_OBS = 40
REF_WINDOW = 252
REF_MIN_VALUES = 200
ANNIV_DAYS = 364
LEAD_SESSIONS = 2
NO_RELEASE_SESSIONS = 30
PREV_MIN, PREV_MAX = 40, 120


def _daily_returns(data):
    priced = np.isfinite(np.asarray(data.close))
    valid = np.zeros_like(priced)
    valid[1:] = priced[1:] & priced[:-1]
    valid &= np.asarray(data.listed, dtype=bool)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = np.where(valid & np.isfinite(r), r, np.nan)
    return r


def _member_mean(x, member):
    ok = member & np.isfinite(x)
    cnt = ok.sum(axis=1)
    s = np.where(ok, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


def release_sessions(ea):
    flag = np.asarray(ea) == 1.0
    T, N = flag.shape
    rows = np.arange(T)
    cflag = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(flag, axis=0, out=cflag[1:])
    lo_idx = np.maximum(rows - DEDUP_SESSIONS, 0)
    prev_flags = cflag[rows] - cflag[lo_idx]
    return flag & (prev_flags == 0)


def anniversary_source_rows(dates):
    dates = np.asarray(dates)
    T = dates.shape[0]
    rows = np.arange(T)
    d_int = dates.astype(np.int64)
    target = np.busday_offset(dates, LEAD_SESSIONS, roll="forward").astype(np.int64)
    src = target - ANNIV_DAYS
    pos = np.searchsorted(d_int, src)
    pos_c = np.minimum(pos, max(T - 1, 0))
    has = (pos < T) & (d_int[pos_c] == src) & (pos_c < rows)
    return pos_c, has


def _pre_release_vol(r, mkt1, Rr, Rj, chunk=20000):
    offs = np.arange(-(VOL_WINDOW + 1), -1)
    sd = np.full(Rr.size, np.nan)
    n_ok = np.zeros(Rr.size, dtype=np.int64)
    for s in range(0, Rr.size, chunk):
        rr = Rr[s:s + chunk]
        jj = Rj[s:s + chunk]
        wr = rr[:, None] + offs[None, :]
        inb = wr >= 0
        wr_c = np.where(inb, wr, 0)
        adj = r[wr_c, jj[:, None]] - mkt1[wr_c]
        ok = inb & np.isfinite(adj)
        n = ok.sum(axis=1)
        a0 = np.where(ok, adj, 0.0)
        mean = a0.sum(axis=1) / np.maximum(n, 1)
        dev = np.where(ok, adj - mean[:, None], 0.0)
        var = (dev * dev).sum(axis=1) / np.maximum(n - 1, 1)
        sd[s:s + chunk] = np.sqrt(var)
        n_ok[s:s + chunk] = n
    return sd, n_ok


def compute_events(data, params):
    leg_fraction = float(params["leg_fraction"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0197 needs the LIQ-500 point-in-time universe (data.universe is None)")
    member = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"])
    T, N = member.shape
    rows = np.arange(T, dtype=np.int32)

    rel = release_sessions(ea)
    del ea

    r = _daily_returns(data)
    mkt1 = _member_mean(r, member)
    r2 = np.full((T, N), np.nan)
    if T > 1:
        r2[1:] = (1.0 + r[:-1]) * (1.0 + r[1:]) - 1.0
    mkt2 = _member_mean(r2, member)
    del r2

    Rr, Rj = np.nonzero(rel)
    react_at_R = np.full((T, N), np.nan, dtype=np.float64)
    keep = Rr + 1 < T
    Rr_k, Rj_k = Rr[keep], Rj[keep]
    if Rr_k.size:
        K = Rr_k + 1
        stock2 = (1.0 + r[Rr_k, Rj_k]) * (1.0 + r[K, Rj_k]) - 1.0
        excess = stock2 - mkt2[K]
        sd, n_ok = _pre_release_vol(r, mkt1, Rr_k, Rj_k)
        good = (n_ok >= VOL_MIN_OBS) & (sd > 0) & np.isfinite(excess)
        val = np.full(Rr_k.size, np.nan)
        val[good] = excess[good] / (np.sqrt(2.0) * sd[good])
        react_at_R[Rr_k, Rj_k] = val
        ref_mask = np.isfinite(val) & member[K, Rj_k]
        ref_rows = K[ref_mask]
        ref_vals = val[ref_mask]
        order = np.argsort(ref_rows, kind="stable")
        ref_rows, ref_vals = ref_rows[order], ref_vals[order]
    else:
        ref_rows = np.zeros(0, dtype=np.int64)
        ref_vals = np.zeros(0)

    pos_c, has_src = anniversary_source_rows(data.dates)

    last_rel = np.where(rel, rows[:, None], -1)
    np.maximum.accumulate(last_rel, axis=0, out=last_rel)

    side = np.zeros((T, N), dtype=np.float64)
    tt = rows[has_src]
    if tt.size == 0:
        return side
    cand = rel[pos_c[has_src]]
    ci, cj = np.nonzero(cand)
    et = tt[ci]
    ej = cj
    lr = last_rel[et, ej]
    gap = et - lr
    ok = (member[et, ej] & tradable[et, ej] & np.isfinite(r[et, ej])
          & (lr >= 0) & (gap >= NO_RELEASE_SESSIONS) & (gap >= PREV_MIN) & (gap <= PREV_MAX))
    et, ej, lr = et[ok], ej[ok], lr[ok]
    if et.size == 0:
        return side
    rv = react_at_R[lr, ej]
    fin = np.isfinite(rv)
    et, ej, rv = et[fin], ej[fin], rv[fin]
    if et.size == 0:
        return side

    uniq_t = np.unique(et)
    lo_q = np.full(T, np.nan)
    hi_q = np.full(T, np.nan)
    a = np.searchsorted(ref_rows, uniq_t - (REF_WINDOW - 1), side="left")
    b = np.searchsorted(ref_rows, uniq_t, side="right")
    for t, s0, s1 in zip(uniq_t, a, b):
        if s1 - s0 >= REF_MIN_VALUES:
            q = np.quantile(ref_vals[s0:s1], [leg_fraction, 1.0 - leg_fraction])
            lo_q[t], hi_q[t] = q[0], q[1]
    lo_e, hi_e = lo_q[et], hi_q[et]
    is_long = np.isfinite(hi_e) & (rv >= hi_e)
    is_short = np.isfinite(lo_e) & (rv <= lo_e) & ~is_long
    side[et[is_long], ej[is_long]] = 1.0
    side[et[is_short], ej[is_short]] = -1.0
    return side


def events(data, params):
    side = compute_events(data, params)
    mask = side != 0
    return mask, side
