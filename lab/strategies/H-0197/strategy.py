"""H-0197: last quarter's 8-K earnings reaction traded around the predicted next release.

LIQ-500, dollar-neutral. Long stocks whose previous release reaction was in the top leg_fraction of
the trailing-252-session LIQ-500 reaction distribution, short the bottom leg_fraction, entered at the
open of the session before the predicted release (anniversary of last year's release) and held
hold_days sessions. The net of the two legs is offset by an equal-weight basket of all other tradable
LIQ-500 members.
"""
import numpy as np

PARAMS = {"hold_days": 5, "leg_fraction": 0.25}

DEDUP_SESSIONS = 20        # flagged session within 20 sessions after an earlier flag is ignored
VOL_WINDOW = 60            # sessions of market-adjusted returns ending at R-2
VOL_MIN_OBS = 40
REF_WINDOW = 252           # reference distribution window (by R+1)
REF_MIN_VALUES = 200
ANNIV_DAYS = 364
LEAD_SESSIONS = 2          # anniversary must be the second session after t
NO_RELEASE_SESSIONS = 30   # no release in the 30 sessions up to and including t
PREV_MIN, PREV_MAX = 40, 120
LEG_CAP = 0.05
LEG_BUDGET = 0.5


def _daily_returns(data):
    """Close-to-close total return of row t, NaN where row t or t-1 has no price."""
    priced = np.isfinite(np.asarray(data.close))
    valid = np.zeros_like(priced)
    valid[1:] = priced[1:] & priced[:-1]
    valid &= np.asarray(data.listed, dtype=bool)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = np.where(valid & np.isfinite(r), r, np.nan)
    return r


def _member_mean(x, member):
    """Equal-weight mean of x over members with a finite value, per row (NaN if none)."""
    ok = member & np.isfinite(x)
    cnt = ok.sum(axis=1)
    s = np.where(ok, x, 0.0).sum(axis=1)
    out = np.full(x.shape[0], np.nan)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


def release_sessions(ea):
    """Rule (1): ea_reaction_day == 1 is a release unless another flag lies in the previous 20 sessions."""
    flag = np.asarray(ea) == 1.0                         # NaN (before the first filing) -> no release
    T, N = flag.shape
    rows = np.arange(T)
    cflag = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(flag, axis=0, out=cflag[1:])
    lo_idx = np.maximum(rows - DEDUP_SESSIONS, 0)
    prev_flags = cflag[rows] - cflag[lo_idx]            # flags in rows t-20 .. t-1
    return flag & (prev_flags == 0)


def anniversary_source_rows(dates):
    """Rule (4)/(5): for each row t the row of a past session L with date(L) + 364 == busday_offset(date_t, 2).

    Returns (pos, has): has[t] True when such a session exists in rows < t, pos[t] its row.
    Weekday calendar (np.busday_offset, no exchange holidays): A(L) = date(L) + 364 is a weekday
    because L is a session, and "the second session after t" is the second weekday after date_t.
    """
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
    """Sample std (ddof=1) and count of valid market-adjusted daily returns over rows R-61 .. R-2."""
    offs = np.arange(-(VOL_WINDOW + 1), -1)              # -61 .. -2 (60 rows, ending at R-2)
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
    """Event sides (T, N): +1 long event, -1 short event, 0 none, decided after the close of row t."""
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

    # (2) reaction of each release R, known at row R+1
    r = _daily_returns(data)
    mkt1 = _member_mean(r, member)
    r2 = np.full((T, N), np.nan)
    if T > 1:
        r2[1:] = (1.0 + r[:-1]) * (1.0 + r[1:]) - 1.0   # sessions s-1 and s
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
        # (3) reference values: members at R+1 with a reaction, ordered by R+1
        ref_mask = np.isfinite(val) & member[K, Rj_k]
        ref_rows = K[ref_mask]
        ref_vals = val[ref_mask]
        order = np.argsort(ref_rows, kind="stable")
        ref_rows, ref_vals = ref_rows[order], ref_vals[order]
    else:
        ref_rows = np.zeros(0, dtype=np.int64)
        ref_vals = np.zeros(0)

    # (4) predicted release: anniversary A(L) = first session on/after date(L)+364 (weekday calendar).
    # Event row t needs A(L) == second session after t, i.e. date(L) + 364 == busday_offset(date_t, 2).
    pos_c, has_src = anniversary_source_rows(data.dates)

    # most recent release row (<= t), -1 if none
    last_rel = np.where(rel, rows[:, None], -1)
    np.maximum.accumulate(last_rel, axis=0, out=last_rel)

    side = np.zeros((T, N), dtype=np.float64)
    tt = rows[has_src]
    if tt.size == 0:
        return side
    cand = rel[pos_c[has_src]]                           # (n_t, N): release on the anniversary source date
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

    # quantiles of the reference distribution on each event row
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


def target_weights(data, params):
    hold = int(params["hold_days"])
    side_ev = compute_events(data, params)
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    return book_weights(side_ev, member, tradable, hold)


def book_weights(side_ev, member, tradable, hold):
    """Rules (6) and (7): open positions from events, leg weights and the hedge basket."""
    T, N = side_ev.shape
    rows = np.arange(T, dtype=np.int32)

    # (6) open positions: latest event (restarts the hold), held for rows ev .. ev+hold-1,
    # closed for good once the stock leaves LIQ-500 after the event.
    has_ev = side_ev != 0
    last_ev = np.where(has_ev, rows[:, None], -1)
    np.maximum.accumulate(last_ev, axis=0, out=last_ev)
    last_out = np.where(~member, rows[:, None], -1)
    np.maximum.accumulate(last_out, axis=0, out=last_out)
    age = rows[:, None] - last_ev
    active = (last_ev >= 0) & (age < hold) & (last_out < last_ev) & member
    pos_side = np.take_along_axis(side_ev, np.maximum(last_ev, 0), axis=0)
    longs = active & (pos_side > 0)
    shorts = active & (pos_side < 0)
    del last_ev, last_out, age, active, pos_side

    # (7) weights
    n_l = longs.sum(axis=1).astype(np.float64)
    n_s = shorts.sum(axis=1).astype(np.float64)
    wl = np.where(n_l > 0, np.minimum(LEG_BUDGET / np.maximum(n_l, 1.0), LEG_CAP), 0.0)
    ws = np.where(n_s > 0, np.minimum(LEG_BUDGET / np.maximum(n_s, 1.0), LEG_CAP), 0.0)
    net = n_l * wl - n_s * ws
    basket = member & tradable & ~longs & ~shorts
    n_b = basket.sum(axis=1).astype(np.float64)
    wb = np.where(n_b > 0, -net / np.maximum(n_b, 1.0), 0.0)

    w = np.zeros((T, N), dtype=np.float64)
    w += np.where(longs, wl[:, None], 0.0)
    w -= np.where(shorts, ws[:, None], 0.0)
    w += np.where(basket, wb[:, None], 0.0)
    return w
