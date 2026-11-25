"""H-0204 Expectation ratchet at the next release (LIQ-500).

Long last quarter's big negative 8-K reactors, short big positive ones, for 5 sessions,
entered lead_sessions sessions before the predicted next release (last year's reaction
session + 364 calendar days).

Rules of the card -> code:
 (1) reaction z of a release R: 2-session total return close R-1 -> close R+1 minus the
     equal-weight mean of LIQ-500 members (membership at R-1) over the same two sessions,
     divided by sqrt(2) * std of the stock's daily total return over the 63 sessions
     ending at R-2 (>= 40 valid returns). Stored at R, used only on rows t >= R+1.
 (2) R_last(j, t) = most recent reaction session with R_last + 1 <= t; eligible iff
     40 <= t - R_last <= 110.
 (3) predicted release = first weekday on or after date(R_old) + 364 days for reaction
     sessions R_old with 230 <= t - R_old <= 290; event iff that date is exactly
     lead_sessions weekdays after date(t) (weekday arithmetic, future holidays unknown).
     No event if ea_reaction_day is NaN on any of the rows t-290 .. t-1.
 (4) side -1 if z(R_last) >= z_min, +1 if z(R_last) <= -z_min.
 (5) fixed slots of 0.04 * side on decision rows t .. t+4 (open t+1 -> open t+6), one slot
     per stock, gross cap 1.0 (25 slots), new slots by |z| descending, slots closed early
     when the stock leaves the universe or is no longer listed.
"""
import math

import numpy as np

PARAMS = {"lead_sessions": 2, "z_min": 2.0}

SD_WINDOW = 63        # sessions ending at R-2
SD_MIN_VALID = 40
GAP_MIN = 40          # t - R_last
GAP_MAX = 110
OLD_MIN = 230         # t - R_old
OLD_MAX = 290
YEAR_DAYS = 364       # 52 weeks, same weekday
SLOT_WEIGHT = 0.04
HOLD = 5
GROSS_CAP = 1.0
MAX_SLOTS = int(math.floor(GROSS_CAP / SLOT_WEIGHT + 1e-9))   # 25


def _daily_returns(data):
    """Daily total return close t-1 -> close t and its validity (price on both days)."""
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(close) & np.isfinite(r)
    valid = np.zeros_like(ok)
    valid[1:] = ok[1:] & ok[:-1]
    r = np.where(valid, r, 0.0)
    return r, valid


def reaction_z(data):
    """(T, N) float: z of each reaction session R stored on row R (NaN elsewhere).

    Uses rows up to R+1 only; the caller may use it only on rows t >= R+1."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0204 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = flag.shape
    r, valid = _daily_returns(data)

    # two-session return ending at row k: close k-2 -> close k
    g2 = np.full((T, N), np.nan)
    v2 = np.zeros((T, N), dtype=bool)
    if T >= 2:
        v2[1:] = valid[1:] & valid[:-1]
        g2[1:] = (1.0 + r[:-1]) * (1.0 + r[1:]) - 1.0
    # equal-weight mean of LIQ-500 members at the window start (row k-2)
    m2 = np.full(T, np.nan)
    if T >= 3:
        mem = uni[:-2] & v2[2:]
        cnt = mem.sum(axis=1)
        s = np.where(mem, g2[2:], 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m2[2:] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    z = np.full((T, N), np.nan, dtype=np.float64)
    R, J = np.nonzero(flag == 1.0)
    keep = (R + 1 <= T - 1) & (R - 2 >= 0)
    R, J = R[keep], J[keep]
    if R.size == 0:
        return z
    k = R + 1
    abn = g2[k, J] - m2[k]
    ok2 = v2[k, J] & np.isfinite(abn)
    # std of daily returns over rows R-64 .. R-2 (63 sessions ending at R-2)
    off = np.arange(-(SD_WINDOW + 1), -1)          # -64 .. -2
    idx = R[:, None] + off[None, :]
    inside = idx >= 0
    idxc = np.maximum(idx, 0)
    vv = valid[idxc, J[:, None]] & inside
    xx = np.where(vv, r[idxc, J[:, None]], 0.0)
    n = vv.sum(axis=1)
    nn = np.maximum(n, 2)
    mean = xx.sum(axis=1) / nn
    dev = np.where(vv, xx - mean[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / (nn - 1)
    sd = np.sqrt(var)
    good = ok2 & (n >= SD_MIN_VALID) & (sd > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = abn / (math.sqrt(2.0) * sd)
    z[R[good], J[good]] = zz[good]
    return z


def compute_events(data, params):
    """Event mask, side (+1/-1) and |z(R_last)| per (row, stock), point-in-time."""
    lead = int(params["lead_sessions"])
    z_min = float(params["z_min"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0204 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = flag.shape
    rows = np.arange(T)
    is_ev = flag == 1.0

    # (1) z of every reaction session
    z = reaction_z(data)

    # (2) last release with R_last + 1 <= t
    last_incl = np.where(is_ev, rows[:, None], -1).astype(np.int64)
    np.maximum.accumulate(last_incl, axis=0, out=last_incl)
    r_last = np.full((T, N), -1, dtype=np.int64)
    if T >= 2:
        r_last[1:] = last_incl[:-1]
    gap = rows[:, None] - r_last
    elig = (r_last >= 0) & (gap >= GAP_MIN) & (gap <= GAP_MAX)
    z_last = z[np.maximum(r_last, 0), np.arange(N)[None, :]]
    z_last = np.where(r_last >= 0, z_last, np.nan)

    # NaN flag on any needed session (rows t-290 .. t-1) -> no event
    nan_c = np.zeros((T + 1, N), dtype=np.int64)
    nan_c[1:] = np.cumsum(~np.isfinite(flag), axis=0)
    lo = rows - OLD_MAX
    has_hist = lo >= 0
    nan_in = nan_c[rows] - nan_c[np.maximum(lo, 0)]
    flag_ok = has_hist[:, None] & (nan_in == 0)

    # (3) predicted release from last year's reaction sessions (weekday arithmetic)
    d = np.asarray(data.dates).astype("datetime64[D]")
    d_year = (d.astype(np.int64) + YEAR_DAYS).astype("datetime64[D]")   # + 364 calendar days
    pw = np.busday_offset(d_year, 0, roll="forward")
    tgt = np.busday_offset(d, lead, roll="forward")
    kk = np.searchsorted(pw, tgt, side="left")
    kc = np.minimum(kk, T - 1)
    hit = (kk < T) & (pw[kc] == tgt) & (rows - kc >= OLD_MIN) & (rows - kc <= OLD_MAX)
    pred = hit[:, None] & is_ev[kc, :]

    # (4) side from z(R_last)
    zl_ok = np.isfinite(z_last)
    zf = np.where(zl_ok, z_last, 0.0)
    big = zl_ok & (np.abs(zf) >= z_min)
    event = pred & elig & flag_ok & uni & big
    side = np.where(zf >= z_min, -1.0, np.where(zf <= -z_min, 1.0, 0.0))
    side = np.where(event, side, 0.0)
    absz = np.where(event, np.abs(zf), 0.0)
    return event, side, absz


def build_book(event, side, absz, universe, listed):
    """Fixed 0.04 slots held 5 decision rows, one per stock, gross cap 1.0."""
    T, N = event.shape
    w = np.zeros((T, N), dtype=np.float64)
    remaining = np.zeros(N, dtype=np.int64)
    slot_w = np.zeros(N, dtype=np.float64)
    for s in range(T):
        # close slots of stocks that left the universe or are no longer listed
        gone = ~(universe[s] & listed[s])
        remaining[gone] = 0
        n_open = int((remaining > 0).sum())
        cand = np.nonzero(event[s] & (remaining == 0))[0]
        room = MAX_SLOTS - n_open
        if cand.size and room > 0:
            order = np.argsort(-absz[s, cand], kind="stable")
            take = cand[order[:room]]
            remaining[take] = HOLD
            slot_w[take] = SLOT_WEIGHT * side[s, take]
        active = remaining > 0
        w[s] = np.where(active, slot_w, 0.0)
        remaining[active] -= 1
    return w


def target_weights(data, params):
    event, side, absz = compute_events(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    return build_book(event, side, absz, uni, listed)
