"""H-0068 Post-release attention unwind in past winners (LIQ-500, market-neutral).

Short 12-month winners for 5 sessions after a real 8-K Item 2.02 release session
whose reaction is non-extreme (|z| <= 2); hedge long the 100 most liquid
universe members not in an active slot, so that net weight is zero.
"""
import numpy as np

PARAMS = {"formation_days": 252, "winner_quantile": 0.2}

# fixed rules of the card (not parameters)
WINDOW_END_LAG = 6        # formation window ends at t-6
PRE_RELEASE = 5           # window starts at t - formation_days - 5
MIN_COVERAGE = 0.9        # >= 90 % non-missing returns in the formation window
SD_LO, SD_HI = 65, 6      # residual sd over sessions t-65 .. t-6
SD_MIN_OBS = 40           # at least 40 valid residuals
Z_MAX = 2.0               # |z| <= 2.0
HOLD = 5                  # slot held 5 sessions (decision rows t .. t+4, closed at row t+5)
MAX_SHORT = 0.04          # per-name short cap
SHORT_TOTAL = 0.5         # total short leg cap
HEDGE_N = 100             # hedge: 100 best liq_rank members not in a slot


def daily_returns(ret_co, ret_oc, listed, close):
    """r = (1+ret_co)(1+ret_oc)-1 and the validity mask (listed with a price)."""
    co = np.asarray(ret_co, dtype=np.float64)
    oc = np.asarray(ret_oc, dtype=np.float64)
    valid = (np.asarray(listed, dtype=bool) & np.isfinite(co) & np.isfinite(oc)
             & np.isfinite(np.asarray(close, dtype=np.float64)))
    r = np.where(valid, (1.0 + np.where(valid, co, 0.0)) * (1.0 + np.where(valid, oc, 0.0)) - 1.0, 0.0)
    return r, valid


def window_sum(x, lo, hi):
    """out[t] = sum of x[t-lo .. t-hi] (inclusive), NaN where t-lo < 0. Uses rows <= t-hi only."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    c[1:] = np.cumsum(x, axis=0)
    out = np.full(x.shape, np.nan)
    t = np.arange(T)
    a = t - lo
    b = t - hi
    ok = a >= 0
    out[ok] = c[b[ok] + 1] - c[a[ok]]
    return out


def formation_return(r, valid, formation_days):
    """F[t] = prod(1+r) over t-formation_days-5 .. t-6, minus 1; NaN if coverage < 90 %."""
    fd = int(formation_days)
    logs = np.where(valid, np.log(np.maximum(1.0 + r, 1e-12)), 0.0)
    lo = fd + PRE_RELEASE
    s = window_sum(logs, lo, WINDOW_END_LAG)
    n = window_sum(valid.astype(np.float64), lo, WINDOW_END_LAG)
    ok = np.isfinite(n) & (n >= MIN_COVERAGE * fd)
    return np.where(ok, np.expm1(np.where(ok, s, 0.0)), np.nan)


def winner_flag(F, member, winner_quantile):
    """Among members with a valid F on t: F >= the (1 - q) cross-sectional quantile."""
    elig = member & np.isfinite(F)
    Fm = np.where(elig, F, np.nan)
    thr = np.full(F.shape[0], np.nan)
    rows = np.flatnonzero(elig.any(axis=1))
    if rows.size:
        thr[rows] = np.nanquantile(Fm[rows], 1.0 - float(winner_quantile), axis=1)
    with np.errstate(invalid="ignore"):
        return elig & (Fm >= thr[:, None])


def reaction_ok(r, valid, member):
    """|z| <= 2 with z = e[t]/sd(e over t-65..t-6, >= 40 obs), e = r - equal-weight member mean."""
    mv = member & valid
    cnt = mv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, np.where(mv, r, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    ev = valid & np.isfinite(m)[:, None]
    e = np.where(ev, r - np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
    s1 = window_sum(e, SD_LO, SD_HI)
    s2 = window_sum(e * e, SD_LO, SD_HI)
    n = window_sum(ev.astype(np.float64), SD_LO, SD_HI)
    ok = np.isfinite(n) & (n >= SD_MIN_OBS)
    nn = np.where(ok, n, 2.0)
    var = np.where(ok, (np.where(ok, s2, 0.0) - np.where(ok, s1, 0.0) ** 2 / nn) / (nn - 1.0), 0.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    ok &= sd > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(ok & ev, e / np.where(sd > 0, sd, 1.0), np.nan)
    return ok & ev & (np.abs(z) <= Z_MAX)


def compute_events(data, params):
    """(T, N) bool: event after the close of t (rows <= t only)."""
    if data.universe is None:
        raise ValueError("H-0068 needs the point-in-time LIQ-500 universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    r, valid = daily_returns(data.ret_co, data.ret_oc, data.listed, data.close)
    F = formation_return(r, valid, params["formation_days"])
    win = winner_flag(F, member, params["winner_quantile"])
    calm = reaction_ok(r, valid, member)
    with np.errstate(invalid="ignore"):
        flag = ea == 1.0          # NaN counts as no event
    return flag & member & tradable & win & calm


def active_slots(event, member, tradable):
    """Slot opened at decision row t is active on rows t..t+4; restart on re-trigger;
    closed at the first row where the stock is not a member or not tradable."""
    T, N = event.shape
    active = np.zeros((T, N), dtype=bool)
    rem = np.zeros(N, dtype=np.int64)
    for t in range(T):
        rem = np.where(rem > 0, rem, 0)
        rem[~(member[t] & tradable[t])] = 0
        rem[event[t]] = HOLD
        active[t] = rem > 0
        rem = rem - 1
    return active


def book(active, member, liq_rank):
    """Shorts -min(0.04, 0.5/n) each; long equal-weight hedge of equal total on the 100 best
    liq_rank members not in a slot. Flat when no slot is active."""
    T, N = active.shape
    w = np.zeros((T, N), dtype=np.float64)
    n_act = active.sum(axis=1)
    liq = np.asarray(liq_rank, dtype=np.float64)
    for t in np.flatnonzero(n_act > 0):
        n = int(n_act[t])
        ws = min(MAX_SHORT, SHORT_TOTAL / n)
        w[t, active[t]] = -ws
        cand = np.flatnonzero(member[t] & ~active[t] & np.isfinite(liq[t]))
        if cand.size:
            order = cand[np.argsort(liq[t, cand], kind="stable")][:HEDGE_N]
            w[t, order] = ws * n / order.size
    return w


def target_weights(data, params):
    event = compute_events(data, params)
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    active = active_slots(event, member, tradable)
    return book(active, member, data.extras["liq_rank"])
