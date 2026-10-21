"""H-0068 mechanism diagnostic: the same events as strategy.py (code duplicated because the
static rules do not allow importing strategy.py here)."""
import numpy as np

WINDOW_END_LAG = 6
PRE_RELEASE = 5
MIN_COVERAGE = 0.9
SD_LO, SD_HI = 65, 6
SD_MIN_OBS = 40
Z_MAX = 2.0


def _daily_returns(ret_co, ret_oc, listed, close):
    co = np.asarray(ret_co, dtype=np.float64)
    oc = np.asarray(ret_oc, dtype=np.float64)
    valid = (np.asarray(listed, dtype=bool) & np.isfinite(co) & np.isfinite(oc)
             & np.isfinite(np.asarray(close, dtype=np.float64)))
    r = np.where(valid, (1.0 + np.where(valid, co, 0.0)) * (1.0 + np.where(valid, oc, 0.0)) - 1.0, 0.0)
    return r, valid


def _window_sum(x, lo, hi):
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


def _formation_return(r, valid, formation_days):
    fd = int(formation_days)
    logs = np.where(valid, np.log(np.maximum(1.0 + r, 1e-12)), 0.0)
    lo = fd + PRE_RELEASE
    s = _window_sum(logs, lo, WINDOW_END_LAG)
    n = _window_sum(valid.astype(np.float64), lo, WINDOW_END_LAG)
    ok = np.isfinite(n) & (n >= MIN_COVERAGE * fd)
    return np.where(ok, np.expm1(np.where(ok, s, 0.0)), np.nan)


def _winner_flag(F, member, winner_quantile):
    elig = member & np.isfinite(F)
    Fm = np.where(elig, F, np.nan)
    thr = np.full(F.shape[0], np.nan)
    rows = np.flatnonzero(elig.any(axis=1))
    if rows.size:
        thr[rows] = np.nanquantile(Fm[rows], 1.0 - float(winner_quantile), axis=1)
    with np.errstate(invalid="ignore"):
        return elig & (Fm >= thr[:, None])


def _reaction_ok(r, valid, member):
    mv = member & valid
    cnt = mv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, np.where(mv, r, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    ev = valid & np.isfinite(m)[:, None]
    e = np.where(ev, r - np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
    s1 = _window_sum(e, SD_LO, SD_HI)
    s2 = _window_sum(e * e, SD_LO, SD_HI)
    n = _window_sum(ev.astype(np.float64), SD_LO, SD_HI)
    ok = np.isfinite(n) & (n >= SD_MIN_OBS)
    nn = np.where(ok, n, 2.0)
    var = np.where(ok, (np.where(ok, s2, 0.0) - np.where(ok, s1, 0.0) ** 2 / nn) / (nn - 1.0), 0.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    ok &= sd > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(ok & ev, e / np.where(sd > 0, sd, 1.0), np.nan)
    return ok & ev & (np.abs(z) <= Z_MAX)


def events(data, params):
    """Event: LIQ-500 member, tradable, ea_reaction_day == 1 on t, top winner_quantile formation
    return (window ending at t-6), |z| <= 2 release reaction. Side -1 (short leg)."""
    if data.universe is None:
        raise ValueError("H-0068 needs the point-in-time LIQ-500 universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    r, valid = _daily_returns(data.ret_co, data.ret_oc, data.listed, data.close)
    F = _formation_return(r, valid, params["formation_days"])
    win = _winner_flag(F, member, params["winner_quantile"])
    calm = _reaction_ok(r, valid, member)
    with np.errstate(invalid="ignore"):
        flag = ea == 1.0
    mask = flag & member & tradable & win & calm
    side = np.where(mask, -1.0, 0.0)
    return mask, side
