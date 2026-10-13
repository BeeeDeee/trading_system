"""H-0030 mechanism diagnostic: the daily no-news shock cohorts of the strategy.

Event on (t, j): after the close of t, j is an eligible S&P 500 member (tradable on t, >= 252 sessions of
return history, peers this month, z defined, ea_reaction_day == 0 on t and != 1 on t-1) and is in the bottom
leg_pct (side +1, long leg) or top leg_pct (side -1, short leg) of the day's eligible members by the
peer-adjusted, volatility-standardized one-day return z. Identical code to strategy.cohorts (copied, since the
diagnostic may only import numpy and lab.framework).
"""
import numpy as np
from lab.framework.api import period_starts

HISTORY_MIN = 252
CORR_WINDOW = 252
MIN_COMMON = 126
N_PEERS = 10
VOL_WINDOW = 63
MIN_VOL_OBS = 40
_VAR_EPS = 1e-12


def _returns(data):
    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    valid = (np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
             & np.isfinite(ret_co) & np.isfinite(ret_oc))
    r = np.where(valid, (1.0 + np.where(valid, ret_co, 0.0)) * (1.0 + np.where(valid, ret_oc, 0.0)) - 1.0, np.nan)
    return r, valid


def _month_start_rows(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return np.flatnonzero(ps)
    return np.asarray(ps, dtype=int)


def _peers(r, valid, member_row, s):
    idx = np.flatnonzero(member_row)
    lo = max(0, s - CORR_WINDOW)
    if s - lo < MIN_COMMON or idx.size <= N_PEERS:
        return idx[:0], None
    m = valid[lo:s][:, idx].astype(float)
    x = np.where(m > 0, r[lo:s][:, idx], 0.0)
    n = m.T @ m
    sx = x.T @ m
    sxx = (x * x).T @ m
    sxy = x.T @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx * sx / n
        vy = vx.T
        corr = cov / np.sqrt(vx * vy)
    ok = (n >= MIN_COMMON) & (vx > _VAR_EPS) & (vy > _VAR_EPS) & np.isfinite(corr)
    np.fill_diagonal(ok, False)
    corr = np.where(ok, corr, -np.inf)
    order = np.argsort(-corr, axis=1, kind="stable")[:, :N_PEERS]
    has = ok.sum(axis=1) >= N_PEERS
    return idx[has], idx[order[has]]


def _shocks(data, r, valid, member):
    T, N = r.shape
    e = np.full((T, N), np.nan)
    starts = _month_start_rows(data.dates)
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else T
        cols, pcols = _peers(r, valid, member[s], s)
        if pcols is None or cols.size == 0:
            continue
        rr = r[s:end]
        pr = rr[:, pcols]
        fin = np.isfinite(pr)
        cnt = fin.sum(axis=2)
        sm = np.where(fin, pr, 0.0).sum(axis=2)
        pm = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        e[s:end, cols] = rr[:, cols] - pm
    return e


def _zscores(e):
    T, N = e.shape
    fin = np.isfinite(e)
    x = np.where(fin, e, 0.0)
    c0 = np.zeros((T + 1, N))
    np.cumsum(fin, axis=0, out=c0[1:])
    c1 = np.zeros((T + 1, N))
    np.cumsum(x, axis=0, out=c1[1:])
    c2 = np.zeros((T + 1, N))
    np.cumsum(x * x, axis=0, out=c2[1:])
    hi = np.arange(T)
    lo = np.maximum(hi - VOL_WINDOW, 0)
    n = c0[hi] - c0[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    del c0, c1, c2
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        z = e / sd
    ok = fin & (n >= MIN_VOL_OBS) & (var > _VAR_EPS) & np.isfinite(z)
    return np.where(ok, z, np.nan)


def _member(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0030 needs the point-in-time S&P 500 universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def _eligible(data, z, valid, member):
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=float)
    release = ea >= 1.0
    release_prev = np.zeros_like(release)
    release_prev[1:] = release[:-1]
    hist = np.cumsum(valid, axis=0) >= HISTORY_MIN
    news_ok = np.isfinite(ea) & ~release & ~release_prev
    tradable = np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
    return member & tradable & hist & news_ok & np.isfinite(z)


def events(data, params):
    leg = float(params["leg_pct"])
    if not (0.0 < leg < 0.5):
        raise ValueError("bad leg_pct")
    member = _member(data)
    r, valid = _returns(data)
    e = _shocks(data, r, valid, member)
    z = _zscores(e)
    del e
    elig = _eligible(data, z, valid, member)
    T, N = z.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for t in range(T):
        cols = np.flatnonzero(elig[t])
        n = cols.size
        if n == 0:
            continue
        k = max(1, int(np.floor(leg * n + 0.5)))
        if n < 2 * k:
            continue
        order = np.lexsort((cols, z[t, cols]))
        longs = cols[order[:k]]
        shorts = cols[order[n - k:]]
        mask[t, longs] = True
        side[t, longs] = 1.0
        mask[t, shorts] = True
        side[t, shorts] = -1.0
    return mask, side
