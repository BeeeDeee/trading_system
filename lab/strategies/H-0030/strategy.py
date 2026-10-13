"""H-0030 No-news shock reversal in the S&P 500.

Daily decision after the close of day t (rows 0..t only), executed at the open of t+1.

Rules of the card and where they are implemented:
 (U) universe: PIT S&P 500 member on t, tradable on t, >= 252 sessions of return history  -> _eligible
 (1) r = (1+ret_co)(1+ret_oc)-1                                                             -> _returns
 (2) peers: first session of each month, corr over the previous 252 sessions (rows s-252..s-1),
     >= 126 common obs per pair, 10 highest-correlated other members; fewer than 10 -> no peers -> _peers
 (3) e = r - mean r of the peers that have a return on t                                     -> _shocks
 (4) z = e / std(e over t-63..t-1), >= 40 obs                                                -> _zscores
 (5) news exclusion: ea_reaction_day == 1 on t or t-1, or NaN on t -> not eligible           -> _eligible
 (6) top leg_pct by z short, bottom leg_pct long (round half up, >= 1 name)                  -> cohorts
 (7) staggered cohorts, each +-1/(2*hold_days) split equally per leg, held hold_days sessions -> target_weights
 (8) a name that stops being a member / stops trading / delists is closed and its slot stays cash
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 5, "leg_pct": 0.02}

HISTORY_MIN = 252      # sessions of return history to be in the universe
CORR_WINDOW = 252      # sessions for the peer correlation
MIN_COMMON = 126       # common observations for a pair correlation
N_PEERS = 10
VOL_WINDOW = 63        # sessions t-63..t-1 for the shock std
MIN_VOL_OBS = 40
_VAR_EPS = 1e-12


def _returns(data):
    """Close-to-close total return and its validity mask (stock listed and trading on t)."""
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
    """Peers decided on month-start row s from returns of rows s-252..s-1.

    Returns (cols, peer_cols): the member columns that have 10 peers and their (m, 10) peer columns.
    """
    idx = np.flatnonzero(member_row)
    lo = max(0, s - CORR_WINDOW)
    if s - lo < MIN_COMMON or idx.size <= N_PEERS:
        return idx[:0], None
    m = valid[lo:s][:, idx].astype(float)
    x = np.where(m > 0, r[lo:s][:, idx], 0.0)
    n = m.T @ m                       # common observations per pair
    sx = x.T @ m                      # sum of x_i over the rows where i and j both trade
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
    """Peer-adjusted shock e (T, N); NaN where the stock has no peers that month or no return."""
    T, N = r.shape
    e = np.full((T, N), np.nan)
    starts = _month_start_rows(data.dates)
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else T
        cols, pcols = _peers(r, valid, member[s], s)
        if pcols is None or cols.size == 0:
            continue
        rr = r[s:end]
        pr = rr[:, pcols]                              # (L, m, 10)
        fin = np.isfinite(pr)
        cnt = fin.sum(axis=2)
        sm = np.where(fin, pr, 0.0).sum(axis=2)
        pm = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        e[s:end, cols] = rr[:, cols] - pm
    return e


def _zscores(e):
    """z = e_t / sample std of e over rows t-63..t-1 (>= 40 finite observations)."""
    T, N = e.shape
    fin = np.isfinite(e)
    x = np.where(fin, e, 0.0)
    c0 = np.zeros((T + 1, N))
    np.cumsum(fin, axis=0, out=c0[1:])
    c1 = np.zeros((T + 1, N))
    np.cumsum(x, axis=0, out=c1[1:])
    c2 = np.zeros((T + 1, N))
    np.cumsum(x * x, axis=0, out=c2[1:])
    hi = np.arange(T)                     # rows < t
    lo = np.maximum(hi - VOL_WINDOW, 0)   # rows >= t-63
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
    release = ea >= 1.0                                     # the flag is 1 on a reaction session, 0 otherwise
    release_prev = np.zeros_like(release)
    release_prev[1:] = release[:-1]
    hist = np.cumsum(valid, axis=0) >= HISTORY_MIN          # sessions with a return up to and incl. t
    # excluded: NaN on t (no EDGAR coverage), release on t, release on t-1 (identical to ea==0 & ea[t-1]!=1 on 0/1/NaN data)
    news_ok = np.isfinite(ea) & ~release & ~release_prev
    tradable = np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
    return member & tradable & hist & news_ok & np.isfinite(z)


def cohorts(data, params):
    """(T, N) cohort weights formed after the close of t: +1/(2H k) longs, -1/(2H k) shorts."""
    hold = int(params["hold_days"])
    leg = float(params["leg_pct"])
    if hold < 1 or not (0.0 < leg < 0.5):
        raise ValueError("bad params")
    member = _member(data)
    r, valid = _returns(data)
    e = _shocks(data, r, valid, member)
    z = _zscores(e)
    del e
    elig = _eligible(data, z, valid, member)
    T, N = z.shape
    C = np.zeros((T, N))
    for t in range(T):
        cols = np.flatnonzero(elig[t])
        n = cols.size
        if n == 0:
            continue
        k = max(1, int(np.floor(leg * n + 0.5)))
        if n < 2 * k:
            continue
        order = np.lexsort((cols, z[t, cols]))   # ascending z, ties by column
        longs = cols[order[:k]]
        shorts = cols[order[n - k:]]
        C[t, longs] = 1.0 / (2.0 * hold * k)
        C[t, shorts] = -1.0 / (2.0 * hold * k)
    return C


def target_weights(data, params):
    hold = int(params["hold_days"])
    C = cohorts(data, params)
    member = _member(data)
    alive = member & np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
    T, N = C.shape
    run = np.zeros((T, N), dtype=np.int32)     # consecutive alive rows ending at t
    prev = np.zeros(N, dtype=np.int32)
    for t in range(T):
        prev = np.where(alive[t], prev + 1, 0).astype(np.int32)
        run[t] = prev
    W = np.zeros((T, N))
    for a in range(hold):
        # cohort formed at t-a is in the target rows t-a .. t-a+hold-1 (held open t-a+1 .. open t-a+1+hold)
        if a >= T:
            break
        keep = run[a:] >= a + 1                  # alive on every row from formation to t
        W[a:] += np.where(keep, C[:T - a], 0.0)
    return W
