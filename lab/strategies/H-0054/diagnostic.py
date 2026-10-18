"""H-0054 mechanism diagnostic: leg entries at monthly rebalances.

Event: at a rebalance row t (last session of the month), a stock ENTERS the long leg (in the top leg_quantile
of the residual-momentum score at t, not in the long leg at the previous monthly rebalance; side +1) or ENTERS
the short leg (bottom leg_quantile, not in the short leg at the previous rebalance; side -1). Steps 1-3 are a
verbatim copy of strategy/strategy.py (the diagnostic may not import the strategy module).
"""
import numpy as np

BLOCK = 21
N_BLOCKS = 36
MIN_VALID_SESSIONS = 15
MIN_REG_BLOCKS = 24
MIN_LEG = 10
FIRST_ROW = BLOCK * N_BLOCKS - 1


def _good_fridays_and_memorial_days(years):
    ys = np.asarray(years, dtype=np.int64)
    a = ys % 19
    b = ys // 100
    c = ys % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    month_start = ((ys - 1970) * 12 + month - 1).astype("datetime64[M]").astype("datetime64[D]")
    one = np.ones(1, dtype=np.int64).astype("timedelta64[D]")[0]
    easter = month_start + (day - 1).astype("timedelta64[D]")
    good_friday = easter - 2 * one
    june1 = ((ys - 1970) * 12 + 5).astype("datetime64[M]").astype("datetime64[D]")
    memorial = np.busday_offset(june1 - one, 0, roll="backward", weekmask="Mon")
    return np.sort(np.concatenate([good_friday, memorial]))


def rebalance_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    flags = np.zeros(T, dtype=bool)
    if T == 0:
        return flags
    years = d.astype("datetime64[Y]").astype(np.int64) + 1970
    hol = _good_fridays_and_memorial_days(np.arange(years.min(), years.max() + 2))
    one = np.ones(1, dtype=np.int64).astype("timedelta64[D]")[0]
    nxt = np.busday_offset(d + one, 0, roll="forward", holidays=hol)
    mon = d.astype("datetime64[M]")
    natural = nxt.astype("datetime64[M]") != mon
    flags[:] = natural
    flagged_this_month = False
    for t in range(T):
        if t > 0 and mon[t] != mon[t - 1]:
            if not flagged_this_month:
                flags[t] = True
            flagged_this_month = False
        if natural[t]:
            flagged_this_month = True
    return flags


def _cumulate(data):
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + co) * (1.0 + oc) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64)) & np.isfinite(r)
    T, N = r.shape
    wipe = valid & (r <= -1.0)
    good = valid & ~wipe
    lg = np.log1p(np.where(good, r, 0.0))
    C = np.zeros((T + 1, N), dtype=np.float64)
    np.cumsum(lg, axis=0, out=C[1:])
    CV = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(valid, axis=0, out=CV[1:])
    CW = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(wipe, axis=0, out=CW[1:])
    return C, CV, CW


def _block_returns(t, C, CV, CW):
    k = np.arange(1, N_BLOCKS + 1)
    end = t - BLOCK * (k - 1)
    start = end - BLOCK + 1
    lr = C[end + 1] - C[start]
    nv = CV[end + 1] - CV[start]
    nw = CW[end + 1] - CW[start]
    br = np.where(nw > 0, -1.0, np.expm1(lr))
    bvalid = nv >= MIN_VALID_SESSIONS
    return br, bvalid, start


def _score_from_blocks(br, bvalid, member_first, eligible, formation_months):
    br = np.asarray(br, dtype=np.float64)
    bvalid = np.asarray(bvalid, dtype=bool)
    fm = int(formation_months)
    if fm < 3 or fm > N_BLOCKS:
        raise ValueError("formation_months must be in [3, 36]")
    mk = bvalid & np.asarray(member_first, dtype=bool)
    cnt = mk.sum(axis=1)
    msum = np.where(mk, br, 0.0).sum(axis=1)
    m = np.full(br.shape[0], np.nan)
    np.divide(msum, cnt, out=m, where=cnt > 0)
    mvalid = np.isfinite(m)
    elig = np.asarray(eligible, dtype=bool)
    W = bvalid & mvalid[:, None] & elig[None, :]
    n = W.sum(axis=0).astype(np.float64)
    m0 = np.where(mvalid, m, 0.0)[:, None]
    rf = np.where(W, br, 0.0)
    mf = np.where(W, m0, 0.0)
    nn = np.maximum(n, 1.0)
    mx = mf.sum(axis=0) / nn
    ry = rf.sum(axis=0) / nn
    dm = np.where(W, m0 - mx[None, :], 0.0)
    dr = np.where(W, br - ry[None, :], 0.0)
    sxx = (dm * dm).sum(axis=0)
    sxy = (dm * dr).sum(axis=0)
    ok = elig & (n >= MIN_REG_BLOCKS) & (sxx > 0)
    beta = np.full(br.shape[1], np.nan)
    np.divide(sxy, sxx, out=beta, where=ok)
    alpha = ry - beta * mx
    form_valid = W[1:fm].all(axis=0)
    e = np.where(W[1:fm], br[1:fm] - alpha[None, :] - beta[None, :] * m0[1:fm], 0.0)
    s = e.sum(axis=0)
    sd = e.std(axis=0, ddof=1)
    good = ok & form_valid & (sd > 0) & np.isfinite(s) & np.isfinite(sd)
    score = np.full(br.shape[1], np.nan)
    np.divide(s, sd, out=score, where=good)
    return score


def _legs(score, leg_quantile):
    idx = np.flatnonzero(np.isfinite(score))
    ns = idx.size
    nleg = max(int(np.floor(float(leg_quantile) * ns + 1e-9)), MIN_LEG)
    if 2 * nleg > ns:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = idx[np.argsort(-score[idx], kind="stable")]
    return order[:nleg], order[ns - nleg:]


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0054 needs the point-in-time LIQ-500 universe (data.universe is None)")
    U = np.asarray(data.universe, dtype=bool)
    TR = np.asarray(data.tradable, dtype=bool)
    fm = int(params["formation_months"])
    q = float(params["leg_quantile"])
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    C, CV, CW = _cumulate(data)
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    for t in np.flatnonzero(rebalance_rows(data.dates)):
        if t < FIRST_ROW:
            continue
        br, bvalid, start = _block_returns(t, C, CV, CW)
        elig = U[t] & TR[t]
        score = _score_from_blocks(br, bvalid, U[start], elig, fm)
        li, si = _legs(score, q)
        cur_long = np.zeros(N, dtype=bool)
        cur_short = np.zeros(N, dtype=bool)
        cur_long[li] = True
        cur_short[si] = True
        new_long = cur_long & ~prev_long & elig
        new_short = cur_short & ~prev_short & elig
        mask[t] = new_long | new_short
        side[t] = np.where(new_long, 1.0, np.where(new_short, -1.0, 0.0))
        prev_long = cur_long
        prev_short = cur_short
    return mask, side
