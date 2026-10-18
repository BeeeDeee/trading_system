"""H-0054 Residual momentum, beta-neutral, LIQ-500, monthly.

Long the top / short the bottom leg_quantile of the volatility-scaled market-adjusted residual return over
blocks 2..formation_months (21-session blocks counted back from the rebalance close), legs sized to zero
estimated beta (with a dollar-neutral fallback). See the IMPL_DONE summary for the rule-to-code mapping.
"""
import numpy as np

PARAMS = {"formation_months": 12, "leg_quantile": 0.2}

BLOCK = 21                # sessions per block
N_BLOCKS = 36             # regression window in blocks (fixed by the card, not a parameter)
MIN_VALID_SESSIONS = 15   # a block return needs at least this many valid sessions
MIN_REG_BLOCKS = 24       # OLS needs at least this many valid blocks
MIN_LEG = 10              # at least 10 names per leg
MIN_LEG_BETA = 0.3        # below this leg beta -> dollar-neutral
G_LO, G_HI = 0.35, 0.65   # allowed range of the beta-neutral leg gross
FIRST_ROW = BLOCK * N_BLOCKS - 1   # first row index at which all 36 blocks exist in the data


# ----------------------------------------------------------------------------------------------- calendar
def _good_fridays_and_memorial_days(years):
    """Good Friday (Gregorian computus) and Memorial Day (last Monday of May) for each year, datetime64[D]."""
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
    """Bool (T,): row t is the last XNYS session of its calendar month, from the date of row t alone.

    t is flagged when the next weekday after t that is not a Good Friday or Memorial Day lies in a later month
    (the only regular NYSE holidays that can fall on the last weekday of a month; NYSE never closes on Dec 31).
    Fallback for any other unforeseen closure on the last weekday: if a month ended without a flagged row, the
    first row of the next month is flagged (one session late). Uses rows <= t only.
    """
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
    # fallback: a month (with rows) that got no natural flag -> flag the first row of the following month
    flagged_this_month = False
    for t in range(T):
        if t > 0 and mon[t] != mon[t - 1]:
            if not flagged_this_month:
                flags[t] = True
            flagged_this_month = False
        if natural[t]:
            flagged_this_month = True
    return flags


# ----------------------------------------------------------------------------------------------- blocks
def daily_inputs(data):
    """Cumulative sums used for block returns. Returns (C, CV, CW): cumulative log(1+r) over valid sessions with
    r > -1, cumulative count of valid sessions, cumulative count of valid wipe-out sessions (r <= -1); each
    (T+1, N) with a leading zero row. A valid session: listed, has a close and a finite total return."""
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + co) * (1.0 + oc) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64)) & np.isfinite(r)
    return cumulate(r, valid)


def cumulate(r, valid):
    r = np.asarray(r, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
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


def block_returns(t, C, CV, CW):
    """Block k = 1..36 (row k-1): the 21 sessions ending at row t - 21 (k - 1). Returns (br, bvalid, first_row)."""
    k = np.arange(1, N_BLOCKS + 1)
    end = t - BLOCK * (k - 1)
    start = end - BLOCK + 1
    lr = C[end + 1] - C[start]
    nv = CV[end + 1] - CV[start]
    nw = CW[end + 1] - CW[start]
    br = np.where(nw > 0, -1.0, np.expm1(lr))
    bvalid = nv >= MIN_VALID_SESSIONS
    return br, bvalid, start


def score_from_blocks(br, bvalid, member_first, eligible, formation_months):
    """Steps 1-2. br, bvalid, member_first: (36, N), row k-1 = block k; eligible (N,): member and tradable at t.
    Returns (score, beta), NaN where the stock gets no score / no beta."""
    br = np.asarray(br, dtype=np.float64)
    bvalid = np.asarray(bvalid, dtype=bool)
    fm = int(formation_months)
    if fm < 3 or fm > N_BLOCKS:
        raise ValueError("formation_months must be in [3, 36]")
    # market block return: EW mean over stocks that were members on the block's first session with a valid return
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
    # step 2: residuals over formation blocks k = 2..fm (rows 1..fm-1), all must be valid
    form_valid = W[1:fm].all(axis=0)
    e = np.where(W[1:fm], br[1:fm] - alpha[None, :] - beta[None, :] * m0[1:fm], 0.0)
    s = e.sum(axis=0)
    sd = e.std(axis=0, ddof=1)
    good = ok & form_valid & (sd > 0) & np.isfinite(s) & np.isfinite(sd)
    score = np.full(br.shape[1], np.nan)
    np.divide(s, sd, out=score, where=good)
    beta = np.where(ok, beta, np.nan)
    return score, beta


def legs(score, leg_quantile):
    """Step 3. Returns (long_idx, short_idx); both empty if fewer than 2 * leg size stocks are scored."""
    idx = np.flatnonzero(np.isfinite(score))
    ns = idx.size
    nleg = max(int(np.floor(float(leg_quantile) * ns + 1e-9)), MIN_LEG)
    if 2 * nleg > ns:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = idx[np.argsort(-score[idx], kind="stable")]
    return order[:nleg], order[ns - nleg:]


def leg_gross(beta, long_idx, short_idx):
    """Step 4. Beta-neutral leg gross (g_L, g_S) with the dollar-neutral fallback."""
    b_l = float(np.mean(beta[long_idx]))
    b_s = float(np.mean(beta[short_idx]))
    if not (b_l >= MIN_LEG_BETA and b_s >= MIN_LEG_BETA):
        return 0.5, 0.5
    g_l = b_s / (b_l + b_s)
    g_s = b_l / (b_l + b_s)
    if not (G_LO <= g_l <= G_HI and G_LO <= g_s <= G_HI):
        return 0.5, 0.5
    return g_l, g_s


def rebalance_legs(data, params):
    """Yields (t, long_idx, short_idx, beta) for every rebalance row t with all 36 blocks in the data."""
    if data.universe is None:
        raise ValueError("H-0054 needs the point-in-time LIQ-500 universe (data.universe is None)")
    U = np.asarray(data.universe, dtype=bool)
    TR = np.asarray(data.tradable, dtype=bool)
    fm = int(params["formation_months"])
    q = float(params["leg_quantile"])
    C, CV, CW = daily_inputs(data)
    flags = rebalance_rows(data.dates)
    out = []
    for t in np.flatnonzero(flags):
        if t < FIRST_ROW:
            continue
        br, bvalid, start = block_returns(t, C, CV, CW)
        score, beta = score_from_blocks(br, bvalid, U[start], U[t] & TR[t], fm)
        li, si = legs(score, q)
        out.append((int(t), li, si, beta))
    return out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    U = np.asarray(data.universe, dtype=bool) if data.universe is not None else None
    TR = np.asarray(data.tradable, dtype=bool)
    reb = rebalance_legs(data, params)
    for j, (t, li, si, beta) in enumerate(reb):
        row = np.zeros(N)
        if li.size > 0:
            g_l, g_s = leg_gross(beta, li, si)
            row[li] = g_l / li.size
            row[si] = -g_s / si.size
        w[t] = row
        # between rebalances: a held stock that leaves the universe or stops being tradable is closed at the
        # next open (weight 0 on the first such row), and stays in cash until the next rebalance
        a = t + 1
        b = reb[j + 1][0] if j + 1 < len(reb) else T
        held = np.flatnonzero(row != 0)
        if a < b and held.size > 0:
            bad = ~(U[a:b][:, held] & TR[a:b][:, held])
            first = bad & (np.cumsum(bad, axis=0) == 1)
            rr, cc = np.nonzero(first)
            w[a + rr, held[cc]] = 0.0
    return w
