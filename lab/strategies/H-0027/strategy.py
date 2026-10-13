"""H-0027 Release-anchored trading-window net flow (LIQ-500, dollar-neutral).

After the close of t: for every LIQ-500 member with a score,
    net_flow = (-ncfcommon - sbcomp) / marketcap   (sbcomp NaN -> 0; ncfcommon/marketcap NaN or marketcap <= 0 -> no score)
If >= 100 members have a score, the leg_quantile / 1 - leg_quantile quantiles of net_flow over all scored members are
the thresholds. A member with ea_reaction_day[t] == 1 and net_flow >= upper is a long event, <= lower a short event.
Events are held hold_days sessions (decision rows t .. t+hold_days-1), a new event restarts the clock with its side.
Weights each row: longs +min(0.05, 0.5/n_long), shorts -min(0.05, 0.5/n_short), residual net offset by an
equal-weight basket of all other LIQ-500 members (-net / M). No open event -> all zero (cash).
"""
import numpy as np

PARAMS = {"hold_days": 5, "leg_quantile": 0.2}

MIN_SCORED = 100      # card: fewer than 100 scored members -> no event on t
MAX_NAME_WEIGHT = 0.05
LEG_BUDGET = 0.5


def net_flow_score(data):
    """(T, N) float64 net_flow, NaN where there is no score (or not a LIQ-500 member)."""
    member = np.asarray(data.universe, dtype=bool)
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=np.float64)
    sbc = np.asarray(data.extras["sf1_art_sbcomp"], dtype=np.float64)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=np.float64)
    sbc0 = np.where(np.isnan(sbc), 0.0, sbc)
    ok = member & ~np.isnan(ncf) & ~np.isnan(mcap) & (mcap > 0)
    safe_mcap = np.where(ok, mcap, 1.0)
    score = np.where(ok, (-ncf - sbc0) / safe_mcap, np.nan)
    return score


def event_sides(data, params):
    """(T, N) float: +1 long event, -1 short event, 0 none (decided after the close of t from rows <= t)."""
    if data.universe is None:
        raise ValueError("H-0027 needs the LIQ-500 universe (data.universe is None)")
    q = float(params["leg_quantile"])
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    score = net_flow_score(data)
    T, N = score.shape
    n_scored = np.sum(~np.isnan(score), axis=1)
    rows = np.flatnonzero(n_scored >= MIN_SCORED)
    lower = np.full(T, np.nan)
    upper = np.full(T, np.nan)
    if rows.size:
        qs = np.nanquantile(score[rows], [q, 1.0 - q], axis=1)
        lower[rows] = qs[0]
        upper[rows] = qs[1]
    valid = ~np.isnan(score) & ~np.isnan(lower)[:, None]
    announce = member & tradable & (ea == 1.0) & valid
    with np.errstate(invalid="ignore"):
        is_long = announce & (score >= upper[:, None])
        is_short = announce & (score <= lower[:, None])
    both = is_long & is_short          # only if upper == lower (degenerate ties): no event
    side = np.zeros((T, N), dtype=np.float64)
    side[is_long & ~both] = 1.0
    side[is_short & ~both] = -1.0
    return side


def target_weights(data, params):
    hold = int(params["hold_days"])
    side_ev = event_sides(data, params)
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = side_ev.shape
    w = np.zeros((T, N), dtype=np.float64)
    cur_side = np.zeros(N, dtype=np.float64)
    start = np.full(N, -(10 ** 9), dtype=np.int64)
    for t in range(T):
        ev = side_ev[t]
        new = ev != 0.0
        if new.any():
            cur_side[new] = ev[new]          # a new event restarts the clock with its side
            start[new] = t
        expired = (t - start) >= hold        # held on decision rows t0 .. t0+hold-1
        cur_side[expired] = 0.0
        gone = ~(member[t] & tradable[t])    # not tradable any more / left LIQ-500: close at next possible open
        cur_side[gone] = 0.0
        longs = cur_side > 0
        shorts = cur_side < 0
        n_l = int(longs.sum())
        n_s = int(shorts.sum())
        if n_l == 0 and n_s == 0:
            continue                          # cash
        row = w[t]
        w_l = min(MAX_NAME_WEIGHT, LEG_BUDGET / n_l) if n_l else 0.0
        w_s = min(MAX_NAME_WEIGHT, LEG_BUDGET / n_s) if n_s else 0.0
        row[longs] = w_l
        row[shorts] = -w_s
        net = n_l * w_l - n_s * w_s
        basket = member[t] & ~longs & ~shorts
        m = int(basket.sum())
        if m > 0 and net != 0.0:
            row[basket] = -net / m
        gross = np.abs(row).sum()
        if gross > 1.0:                       # float rounding only (max gross is exactly 1 by construction)
            row *= 1.0 / gross
    return w
