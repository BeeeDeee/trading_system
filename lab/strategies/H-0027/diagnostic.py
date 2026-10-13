"""H-0027 mechanism event: a LIQ-500 member (tradable) has its 8-K Item 2.02 reaction session on t and its
trailing net equity flow (-ncfcommon - sbcomp) / marketcap is in the top leg_quantile (+1) or bottom leg_quantile (-1)
of all LIQ-500 members with a score on t (no event if fewer than 100 members have a score).
Same rule as strategy.event_sides (duplicated because the diagnostic may only import numpy / lab.framework)."""
import numpy as np

MIN_SCORED = 100


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0027 needs the LIQ-500 universe (data.universe is None)")
    q = float(params["leg_quantile"])
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=np.float64)
    sbc = np.asarray(data.extras["sf1_art_sbcomp"], dtype=np.float64)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=np.float64)
    sbc0 = np.where(np.isnan(sbc), 0.0, sbc)
    ok = member & ~np.isnan(ncf) & ~np.isnan(mcap) & (mcap > 0)
    safe_mcap = np.where(ok, mcap, 1.0)
    score = np.where(ok, (-ncf - sbc0) / safe_mcap, np.nan)
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
    both = is_long & is_short
    mask = (is_long | is_short) & ~both
    side = np.where(is_long & ~both, 1.0, np.where(is_short & ~both, -1.0, 0.0))
    return mask, side
