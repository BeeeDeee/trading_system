"""H-0019 mechanism diagnostic: new usable ART filing of a LIQ-500 stock with net window-flow score
F = (-ncfcommon - sbcomp) / marketcap in the top `quantile` (+1) or bottom `quantile` (-1) of the
LIQ-500 cross-section of F on that day. Same event logic as strategy.compute_events."""
import numpy as np


def _mid_rank_pct(vals):
    n = vals.shape[0]
    u, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg_rank = cum - (cnt - 1) / 2.0
    return (avg_rank[inv.reshape(-1)] - 0.5) / n


def events(data, params):
    q = float(params["quantile"])
    if data.universe is None:
        raise ValueError("H-0019 needs the LIQ-500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=np.float64)
    sbc = np.asarray(data.extras["sf1_art_sbcomp"], dtype=np.float64)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=np.float64)

    valid = np.isfinite(ncf) & np.isfinite(sbc) & np.isfinite(mcap) & (mcap > 0)
    F = np.full(mcap.shape, np.nan)
    F[valid] = (-ncf[valid] - sbc[valid]) / mcap[valid]

    T, N = F.shape
    nf = np.zeros((T, N), dtype=bool)
    if T > 1:
        nf[1:] = np.isfinite(mcap[1:]) & np.isfinite(mcap[:-1]) & (mcap[1:] != mcap[:-1])

    cand = nf & member & tradable & np.isfinite(F)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.nonzero(cand.any(axis=1))[0]:
        idx = np.nonzero(member[t] & np.isfinite(F[t]))[0]
        if idx.size == 0:
            continue
        pct = _mid_rank_pct(F[t, idx])
        row_side = np.zeros(N)
        row_side[idx[pct >= 1.0 - q]] = 1.0
        row_side[idx[pct <= q]] = -1.0
        side[t] = np.where(cand[t], row_side, 0.0)
    mask = side != 0.0
    return mask, side
