"""H-0013 Peer lead-lag catch-up (LIQ-500, market neutral, weekly).

Decision rows: the last weekday of each calendar week (weekday arithmetic, see _week_end_rows).
After the close of t: eligible set E(t), correlation peer map on market-demeaned daily returns over the
trailing corr_window rows, own / peer formation returns over formation_days rows, keep the middle 60 % of
own returns M(t), long the top quintile of M(t) by peer return, short the bottom quintile.
"""
import numpy as np

PARAMS = {"corr_window": 252, "formation_days": 5, "peers_k": 10}

MIN_M = 50            # fewer names in M(t): flat book
N_QUANTILES = 5       # quintiles
ELIG_FRAC = 0.75      # min share of non-missing returns in the corr window
OVERLAP_FRAC = 0.60   # min share of overlapping days of a pair (of corr_window)
OWN_LO = 1            # own percentile rank in [OWN_LO / 5, OWN_HI / 5] = [0.2, 0.8]
OWN_HI = 4


def _week_end_rows(dates):
    """True where row t is the last weekday (Mon-Fri) of its calendar week.

    Point-in-time: uses only the date of row t. The next weekday (np.busday_offset, which ignores
    exchange holidays) lies in another Monday-based week exactly when t is a Friday. A week whose Friday
    is an exchange holiday has no decision row (positions are held to the next week's decision)."""
    days = dates.astype(np.int64)
    nxt = np.busday_offset(dates, 1, roll="forward").astype(np.int64)
    # day 0 of the epoch was a Thursday, so (day + 3) // 7 is a Monday-based week number.
    return (nxt + 3) // 7 != (days + 3) // 7


def _id_rank(instruments):
    """Rank of each instrument id; ids 'E<number>' sort numerically via (length, string)."""
    order = sorted(range(len(instruments)), key=lambda j: (len(instruments[j]), instruments[j]))
    rank = np.empty(len(instruments), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(instruments))
    return rank


def _legs(data, params):
    """Returns (decision rows (T,), long mask (T, N), short mask (T, N))."""
    W = int(params["corr_window"])
    F = int(params["formation_days"])
    K = int(params["peers_k"])

    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    if data.universe is None:
        raise ValueError("H-0013 needs the LIQ-500 point-in-time universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    T, N = ret_co.shape

    # a daily return exists where the stock is listed and has a price on that day
    valid = listed & np.isfinite(close) & np.isfinite(ret_co) & np.isfinite(ret_oc)
    r = np.where(valid, (1.0 + np.where(valid, ret_co, 0.0)) * (1.0 + np.where(valid, ret_oc, 0.0)) - 1.0, 0.0)
    csum = np.vstack([np.zeros((1, N), dtype=np.int64), np.cumsum(valid, axis=0, dtype=np.int64)])

    decision = _week_end_rows(np.asarray(data.dates))
    idrank = _id_rank(tuple(data.instruments))
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)

    for t in np.flatnonzero(decision):
        if t + 1 < F:
            continue
        lo = max(t + 1 - W, 0)
        n_form = csum[t + 1] - csum[t + 1 - F]
        n_corr = csum[t + 1] - csum[lo]            # rows before the data start count as missing
        elig = member[t] & tradable[t] & (n_form == F) & (n_corr >= ELIG_FRAC * W)
        cols = np.flatnonzero(elig)
        if cols.size < MIN_M:
            continue
        cols = cols[np.argsort(idrank[cols], kind="stable")]   # id order: ties go to the lower id
        n_e = cols.size

        # Step 1: peer map on market-demeaned returns, pairwise-complete Pearson correlation
        Mk = valid[lo:t + 1][:, cols].astype(np.float64)
        X = r[lo:t + 1][:, cols] * Mk
        cnt = Mk.sum(axis=1)
        mkt = np.where(cnt > 0, X.sum(axis=1) / np.maximum(cnt, 1.0), 0.0)
        X = (X - mkt[:, None]) * Mk
        n_ov = Mk.T @ Mk                 # overlapping days of (i, j)
        Sx = X.T @ Mk                    # sum of x_i over days where both are valid
        Sxx = (X * X).T @ Mk
        Sxy = X.T @ X
        with np.errstate(divide="ignore", invalid="ignore"):
            nn = np.where(n_ov > 0, n_ov, np.nan)
            cov = Sxy - Sx * Sx.T / nn
            vi = Sxx - Sx * Sx / nn
            vj = vi.T
            C = cov / np.sqrt(vi * vj)
        bad = (n_ov < OVERLAP_FRAC * W) | ~np.isfinite(C) | ~(vi > 0) | ~(vj > 0)
        C[bad] = -np.inf
        np.fill_diagonal(C, -np.inf)

        # Step 2: formation returns
        own = np.prod(1.0 + r[t + 1 - F:t + 1][:, cols], axis=0) - 1.0
        own_d = own - own.mean()
        k = min(K, n_e - 1)
        nb = np.argsort(-C, axis=1, kind="stable")[:, :k]   # highest correlation first, ties by id
        ok = np.isfinite(np.take_along_axis(C, nb, axis=1))
        n_p = ok.sum(axis=1)
        peer = np.where(n_p > 0, (own[nb] * ok).sum(axis=1) / np.maximum(n_p, 1), np.nan)
        has_peer = np.isfinite(peer)
        if not has_peer.any():
            continue
        peer_d = peer - peer[has_peer].mean()

        # Step 3: own percentile rank (rank / (n_e - 1), ties by id) within [0.2, 0.8]
        o1 = np.lexsort((np.arange(n_e), own_d))
        rk = np.empty(n_e, dtype=np.int64)
        rk[o1] = np.arange(n_e)
        mid = (N_QUANTILES * rk >= OWN_LO * (n_e - 1)) & (N_QUANTILES * rk <= OWN_HI * (n_e - 1)) & has_peer
        mc = np.flatnonzero(mid)
        m = mc.size
        if m < MIN_M:
            continue

        # Step 4: rank M(t) by peer return, quintiles
        o2 = mc[np.lexsort((mc, peer_d[mc]))]      # ascending peer return, ties by id
        q = m // N_QUANTILES
        short_m[t, cols[o2[:q]]] = True
        long_m[t, cols[o2[m - q:]]] = True
    return decision, long_m, short_m


def target_weights(data, params):
    decision, long_m, short_m = _legs(data, params)
    T, N = long_m.shape
    w = np.full((T, N), np.nan)
    w[decision] = 0.0
    n_l = long_m.sum(axis=1, keepdims=True)
    n_s = short_m.sum(axis=1, keepdims=True)
    w = np.where(long_m, 0.5 / np.maximum(n_l, 1), w)
    w = np.where(short_m, -0.5 / np.maximum(n_s, 1), w)
    return w
