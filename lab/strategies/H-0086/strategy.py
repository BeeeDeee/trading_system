"""H-0086: net share repurchasers held for a year.

LIQ-500 long-only book of overlapping monthly cohorts. Each month-end decision date forms a cohort of the
top `quantile` fraction of eligible LIQ-500 members ranked by trailing-twelve-month net repurchase yield
NRY = -sf1_art_ncfcommon / sf1_art_marketcap. The live cohorts (current and previous hold_months - 1) share the
capital equally; inside a cohort the share is split equally over its remaining names.
"""
import numpy as np

PARAMS = {"hold_months": 12, "quantile": 0.2}


def month_end_decisions(dates):
    """Boolean (T,) mask of decision rows: the last trading day of each month, point-in-time.

    Row t is a decision row if the next weekday after date t falls in a new calendar month (weekday arithmetic,
    exchange holidays ignored). Fallback for a month whose last weekday is a holiday (so no row was marked):
    the first row of the following month becomes the decision row (one trading day late). Both rules use the
    date of row t and earlier rows only.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    month = dates.astype("datetime64[M]")
    nxt = np.busday_offset(dates, 1, roll="forward")
    is_last = nxt.astype("datetime64[M]") != month
    first = np.zeros(T, dtype=bool)
    first[1:] = month[1:] != month[:-1]
    group = np.cumsum(first)                      # 0, 1, 2, ... per calendar month present in the data
    n_groups = int(group[-1]) + 1
    has_dec = np.bincount(group, weights=is_last.astype(float), minlength=n_groups) > 0
    dec = is_last.copy()
    rows = np.flatnonzero(first)
    for t in rows:                                # previous month lies entirely in rows < t
        if not has_dec[group[t] - 1]:
            dec[t] = True
    return dec


def _top_k(n, q):
    """Cohort size: top `q` fraction of n names, rounded down, at least 1 (0 when n == 0)."""
    if n <= 0:
        return 0
    return max(1, int(np.floor(q * n + 1e-9)))


def target_weights(data, params):
    hold = int(params["hold_months"])
    q = float(params["quantile"])
    if hold < 1:
        raise ValueError("hold_months must be >= 1")

    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=float)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=float)
    if data.universe is None:
        raise ValueError("H-0086 needs the LIQ-500 point-in-time universe")
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    delisting = np.asarray(data.delisting, dtype=bool)
    T, N = uni.shape

    valid = uni & np.isfinite(ncf) & np.isfinite(mcap) & (mcap > 0)
    nry = np.full((T, N), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        nry[valid] = -ncf[valid] / mcap[valid]

    # a held name is sold when it leaves the universe, stops being listed or is delisted
    bad = (~uni) | (~listed) | delisting

    W = np.full((T, N), np.nan)
    dec_rows = np.flatnonzero(month_end_decisions(data.dates))
    cols = np.arange(N)

    cohorts = []                                  # list of [idx (int array), alive (bool array)]
    held = np.zeros(N)
    prev = -1

    def _exits(lo, hi, held_now):
        # non-decision rows lo..hi-1: sell (target 0) held names on rows where they are out of the universe
        if hi <= lo:
            return
        hj = held_now > 0
        if not hj.any():
            return
        sb = bad[lo:hi] & hj[None, :]
        if sb.any():
            seg = W[lo:hi]
            seg[sb] = 0.0

    for t in dec_rows:
        if prev >= 0:
            _exits(prev + 1, t, held)
        # names that left the universe / stopped being listed since the last decision leave their cohorts
        lo = prev + 1 if prev >= 0 else 0
        seg_bad = bad[lo:t + 1].any(axis=0)
        for c in cohorts:
            c[1] &= ~seg_bad[c[0]]

        # form the new cohort from eligible members, NRY descending, ties by instrument column order
        e = cols[valid[t]]
        k = _top_k(e.size, q)
        if k > 0:
            order = np.lexsort((e, -nry[t, e]))
            idx = e[order[:k]]
        else:
            idx = np.zeros(0, dtype=int)
        cohorts.append([idx, np.ones(idx.size, dtype=bool)])
        cohorts = cohorts[-hold:]

        # names not tradable on the decision date drop out of their cohort
        live = []
        for c in cohorts:
            c[1] &= tradable[t, c[0]]
            if c[1].any():
                live.append(c[0][c[1]])

        w = np.zeros(N)
        if live:
            share = 1.0 / len(live)
            for idx_live in live:
                w[idx_live] += share / idx_live.size
        W[t] = w
        held = w
        prev = t

    if prev >= 0:
        _exits(prev + 1, T, held)
    return W


def events(data, params):
    """Mechanism-test events (diagnostic.py).

    On each decision date rank eligible LIQ-500 members by NRY (descending, ties by instrument column order).
    +1: a member in the top `quantile` now that was not in it at the previous decision date.
    -1: a member in the bottom `quantile` now that was not in it at the previous decision date.
    No events on the first decision date; events only where the name is a universe member and tradable.
    """
    q = float(params["quantile"])
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"], dtype=float)
    mcap = np.asarray(data.extras["sf1_art_marketcap"], dtype=float)
    if data.universe is None:
        raise ValueError("H-0086 needs the LIQ-500 point-in-time universe")
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = uni.shape

    valid = uni & np.isfinite(ncf) & np.isfinite(mcap) & (mcap > 0)
    nry = np.full((T, N), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        nry[valid] = -ncf[valid] / mcap[valid]

    mask = np.zeros((T, N), dtype=bool)
    side = np.ones((T, N), dtype=float)
    cols = np.arange(N)
    prev_top = None
    prev_bot = None
    for t in np.flatnonzero(month_end_decisions(data.dates)):
        e = cols[valid[t]]
        k = _top_k(e.size, q)
        top = np.zeros(N, dtype=bool)
        bot = np.zeros(N, dtype=bool)
        if k > 0:
            order = e[np.lexsort((e, -nry[t, e]))]
            top[order[:k]] = True
            bot[order[-k:]] = True
            bot &= ~top                           # overlap only possible with very few eligible names
        if prev_top is not None:
            ok = uni[t] & tradable[t]
            new_top = top & ~prev_top & ok
            new_bot = bot & ~prev_bot & ok
            mask[t] = new_top | new_bot
            side[t, new_bot] = -1.0
        prev_top = top
        prev_bot = bot
    return mask, side
