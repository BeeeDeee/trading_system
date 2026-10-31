"""H-0086 mechanism-test events.

Verbatim copy of `month_end_decisions`, `_top_k` and `events` from strategy.py (the static rules forbid importing
strategy here and importing diagnostic in the tests, so the tests exercise the identical copy in strategy.py).
"""
import numpy as np


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
