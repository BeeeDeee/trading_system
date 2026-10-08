"""H-0011: 52-week-high anchoring, LIQ-500 market-neutral.

Long stocks that closed above a stale 52-week high of their total-return index (old high set at
least gap_days trading days ago, not on a top-decile move day), held hold_days trading days;
short all other LIQ-500 stocks with a price today. +0.5 / -0.5, daily rebalance.
"""
import numpy as np

PARAMS = {"gap_days": 21, "hold_days": 5}

WINDOW = 252          # 52-week anchor window t-252 .. t-1 (definition of the anchor, not a param)
MIN_OBS = 240         # stock must have a price on at least 240 of those 252 days
TOP_DECILE = 0.9      # close-to-close returns above the 90th percentile of LIQ-500 are excluded
CHUNK = 256           # columns per block for the rolling max (memory)


def _rolling_max_last(v, w):
    """Max over rows t-w+1..t and the row index of its most recent occurrence.

    v: (T, n) float, -inf where there is no value. Rows before row 0 count as -inf.
    Sparse-table doubling; on ties the later window wins, so the index is the most recent max.
    """
    T, n = v.shape
    m = v.copy()
    idx = np.empty((T, n), dtype=np.int64)
    idx[:] = np.arange(T, dtype=np.int64)[:, None]
    length = 1

    def combine(m, idx, shift):
        m2 = np.full_like(m, -np.inf)
        i2 = np.full_like(idx, -1)
        if shift < T:
            m2[shift:] = m[:T - shift]
            i2[shift:] = idx[:T - shift]
        take = m2 > m          # strict: ties keep the later (current) window
        return np.where(take, m2, m), np.where(take, i2, idx)

    while 2 * length <= w:
        m, idx = combine(m, idx, length)
        length *= 2
    rest = w - length
    if rest > 0:
        m, idx = combine(m, idx, rest)
    return m, idx


def breakout_events(data, params):
    """(T, N) bool: stale-52-week-high breakthrough of stock j after the close of day t."""
    gap_days = int(params["gap_days"])
    universe = data.universe
    if universe is None:
        raise ValueError("H-0011 needs the LIQ-500 point-in-time universe (data.universe is None)")
    universe = np.asarray(universe, dtype=bool)

    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = ret_co.shape

    # close-to-close total return of day t and the total-return index P (adjusted closing level)
    rc = np.where(np.isfinite(ret_co), ret_co, 0.0)
    ro = np.where(np.isfinite(ret_oc), ret_oc, 0.0)
    r_cc = (1.0 + rc) * (1.0 + ro) - 1.0
    has_price = np.isfinite(close)
    P = np.cumprod(1.0 + r_cc, axis=0)
    P = np.where(has_price, P, np.nan)

    # number of priced days in t-252 .. t-1
    cs = np.zeros((T + 1, N), dtype=np.int64)
    np.cumsum(has_price, axis=0, out=cs[1:])
    count = np.zeros((T, N), dtype=np.int64)
    if T > WINDOW:
        count[WINDOW:] = cs[WINDOW:T] - cs[0:T - WINDOW]

    # top decile of the LIQ-500 cross-section of close-to-close returns on day t
    in_xs = universe & has_price
    xs = np.where(in_xs, r_cc, np.nan)
    any_xs = in_xs.any(axis=1)
    q = np.full(T, np.inf)
    if any_xs.any():
        q[any_xs] = np.nanquantile(xs[any_xs], TOP_DECILE, axis=1)
    top_decile = r_cc > q[:, None]

    rows = np.arange(T, dtype=np.int64)[:, None]
    ev = np.zeros((T, N), dtype=bool)
    if T <= WINDOW:
        return ev
    for c0 in range(0, N, CHUNK):
        c1 = min(N, c0 + CHUNK)
        Pc = P[:, c0:c1]
        v = np.where(np.isfinite(Pc), Pc, -np.inf)
        m, idx = _rolling_max_last(v, WINDOW)
        # old high over t-252 .. t-1 = window ending at t-1
        H = np.full_like(m, -np.inf)
        H[1:] = m[:-1]
        setday = np.full_like(idx, -1)
        setday[1:] = idx[:-1]
        stale = (rows - setday) >= gap_days
        with np.errstate(invalid="ignore"):
            above = np.isfinite(Pc) & np.isfinite(H) & (Pc > H)
        ev[:, c0:c1] = above & stale & (count[:, c0:c1] >= MIN_OBS)
    ev[:WINDOW] = False
    ev &= has_price & ~top_decile & universe & tradable
    return ev


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    ev = breakout_events(data, params)
    T, N = ev.shape
    universe = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    has_price = np.isfinite(np.asarray(data.close, dtype=np.float64))
    # proxy for "tradable at the next open" (not knowable at t): listed and priced on t
    can_trade_next = listed & has_price

    # event on any of t-hold_days+1 .. t
    cs = np.zeros((T + 1, N), dtype=np.int64)
    np.cumsum(ev, axis=0, out=cs[1:])
    lo = np.maximum(np.arange(T) + 1 - hold_days, 0)
    recent = (cs[1:] - cs[lo]) > 0

    # Card rule (5) wants a stock that drops out of LIQ-500 to keep its long position until its
    # scheduled exit; the lab contract (G0 g0_outside_universe) forbids weights on non-members, so
    # such a stock is closed when it leaves the universe (documented deviation).
    long_set = recent & can_trade_next & universe
    short_set = universe & can_trade_next & ~long_set
    n_long = long_set.sum(axis=1)
    n_short = short_set.sum(axis=1)
    active = (n_long > 0) & (n_short > 0)

    w = np.zeros((T, N), dtype=np.float64)
    wl = np.where(active, 0.5 / np.maximum(n_long, 1), 0.0)
    ws = np.where(active, -0.5 / np.maximum(n_short, 1), 0.0)
    w = np.where(long_set, wl[:, None], w)
    w = np.where(short_set, ws[:, None], w)
    return w
