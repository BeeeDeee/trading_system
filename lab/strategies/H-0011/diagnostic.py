"""H-0011 mechanism test event: breakthrough of a stale 52-week high on a non-top-decile day.

Identical to the strategy's event (strategy.breakout_events, duplicated because the static scan
forbids importing the strategy module): in LIQ-500, tradable, priced on t, P_t > max(P over
t-252..t-1) with >= 240 priced days, high set >= gap_days days before t, close-to-close return
not in the LIQ-500 top decile of day t. side = +1 (long leg).
"""
import numpy as np

WINDOW = 252
MIN_OBS = 240
TOP_DECILE = 0.9
CHUNK = 256


def _rolling_max_last(v, w):
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
        take = m2 > m
        return np.where(take, m2, m), np.where(take, i2, idx)

    while 2 * length <= w:
        m, idx = combine(m, idx, length)
        length *= 2
    rest = w - length
    if rest > 0:
        m, idx = combine(m, idx, rest)
    return m, idx


def _breakout_events(data, params):
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

    rc = np.where(np.isfinite(ret_co), ret_co, 0.0)
    ro = np.where(np.isfinite(ret_oc), ret_oc, 0.0)
    r_cc = (1.0 + rc) * (1.0 + ro) - 1.0
    has_price = np.isfinite(close)
    P = np.cumprod(1.0 + r_cc, axis=0)
    P = np.where(has_price, P, np.nan)

    cs = np.zeros((T + 1, N), dtype=np.int64)
    np.cumsum(has_price, axis=0, out=cs[1:])
    count = np.zeros((T, N), dtype=np.int64)
    if T > WINDOW:
        count[WINDOW:] = cs[WINDOW:T] - cs[0:T - WINDOW]

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


def events(data, params):
    mask = _breakout_events(data, params)
    return mask, None
