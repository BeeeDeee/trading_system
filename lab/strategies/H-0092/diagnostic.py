"""H-0092 mechanism event: a LIQ-500 stock's new 10-Q/10-K filing becomes usable on session t
(as-reported quarterly netinc, ncfo, assets change; first accepted change in 40 sessions; not in the
first 63 sessions) and its quarterly accruals (netinc - ncfo) / assets fall in the bottom leg_quantile
(+1) or top leg_quantile (-1) of the current LIQ-500 cross-section of latest accruals (>= 100 members).
Same event logic as strategy.accrual_events (duplicated: the diagnostic may not import the strategy)."""
import numpy as np

WARMUP_ROWS = 63
GAP_SESSIONS = 40
MIN_MEMBERS = 100
FIELDS = ("sf1_arq_netinc", "sf1_arq_ncfo", "sf1_arq_assets")


def _changed(x):
    ch = np.zeros(x.shape, dtype=bool)
    prev = x[:-1]
    cur = x[1:]
    pn = np.isnan(prev)
    cn = np.isnan(cur)
    ch[1:] = (pn != cn) | (~pn & ~cn & (prev != cur))
    return ch


def _row_quantile(sorted_rows, n, p):
    pos = p * (n - 1).astype(np.float64)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    r = np.arange(sorted_rows.shape[0])
    v0 = sorted_rows[r, i0]
    v1 = sorted_rows[r, i1]
    return v0 + frac * (v1 - v0)


def _side(data, params):
    q = float(params["leg_quantile"])
    ni = np.asarray(data.extras[FIELDS[0]])
    cf = np.asarray(data.extras[FIELDS[1]])
    at = np.asarray(data.extras[FIELDS[2]])
    if data.universe is None:
        raise ValueError("H-0092 needs the LIQ-500 point-in-time universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    T, N = ni.shape

    have = ~np.isnan(ni) & ~np.isnan(cf) & ~np.isnan(at)
    have &= np.where(have, at, 0.0) > 0
    member = have & uni

    cand = (_changed(ni) | _changed(cf) | _changed(at)) & member
    cand[:WARMUP_ROWS] = False
    filing = np.zeros((T, N), dtype=bool)
    cols, rows = np.nonzero(cand.T)
    last_col = -1
    last_row = -(10 ** 9)
    for c, r in zip(cols.tolist(), rows.tolist()):
        if c != last_col:
            last_col = c
            last_row = -(10 ** 9)
        if r - last_row > GAP_SESSIONS:
            filing[r, c] = True
            last_row = r

    acc = np.full((T, N), np.nan)
    ii, jj = np.nonzero(member)
    acc[ii, jj] = (ni[ii, jj].astype(np.float64) - cf[ii, jj].astype(np.float64)) / at[ii, jj].astype(np.float64)
    count = member.sum(axis=1)

    side = np.zeros((T, N))
    need = np.nonzero(filing.any(axis=1) & (count >= MIN_MEMBERS))[0]
    if need.size:
        a = acc[need]
        s = np.sort(a, axis=1)
        n = count[need]
        lo = _row_quantile(s, n, q)
        hi = _row_quantile(s, n, 1.0 - q)
        f = filing[need]
        with np.errstate(invalid="ignore"):
            long_ev = f & (a <= lo[:, None])
            short_ev = f & (a >= hi[:, None]) & ~long_ev
        sub = np.zeros(a.shape)
        sub[long_ev] = 1.0
        sub[short_ev] = -1.0
        side[need] = sub
    return side, uni


def events(data, params):
    side, uni = _side(data, params)
    trad = np.asarray(data.tradable, dtype=bool)
    mask = (side != 0) & uni & trad
    return mask, np.where(mask, side, 0.0)
