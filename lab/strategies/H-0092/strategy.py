"""H-0092: accrual information at the 10-Q/10-K filing, LIQ-500 long/short event book.

Rules of the card and where they live:
(1) filing event      -> accrual_events(): change of the (netinc, ncfo, assets) tuple vs the previous row
                         (NaN -> value counts), all three non-NaN, assets > 0, LIQ-500 member at t,
                         no accepted filing event in the previous 40 sessions, none in the first 63 rows.
(2) score             -> acc = (netinc - ncfo) / assets of the latest filing (as delivered by the loader).
(3) breakpoints       -> per row: quantiles (numpy 'linear') of acc over current LIQ-500 members with values;
                         fewer than 100 members -> no events that row.
(4) side              -> filing event with acc <= lo: +1, acc >= hi: -1, else none.
(5) book              -> target_weights(): fixed +/-0.04 from decision row t (fill at open t+1), held
                         hold_days rows, target 0 on row t+hold_days (closed at open t+1+hold_days); closed
                         earlier on the first row the stock is out of LIQ-500 or not tradable; held stocks
                         get no second position; a session's new entries are scaled together so gross = 1.0
                         when they would take it above 1.0. Weights are constant while held (no re-equalizing).
"""
import numpy as np

PARAMS = {"hold_days": 5, "leg_quantile": 0.2}

WARMUP_ROWS = 63      # no events in the first 63 sessions of the view
GAP_SESSIONS = 40     # no accepted filing event in the previous 40 sessions
MIN_MEMBERS = 100     # minimum cross-section for breakpoints
SLOT = 0.04           # fixed weight per event

FIELDS = ("sf1_arq_netinc", "sf1_arq_ncfo", "sf1_arq_assets")


def _changed(x):
    """(T, N) bool: row t differs from row t-1 (NaN -> value and value -> NaN count). Row 0: False."""
    ch = np.zeros(x.shape, dtype=bool)
    prev = x[:-1]
    cur = x[1:]
    pn = np.isnan(prev)
    cn = np.isnan(cur)
    ch[1:] = (pn != cn) | (~pn & ~cn & (prev != cur))
    return ch


def _row_quantile(sorted_rows, n, p):
    """numpy 'linear' quantile p of the first n[r] (non-NaN, sorted) entries of each row."""
    pos = p * (n - 1).astype(np.float64)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    r = np.arange(sorted_rows.shape[0])
    v0 = sorted_rows[r, i0]
    v1 = sorted_rows[r, i1]
    return v0 + frac * (v1 - v0)


def accrual_events(data, params):
    """Return (filing, side): filing (T, N) bool accepted filing events (rule 1),
    side (T, N) float +1 long event / -1 short event / 0 (rules 2-4)."""
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

    # (1) candidate filing events, then the 40-session rule (causal, per stock)
    cand = (_changed(ni) | _changed(cf) | _changed(at)) & member
    cand[:WARMUP_ROWS] = False
    filing = np.zeros((T, N), dtype=bool)
    cols, rows = np.nonzero(cand.T)          # sorted by stock, then by row
    last_col = -1
    last_row = -(10 ** 9)
    for c, r in zip(cols.tolist(), rows.tolist()):
        if c != last_col:
            last_col = c
            last_row = -(10 ** 9)
        if r - last_row > GAP_SESSIONS:
            filing[r, c] = True
            last_row = r

    # (2) accruals of the latest filing for every member with values
    acc = np.full((T, N), np.nan)
    ii, jj = np.nonzero(member)
    acc[ii, jj] = (ni[ii, jj].astype(np.float64) - cf[ii, jj].astype(np.float64)) / at[ii, jj].astype(np.float64)
    count = member.sum(axis=1)

    # (3)-(4) breakpoints on rows that have filing events and enough members
    side = np.zeros((T, N))
    need = np.nonzero(filing.any(axis=1) & (count >= MIN_MEMBERS))[0]
    if need.size:
        a = acc[need]
        s = np.sort(a, axis=1)               # NaN sorts last
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
    return filing, side


def target_weights(data, params):
    hold = int(params["hold_days"])
    _, side = accrual_events(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    T, N = side.shape

    w = np.zeros((T, N))
    cur = np.zeros(N)
    exit_row = np.full(N, -1, dtype=np.int64)
    ev_row = (side != 0).any(axis=1)
    for t in range(T):
        held = cur != 0
        if held.any():
            close = held & ((exit_row == t) | ~uni[t] | ~trad[t])
            if close.any():
                cur[close] = 0.0
                exit_row[close] = -1
                held = cur != 0
        if ev_row[t]:
            s = side[t]
            new = (s != 0) & ~held
            k = int(new.sum())
            if k:
                gross = float(np.abs(cur).sum())
                scale = 1.0
                if gross + SLOT * k > 1.0:
                    # scale this session's entries so gross = 1.0 (1e-9 margin against float round-up);
                    # a book that is already full (room < 1e-6) takes no new entries
                    room = 1.0 - gross
                    scale = room / (SLOT * k) * (1.0 - 1e-9) if room > 1e-6 else 0.0
                if scale > 0:
                    cur[new] = SLOT * scale * s[new]
                    exit_row[new] = t + hold
        w[t] = cur
    return w
