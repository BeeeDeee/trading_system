"""H-0021 IPO lockup-expiry supply overhang (LIQ-500, beta-hedged short, event hold).

Rules of the card and where they are implemented:
(1) listing date F = first row with a bar (listed and finite close) in the loaded panel;
    F within the first 30 rows of the panel -> pre-existing listing, never an event.   -> _event_rows
(2) event at t = the FIRST row whose calendar distance date(t) - date(F) >= 180 - lead_days,
    if j is a universe member and tradable at t (and has a close at t). At most once.  -> _event_rows
(3) short j on decision rows t .. t+hold_days-1 (held from the open of t+1 for hold_days
    sessions, exit at the open after the last held day); weight min(0.25, 0.5/n_active),
    recomputed only when the active set changes.                                       -> target_weights
(4) hedge: equal-weight long in all other members tradable at the change row and not in an
    event window, total notional sum |w_j| * beta_j; beta = OLS of (ret_co+ret_oc) on the EW
    universe return over the rows since F up to t (<=126, >=60 else 1.0), clipped [0.5, 2].
(5) gross > 1 -> scale to 1; flat when nothing is active; an event stock that is untradable
    or outside the universe during the hold is dropped (short and its hedge closed).
"""
import numpy as np

PARAMS = {"hold_days": 5, "lead_days": 4}

LOCKUP_DAYS = 180          # industry convention, fixed by the card (not a parameter)
PRE_EXISTING_ROWS = 30     # first bars in the first 30 panel rows are not IPOs
BETA_MAX_ROWS = 126
BETA_MIN_ROWS = 60
BETA_LO, BETA_HI = 0.5, 2.0
MAX_SHORT = 0.25
SHORT_BUDGET = 0.5


def _has_bar(data):
    return np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=float))


def _first_bar(has_bar):
    """Row index of the first bar of each column, T where the column never has one."""
    T = has_bar.shape[0]
    any_bar = has_bar.any(axis=0)
    first = np.argmax(has_bar, axis=0)
    return np.where(any_bar, first, T)


def _event_rows(data, params, has_bar, universe, tradable):
    """Per column: the event row t (or -1). Uses only rows <= t for an event at t."""
    T, N = has_bar.shape
    lead = int(params["lead_days"])
    thr = LOCKUP_DAYS - lead
    day = np.asarray(data.dates).astype(np.int64)          # calendar days since epoch
    first = _first_bar(has_bar)
    ev = np.full(N, -1, dtype=np.int64)
    ok = (first >= PRE_EXISTING_ROWS) & (first < T)
    cols = np.nonzero(ok)[0]
    if cols.size == 0:
        return ev, first
    target = day[first[cols]] + thr
    t = np.searchsorted(day, target, side="left")           # first row with distance >= thr
    inside = t < T
    cols, t = cols[inside], t[inside]
    fire = universe[t, cols] & tradable[t, cols] & has_bar[t, cols]
    ev[cols[fire]] = t[fire]
    return ev, first


def _daily_returns(data):
    r = np.asarray(data.ret_co, dtype=float) + np.asarray(data.ret_oc, dtype=float)
    return r


def _market_return(r, has_bar, universe):
    """Equal-weight return of the universe members with a bar on each row (rows <= t only)."""
    T = r.shape[0]
    m = np.full(T, np.nan)
    step = 256
    for a in range(0, T, step):
        b = min(T, a + step)
        rr = r[a:b]
        msk = universe[a:b] & has_bar[a:b] & np.isfinite(rr)
        cnt = msk.sum(axis=1)
        s = np.where(msk, rr, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m[a:b] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return m


def _beta(r, has_bar, m, j, first, t):
    """OLS beta of stock j on the market over rows first+1 .. t (last <= 126 rows)."""
    lo = max(first + 1, t - BETA_MAX_ROWS + 1)
    if t < lo:
        return 1.0
    y = r[lo:t + 1, j]
    x = m[lo:t + 1]
    ok = has_bar[lo:t + 1, j] & np.isfinite(y) & np.isfinite(x)
    if int(ok.sum()) < BETA_MIN_ROWS:
        return 1.0
    y = y[ok]
    x = x[ok]
    xc = x - x.mean()
    var = float((xc * xc).sum())
    if var <= 0.0:
        return 1.0
    b = float((xc * (y - y.mean())).sum()) / var
    return float(min(BETA_HI, max(BETA_LO, b)))


def _event_matrix(data, params):
    """(T, N) bool event mask (shared with the diagnostic)."""
    if data.universe is None:
        raise ValueError("H-0021 needs the LIQ-500 point-in-time universe")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    has_bar = _has_bar(data)
    ev, first = _event_rows(data, params, has_bar, universe, tradable)
    T, N = has_bar.shape
    mask = np.zeros((T, N), dtype=bool)
    cols = np.nonzero(ev >= 0)[0]
    mask[ev[cols], cols] = True
    return mask, ev, first, has_bar, universe, tradable


def target_weights(data, params):
    hold = int(params["hold_days"])
    if hold < 1:
        raise ValueError("hold_days must be >= 1")
    _, ev, first, has_bar, universe, tradable = _event_matrix(data, params)
    T, N = has_bar.shape
    W = np.zeros((T, N), dtype=float)

    ev_cols = np.nonzero(ev >= 0)[0]
    if ev_cols.size == 0:
        return W

    r = _daily_returns(data)
    m = _market_return(r, has_bar, universe)

    starts = {}
    for j in ev_cols:
        starts.setdefault(int(ev[j]), []).append(int(j))

    active = {}        # column -> (end row exclusive, beta)
    w = np.zeros(N)
    for s in range(T):
        changed = False
        # exits: window over, or the event stock left the universe / became untradable
        for j in list(active.keys()):
            end, _ = active[j]
            if s >= end or not universe[s, j] or not tradable[s, j]:
                del active[j]
                changed = True
        # entries
        for j in starts.get(s, ()):
            beta = _beta(r, has_bar, m, j, int(first[j]), s)
            active[j] = (s + hold, beta)
            changed = True
        if not active:
            w = np.zeros(N)
            continue
        if changed:
            n_act = len(active)
            ws = min(MAX_SHORT, SHORT_BUDGET / n_act)
            w = np.zeros(N)
            act_cols = np.fromiter(active.keys(), dtype=np.int64, count=n_act)
            long_total = ws * sum(b for (_, b) in active.values())
            basket = universe[s] & tradable[s]
            basket[act_cols] = False
            nb = int(basket.sum())
            if nb > 0:
                w[basket] = long_total / nb
            w[act_cols] = -ws
            gross = float(np.abs(w).sum())
            if gross > 1.0:
                w = w / gross
        # never hold a non-member (hedge names that leave the universe are dropped, not replaced)
        w = np.where(universe[s], w, 0.0)
        W[s] = w
    return W
