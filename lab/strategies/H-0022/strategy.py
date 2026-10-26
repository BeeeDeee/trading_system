"""H-0022: S&P 500 change price-pressure reversal read inside LIQ-500.

Long liquid stocks on their first session as S&P 500 non-members (deletions, side +1), short liquid stocks on
their first session as members (additions, side -1). Event leg gross 0.5 split equally over the active event
stocks, hedged dollar-neutral with an equal-weight basket of the other tradable LIQ-500 members. Each event is
held for hold_days sessions (decision rows t .. t+hold_days-1, i.e. sessions t+1 .. t+hold_days). Targets are
recomputed only when the active set A changes.
"""
import numpy as np

PARAMS = {"hold_days": 5}

MIN_LISTED_SESSIONS = 63
EVENT_GROSS = 0.5


def _sp500_mask(data):
    """The PIT S&P 500 membership extra; fail loudly if it is missing or not a (T, N) panel.

    Values other than exactly 0 or 1 (NaN) are 'unknown' and never form an event. A value-based raise over all
    rows would depend on future rows (point-in-time), so the check is per element instead.
    """
    sp = np.asarray(data.extras["sp500_member"], dtype=np.float64)   # KeyError if the extra is missing
    T, N = len(data.dates), len(data.instruments)
    if sp.shape != (T, N):
        raise ValueError("extras['sp500_member'] has shape %r, expected %r" % (sp.shape, (T, N)))
    return sp


def _events(data):
    """(T, N) float: -1 addition, +1 deletion, 0 no event, decided after the close of row t (rows <= t only)."""
    sp = _sp500_mask(data)
    if data.universe is None:
        raise ValueError("H-0022 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = sp.shape
    ev = np.zeros((T, N), dtype=np.float64)
    if T < 2:
        return ev

    prev, cur = sp[:-1], sp[1:]
    add = (prev == 0.0) & (cur == 1.0)          # NaN on either day -> no event
    dele = (prev == 1.0) & (cur == 0.0)
    both = add & dele                           # id artifact: skip (cannot happen within one column)
    add &= ~both
    dele &= ~both

    # listing age: number of listed sessions strictly before t
    age = np.zeros((T, N), dtype=np.int64)
    age[1:] = np.cumsum(listed, axis=0)[:-1]

    valid_ret = (listed & np.isfinite(np.asarray(data.ret_co, dtype=np.float64))
                 & np.isfinite(np.asarray(data.ret_oc, dtype=np.float64))
                 & np.isfinite(np.asarray(data.close, dtype=np.float64)))

    elig = (uni[1:] & trad[:-1] & trad[1:] & (age[1:] >= MIN_LISTED_SESSIONS) & valid_ret[1:])
    ev[1:][add & elig] = -1.0
    ev[1:][dele & elig] = 1.0
    return ev


def _active_side(ev, hold):
    """Side of the most recent event of each stock within decision rows r-hold+1 .. r (0 if none)."""
    T = ev.shape[0]
    act = np.zeros_like(ev)
    for k in range(hold - 1, -1, -1):          # oldest first, newest overwrites
        if k >= T:
            continue
        src = ev[: T - k]
        dst = act[k:]
        nz = src != 0.0
        dst[nz] = src[nz]
    return act


def target_weights(data, params):
    hold = int(params["hold_days"])
    if hold < 1:
        raise ValueError("hold_days must be >= 1")
    ev = _events(data)
    T, N = ev.shape
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)

    act = _active_side(ev, hold)
    # A on row r: active event stocks that are LIQ-500 members and tradable that day
    a_side = np.where(uni & trad, act, 0.0)

    out = np.full((T, N), np.nan)
    w_cur = np.zeros(N)
    prev_side = None
    for r in range(T):
        s = a_side[r]
        if prev_side is None or not np.array_equal(s, prev_side):
            w = np.zeros(N)
            in_a = s != 0.0
            n_a = int(in_a.sum())
            if n_a > 0:
                w[in_a] = s[in_a] * EVENT_GROSS / n_a
                net = w.sum()
                hedge = uni[r] & trad[r] & ~in_a
                n_h = int(hedge.sum())
                if n_h > 0 and net != 0.0:
                    w[hedge] = -net / n_h
            out[r] = w
            w_cur = w
            prev_side = s.copy()
        else:
            # A unchanged: keep targets, except positions in stocks that left the universe (weight 0 from next open)
            gone = (w_cur != 0.0) & ~uni[r]
            if gone.any():
                w_cur = w_cur.copy()
                w_cur[gone] = 0.0
                out[r, gone] = 0.0
    return out
