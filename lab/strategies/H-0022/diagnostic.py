"""H-0022 mechanism-test events: S&P 500 additions (side -1) and deletions (side +1) of LIQ-500 stocks.

Identical event definition to strategy._events (copied, since modules may not import each other): membership
change between t-1 and t in extras['sp500_member'], stock in LIQ-500 on t, tradable on t-1 and t, listed >= 63
sessions before t, valid return on t.
"""
import numpy as np

MIN_LISTED_SESSIONS = 63


def _sp500_mask(data):
    sp = np.asarray(data.extras["sp500_member"], dtype=np.float64)   # KeyError if the extra is missing
    T, N = len(data.dates), len(data.instruments)
    if sp.shape != (T, N):
        raise ValueError("extras['sp500_member'] has shape %r, expected %r" % (sp.shape, (T, N)))
    return sp


def _events(data):
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
    add = (prev == 0.0) & (cur == 1.0)
    dele = (prev == 1.0) & (cur == 0.0)
    both = add & dele
    add &= ~both
    dele &= ~both
    age = np.zeros((T, N), dtype=np.int64)
    age[1:] = np.cumsum(listed, axis=0)[:-1]
    valid_ret = (listed & np.isfinite(np.asarray(data.ret_co, dtype=np.float64))
                 & np.isfinite(np.asarray(data.ret_oc, dtype=np.float64))
                 & np.isfinite(np.asarray(data.close, dtype=np.float64)))
    elig = (uni[1:] & trad[:-1] & trad[1:] & (age[1:] >= MIN_LISTED_SESSIONS) & valid_ret[1:])
    ev[1:][add & elig] = -1.0
    ev[1:][dele & elig] = 1.0
    return ev


def events(data, params):
    ev = _events(data)
    mask = ev != 0.0
    side = np.where(mask, ev, 1.0)
    return mask, side
