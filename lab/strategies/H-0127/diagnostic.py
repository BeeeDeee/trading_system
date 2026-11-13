"""H-0127 mechanism events: PRE (side -1) and POST (side +1) on TLT and IEF.

Same event logic and gating as strategy.py (copied, separate file).
"""
import numpy as np

TRADED = ("TLT", "IEF")
NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"


def _calendar_events(data, h):
    nxt = np.asarray(data.series[NEXT_KEY], dtype=float)
    lst = np.asarray(data.series[LAST_KEY], dtype=float)
    with np.errstate(invalid="ignore"):
        pre = (nxt == h + 1) & (np.isnan(lst) | (lst >= 3))
        post = (lst == 0) & (np.isnan(nxt) | (nxt >= 3))
    pre &= ~np.isnan(nxt)
    post &= ~np.isnan(lst)
    return pre, post


def _instrument_ok(data, j):
    close = np.asarray(data.close, dtype=float)[:, j]
    ok = ~np.isnan(close) & np.asarray(data.tradable, dtype=bool)[:, j]
    ok &= np.asarray(data.listed, dtype=bool)[:, j]
    if data.universe is not None:
        ok &= np.asarray(data.universe, dtype=bool)[:, j]
    return ok


def events(data, params):
    h = int(params["hold_sessions"])
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    pre, post = _calendar_events(data, h)
    for name in TRADED:
        j = data.col(name)
        ok = _instrument_ok(data, j)
        m_pre = pre & ok
        m_post = post & ok
        mask[:, j] = m_pre | m_post
        side[:, j] = np.where(m_pre, -1.0, np.where(m_post, 1.0, 0.0))
    return mask, side
