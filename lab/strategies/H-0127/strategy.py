"""H-0127 Treasury auction supply cycle.

PRE event after the close of t: next_coupon5_sessions[t] == h + 1 and
(last_coupon5_sessions[t] absent or >= 3)  -> TLT -0.5, IEF -0.5 for h sessions.
POST event after the close of t: last_coupon5_sessions[t] == 0 and
(next_coupon5_sessions[t] absent or >= 3)  -> TLT +0.5, IEF +0.5 for h sessions.
A window decided on row t occupies weight rows t .. t+h-1 (held from the open of
t+1 to the open of t+h+1). Overlapping windows add per instrument, clipped to
[-0.5, 0.5]. Peers (SHY, SHV, BIL, VGSH, SCHO) are always 0. Flat otherwise.
"""
import numpy as np

PARAMS = {"hold_sessions": 3}

LEG = 0.5
TRADED = ("TLT", "IEF")
NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"


def _calendar_events(data, h):
    """(T,) bool PRE and POST events from the calendar keys at row t only."""
    nxt = np.asarray(data.series[NEXT_KEY], dtype=float)
    lst = np.asarray(data.series[LAST_KEY], dtype=float)
    with np.errstate(invalid="ignore"):
        pre = (nxt == h + 1) & (np.isnan(lst) | (lst >= 3))
        post = (lst == 0) & (np.isnan(nxt) | (nxt >= 3))
    pre &= ~np.isnan(nxt)
    post &= ~np.isnan(lst)
    return pre, post


def _instrument_ok(data, j):
    """Window may open on row t: price on t and tradable at the open of t+1.

    tradable[t+1] is not in the data at the close of t, so the latest known
    tradability (tradable[t], listed[t]) is used as its point-in-time proxy.
    """
    close = np.asarray(data.close, dtype=float)[:, j]
    ok = ~np.isnan(close) & np.asarray(data.tradable, dtype=bool)[:, j]
    ok &= np.asarray(data.listed, dtype=bool)[:, j]
    if data.universe is not None:
        ok &= np.asarray(data.universe, dtype=bool)[:, j]
    return ok


def _window_count(starts, h):
    """Number of windows started on rows t-h+1 .. t (causal moving sum)."""
    cs = np.cumsum(starts.astype(float))
    out = cs.copy()
    if h < len(cs):
        out[h:] = cs[h:] - cs[:-h]
    return out


def target_weights(data, params):
    h = int(params["hold_sessions"])
    if h < 1:
        raise ValueError("hold_sessions must be >= 1")
    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)
    pre, post = _calendar_events(data, h)
    for name in TRADED:
        j = data.col(name)
        ok = _instrument_ok(data, j)
        n_post = _window_count(post & ok, h)
        n_pre = _window_count(pre & ok, h)
        w[:, j] = np.clip(LEG * (n_post - n_pre), -LEG, LEG)
    return w
