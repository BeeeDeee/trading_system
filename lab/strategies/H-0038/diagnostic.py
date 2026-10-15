"""H-0038 mechanism-test events (TLT only, SHY never has an event).

Same state logic as strategy.py (duplicated here so the file is self-contained):
  short state: 1 <= next_coupon5_sessions <= window_days
  long state:  0 <= last_coupon5_sessions <= window_days - 1 and not short

Pre-auction (side -1): after the close of the session on which the strategy first enters the short
state of an auction cluster (short state today, not short on the previous session).
Post-auction (side +1): after the close of an auction day (last_coupon5_sessions == 0) with no
further qualifying auction within window_days sessions (rule (1) does not fire, the long starts at
the next open).
Events are marked only where TLT is tradable (and a universe member if a universe is given).
"""
import numpy as np

NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"


def events(data, params):
    tlt = data.col("TLT")
    data.col("SHY")
    T = len(data.dates)
    N = len(data.instruments)
    w = int(params["window_days"])

    nxt = np.asarray(data.series[NEXT_KEY], dtype=float).reshape(-1)
    lst = np.asarray(data.series[LAST_KEY], dtype=float).reshape(-1)
    if nxt.shape[0] != T or lst.shape[0] != T:
        raise ValueError("auction series not aligned with data.dates")
    with np.errstate(invalid="ignore"):
        short = (nxt >= 1) & (nxt <= w)
        long_ = (lst >= 0) & (lst <= w - 1) & ~short
        auction_day = lst == 0

    prev_short = np.zeros(T, dtype=bool)
    prev_short[1:] = short[:-1]
    pre_ev = short & ~prev_short
    post_ev = long_ & auction_day

    ok = np.asarray(data.tradable, dtype=bool)[:, tlt]
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)[:, tlt]

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    mask[:, tlt] = (pre_ev | post_ev) & ok
    side[:, tlt] = np.where(pre_ev, -1.0, 1.0)
    side[~mask] = 0.0
    return mask, side
