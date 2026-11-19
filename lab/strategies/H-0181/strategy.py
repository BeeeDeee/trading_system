"""H-0181: slow S&P 500 change reversal, 12-month hold, LIQ-500, dollar-neutral slot book.

Events (after the close of row t, t >= 5): deletion when sp500_member[t-1] == 1 and sp500_member[t] == 0,
addition when 0 -> 1, counted only for LIQ-500 members tradable on row t. Each event opens a slot from the
open of t+1 (long 0.05 for a deletion, short 0.05 for an addition). A slot is closed at the open after
hold_days sessions (decision row t + hold_days) or earlier at the first decision where the stock is out of
LIQ-500, not tradable, not listed or delisting. A new event on a stock with a slot replaces the old slot.
Each side is capped at 0.45 gross (pro-rata scaling). The net of the slots is hedged with an equal-weight
basket of the other tradable LIQ-500 members without a slot. Decision rows only on slot-book changes and on
the first session of each month; NaN rows otherwise (keep positions).
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 252}

SLOT_WEIGHT = 0.05      # fixed slot weight (card)
SIDE_CAP = 0.45         # max gross of the long slots and of the short slots (card)
SKIP_ROWS = 5           # the first 5 sessions of the view are ignored (card)


def _bool(x):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    a = np.asarray(a, dtype=float)
    return np.nan_to_num(a, nan=0.0) > 0


def slot_book(member, tradable, alive, sp, month_start, hold_days):
    """Core slot/hedge logic on plain arrays. Returns (T, N) weights with NaN on non-decision rows."""
    T, N = member.shape
    hold = int(hold_days)
    W = np.full((T, N), np.nan)
    side = np.zeros(N)                     # +1 long slot (deletion), -1 short slot (addition), 0 none
    start = np.zeros(N, dtype=np.int64)    # row of the event that opened the slot
    for t in range(T):
        ok = member[t] & tradable[t] & alive[t]
        active = side != 0
        close = active & ((~ok) | ((t - start) >= hold))
        changed = bool(close.any())
        if changed:
            side[close] = 0.0
        if t >= SKIP_ROWS:
            prev = sp[t - 1]
            cur = sp[t]
            ev_ok = member[t] & tradable[t]
            dele = (prev == 1) & (cur == 0) & ev_ok
            addi = (prev == 0) & (cur == 1) & ev_ok
            ev = dele | addi
            if ev.any():
                side[dele] = 1.0
                side[addi] = -1.0
                start[ev] = t
                changed = True
        if not (changed or month_start[t] or t == 0):
            continue
        row = np.zeros(N)
        longs = side > 0
        shorts = side < 0
        n_l = int(longs.sum())
        n_s = int(shorts.sum())
        w_l = SLOT_WEIGHT if n_l * SLOT_WEIGHT <= SIDE_CAP else SIDE_CAP / n_l
        w_s = SLOT_WEIGHT if n_s * SLOT_WEIGHT <= SIDE_CAP else SIDE_CAP / n_s
        row[longs] = w_l
        row[shorts] = -w_s
        net = n_l * w_l - n_s * w_s
        basket = member[t] & tradable[t] & (side == 0)
        n_b = int(basket.sum())
        if abs(net) > 1e-12 and n_b > 0:
            row[basket] = -net / n_b
        W[t] = row
    return W


def target_weights(data, params):
    member = data.universe
    if member is None:
        raise ValueError("H-0181 needs the LIQ-500 point-in-time universe")
    member = _bool(member)
    sp = np.asarray(data.extras["sp500_member"], dtype=float)
    tradable = _bool(data.tradable)
    alive = _bool(data.listed) & ~_bool(data.delisting)
    T = len(data.dates)
    month_start = np.zeros(T, dtype=bool)
    month_start[period_starts(data.dates, "M")] = True
    return slot_book(member, tradable, alive, sp, month_start, int(params["hold_days"]))
