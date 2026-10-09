"""H-0017 mechanism diagnostic: S&P 500 additions.

Event (card mechanism_test.event): j is a member on t, not on t-1, with at least 21 valid closes in the
30 trading days before t (same seasoning rule and the same no-pre-membership-prices fallback as the
strategy). Marked only where j is in the universe, tradable and has a valid close on t (the conditions
under which the strategy shorts it). Side -1 (short leg).
"""
import numpy as np


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0017 needs the point-in-time sp500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = member.shape
    valid = np.isfinite(close) & (close > 0)

    prev_member = np.zeros_like(member)
    prev_member[1:] = member[:-1]
    first_day = member & ~prev_member
    first_day[0] = False

    cs = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(valid, axis=0, dtype=np.int32, out=cs[1:])
    idx = np.arange(T)
    n_valid_before = cs[idx] - cs[np.maximum(idx - 30, 0)]
    seasoned = n_valid_before >= 21

    nonmember_price_row = np.any(valid & ~member, axis=1)
    filter_on = np.cumsum(nonmember_price_row) > 0
    season_ok = seasoned | ~filter_on[:, None]

    mask = first_day & season_ok & member & tradable & valid
    side = np.where(mask, -1.0, 0.0)
    return mask, side
