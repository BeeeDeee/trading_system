"""H-0017 Index-inclusion price-pressure reversal (S&P 500 universe).

Decision after the close of day t (rows 0..t only), executed at the open of t+1.

Rules of the card and their implementation:
(1) Addition event of j on day t: member on t, not a member on t-1 (no event on row 0, where t-1 is
    unknown), and a valid close (finite, > 0) on at least 21 of the 30 rows t-30..t-1 (fewer rows
    available early in the panel: counted over the rows that exist). Seasoning fallback required by
    the card: the filter is applied only once the panel has shown, up to row t, at least one valid
    close on a non-member row; if the panel holds no pre-membership prices the filter is dropped
    instead of silently removing every event (point-in-time check, rows <= t).
(2) Active short set A_t: an event on some e with t-hold_days+1 <= e <= t, and j is a member, tradable
    and has a valid close on t.
(3) A_t non-empty: -0.5/|A_t| on A_t, +0.5/N_t on every other member tradable with a valid close on t.
    A_t empty: all weights 0 (cash).
(4) Decided every row (daily rebalance).
"""
import numpy as np

PARAMS = {"hold_days": 5}

SEASON_WINDOW = 30      # trading days before the event day (card rule 1, fixed in the card)
SEASON_MIN_VALID = 21   # valid closes required in that window (card rule 1, fixed in the card)


def _window_count(x, window, include_current):
    """Rolling count of True in x over rows [t-window, t-1] (include_current=False)
    or [t-window+1, t] (include_current=True), trailing only."""
    T, N = x.shape
    cs = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(x, axis=0, dtype=np.int32, out=cs[1:])
    idx = np.arange(T)
    if include_current:
        hi = idx + 1
        lo = np.maximum(idx + 1 - window, 0)
    else:
        hi = idx
        lo = np.maximum(idx - window, 0)
    return cs[hi] - cs[lo]


def _components(data, params):
    hold_days = int(params["hold_days"])
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if data.universe is None:
        raise ValueError("H-0017 needs the point-in-time sp500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = member.shape

    valid = np.isfinite(close) & (close > 0)

    # (1) first membership day
    prev_member = np.zeros_like(member)
    prev_member[1:] = member[:-1]
    first_day = member & ~prev_member
    first_day[0] = False

    # seasoning: valid closes on rows t-30..t-1
    n_valid_before = _window_count(valid, SEASON_WINDOW, include_current=False)
    seasoned = n_valid_before >= SEASON_MIN_VALID

    # fallback of the card: apply the filter only if pre-membership prices exist in the panel
    # (point-in-time: a valid close on a non-member row seen at or before row t)
    nonmember_price_row = np.any(valid & ~member, axis=1)
    filter_on = np.cumsum(nonmember_price_row) > 0
    season_ok = seasoned | ~filter_on[:, None]

    event = first_day & season_ok
    return hold_days, member, tradable, valid, event


def addition_events(data, params):
    """(T, N) bool: addition event of j after the close of t (rule 1)."""
    return _components(data, params)[4]


def target_weights(data, params):
    hold_days, member, tradable, valid, event = _components(data, params)
    T, N = member.shape

    eligible = member & tradable & valid          # usable in either leg on t
    recent = _window_count(event, hold_days, include_current=True) > 0
    active = recent & eligible                    # (2) A_t
    longs = eligible & ~active                    # (3) every other eligible member

    n_short = active.sum(axis=1)
    n_long = longs.sum(axis=1)
    on = (n_short > 0) & (n_long > 0)

    w = np.zeros((T, N), dtype=np.float64)
    s_short = np.where(on, -0.5 / np.maximum(n_short, 1), 0.0)
    s_long = np.where(on, 0.5 / np.maximum(n_long, 1), 0.0)
    w += active * s_short[:, None]
    w += longs * s_long[:, None]
    return w
