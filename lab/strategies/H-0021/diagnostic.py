"""H-0021 mechanism event: first row at which a post-panel-start listing reaches a calendar-day
distance of at least 180 - lead_days from its first bar, if member and tradable at that row.
Identical to the strategy's entry rule (strategy._event_rows); side -1 (short leg)."""
import numpy as np

LOCKUP_DAYS = 180
PRE_EXISTING_ROWS = 30


def events(data, params):
    if data.universe is None:
        raise ValueError("H-0021 needs the LIQ-500 point-in-time universe")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    has_bar = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=float))
    T, N = has_bar.shape
    thr = LOCKUP_DAYS - int(params["lead_days"])
    day = np.asarray(data.dates).astype(np.int64)
    any_bar = has_bar.any(axis=0)
    first = np.where(any_bar, np.argmax(has_bar, axis=0), T)
    mask = np.zeros((T, N), dtype=bool)
    cols = np.nonzero((first >= PRE_EXISTING_ROWS) & (first < T))[0]
    if cols.size:
        t = np.searchsorted(day, day[first[cols]] + thr, side="left")
        inside = t < T
        cols, t = cols[inside], t[inside]
        fire = universe[t, cols] & tradable[t, cols] & has_bar[t, cols]
        mask[t[fire], cols[fire]] = True
    side = -np.ones((T, N), dtype=float)
    return mask, side
