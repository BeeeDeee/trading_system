"""H-0082 mechanism events: a coin in the short leg at a weekly (Sunday UTC close) decision.

Self-contained copy of the strategy's event logic (the diagnostic may not import the strategy module):
PIT top-50 member with a bar on t, perp age 7 <= t - L_j <= 7 + hold_days, perp launched >= 30 days after the
perp panel start, matching spot pair (perp id without '.P') first bar >= min_spot_age_days before L_j.
Marked only on the rows the strategy holds short (k > 0 and a non-empty hedge), and only if tradable. Side -1.
"""
import numpy as np

ENTRY_LAG_DAYS = 7
PREEXISTING_DAYS = 30
PERP_SUFFIX = ".P"
PERP_CLASS = "crypto_perp"
SUNDAY = 6


def _day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def _first_bar_rows(data):
    has = np.asarray(data.listed, dtype=bool) | np.isfinite(np.asarray(data.close, dtype=float))
    first = np.argmax(has, axis=0)
    return np.where(has.any(axis=0), first, -1)


def _launch_days(data, params):
    min_age = int(params["min_spot_age_days"])
    n = len(data.instruments)
    dnum = _day_numbers(data.dates)
    is_perp = np.array([a == PERP_CLASS for a in data.asset_class], dtype=bool)
    if not is_perp.any():
        raise ValueError("H-0082: no crypto_perp instrument in the view")
    first = _first_bar_rows(data)
    eligible = np.zeros(n, dtype=bool)
    launch = np.zeros(n, dtype=np.int64)
    perp_first = first[is_perp & (first >= 0)]
    if perp_first.size == 0:
        return eligible, launch, is_perp
    panel_start = dnum[perp_first.min()]
    name_to_col = {name: j for j, name in enumerate(data.instruments)}
    suffixed = [j for j in np.flatnonzero(is_perp) if data.instruments[j].endswith(PERP_SUFFIX)]
    if suffixed:
        spot_cols = [name_to_col.get(data.instruments[j][:-len(PERP_SUFFIX)]) for j in suffixed]
        if not any(s is not None and not is_perp[s] for s in spot_cols):
            raise ValueError("H-0082: no spot pair of any perp is in the view (binance_spot_1d missing?)")
    # left-censored spot history: seasoning counts only when proven (same rule as the strategy)
    for j in suffixed:
        if first[j] < 0:
            continue
        lday = dnum[first[j]]
        if lday - panel_start < PREEXISTING_DAYS:
            continue
        s = name_to_col.get(data.instruments[j][:-len(PERP_SUFFIX)])
        if s is None or is_perp[s] or first[s] < 0:
            continue
        if dnum[first[s]] <= lday - min_age:
            eligible[j] = True
            launch[j] = lday
    return eligible, launch, is_perp


def events(data, params):
    hold = int(params["hold_days"])
    dnum = _day_numbers(data.dates)
    univ = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    has_bar = np.isfinite(np.asarray(data.close, dtype=float))
    eligible, launch, is_perp = _launch_days(data, params)
    age = dnum[:, None] - launch[None, :]
    in_window = eligible[None, :] & (age >= ENTRY_LAG_DAYS) & (age <= ENTRY_LAG_DAYS + hold)
    candidate = univ & has_bar & is_perp[None, :]
    short = candidate & in_window
    hedge = candidate & ~short
    held = (short.sum(axis=1) > 0) & (hedge.sum(axis=1) > 0)
    sunday = ((dnum + 3) % 7) == SUNDAY
    mask = short & (sunday & held)[:, None] & tradable
    side = np.where(mask, -1.0, 0.0)
    return mask, side
