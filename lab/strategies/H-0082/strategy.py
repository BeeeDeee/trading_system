"""H-0082 perp-listing short-constraint relief.

Weekly (after the Sunday UTC close): short the PIT top-50 perps whose perp launched 7..7+hold_days days ago
and whose matching spot pair (perp id without '.P') had its first bar at least min_spot_age_days before the
perp launch; long the other universe perps with the same total amount, equal weights.
"""
import numpy as np

PARAMS = {"hold_days": 56, "min_spot_age_days": 180}

ENTRY_LAG_DAYS = 7          # card: 7 <= a_j(t) <= 7 + hold_days
PREEXISTING_DAYS = 30       # card: perps whose first bar is within the first 30 days of the panel never count
SHORT_TOTAL = 0.5           # card: each short gets -min(0.5 / k, 0.10)
SHORT_CAP = 0.10
PERP_SUFFIX = ".P"
PERP_CLASS = "crypto_perp"
SUNDAY = 6                  # weekday with Monday = 0


def _day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def _first_bar_rows(data):
    """Row of the first bar of every column (-1 if none in the view). Point-in-time: only used once <= t."""
    has = np.asarray(data.listed, dtype=bool) | np.isfinite(np.asarray(data.close, dtype=float))
    first = np.argmax(has, axis=0)
    return np.where(has.any(axis=0), first, -1)


def launch_days(data, params):
    """Per column: (eligible, launch_day). eligible = perp that is a launch event (rules 1 and 2)."""
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
        # fail loudly: the seasoning rule needs the spot panel (binance_spot_1d) in the view ...
        spot_cols = [name_to_col.get(data.instruments[j][:-len(PERP_SUFFIX)]) for j in suffixed]
        if not any(s is not None and not is_perp[s] for s in spot_cols):
            raise ValueError("H-0082: no spot pair of any perp is in the view (binance_spot_1d missing?)")
    # Left-censoring (G0 fix): the gate's view may start at the perp panel, without the spot history before it.
    # A spot pair with a bar on the first row of the view was listed on or before that row, so dnum[0] is a lower
    # bound of its age. Seasoning counts only when it is proven: dnum[first spot row] <= L_j - min_age. A launch
    # with a censored spot pair whose seasoning cannot be verified is not an event (the rule is never relaxed).
    for j in suffixed:
        if first[j] < 0:
            continue
        lday = dnum[first[j]]
        if lday - panel_start < PREEXISTING_DAYS:
            continue                                      # pre-existing perp, never an event
        s = name_to_col.get(data.instruments[j][:-len(PERP_SUFFIX)])
        if s is None or is_perp[s] or first[s] < 0:
            continue                                      # no spot pair matched: never an event
        if dnum[first[s]] <= lday - min_age:
            eligible[j] = True
            launch[j] = lday
    return eligible, launch, is_perp


def decision_weights(data, params):
    """(T, N) weights the book would set at each row if it were a decision row, and the Sunday flags."""
    hold = int(params["hold_days"])
    dnum = _day_numbers(data.dates)
    univ = np.asarray(data.universe, dtype=bool)          # crypto_top_n: raises if the universe is missing
    has_bar = np.isfinite(np.asarray(data.close, dtype=float))
    eligible, launch, is_perp = launch_days(data, params)
    age = dnum[:, None] - launch[None, :]
    in_window = eligible[None, :] & (age >= ENTRY_LAG_DAYS) & (age <= ENTRY_LAG_DAYS + hold)
    candidate = univ & has_bar & is_perp[None, :]
    short = candidate & in_window
    hedge = candidate & ~short
    k = short.sum(axis=1).astype(float)
    m = hedge.sum(axis=1).astype(float)
    active = (k > 0) & (m > 0)
    w_short = np.where(active, np.minimum(SHORT_TOTAL / np.maximum(k, 1.0), SHORT_CAP), 0.0)
    w_long = np.where(active, w_short * k / np.maximum(m, 1.0), 0.0)
    w = np.where(short, -w_short[:, None], 0.0) + np.where(hedge, w_long[:, None], 0.0)
    sunday = ((dnum + 3) % 7) == SUNDAY                   # 1970-01-01 was a Thursday
    return w, sunday, univ


def target_weights(data, params):
    w, sunday, univ = decision_weights(data, params)
    t_n = w.shape[0]
    out = np.full(w.shape, np.nan)
    out[sunday] = w[sunday]
    # point-in-time universe rule (K-0051): a held position whose instrument leaves the universe is closed
    # on the first row it is not a member; the other positions are kept (NaN) until the next Sunday
    rows = np.arange(t_n)
    last_dec = np.maximum.accumulate(np.where(sunday, rows, -1))
    mid = (~sunday) & (last_dec >= 0)
    if mid.any():
        held = w[last_dec[mid]] != 0.0
        left = held & ~univ[mid]
        block = out[mid]
        block[left] = 0.0
        out[mid] = block
    return out
