"""H-0004 Crypto new-listing supply overhang.

Weekly (after the close of each Sunday UTC bar): short an equal-weight basket of eligible pairs whose listing
age is in [skip_days, max_age_days], long an equal-weight basket of eligible pairs with listing age >= 365 days,
0.5 gross per leg; cash if either leg has fewer than 3 pairs.
"""
import numpy as np

PARAMS = {"max_age_days": 90, "skip_days": 14}

SEASONED_DAYS = 365      # fixed constant of the card (not a parameter)
MIN_LEG = 3              # minimum pairs per leg
TOP_N = 100              # crypto_top_n(100)
RANK_WINDOW_DAYS = 30    # fallback ranking window (calendar days ending at t)


def _fallback_universe_row(dv, day_num, t, mask):
    """Top-N of `mask` columns by median daily quote volume over the 30 calendar days ending at t (rows <= t)."""
    cols = np.flatnonzero(mask)
    if cols.size <= TOP_N:
        return mask.copy()
    start = int(np.searchsorted(day_num, day_num[t] - (RANK_WINDOW_DAYS - 1), side="left"))
    win = dv[start:t + 1][:, cols]
    finite = np.isfinite(win)
    score = np.full(cols.size, -np.inf)
    ok = finite.any(axis=0)
    if ok.any():
        score[ok] = np.nanmedian(win[:, ok], axis=0)
    # stable sort: ties broken by column order (deterministic)
    order = np.argsort(-score, kind="stable")
    out = np.zeros_like(mask)
    out[cols[order[:TOP_N]]] = True
    return out


def target_weights(data, params):
    max_age = int(params["max_age_days"])
    skip = int(params["skip_days"])

    dates = np.asarray(data.dates).astype("datetime64[D]")
    T = dates.shape[0]
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    if T == 0:
        return w

    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close)
    if data.listed is not None:
        has_bar &= np.asarray(data.listed, dtype=bool)

    # listing date = first row with a bar (point-in-time: only used where that row is <= t)
    ever = np.logical_or.accumulate(has_bar, axis=0)
    first = np.argmax(has_bar, axis=0)
    day_num = (dates - dates[0]).astype("int64")
    age = day_num[:, None] - day_num[first][None, :]          # calendar days since listing

    # a pair handled as delisted by the loader is never re-entered
    if data.delisting is not None:
        delisted = np.logical_or.accumulate(np.asarray(data.delisting, dtype=bool), axis=0)
    else:
        delisted = np.zeros((T, N), dtype=bool)

    base = has_bar & ever & ~delisted
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    dv = None if universe is not None else np.asarray(data.dollar_volume, dtype=float)

    # decision days: Sunday UTC bars (weekmask Mon..Sun, only Sunday set)
    sundays = np.flatnonzero(np.is_busday(dates, weekmask="0000001"))
    for t in sundays:
        mask = base[t].copy()
        if universe is not None:
            mask &= universe[t]
        else:
            mask = _fallback_universe_row(dv, day_num, t, mask)
        a = age[t]
        short_leg = mask & (a >= skip) & (a <= max_age)
        long_leg = mask & (a >= SEASONED_DAYS)
        row = np.zeros(N)
        ns, nl = int(short_leg.sum()), int(long_leg.sum())
        if ns >= MIN_LEG and nl >= MIN_LEG:
            row[short_leg] = -0.5 / ns
            row[long_leg] = 0.5 / nl
        w[t] = row
    return w
