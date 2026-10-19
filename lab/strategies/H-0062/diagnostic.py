"""H-0062 mechanism events: at each monthly rebalance, the currency with the highest value score v
(cheapest against its trailing average) is an event with side +1, the one with the lowest v (richest)
an event with side -1. Same inputs, eligibility and rebalance rows as strategy.py (code duplicated
because the diagnostic may only import numpy and lab.framework)."""
import numpy as np

CURRENCIES = (
    ("FXE", "DEXUSEU", False),
    ("FXY", "DEXJPUS", True),
    ("FXB", "DEXUSUK", False),
    ("FXA", "DEXUSAL", False),
    ("FXC", "DEXCAUS", True),
    ("FXF", "DEXSZUS", True),
)
FFILL_MAX = 5
COVERAGE = 0.90
MIN_ELIGIBLE = 5


def _ffill_limited(x, limit):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)
    out = np.full_like(x, np.nan)
    ok = (last >= 0) & ((np.arange(T)[:, None] - last) <= limit)
    cols = np.broadcast_to(np.arange(x.shape[1]), x.shape)
    out[ok] = x[last[ok], cols[ok]]
    return out


def _log_prices(data):
    T = len(data.dates)
    raw = np.empty((T, len(CURRENCIES)), dtype=np.float64)
    for k, (_, key, invert) in enumerate(CURRENCIES):
        x = np.asarray(data.series["fred_macro." + key], dtype=np.float64).reshape(T)
        x = np.abs(x)  # |x| == x for real FX fixings; only matters on the synthetic market
        x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
        raw[:, k] = -np.log(x) if invert else np.log(x)
    return _ffill_limited(raw, FFILL_MAX)


def _value_scores(lp, window):
    T = lp.shape[0]
    ok = np.isfinite(lp)
    vals = np.where(ok, lp, 0.0)
    zero = np.zeros((1, lp.shape[1]))
    csum = np.vstack([zero, np.cumsum(vals, axis=0)])
    ccnt = np.vstack([zero, np.cumsum(ok, axis=0).astype(np.float64)])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - window, 0)
    s = csum[hi] - csum[lo]
    n = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
    return mean - lp, n


def _month_end_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = len(d)
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    m = d.astype("datetime64[M]")
    last = nxt.astype("datetime64[M]") != m
    first_of_month = np.zeros(T, dtype=bool)
    first_of_month[1:] = m[1:] != m[:-1]
    missed = np.zeros(T, dtype=bool)
    missed[1:] = first_of_month[1:] & ~last[:-1]
    return last | missed


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    window = int(round(float(params["lookback_years"]) * 252))
    v, n = _value_scores(_log_prices(data), window)
    cols = [data.col(etf) for etf, _, _ in CURRENCIES]
    elig = (np.isfinite(v) & (n >= COVERAGE * window)
            & np.asarray(data.tradable, dtype=bool)[:, cols]
            & np.asarray(data.listed, dtype=bool)[:, cols])
    if data.universe is not None:
        elig &= np.asarray(data.universe, dtype=bool)[:, cols]
    cnt = elig.sum(axis=1)
    started = np.maximum.accumulate(cnt >= MIN_ELIGIBLE) if T else np.zeros(0, dtype=bool)
    days = _month_end_rows(data.dates) & started
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(days):
        idx = np.flatnonzero(elig[t])
        if len(idx) < MIN_ELIGIBLE:
            continue  # same as strategy.py: rebalance skipped, no event on this date
        order = idx[np.argsort(-v[t, idx], kind="stable")]
        hi, lo = cols[order[0]], cols[order[-1]]
        mask[t, hi] = True
        side[t, hi] = 1.0
        mask[t, lo] = True
        side[t, lo] = -1.0
    return mask, side
