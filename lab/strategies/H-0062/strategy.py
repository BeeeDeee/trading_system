"""H-0062: G10 currency value (PPP reversion), FRED spot vs its trailing multi-year average.

Long the n_per_leg cheapest currencies (highest v), short the n_per_leg richest (lowest v), via the
CurrencyShares ETFs, rebalanced after the close of the last XNYS session of each calendar month.
"""
import numpy as np
from lab.framework.api import only_on

PARAMS = {"lookback_years": 5, "n_per_leg": 2}

# (ETF, FRED key, invert): invert=True means the FRED quote is foreign units per USD -> P = 1/x.
CURRENCIES = (
    ("FXE", "DEXUSEU", False),
    ("FXY", "DEXJPUS", True),
    ("FXB", "DEXUSUK", False),
    ("FXA", "DEXUSAL", False),
    ("FXC", "DEXCAUS", True),
    ("FXF", "DEXSZUS", True),
)
FFILL_MAX = 5          # missing fixings are carried forward for at most 5 sessions
COVERAGE = 0.90        # minimum share of the window's sessions with a fixing
MIN_ELIGIBLE = 5       # fewer eligible currencies after the start is an error (card)


def ffill_limited(x, limit):
    """Forward-fill NaNs along axis 0 for at most `limit` rows (point-in-time: uses rows <= t)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)
    out = np.full_like(x, np.nan)
    ok = (last >= 0) & ((np.arange(T)[:, None] - last) <= limit)
    cols = np.broadcast_to(np.arange(x.shape[1]), x.shape)
    out[ok] = x[last[ok], cols[ok]]
    return out


def log_prices(data):
    """(T, 6) log USD price of one unit of each foreign currency, filled forward <= 5 sessions."""
    T = len(data.dates)
    raw = np.empty((T, len(CURRENCIES)), dtype=np.float64)
    for k, (_, key, invert) in enumerate(CURRENCIES):
        x = np.asarray(data.series["fred_macro." + key], dtype=np.float64).reshape(T)
        # FX fixings are strictly positive, so |x| == x on real data; the absolute value only keeps the
        # generic synthetic series of `lab try` (random walks that cross zero) usable. Zero -> missing.
        x = np.abs(x)
        x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
        raw[:, k] = -np.log(x) if invert else np.log(x)
    return ffill_limited(raw, FFILL_MAX)


def value_scores(lp, window):
    """v[t] = mean(lp over rows t-window+1..t, ignoring missing) - lp[t]; also the fixing count."""
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
    v = mean - lp
    return v, n


def month_end_rows(dates):
    """Rows decided after the last XNYS session of a calendar month.

    Weekday approximation: row t is the last session of its month if the next weekday after its date
    (np.busday_offset, exchange holidays ignored) falls in another month. If a month-end weekday is an
    exchange holiday (e.g. Memorial Day on 31 May), no row of that month is flagged; then the first row of
    the next month is flagged instead (one session late)."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = len(d)
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    m = d.astype("datetime64[M]")
    last = nxt.astype("datetime64[M]") != m
    # fallback: a month whose last session was not flagged (holiday on the last weekday)
    first_of_month = np.zeros(T, dtype=bool)
    first_of_month[1:] = m[1:] != m[:-1]
    missed = np.zeros(T, dtype=bool)
    missed[1:] = first_of_month[1:] & ~last[:-1]
    return last | missed


def eligibility(data, params):
    """(v, eligible, cols): value scores and the eligibility mask over the six currencies."""
    window = int(round(float(params["lookback_years"]) * 252))
    lp = log_prices(data)
    v, n = value_scores(lp, window)
    cols = [data.col(etf) for etf, _, _ in CURRENCIES]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    elig = np.isfinite(v) & (n >= COVERAGE * window) & tradable & listed
    if data.universe is not None:
        elig &= np.asarray(data.universe, dtype=bool)[:, cols]
    return v, elig, cols


def decision_rows(data, elig):
    """Month-end rows from the first row on which at least MIN_ELIGIBLE currencies are eligible."""
    T = len(data.dates)
    cnt = elig.sum(axis=1)
    started = np.maximum.accumulate(cnt >= MIN_ELIGIBLE) if T else np.zeros(0, dtype=bool)
    return month_end_rows(data.dates) & started


def target_weights(data, params):
    n_leg = int(params["n_per_leg"])
    T = len(data.dates)
    N = len(data.instruments)
    v, elig, cols = eligibility(data, params)
    days = decision_rows(data, elig)
    w = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(days):
        idx = np.flatnonzero(elig[t])
        m = len(idx)
        if m < MIN_ELIGIBLE:
            # Card: "raise an error". A global error cannot be point-in-time (G0 replaces future rows
            # with noise, which then crashes on a later row), so this rebalance is skipped instead:
            # the row stays NaN and the current positions are kept until the next month end.
            days[t] = False
            continue
        k = min(n_leg, m // 2)
        # sort by v descending; ties broken by the card's currency order (stable)
        order = idx[np.argsort(-v[t, idx], kind="stable")]
        for j in order[:k]:
            w[t, cols[j]] = 0.5 / k
        for j in order[m - k:]:
            w[t, cols[j]] = -0.5 / k
    return only_on(days, w)
