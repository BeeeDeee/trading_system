"""H-0088: currency value (5-year average spot) into country equity ETFs.

Long the n_per_leg country ETFs whose currency is cheapest against its own
lookback average, short the n_per_leg richest; overlapping monthly cohorts held
hold_months decisions.
"""
import numpy as np

PARAMS = {"hold_months": 12, "lookback_months": 60, "n_per_leg": 2}

# ETF -> FRED series -> sign s with ln P = s * ln x (P = USD price of one unit of the foreign currency)
COUNTRIES = (
    ("EWA", "DEXUSAL", 1.0),
    ("EWC", "DEXCAUS", -1.0),
    ("EWD", "DEXSDUS", -1.0),
    ("EWJ", "DEXJPUS", -1.0),
    ("EWL", "DEXSZUS", -1.0),
    ("EWS", "DEXSIUS", -1.0),
    ("EWU", "DEXUSUK", 1.0),
    ("EZU", "DEXUSEU", 1.0),
    ("NORW", "DEXNOUS", -1.0),
    ("ENZL", "DEXUSNZ", 1.0),
)
SESSIONS_PER_MONTH = 21
MIN_COVERAGE_NUM, MIN_COVERAGE_DEN = 4, 5   # at least 80 % of the window's sessions have an observation
MAX_STALENESS = 10                          # latest observation at most 10 sessions old


def decision_rows(dates):
    """Monthly decision rows: the last weekday of the month (weekday arithmetic,
    exchange holidays ignored). If a month has no such row in the data (its last
    weekday was an exchange holiday), the decision is taken on the first session of
    the next month (known point-in-time: the previous month is complete)."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    nxt = np.busday_offset(d, 1, roll="forward")
    m_now = d.astype("datetime64[M]")
    last = nxt.astype("datetime64[M]") != m_now
    mid = (m_now - m_now[0]).astype(np.int64)
    has_dec = np.bincount(mid[last], minlength=int(mid[-1]) + 1) > 0
    dec = last.copy()
    if T > 1:
        new_month = np.zeros(T, dtype=bool)
        new_month[1:] = mid[1:] != mid[:-1]
        prev_mid = np.zeros(T, dtype=np.int64)
        prev_mid[1:] = mid[:-1]
        dec |= new_month & ~has_dec[prev_mid]
    return dec


def _tradable_now(data, j):
    ok = np.asarray(data.tradable[:, j], dtype=bool) & np.asarray(data.listed[:, j], dtype=bool)
    if data.universe is not None:
        ok &= np.asarray(data.universe[:, j], dtype=bool)
    return ok


def value_scores(data, lookback_months):
    """Returns (V, eligible, cols): V (T, C) value score, eligible (T, C) bool, cols (C,) column ids."""
    T = len(data.dates)
    L = int(lookback_months) * SESSIONS_PER_MONTH
    C = len(COUNTRIES)
    idx = np.arange(T)
    V = np.full((T, C), np.nan)
    elig = np.zeros((T, C), dtype=bool)
    cols = np.zeros(C, dtype=np.int64)
    for k, (etf, key, sgn) in enumerate(COUNTRIES):
        j = data.col(etf)
        cols[k] = j
        x = np.asarray(data.series["fred_macro." + key], dtype=np.float64)
        if x.shape[0] != T:
            raise ValueError("series fred_macro.%s has the wrong length" % key)
        good = np.isfinite(x) & (x > 0)
        # A row is a new FX observation only if the value is present and differs from the
        # previous row's value (an aligned series may carry the last value forward; carried
        # values are not observations, so the 80 % / 10-session rules see real fixings only).
        prev = np.full(T, np.nan)
        prev[1:] = x[:-1]
        obs = good & ~(np.isfinite(prev) & (x == prev))
        lp = np.where(obs, sgn * np.log(np.where(good, x, 1.0)), 0.0)
        cs_n = np.concatenate([[0], np.cumsum(obs.astype(np.int64))])
        cs_v = np.concatenate([[0.0], np.cumsum(lp)])
        n = np.zeros(T, dtype=np.int64)
        s = np.zeros(T)
        if T >= L:
            n[L - 1:] = cs_n[L:] - cs_n[:T - L + 1]
            s[L - 1:] = cs_v[L:] - cs_v[:T - L + 1]
        last = np.maximum.accumulate(np.where(obs, idx, -1))
        lastv = lp[np.maximum(last, 0)]
        ok_fx = ((idx >= L - 1) & (n * MIN_COVERAGE_DEN >= MIN_COVERAGE_NUM * L)
                 & (last >= 0) & (idx - last <= MAX_STALENESS) & (n > 0))
        ok = ok_fx & _tradable_now(data, j)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = s / np.maximum(n, 1) - lastv
        V[:, k] = np.where(ok, v, np.nan)
        elig[:, k] = ok
    return V, elig, cols


def rank_legs(v, ok, n_per_leg):
    """Indices (into the country list) of the long (cheapest, highest V) and short legs."""
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(-v[cand], kind="stable")]
    return order[:n_per_leg], order[len(order) - n_per_leg:]


def target_weights(data, params):
    H = int(params["hold_months"])
    n_leg = int(params["n_per_leg"])
    T = len(data.dates)
    N = len(data.instruments)
    V, elig, cols = value_scores(data, params["lookback_months"])
    tradable = np.zeros((T, N), dtype=bool)
    for j in cols:
        tradable[:, j] = _tradable_now(data, j)
    dec = decision_rows(data.dates)
    W = np.full((T, N), np.nan)
    cohorts = []
    for t in np.flatnonzero(dec):
        # slots of an ETF without a tradable bar go to cash in every live cohort until it expires
        for c in cohorts:
            c[~tradable[t]] = 0.0
        new = np.zeros(N)
        ok = elig[t]
        if ok.sum() >= 2 * n_leg + 1:
            lo, sh = rank_legs(V[t], ok, n_leg)
            new[cols[lo]] += 0.5 / n_leg / H
            new[cols[sh]] -= 0.5 / n_leg / H
        cohorts.append(new)
        if len(cohorts) > H:
            cohorts.pop(0)
        W[t] = np.sum(cohorts, axis=0)
    return W
