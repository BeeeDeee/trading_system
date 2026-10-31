"""H-0088 mechanism-test events: at each monthly decision date, the n_per_leg country
ETFs with the cheapest currency (highest value score, side +1) and the n_per_leg richest
(side -1), marked only on dates with at least 2 * n_per_leg + 3 eligible countries
(4 marked + at least 3 unmarked peers at n_per_leg = 2). Same score, eligibility and
calendar as strategy.py (code duplicated because the diagnostic may only import numpy)."""
import numpy as np

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
MIN_COVERAGE_NUM, MIN_COVERAGE_DEN = 4, 5
MAX_STALENESS = 10
MIN_UNMARKED_PEERS = 3


def _decision_rows(dates):
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


def _value_scores(data, lookback_months):
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
        v = s / np.maximum(n, 1) - lastv
        V[:, k] = np.where(ok, v, np.nan)
        elig[:, k] = ok
    return V, elig, cols


def events(data, params):
    n_leg = int(params["n_per_leg"])
    T = len(data.dates)
    N = len(data.instruments)
    V, elig, cols = _value_scores(data, params["lookback_months"])
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    for t in np.flatnonzero(_decision_rows(data.dates)):
        ok = elig[t]
        if ok.sum() < 2 * n_leg + MIN_UNMARKED_PEERS:
            continue
        cand = np.flatnonzero(ok)
        order = cand[np.argsort(-V[t, cand], kind="stable")]
        lo, sh = order[:n_leg], order[len(order) - n_leg:]
        mask[t, cols[lo]] = True
        side[t, cols[lo]] = 1.0
        mask[t, cols[sh]] = True
        side[t, cols[sh]] = -1.0
    return mask, side
