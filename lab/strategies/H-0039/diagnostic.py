"""H-0039 mechanism events: every day, the long (+1) and short (-1) legs of steps (1)-(4).

Same selection as strategy.py (duplicated because the diagnostic may import only numpy / lab.framework):
eligible = in universe at t, funding non-missing on rows t-fd+1..t, close non-missing on rows t-6..t;
split at the median of the 7-day compounded return (odd extra name to the high half); within each half
k = max(1, floor(0.25 * half + 0.5)) highest trailing funding sums -> side -1, k lowest -> side +1;
ties by trailing 30-day mean quote volume (more liquid first). Rows with fewer than 8 eligible names
have no events (the strategy's own minimum). Events only where in the universe and tradable at t.
"""
import numpy as np

RET_DAYS = 7
MIN_ELIGIBLE = 8
LEG_FRACTION = 0.25
LIQ_DAYS = 30
UNIVERSE_N = 20   # card: crypto_top_n, n = 20 (used only if the gate supplies no universe mask)


def _lag(x, k, fill):
    if k == 0:
        return x.copy()
    out = np.full_like(x, fill)
    out[k:] = x[:-k]
    return out


def _top_n_universe(close, dv, listed, n=UNIVERSE_N, days=LIQ_DAYS):
    """Same rule as strategy.top_n_universe: PIT top n by quote volume summed over rows t-days+1..t."""
    T, N = close.shape
    qv = np.zeros((T, N), dtype=np.float64)
    for k in range(days):
        dl = _lag(dv, k, np.nan)
        qv += np.where(np.isnan(dl), 0.0, dl)
    cand = listed & ~np.isnan(close) & (qv > 0.0)
    score = np.where(cand, qv, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty((T, N), dtype=np.int64)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return cand & (rank < n)


def _panel(data, funding_days):
    fund = np.asarray(data.extras["funding"], dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
    else:
        uni = _top_n_universe(close, dv, np.asarray(data.listed, dtype=bool))
    T, N = close.shape

    fok = np.ones((T, N), dtype=bool)
    F = np.zeros((T, N), dtype=np.float64)
    for k in range(funding_days):
        fl = _lag(fund, k, np.nan)
        fok &= ~np.isnan(fl)
        F += np.where(np.isnan(fl), 0.0, fl)

    pok = np.ones((T, N), dtype=bool)
    for k in range(RET_DAYS):
        pok &= ~np.isnan(_lag(close, k, np.nan))

    gross = (1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc))
    logg = np.log(gross)
    R = np.zeros((T, N), dtype=np.float64)
    for k in range(RET_DAYS):
        R += _lag(logg, k, 0.0)
    R = np.expm1(R)

    s = np.zeros((T, N), dtype=np.float64)
    c = np.zeros((T, N), dtype=np.float64)
    for k in range(LIQ_DAYS):
        dl = _lag(dv, k, np.nan)
        ok = ~np.isnan(dl)
        s += np.where(ok, dl, 0.0)
        c += ok
    liq = np.where(c > 0, s / np.maximum(c, 1.0), -np.inf)

    elig = uni & fok & pok
    return elig, F, R, liq, uni


def _legs(idx, R, F, liq):
    order = np.lexsort((idx, R[idx]))
    srt = idx[order]
    n_low = len(srt) // 2
    longs, shorts = [], []
    for half in (srt[:n_low], srt[n_low:]):
        m = len(half)
        if m == 0:
            continue
        k = max(1, int(np.floor(LEG_FRACTION * m + 0.5)))
        f = F[half]
        q = liq[half]
        sh = half[np.lexsort((-q, -f))][:k]
        lo_order = half[np.lexsort((-q, f))]
        lo = lo_order[~np.isin(lo_order, sh)][:k]   # a name is never in both legs (ties in F)
        shorts.extend(sh.tolist())
        longs.extend(lo.tolist())
    return np.array(longs, dtype=int), np.array(shorts, dtype=int)


def events(data, params):
    fd = int(params["funding_days"])
    elig, F, R, liq, uni = _panel(data, fd)
    T, N = elig.shape
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    n_elig = elig.sum(axis=1)
    for t in np.flatnonzero(n_elig >= MIN_ELIGIBLE):
        idx = np.flatnonzero(elig[t])
        longs, shorts = _legs(idx, R[t], F[t], liq[t])
        side[t, longs] = 1.0
        side[t, shorts] = -1.0
        mask[t, longs] = True
        mask[t, shorts] = True
    mask &= uni & tradable
    side = np.where(mask, side, 0.0)
    return mask, side
