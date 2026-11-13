"""H-0135 mechanism events: at each weekly formation date, the k = min(n_hold, floor(E/2)) eligible
top-20 coins nearest their trailing high_window_days closing high (same pipeline as strategy.py), side +1."""
import numpy as np

MIN_HISTORY_DAYS = 365
VOL_DAYS = 30
TOP_N = 20
STEP_DAYS = 7


def _rolling_sum(x, n):
    cs = np.cumsum(x, axis=0)
    out = cs.copy()
    if x.shape[0] > n:
        out[n:] = cs[n:] - cs[:-n]
    return out


def _fallback_universe(close, vol30, n):
    T, N = close.shape
    valid = np.isfinite(close) & (vol30 > 0)
    key = np.where(valid, vol30, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N), (T, N)).copy(), axis=1)
    return valid & (ranks < n)


def _day_numbers(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    if d.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return (d - d[0]).astype(np.int64)


def _eligibility(close, uni, day):
    T, N = close.shape
    fin = np.isfinite(close)
    has = fin.any(axis=0)
    first = np.argmax(fin, axis=0)
    first_day = day[first] if T > 0 else np.zeros(N, dtype=np.int64)
    age = day[:, None] - first_day[None, :] + 1
    started = np.arange(T)[:, None] >= first[None, :]
    return uni & fin & has[None, :] & started & (age >= MIN_HISTORY_DAYS)


def _formation_rows(elig, day):
    T = elig.shape[0]
    cand = np.nonzero(elig.sum(axis=1) >= 2)[0]
    if cand.size == 0:
        return np.zeros(T, dtype=bool)
    d0 = day[cand[0]]
    return (day >= d0) & ((day - d0) % STEP_DAYS == 0)


def _select(close, vol30, elig, t, W, n_hold):
    cols = np.nonzero(elig[t])[0]
    k = min(int(n_hold), cols.size // 2)
    if k < 1:
        return cols[:0]
    lo = max(0, t - int(W) + 1)
    hi = np.nanmax(close[lo:t + 1][:, cols], axis=0)
    nh = close[t, cols] / hi
    order = np.lexsort((cols, -vol30[t, cols], -nh))
    return cols[order[:k]]


def events(data, params):
    W = int(params["high_window_days"])
    n_hold = int(params["n_hold"])
    close = np.asarray(data.close, dtype=np.float64)
    vol = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = close.shape
    day = _day_numbers(data.dates)
    v = np.where(np.isfinite(vol), vol, 0.0)
    vol30 = _rolling_sum(v, VOL_DAYS)
    uni = data.universe
    if uni is None:
        uni = _fallback_universe(close, vol30, TOP_N)
    uni = np.asarray(uni, dtype=bool)
    elig = _eligibility(close, uni, day)
    form = _formation_rows(elig, day)
    mask = np.zeros((T, N), dtype=bool)
    for t in np.nonzero(form)[0]:
        sel = _select(close, vol30, elig, t, W, n_hold)
        mask[t, sel] = True
    mask &= uni & np.asarray(data.tradable, dtype=bool)
    side = np.ones((T, N), dtype=np.float64)
    return mask, side
