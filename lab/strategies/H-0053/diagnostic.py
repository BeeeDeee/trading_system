"""H-0053 mechanism events: at each month-end decision, the long leg (+1) and the short leg (-1) of the
seasonal book (same rules and code as strategy.py, duplicated here because the diagnostic may only import
numpy and the lab API). All other scored LIQ-500 members are the unmarked peers."""
import numpy as np

MIN_SESSIONS = 15
MIN_OTHER_NUM = 3
MIN_OTHER_DEN = 4
MIN_PER_LEG = 10


def _month_ids(dates):
    return np.asarray(dates).astype("datetime64[M]").astype(np.int64)


def _decision_rows(dates):
    dates = np.asarray(dates).astype("datetime64[D]")
    T = len(dates)
    if T == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    mid = _month_ids(dates)
    nxt = np.busday_offset(dates, 1, roll="forward")
    is_me = _month_ids(nxt) != mid
    starts = np.flatnonzero(np.r_[True, mid[1:] != mid[:-1]])
    block_has_me = np.logical_or.reduceat(is_me, starts)
    fallback = np.zeros(T, dtype=bool)
    if len(starts) > 1:
        fallback[starts[1:]] = ~block_has_me[:-1]
    rows = np.flatnonzero(is_me | fallback)
    target = np.where(is_me[rows], mid[rows] + 1, mid[rows])
    return rows, target


def _monthly_relative_returns(dates, ret_co, ret_oc, valid):
    mid = _month_ids(dates)
    starts = np.flatnonzero(np.r_[True, mid[1:] != mid[:-1]])
    g = np.asarray(ret_co, dtype=np.float64) + 1.0
    g *= np.asarray(ret_oc, dtype=np.float64) + 1.0
    ok = np.asarray(valid, dtype=bool) & np.isfinite(g)
    g[~ok] = 1.0
    np.maximum(g, 1e-300, out=g)
    np.log(g, out=g)
    sums = np.add.reduceat(g, starts, axis=0)
    del g
    counts = np.add.reduceat(ok.astype(np.int32), starts, axis=0)
    mon = np.expm1(sums)
    mon[counts < MIN_SESSIONS] = np.nan
    have = np.isfinite(mon)
    n_have = have.sum(axis=1)
    tot = np.where(have, mon, 0.0).sum(axis=1)
    mean = np.full(len(starts), np.nan)
    np.divide(tot, n_have, out=mean, where=n_have > 0)
    rel = mon - mean[:, None]
    first = int(mid[starts[0]])
    R = np.full((int(mid[starts[-1]]) - first + 1, rel.shape[1]), np.nan)
    R[mid[starts] - first] = rel
    return first, R


def _lag_rows(R, first, target, lags):
    idx = target - np.asarray(lags, dtype=np.int64) - first
    out = np.full((len(idx), R.shape[1]), np.nan)
    inside = (idx >= 0) & (idx < R.shape[0])
    out[inside] = R[idx[inside]]
    return out


def _seasonal_score(R, first, target, years_back):
    Y = int(years_back)
    ann_lags = [12 * k for k in range(1, Y + 1)]
    oth_lags = [L for L in range(13, 12 * Y + 1) if L % 12 != 0]
    ann = _lag_rows(R, first, target, ann_lags)
    ann_ok = np.isfinite(ann).all(axis=0)
    ann_mean = np.where(ann_ok, np.where(ann_ok, ann, 0.0).sum(axis=0) / Y, np.nan)
    if len(oth_lags) == 0:
        return np.full(R.shape[1], np.nan)
    oth = _lag_rows(R, first, target, oth_lags)
    have = np.isfinite(oth)
    n_have = have.sum(axis=0)
    oth_ok = n_have * MIN_OTHER_DEN >= MIN_OTHER_NUM * len(oth_lags)
    oth_mean = np.full(R.shape[1], np.nan)
    np.divide(np.where(have, oth, 0.0).sum(axis=0), n_have, out=oth_mean, where=oth_ok)
    score = ann_mean - oth_mean
    score[~(ann_ok & oth_ok)] = np.nan
    return score


def events(data, params):
    q = float(params["leg_quantile"])
    Y = int(params["years_back"])
    if data.universe is None:
        raise ValueError("H-0053 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = len(data.dates), len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    first, R = _monthly_relative_returns(data.dates, data.ret_co, data.ret_oc, listed & tradable)
    rows, targets = _decision_rows(data.dates)
    for t, m in zip(rows, targets):
        score = _seasonal_score(R, first, int(m), Y)
        elig = np.flatnonzero(universe[t] & tradable[t] & np.isfinite(score))
        n = int(np.floor(q * len(elig) + 1e-9))
        if n < MIN_PER_LEG:
            continue
        order = elig[np.argsort(score[elig], kind="stable")]
        mask[t, order[-n:]] = True
        side[t, order[-n:]] = 1.0
        mask[t, order[:n]] = True
        side[t, order[:n]] = -1.0
    return mask, side
