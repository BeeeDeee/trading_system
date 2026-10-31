"""H-0083 mechanism test events: at each monthly formation date f, the NEAR-HIGH leg (top leg_fraction of PTH
inside its 12-month-return tercile x volatility tercile cell) is an event with side +1 and the FAR-FROM-HIGH leg
(bottom leg_fraction) an event with side -1. Same cell and leg logic as strategy.py (copied, the diagnostic
cannot import the strategy module)."""
import numpy as np

LOOKBACK = 252
MIN_VALID = 240
MIN_CELL = 5


def formation_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    form = nxt.astype("datetime64[M]") != month
    if T > 1:
        new_month = month[1:] != month[:-1]
        form[1:] = form[1:] | (new_month & ~form[:-1])
    return form


def daily_returns(ret_co, ret_oc):
    return (1.0 + np.asarray(ret_co, dtype=float)) * (1.0 + np.asarray(ret_oc, dtype=float)) - 1.0


def price_index(r, valid):
    step = np.where(valid & np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + step, axis=0)


def id_ranks(instruments):
    names = np.array([str(x) for x in instruments])
    rank = np.empty(names.shape[0], dtype=float)
    rank[np.argsort(names, kind="stable")] = np.arange(names.shape[0], dtype=float)
    return rank


def _tercile(x):
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return np.where(x <= q1, 0, np.where(x <= q2, 1, 2))


def legs_at(f, P, r, valid, member, tradable, high_window, leg_fraction, idrank):
    if f < LOOKBACK:
        return None
    empty = np.zeros(0, dtype=int)
    cand = np.flatnonzero(member[f] & tradable[f])
    if cand.size == 0:
        return empty, empty
    lo = f - LOOKBACK + 1
    v = valid[lo:f + 1][:, cand]
    nvalid = v.sum(axis=0)
    pf = P[f, cand]
    p0 = P[f - LOOKBACK, cand]
    rr = np.where(v, r[lo:f + 1][:, cand], 0.0)
    n = np.maximum(nvalid, 2).astype(float)
    mean = rr.sum(axis=0) / n
    dev = np.where(v, rr - mean, 0.0)
    vol = np.sqrt((dev * dev).sum(axis=0) / (n - 1.0))
    mom = pf / p0 - 1.0
    hl = max(f - int(high_window) + 1, 0)
    win = np.where(valid[hl:f + 1][:, cand], P[hl:f + 1][:, cand], -np.inf)
    high = np.maximum(win.max(axis=0), pf)
    pth = pf / high
    ok = (nvalid >= MIN_VALID) & np.isfinite(pf) & (pf > 0) & np.isfinite(p0) & (p0 > 0)
    ok &= np.isfinite(vol) & np.isfinite(mom) & np.isfinite(pth)
    if not ok.any():
        return empty, empty
    cand, vol, mom, pth = cand[ok], vol[ok], mom[ok], pth[ok]
    cell = 3 * _tercile(mom) + _tercile(vol)
    near, far = [], []
    for c in range(9):
        idx = np.flatnonzero(cell == c)
        m = idx.size
        if m < MIN_CELL:
            continue
        k = max(1, int(np.floor(leg_fraction * m + 1e-9)))
        order = idx[np.lexsort((idrank[cand[idx]], vol[idx], -pth[idx]))]
        near.append(cand[order[:k]])
        far.append(cand[order[m - k:]])
    if not near:
        return empty, empty
    return np.sort(np.concatenate(near)), np.sort(np.concatenate(far))


def events(data, params):
    high_window = int(params["high_window_days"])
    leg_fraction = float(params["leg_fraction"])
    if data.universe is None:
        raise ValueError("H-0083 needs the point-in-time LIQ-500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.nan_to_num(np.asarray(data.tradable, dtype=float), nan=0.0) > 0.5
    listed = np.nan_to_num(np.asarray(data.listed, dtype=float), nan=0.0) > 0.5
    r = daily_returns(data.ret_co, data.ret_oc)
    valid = listed & tradable & np.isfinite(r)
    P = price_index(r, valid)
    T, N = member.shape
    idrank = id_ranks(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    for f in np.flatnonzero(formation_rows(data.dates)):
        res = legs_at(f, P, r, valid, member, tradable, high_window, leg_fraction, idrank)
        if res is None:
            continue
        near, far = res
        mask[f, near] = True
        side[f, near] = 1.0
        mask[f, far] = True
        side[f, far] = -1.0
    return mask, side
