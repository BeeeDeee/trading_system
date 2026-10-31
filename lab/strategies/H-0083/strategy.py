"""H-0083: 52-week-high anchoring net of momentum and volatility (LIQ-500, long-only, monthly).

At each formation date f (last XNYS session of the month, point-in-time approximation below) the eligible
LIQ-500 stocks are split into 3 x 3 cells by independent terciles of the 12-month total return (MOM) and the
252-session daily volatility (VOL). Inside every cell with at least 5 stocks the top leg_fraction by PTH
(P_f / max P over the last high_window_days sessions) is the NEAR-HIGH leg. The book holds the union of the
NEAR-HIGH legs with equal weights summing to 1 from the open after f to the next formation; a held stock that
leaves the universe or is not tradable is sold and its weight stays in cash until the next formation.
"""
import numpy as np

PARAMS = {"high_window_days": 252, "leg_fraction": 0.2}

LOOKBACK = 252   # fixed window of MOM, VOL and the validity rule (card: not a parameter)
MIN_VALID = 240  # valid returns required in the last LOOKBACK sessions
MIN_CELL = 5     # a cell with fewer eligible stocks is skipped


def formation_rows(dates):
    """Bool (T,): row t is a formation date, decided from the date of row t (and t-1) only.

    Primary rule: the next weekday (np.busday_offset, Mon-Fri, ignores exchange holidays) falls in another
    calendar month, i.e. t is the last weekday of its month. Fallback when the last weekday of a month is an
    exchange holiday (e.g. Memorial Day on May 31, Good Friday on March 29-31): no row of that month is flagged,
    so the first session of the new month becomes the formation date (one session late).
    """
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
    """Close-to-close total return r_t = (1 + ret_co)(1 + ret_oc) - 1."""
    return (1.0 + np.asarray(ret_co, dtype=float)) * (1.0 + np.asarray(ret_oc, dtype=float)) - 1.0


def price_index(r, valid):
    """Total-return index P_t = cumprod(1 + r) over valid days; missing days carry the last level."""
    step = np.where(valid & np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + step, axis=0)


def id_ranks(instruments):
    """Rank of each instrument id in ascending (string) order, the last tie-breaker."""
    names = np.array([str(x) for x in instruments])
    rank = np.empty(names.shape[0], dtype=float)
    rank[np.argsort(names, kind="stable")] = np.arange(names.shape[0], dtype=float)
    return rank


def _tercile(x):
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return np.where(x <= q1, 0, np.where(x <= q2, 1, 2))


def legs_at(f, P, r, valid, member, tradable, high_window, leg_fraction, idrank):
    """NEAR-HIGH and FAR-FROM-HIGH legs (column index arrays) at formation row f, rows <= f only.

    Returns None before the 252-session window exists (no decision)."""
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
        # order: PTH descending, then smaller VOL, then instrument id
        order = idx[np.lexsort((idrank[cand[idx]], vol[idx], -pth[idx]))]
        near.append(cand[order[:k]])
        far.append(cand[order[m - k:]])
    if not near:
        return empty, empty
    return np.sort(np.concatenate(near)), np.sort(np.concatenate(far))


def prepare(data):
    """Point-in-time inputs shared by the strategy and the diagnostic."""
    if data.universe is None:
        raise ValueError("H-0083 needs the point-in-time LIQ-500 universe (data.universe is None)")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.nan_to_num(np.asarray(data.tradable, dtype=float), nan=0.0) > 0.5
    listed = np.nan_to_num(np.asarray(data.listed, dtype=float), nan=0.0) > 0.5
    r = daily_returns(data.ret_co, data.ret_oc)
    valid = listed & tradable & np.isfinite(r)
    P = price_index(r, valid)
    return member, tradable, r, valid, P


def target_weights(data, params):
    high_window = int(params["high_window_days"])
    leg_fraction = float(params["leg_fraction"])
    member, tradable, r, valid, P = prepare(data)
    T, N = member.shape
    idrank = id_ranks(data.instruments)
    W = np.full((T, N), np.nan)
    fr = np.flatnonzero(formation_rows(data.dates))
    exit_now = ~(member & tradable)
    for i, f in enumerate(fr):
        res = legs_at(f, P, r, valid, member, tradable, high_window, leg_fraction, idrank)
        if res is None:
            continue
        near = res[0]
        row = np.zeros(N)
        if near.size:
            row[near] = 1.0 / near.size
        W[f] = row
        end = fr[i + 1] if i + 1 < fr.size else T
        if near.size and end > f + 1:
            # sell a held stock from the first row it is out of the universe or not tradable; stays in cash
            gone = np.logical_or.accumulate(exit_now[f + 1:end][:, near], axis=0)
            W[f + 1:end, near] = np.where(gone, 0.0, np.nan)
    return W
