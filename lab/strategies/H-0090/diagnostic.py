"""H-0090 mechanism test events: a twin pair (corr >= 0.95 over 252 sessions, identified at each month
start) with |z| >= z_entry of its 5-session log-return gap against its own 126-session spread volatility,
no event in that pair in the previous hold_days sessions. Laggard +1, leader -1, both on the same date.
Same pair, gap and z code as strategy.py (copied: the static rules allow no import of strategy.py).
No book conditions (slots, already-held stocks): the card's mechanism_test event does not have them."""
import numpy as np

CORR_MIN = 0.95
CORR_WINDOW = 252
CORR_MIN_VALID = 240
GAP_WINDOW = 5
SIGMA_WINDOW = 126


def _inputs(data):
    universe = data.universe
    if universe is None:
        raise ValueError("H-0090 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(universe, dtype=bool)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    has_px = np.isfinite(close)
    valid = listed & tradable & has_px & np.isfinite(ret_co) & np.isfinite(ret_oc)
    valid &= (ret_co > -1.0) & (ret_oc > -1.0)
    valid[1:] &= listed[:-1] & has_px[:-1]
    valid[:1] = False
    return universe, ret_co, ret_oc, tradable, valid


def _log_ret(ret_co, ret_oc, valid):
    co = np.where(valid, ret_co, 0.0)
    oc = np.where(valid, ret_oc, 0.0)
    return np.log1p(co) + np.log1p(oc)


def _month_starts(dates):
    months = np.asarray(dates).astype("datetime64[M]")
    starts = np.zeros(months.shape[0], dtype=bool)
    starts[1:] = months[1:] != months[:-1]
    return np.flatnonzero(starts)


def _month_pairs(m, universe, ret_co, ret_oc, valid):
    lo = m - CORR_WINDOW + 1
    if lo < 0:
        return []
    members = np.flatnonzero(universe[m])
    if members.size < 2:
        return []
    vwin = valid[lo:m + 1][:, members]
    keep = vwin.sum(axis=0) >= CORR_MIN_VALID
    members = members[keep]
    if members.size < 2:
        return []
    mask = vwin[:, keep]
    x = _log_ret(ret_co[lo:m + 1][:, members], ret_oc[lo:m + 1][:, members], mask)
    r = np.where(mask, np.expm1(x), 0.0)
    mf = mask.astype(np.float64)
    n = mf.T @ mf
    sx = r.T @ mf
    sxx = (r * r).T @ mf
    sxy = r.T @ r
    var_a = n * sxx - sx * sx
    var_b = var_a.T
    cov = n * sxy - sx * sx.T
    den = var_a * var_b
    ok = (den > 0.0) & (n >= 2.0)
    corr = np.full(cov.shape, -np.inf)
    corr[ok] = cov[ok] / np.sqrt(den[ok])
    hit = np.triu(corr >= CORR_MIN, 1)
    ia, ib = np.nonzero(hit)
    return [(int(members[a]), int(members[b])) for a, b in zip(ia, ib)]


def _pair_candidates(i, j, m, end, z_entry, universe, tradable, ret_co, ret_oc, valid):
    a = m - SIGMA_WINDOW + 1
    if a < 0 or end <= m:
        return []
    cols = [i, j]
    vij = valid[a:end][:, cols]
    lr = _log_ret(ret_co[a:end][:, cols], ret_oc[a:end][:, cols], vij)
    both = vij[:, 0] & vij[:, 1]
    d = np.where(both, lr[:, 0] - lr[:, 1], 0.0)
    win = np.lib.stride_tricks.sliding_window_view(d, SIGMA_WINDOW)
    full = np.lib.stride_tricks.sliding_window_view(both, SIGMA_WINDOW).all(axis=1)
    gap = win[:, -GAP_WINDOW:].sum(axis=1)
    sigma = win.std(axis=1, ddof=1)
    rows = np.arange(m, end)
    elig = universe[m:end, i] & universe[m:end, j] & tradable[m:end, i] & tradable[m:end, j]
    good = full & elig & (sigma > 0.0)
    z = np.zeros(rows.shape[0])
    z[good] = gap[good] / (sigma[good] * np.sqrt(GAP_WINDOW))
    fire = good & (np.abs(z) >= z_entry)
    out = []
    for k in np.flatnonzero(fire):
        zz = float(z[k])
        lag, lead = (j, i) if zz > 0 else (i, j)
        out.append((int(rows[k]), abs(zz), i, j, lag, lead))
    return out


def events(data, params):
    z_entry = float(params["z_entry"])
    hold_days = int(params["hold_days"])
    universe, ret_co, ret_oc, tradable, valid = _inputs(data)
    T, N = universe.shape
    starts = _month_starts(data.dates)
    cands = []
    for k, m in enumerate(starts):
        end = int(starts[k + 1]) if k + 1 < len(starts) else T
        for i, j in _month_pairs(int(m), universe, ret_co, ret_oc, valid):
            cands.extend(_pair_candidates(i, j, int(m), end, z_entry, universe, tradable, ret_co, ret_oc, valid))
    cands.sort(key=lambda c: (c[0], -c[1], c[2], c[3]))
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    last_event = {}
    for (t, az, i, j, lag, lead) in cands:
        if t - last_event.get((i, j), -(10 ** 9)) <= hold_days:
            continue
        last_event[(i, j)] = t
        # a stock in two events on one date (three share classes) keeps the side of the larger |z|
        if not mask[t, lag]:
            mask[t, lag] = True
            side[t, lag] = 1.0
        if not mask[t, lead]:
            mask[t, lead] = True
            side[t, lead] = -1.0
    return mask, side
