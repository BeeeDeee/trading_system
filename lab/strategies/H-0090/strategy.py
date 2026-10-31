"""H-0090 Dual-class twin-share gap reversion (LIQ-500).

Rules of the card and where they live:
 (1) pairs: at the first session of each calendar month (month of row t differs from row t-1),
     LIQ-500 members with >= 240 valid daily returns in the trailing 252 sessions; pairwise-complete
     Pearson correlation of daily total returns; corr >= 0.95 -> twin pair for that month.   _month_pairs
 (2) gap and z: d_s = log(1+r_i,s) - log(1+r_j,s); g_t = sum d over t-4..t; sigma_t = std of d over
     t-125..t; z = g / (sigma * sqrt(5)); a missing return anywhere in either window -> no event.  _pair_candidates
 (3) event: |z| >= z_entry, both legs in universe and tradable on t, neither stock held, no event in the
     pair in the previous hold_days sessions; laggard (lower 5-session return) +1, leader -1.   _book
 (4) book: at most 2 pair slots, +0.25 laggard / -0.25 leader, largest |z| first, no queue; exit decided
     after the close of entry + hold_days; early exit when a leg leaves the universe, is not tradable,
     not listed or delists.   _book
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_entry": 2.0}

# fixed design constants of the card (not tuned parameters)
CORR_MIN = 0.95
CORR_WINDOW = 252
CORR_MIN_VALID = 240
GAP_WINDOW = 5
SIGMA_WINDOW = 126
MAX_SLOTS = 2
LEG_WEIGHT = 0.25


def _inputs(data):
    universe = data.universe
    if universe is None:
        raise ValueError("H-0090 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(universe, dtype=bool)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    delisting = np.asarray(data.delisting, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    has_px = np.isfinite(close)
    # a daily close-to-close return exists on s if the stock is listed and traded on s and had a
    # close on s-1; returns <= -100 % have no log return and count as missing
    valid = listed & tradable & has_px & np.isfinite(ret_co) & np.isfinite(ret_oc)
    valid &= (ret_co > -1.0) & (ret_oc > -1.0)
    valid[1:] &= listed[:-1] & has_px[:-1]
    valid[:1] = False
    return universe, ret_co, ret_oc, listed, tradable, delisting, valid


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
    """Twin pairs (i, j), i < j, identified after the close of month-start row m (rows <= m only)."""
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
    # daily total return r = (1+ret_co)(1+ret_oc)-1, zero where missing (masked out below)
    r = np.where(mask, np.expm1(x), 0.0)
    mf = mask.astype(np.float64)
    n = mf.T @ mf                      # pairwise-complete day counts
    sx = r.T @ mf                      # sx[a, b] = sum of r_a over days where a and b are valid
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
    """Rows t in [m, end) where pair (i, j) has |z_t| >= z_entry and both legs are eligible."""
    a = m - SIGMA_WINDOW + 1
    if a < 0 or end <= m:
        return []
    cols = [i, j]
    vij = valid[a:end][:, cols]
    lr = _log_ret(ret_co[a:end][:, cols], ret_oc[a:end][:, cols], vij)
    both = vij[:, 0] & vij[:, 1]
    d = np.where(both, lr[:, 0] - lr[:, 1], 0.0)
    win = np.lib.stride_tricks.sliding_window_view(d, SIGMA_WINDOW)          # rows t = m .. end-1
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
        # gap > 0: i had the higher 5-session return -> i is the leader
        lag, lead = (j, i) if zz > 0 else (i, j)
        out.append((int(rows[k]), abs(zz), i, j, lag, lead))
    return out


def candidates(data, params):
    """All (t, |z|, i, j, laggard, leader) with |z_t| >= z_entry on eligible twin pairs, sorted."""
    z_entry = float(params["z_entry"])
    universe, ret_co, ret_oc, listed, tradable, delisting, valid = _inputs(data)
    T = universe.shape[0]
    starts = _month_starts(data.dates)
    out = []
    for k, m in enumerate(starts):
        end = int(starts[k + 1]) if k + 1 < len(starts) else T
        for i, j in _month_pairs(int(m), universe, ret_co, ret_oc, valid):
            out.extend(_pair_candidates(i, j, int(m), end, z_entry, universe, tradable, ret_co, ret_oc, valid))
    out.sort(key=lambda c: (c[0], -c[1], c[2], c[3]))
    return out


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    universe, ret_co, ret_oc, listed, tradable, delisting, valid = _inputs(data)
    T, N = universe.shape
    by_row = {}
    for c in candidates(data, params):
        by_row.setdefault(c[0], []).append(c)

    def broken(t, s):
        return (not universe[t, s]) or (not tradable[t, s]) or (not listed[t, s]) or bool(delisting[t, s])

    w = np.zeros((T, N), dtype=np.float64)
    open_pos = []          # (entry_row, laggard, leader)
    last_event = {}        # (i, j) -> row of the last event of the pair
    for t in range(T):
        # exits decided after the close of t (scheduled after hold_days sessions, or a broken leg)
        open_pos = [p for p in open_pos
                    if t - p[0] < hold_days and not broken(t, p[1]) and not broken(t, p[2])]
        held = set()
        for p in open_pos:
            held.add(p[1])
            held.add(p[2])
        cands = by_row.get(t)
        if cands:
            events = []
            for (_, az, i, j, lag, lead) in cands:      # already sorted by |z| descending
                if t - last_event.get((i, j), -(10 ** 9)) <= hold_days:
                    continue
                if lag in held or lead in held:
                    continue
                events.append((lag, lead))
                last_event[(i, j)] = t
            for lag, lead in events:                    # largest |z| first, no queue
                if len(open_pos) >= MAX_SLOTS:
                    break
                if lag in held or lead in held:
                    continue
                open_pos.append((t, lag, lead))
                held.add(lag)
                held.add(lead)
        for _, lag, lead in open_pos:
            w[t, lag] = LEG_WEIGHT
            w[t, lead] = -LEG_WEIGHT
    return w
