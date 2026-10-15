"""H-0032 Fundamental anomalies resolved at earnings news.

LIQ-500 dollar-neutral: long good-composite / short bad-composite stocks only in a
(2*half_window+1)-session window around their predicted 8-K Item 2.02 release
(last year's reaction session + 364 calendar days).
"""
import numpy as np

PARAMS = {"half_window": 2, "leg_quantile": 0.3}

DEDUP_SESSIONS = 10          # two releases within 10 sessions: keep only the first
PREDICT_DAYS = 364           # predicted release = first session on/after date(r) + 364 days
NO_RECENT_SESSIONS = 30      # no release in the 30 sessions up to and including t
AG_LAG = 252                 # asset growth over 252 sessions
MIN_MEMBERS = 100            # no events on a date with fewer scored members
MIN_LEG_NAMES = 3            # whole book flat if a leg has fewer names
LEG_GROSS = 0.5              # each leg 0.5 gross


def _pct_rank(x):
    """Percentile rank in [0, 1] (average rank for ties): (rank - 1) / (n - 1)."""
    n = x.shape[0]
    if n < 2:
        return np.full(n, 0.5)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    start = np.concatenate((np.zeros(1), np.cumsum(counts)[:-1].astype(np.float64)))
    avg = start + (counts - 1) / 2.0
    return avg[inv] / (n - 1)


def event_sides(data, params):
    """(T, N) float: +1 / -1 on the decision row t = P - half_window - 1 of an event, else 0."""
    hw = int(params["half_window"])
    q = float(params["leg_quantile"])
    ex = data.extras
    ea = np.asarray(ex["ea_reaction_day"], dtype=np.float64)
    netinc = ex["sf1_art_netinc"]
    ncfo = ex["sf1_art_ncfo"]
    assets = ex["sf1_art_assets"]
    gp = ex["sf1_art_gp"]
    if data.universe is None:
        raise ValueError("H-0032 needs the LIQ-500 universe mask")
    U = np.asarray(data.universe, dtype=bool)
    T, N = ea.shape
    side = np.zeros((T, N), dtype=np.float64)
    if T == 0:
        return side

    # (1) release sessions (NaN = no 8-K history = no release)
    rel = ea == 1.0
    cpad = np.zeros((T + 1, N), dtype=np.int64)
    cpad[1:] = np.cumsum(rel, axis=0, dtype=np.int64)    # cpad[k] = releases in rows < k
    idx = np.arange(T)
    prev = cpad[idx] - cpad[np.maximum(idx - DEDUP_SESSIONS, 0)]           # rows r-10 .. r-1
    keep = rel & (prev == 0)
    recent = cpad[idx + 1] - cpad[np.maximum(idx + 1 - NO_RECENT_SESSIONS, 0)]  # rows t-29 .. t

    # predicted release date P = date(r) + 364 days; decision row t = first row whose
    # date + (half_window + 1) weekdays (np.busday_offset, holidays ignored) is >= P
    dint = data.dates.astype(np.int64)
    qdate = np.busday_offset(data.dates, hw + 1, roll="forward").astype(np.int64)
    rr, jj = np.nonzero(keep)
    pdate = dint[rr] + PREDICT_DAYS
    te = np.searchsorted(qdate, pdate, side="left")
    ok = te < T
    cand = np.zeros((T, N), dtype=bool)
    cand[te[ok], jj[ok]] = True
    # (2) eligibility: member at t, no release in the 30 sessions up to t
    cand &= U & (recent == 0)

    # (3) composite score on the decision rows
    rows = np.nonzero(cand.any(axis=1))[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in rows:
            if t < AG_LAG:
                continue
            a = np.asarray(assets[t], dtype=np.float64)
            a0 = np.asarray(assets[t - AG_LAG], dtype=np.float64)
            acc = (np.asarray(netinc[t], dtype=np.float64) - np.asarray(ncfo[t], dtype=np.float64)) / a
            gpa = np.asarray(gp[t], dtype=np.float64) / a
            ag = a / a0 - 1.0
            valid = (U[t] & np.isfinite(acc) & np.isfinite(gpa) & np.isfinite(ag)
                     & (a > 0) & (a0 > 0))
            n = int(valid.sum())
            if n < MIN_MEMBERS:
                continue
            score = (1.0 - _pct_rank(acc[valid]) + _pct_rank(gpa[valid])
                     + 1.0 - _pct_rank(ag[valid])) / 3.0
            hi = np.quantile(score, 1.0 - q)
            lo = np.quantile(score, q)
            s_full = np.full(N, np.nan)
            s_full[valid] = score
            evt = cand[t] & valid
            is_long = evt & (s_full >= hi)
            is_short = evt & (s_full <= lo) & ~is_long
            side[t, is_long] = 1.0
            side[t, is_short] = -1.0
    return side


def target_weights(data, params):
    hw = int(params["half_window"])
    side_ev = event_sides(data, params)
    T, N = side_ev.shape
    U = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    delisting = np.asarray(data.delisting, dtype=bool)
    alive = U & listed & ~delisting

    W = np.zeros((T, N), dtype=np.float64)
    cur = np.zeros(N, dtype=np.float64)
    end = np.full(N, -1, dtype=np.int64)
    for t in range(T):
        cur[end < t] = 0.0                       # window over: exit at the open of P + hw + 1
        new = side_ev[t] != 0.0
        if new.any():
            cur[new] = side_ev[t, new]
            end[new] = t + 2 * hw                # held on weight rows t .. t + 2*hw
        cur[~alive[t]] = 0.0                     # left LIQ-500 / delisted: closed at next open
        end[~alive[t]] = -1
        lg = cur > 0
        sh = cur < 0
        nl = int(lg.sum())
        ns = int(sh.sum())
        if nl >= MIN_LEG_NAMES and ns >= MIN_LEG_NAMES:
            W[t, lg] = LEG_GROSS / nl
            W[t, sh] = -LEG_GROSS / ns
    return W
