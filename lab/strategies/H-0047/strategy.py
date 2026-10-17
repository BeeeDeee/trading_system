"""H-0047 Post-macro-release drift across asset classes.

On each scheduled US macro release session t (CPI, Employment Situation, scheduled FOMC
statement; known from next_release_sessions == 1 at the close of t-1), after the close of t:
rank the eligible ETFs by z = r_t / sigma_63 (sigma over sessions t-63..t-1), long the top_k,
short the bottom_k, inverse-vol weights 0.5 per leg, each capped at 0.25 in absolute value
(excess stays in cash). A cohort is held for hold_days sessions (target rows t..t+hold_days-1,
i.e. entered at the open of t+1, exited at the open of t+hold_days+1). The book is the simple
average of all active cohorts; flat in cash when none is active.
"""
import numpy as np

PARAMS = {"hold_days": 5, "top_k": 3}

WINDOW = 63          # fixed by the card, not a parameter
CAP = 0.25           # per-instrument cap, fixed by the card
MIN_PEERS = 3        # cohort needs at least 2 * top_k + 3 eligible ETFs
SERIES_KEY = "macro_release_calendar_1d.next_release_sessions"


def daily_returns(data):
    """Close-to-close total return of each session and its validity mask.

    r_t = (1 + ret_co_t) * (1 + ret_oc_t) - 1. A return on row t is valid when the ETF is
    listed with a finite close on both rows t-1 and t (ret_co/ret_oc are 0 without trading,
    so validity must come from the listing/price masks). Invalid entries are NaN.
    """
    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    if close.size and not np.isfinite(close).any():
        raise ValueError("no prices in data.close: cannot determine valid returns")
    has =listed & np.isfinite(close)
    valid = np.zeros_like(has)
    valid[1:] = has[1:] & has[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)
    return r, valid


def release_days(data):
    """Row t is a release day when next_release_sessions at the close of t-1 equals 1."""
    s = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    T = len(data.dates)
    if s.shape[0] != T:
        raise ValueError("release calendar series does not match the data calendar")
    rel = np.zeros(T, dtype=bool)
    # The real series is an integer session count (>= 1); rint makes "equals 1" robust to
    # float storage and is identical to == 1 on integer data. NaN -> not a release day.
    with np.errstate(invalid="ignore"):
        rel[1:] = np.rint(s[:-1]) == 1.0
    return rel


def select_cohort(r, valid, t, top_k):
    """Cohort formed after the close of row t, or None.

    Returns (weights (N,), long_cols, short_cols). Uses rows t-63..t only.
    """
    if t < WINDOW:
        return None
    win_valid = valid[t - WINDOW:t]
    elig = valid[t] & win_valid.all(axis=0)
    idx = np.flatnonzero(elig)
    if idx.size == 0:
        return None
    sig = np.std(r[t - WINDOW:t, idx], axis=0, ddof=1)
    ok = np.isfinite(sig) & (sig > 0)
    idx = idx[ok]
    sig = sig[ok]
    if idx.size < 2 * top_k + MIN_PEERS:
        return None
    z = r[t, idx] / sig
    # descending z; ties broken by column order (deterministic)
    order = np.lexsort((idx, -z))
    long_pos = order[:top_k]
    short_pos = order[idx.size - top_k:]
    N = valid.shape[1]
    w = np.zeros(N, dtype=float)
    inv = 1.0 / sig
    wl = 0.5 * inv[long_pos] / inv[long_pos].sum()
    ws = 0.5 * inv[short_pos] / inv[short_pos].sum()
    w[idx[long_pos]] = np.minimum(wl, CAP)
    w[idx[short_pos]] = -np.minimum(ws, CAP)
    return w, idx[long_pos], idx[short_pos]


def build_cohorts(data, params):
    """List of (row, weights, long_cols, short_cols) for every formed cohort."""
    top_k = int(params["top_k"])
    r, valid = daily_returns(data)
    rel = release_days(data)
    out = []
    for t in np.flatnonzero(rel):
        c = select_cohort(r, valid, int(t), top_k)
        if c is not None:
            out.append((int(t), c[0], c[1], c[2]))
    return out


def combine_cohorts(starts, W, T, hold_days):
    """Book = simple average of the cohorts active on each row.

    A cohort formed on row s is active on target rows s .. s+hold_days-1. Rows where the
    active set does not change are NaN (keep positions, no re-equalising of drift); row 0
    and rows where the set changes carry the full target (zeros when no cohort is active).
    """
    N = W.shape[1]
    out = np.full((T, N), np.nan)
    starts = np.asarray(starts, dtype=np.int64)
    prev = None
    for t in range(T):
        lo = int(np.searchsorted(starts, t - hold_days + 1, side="left"))
        hi = int(np.searchsorted(starts, t, side="right"))
        key = (lo, hi) if hi > lo else (0, 0)
        if key != prev:
            out[t] = W[lo:hi].mean(axis=0) if hi > lo else 0.0
            prev = key
    return out


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    T = len(data.dates)
    N = len(data.instruments)
    cohorts = build_cohorts(data, params)
    if cohorts:
        starts = np.array([c[0] for c in cohorts], dtype=np.int64)
        W = np.vstack([c[1] for c in cohorts])
    else:
        starts = np.zeros(0, dtype=np.int64)
        W = np.zeros((0, N))
    w = combine_cohorts(starts, W, T, hold_days)
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
        w = np.where(np.isnan(w), w, np.where(uni, w, 0.0))
    return w
