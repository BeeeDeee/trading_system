"""H-0135: 52-week-high anchoring in liquid crypto (Binance spot top-20, long-only).

Weekly formations; each formation buys the k = min(n_hold, floor(E/2)) eligible coins whose close is
nearest their trailing W-day closing high (NH = close / rolling max close), one quarter of equity
equally weighted, held 28 days; four overlapping cohorts. A coin leaving the PIT universe is sold at
the next open and its slice stays in cash until its cohort ends.
"""
import numpy as np

PARAMS = {"high_window_days": 365, "n_hold": 5}

MIN_HISTORY_DAYS = 365   # pairs younger than a year are never eligible (card, fixed)
VOL_DAYS = 30            # 30-day quote volume (tie-break, fallback universe)
TOP_N = 20               # crypto_top_n, n = 20 (fallback universe only)
STEP_DAYS = 7            # formation every 7th day
HOLD_DAYS = 28           # cohort life
N_COHORTS = 4            # each cohort gets 1 / N_COHORTS of equity


def _rolling_sum(x, n):
    """Trailing n-row sum including row t (fewer rows at the start). x must be finite."""
    cs = np.cumsum(x, axis=0)
    out = cs.copy()
    if x.shape[0] > n:
        out[n:] = cs[n:] - cs[:-n]
    return out


def _fallback_universe(close, vol30, n):
    """Top n by trailing 30-day quote volume among pairs with a valid close on t (only if the gate gives none)."""
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


def eligibility(close, uni, day):
    """Eligible on t: universe member, valid close on t, >= 365 days since the first valid close."""
    T, N = close.shape
    fin = np.isfinite(close)
    has = fin.any(axis=0)
    first = np.argmax(fin, axis=0)
    first_day = day[first] if T > 0 else np.zeros(N, dtype=np.int64)
    age = day[:, None] - first_day[None, :] + 1
    started = np.arange(T)[:, None] >= first[None, :]
    return uni & fin & has[None, :] & started & (age >= MIN_HISTORY_DAYS)


def formation_rows(elig, day):
    """Every 7th calendar day, starting on the first day with at least 2 eligible coins."""
    T = elig.shape[0]
    E = elig.sum(axis=1)
    cand = np.nonzero(E >= 2)[0]
    form = np.zeros(T, dtype=bool)
    if cand.size == 0:
        return form
    d0 = day[cand[0]]
    form = (day >= d0) & ((day - d0) % STEP_DAYS == 0)
    return form


def select(close, vol30, elig, t, W, n_hold):
    """Columns of the k highest-NH eligible coins on row t (ties: higher 30-day quote volume)."""
    cols = np.nonzero(elig[t])[0]
    k = min(int(n_hold), cols.size // 2)
    if k < 1:
        return cols[:0]
    lo = max(0, t - int(W) + 1)
    hi = np.nanmax(close[lo:t + 1][:, cols], axis=0)   # over available history if shorter than W
    nh = close[t, cols] / hi
    order = np.lexsort((cols, -vol30[t, cols], -nh))   # primary: NH desc, then volume desc, then column
    return cols[order[:k]]


def book(close, vol, uni, dates, W, n_hold):
    """Daily (T, N) book weights (sum over active cohorts) and the formation-row mask."""
    close = np.asarray(close, dtype=np.float64)
    T, N = close.shape
    day = _day_numbers(dates)
    v = np.where(np.isfinite(vol), np.asarray(vol, dtype=np.float64), 0.0)
    vol30 = _rolling_sum(v, VOL_DAYS)
    if uni is None:
        uni = _fallback_universe(close, vol30, TOP_N)
    uni = np.asarray(uni, dtype=bool)
    elig = eligibility(close, uni, day)
    form = formation_rows(elig, day)
    wf = np.zeros((T, N), dtype=np.float64)
    for t in np.nonzero(form)[0]:
        sel = select(close, vol30, elig, t, W, n_hold)
        if sel.size == 0:
            continue                                   # empty cohort: its quarter stays in cash
        e = int(np.searchsorted(day, day[t] + HOLD_DAYS, side="left"))   # decision rows t .. (day < t + 28)
        alive = np.logical_and.accumulate(uni[t:e][:, sel], axis=0)       # out of the universe once -> cash
        wf[t:e, sel] += (1.0 / N_COHORTS / sel.size) * alive
    gross = wf.sum(axis=1)
    over = gross > 1.0
    if over.any():                                     # float rounding guard only
        wf[over] /= gross[over][:, None]
    return wf, form


def target_weights(data, params):
    W = int(params["high_window_days"])
    n_hold = int(params["n_hold"])
    close = np.asarray(data.close, dtype=np.float64)
    vol = np.asarray(data.dollar_volume, dtype=np.float64)   # Binance spot quote volume (USDT)
    wf, form = book(close, vol, data.universe, data.dates, W, n_hold)
    T, N = wf.shape
    out = np.full((T, N), np.nan)
    if T == 0:
        return out
    out[form] = wf[form]                               # formation rows: full book
    changed = np.zeros((T, N), dtype=bool)
    changed[1:] = wf[1:] != wf[:-1]
    changed[form] = False
    rows = changed.any(axis=1)                         # other rows: universe exits (and late expiries)
    out[rows] = wf[rows]                               # full row; unchanged targets do not trade
    return out
