"""H-0169 Trend-follower deleveraging spillover.

After a big loss day of a replicated, sector-balanced CTA trend book on 18 ETFs, trade against the trend in
the ETFs whose own trend position did not lose that day (bystanders); each shock day opens a cohort held
5 sessions; the book is the plain average of the active cohorts.
"""
import numpy as np

PARAMS = {"shock_k": 1.25, "trend_lookback_days": 126}

SECTORS = (
    ("SPY", "IWM", "EFA", "EEM", "EWJ"),            # equity
    ("IEF", "TLT"),                                 # bonds
    ("GLD", "SLV", "USO", "DBA", "DBB"),            # commodities
    ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF"),     # currencies
)
VOL_WIN = 63        # volatility window (sessions)
VOL_MIN = 50        # minimum valid returns in the 63 sessions ending t-1
Z_WIN = 252         # window of the P standard deviation (t-252..t-1)
Z_MIN = 200         # minimum valid P values in that window
MIN_ELIG = 6        # minimum eligible instruments for P
MIN_SECTORS = 3     # minimum sectors with an eligible member for P
HOLD = 5            # cohort hold in sessions (fixed by the card, not a parameter)


def _shift(x, k, fill):
    """Row t of the result holds x[t-k] (k >= 0); the first k rows are `fill`."""
    out = np.full(x.shape, fill, dtype=x.dtype)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rolling_count(valid, n):
    """Number of True values in rows t-n+1..t (exact integer arithmetic)."""
    c = np.cumsum(valid.astype(np.int64), axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def _rolling_sum(x, n):
    """Sum of rows t-n+1..t (x has no NaN)."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def _rolling_nanstd(x, valid, n, min_count):
    """Sample std (ddof=1) of the valid values of x in rows t-n+1..t; NaN with fewer than min_count."""
    x0 = np.where(valid, x, 0.0)
    cnt = _rolling_count(valid, n).astype(np.float64)
    s1 = _rolling_sum(x0, n)
    s2 = _rolling_sum(x0 * x0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1.0)
    var = np.maximum(var, 0.0)
    return np.where((cnt >= min_count) & (cnt >= 2), np.sqrt(var), np.nan)


def _bystanders(data, params):
    """Returns (cols, bystander mask (T, n), s[j,t-1] (T, n)) on the card's 18 columns."""
    L = int(params["trend_lookback_days"])
    k = float(params["shock_k"])
    sector_cols = [[data.col(name) for name in sec] for sec in SECTORS]   # raises if an ETF is missing
    cols = [c for sec in sector_cols for c in sec]
    T = len(data.dates)

    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    close = np.asarray(data.close, dtype=np.float64)[:, cols]
    ret_co = np.asarray(data.ret_co, dtype=np.float64)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)[:, cols]

    # valid bar on d: listed, traded (tradable open) and a close price
    bar = listed & tradable & np.isfinite(close)
    # close-to-close total return built from ret_co and ret_oc
    gr = (1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc)) - 1.0
    validret = bar & _shift(bar, 1, False) & np.isfinite(gr)
    r = np.where(validret, gr, np.nan)

    # trend sign over the L sessions ending at d (cumulative total return of the L rows)
    lg = np.log1p(np.where(gr > -1.0, gr, 0.0))
    lg = np.where(np.isfinite(lg), lg, 0.0)
    clog = np.cumsum(lg, axis=0)
    trend = np.full(clog.shape, np.nan)
    if L < T:
        trend[L:] = clog[L:] - clog[:-L]
    s = np.sign(trend)

    # 63-session volatility of valid daily returns ending at d
    sig = _rolling_nanstd(gr, validret, VOL_WIN, 2)
    cnt_all = np.cumsum(validret.astype(np.int64), axis=0)          # valid returns up to d
    cnt_vol = _rolling_count(validret, VOL_WIN)                      # valid returns in the 63 sessions ending d

    s_prev = _shift(s, 1, np.nan)
    sig_prev = _shift(sig, 1, np.nan)
    cnt_all_prev = _shift(cnt_all, 1, 0)
    cnt_vol_prev = _shift(cnt_vol, 1, 0)

    elig = (validret
            & (cnt_all_prev >= L + 1)
            & (cnt_vol_prev >= VOL_MIN)
            & np.isfinite(s_prev) & (s_prev != 0)
            & np.isfinite(sig_prev) & (sig_prev > 0))
    if data.universe is not None:
        elig &= np.asarray(data.universe, dtype=bool)[:, cols]

    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(elig, s_prev * r / sig_prev, 0.0)

    # sector averages over eligible members, then the mean over sectors with an eligible member
    sec_sum = np.zeros(T)
    n_sec = np.zeros(T, dtype=np.int64)
    n_elig = np.zeros(T, dtype=np.int64)
    pos = 0
    for sec in sector_cols:
        sl = slice(pos, pos + len(sec))
        pos += len(sec)
        ne = elig[:, sl].sum(axis=1)
        tot = u[:, sl].sum(axis=1)
        has = ne > 0
        sec_sum += np.where(has, tot / np.maximum(ne, 1), 0.0)
        n_sec += has
        n_elig += ne
    ok = (n_elig >= MIN_ELIG) & (n_sec >= MIN_SECTORS)
    P = np.where(ok, sec_sum / np.maximum(n_sec, 1), np.nan)

    # z[t] = P[t] / std(P[t-252..t-1]), at least 200 valid values
    pv = np.isfinite(P)
    sdP = _shift(_rolling_nanstd(np.where(pv, P, 0.0), pv, Z_WIN, Z_MIN), 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(pv & np.isfinite(sdP) & (sdP > 0), P / sdP, np.nan)
    shock = np.isfinite(z) & (z <= -k)

    with np.errstate(invalid="ignore"):
        not_lost = s_prev * r >= 0
    byst = elig & shock[:, None] & not_lost
    return cols, byst, s_prev


def target_weights(data, params):
    cols, byst, s_prev = _bystanders(data, params)
    T = byst.shape[0]
    n_t = byst.sum(axis=1)
    cohort = n_t > 0
    C = np.where(byst, -s_prev / np.maximum(n_t, 1)[:, None], 0.0)   # cohort weight vector on its open day

    # cohort decided at row t is held on rows t..t+HOLD-1 (filled at the opens of t+1..t+HOLD, closed at t+HOLD+1)
    S = np.zeros(C.shape)
    M = np.zeros(T)
    for i in range(HOLD):
        S += _shift(C, i, 0.0)
        M += _shift(cohort.astype(np.float64), i, 0.0)
    book = np.where(M[:, None] > 0, S / np.maximum(M, 1.0)[:, None], 0.0)

    # guard against floating roundoff only (each cohort has gross 1, the average of cohorts <= 1)
    gross = np.abs(book).sum(axis=1)
    book = np.where(gross[:, None] > 1.0, book / np.maximum(gross, 1.0)[:, None], book)

    W = np.zeros((T, len(data.instruments)))
    W[:, cols] = book
    return W
