"""H-0036 Day-of-week mood in speculative stocks (LIQ-500, beta-neutral).

Decision A (after the close of the session before an F session = last session of a calendar week):
    long the top leg_quantile by IVOL (S) a/|S| each, short the other eligible stocks (H) b/|H| each.
Decision B (after the close of the F session, i.e. the session before an M session = first session of a week):
    mirror book: S short a/|S|, H long b/|H|.
Every other row (incl. decision C after the M session): flat.
a = beta_H / (beta_S + beta_H), b = beta_S / (beta_S + beta_H) (0.5 / 0.5 if a mean beta is not positive).

The F / M sessions are derived from the date of row t alone with a rule-based NYSE holiday calendar
(weekdays minus the regular NYSE holidays), never from data rows after t.
"""
import numpy as np

PARAMS = {
    "ivol_lookback_days": 63,
    "leg_quantile": 0.2,
}

MIN_VALID_SHARE = 0.8   # card: valid returns on at least 80 % of the last L sessions


# ----------------------------------------------------------------------------------------------
# Calendar (rule-based XNYS holidays; known in advance, computed from integers only)
# ----------------------------------------------------------------------------------------------

def _days(x):
    """datetime64 (any unit) or integer day numbers -> int64 days since the epoch."""
    x = np.asarray(x)
    if x.dtype.kind == "M":
        return x.astype("datetime64[D]").astype(np.int64)
    return x.astype(np.int64)


def _ymd_days(y, m, d):
    """Vectorized (year, month, day) -> int64 day number; month may be 13 (= January of y + 1)."""
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _ymd(y, m, d):
    """Vectorized (year, month, day) -> datetime64[D]."""
    return _ymd_days(y, m, d).astype("datetime64[D]")


def _dow(days):
    """Day of week, Monday = 0 ... Sunday = 6."""
    return (_days(days) + 3) % 7


def _week_id(days):
    """Calendar week (Monday start) as an integer."""
    return (_days(days) + 3) // 7


def _nth_weekday(y, m, wd, n):
    first = _ymd_days(y, m, 1)
    return first + ((wd - _dow(first)) % 7) + 7 * (n - 1)


def _last_weekday(y, m, wd):
    last = _ymd_days(y, m + 1, 1) - 1
    return last - ((_dow(last) - wd) % 7)


def _observed(days, saturday_to_friday=True):
    """NYSE observance: Sunday -> Monday, Saturday -> Friday (or dropped)."""
    dw = _dow(days)
    out = days + np.where(dw == 6, 1, 0) - np.where(dw == 5, 1, 0)
    if not saturday_to_friday:
        out = out[dw != 5]
    return out


def _good_friday(y):
    y = np.asarray(y, dtype=np.int64)
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * ll) // 451
    x = h + ll - 7 * mm + 114
    month = x // 31
    day = (x % 31) + 1
    return _ymd_days(y, month, day) - 2


def _nyse_holidays(years):
    """Regular NYSE full-day holidays of the given years (unscheduled closures are not knowable)."""
    ys = np.asarray(years, dtype=np.int64)
    parts = [
        _observed(_ymd_days(ys, 1, 1), saturday_to_friday=False),  # New Year (Saturday: no closure)
        _nth_weekday(ys[ys >= 1998], 1, 0, 3),                     # Martin Luther King Jr. Day
        _nth_weekday(ys, 2, 0, 3),                                 # Washington's Birthday
        _good_friday(ys),                                          # Good Friday
        _last_weekday(ys, 5, 0),                                   # Memorial Day
        _observed(_ymd_days(ys[ys >= 2022], 6, 19)),               # Juneteenth
        _observed(_ymd_days(ys, 7, 4)),                            # Independence Day
        _nth_weekday(ys, 9, 0, 1),                                 # Labor Day
        _nth_weekday(ys, 11, 3, 4),                                # Thanksgiving
        _observed(_ymd_days(ys, 12, 25)),                          # Christmas
    ]
    hol = np.concatenate([np.asarray(p, dtype=np.int64).ravel() for p in parts])
    hol = np.unique(hol[_dow(hol) < 5])
    return hol.astype("datetime64[D]")


def _calendar_flags(dates, holidays=None):
    """(decision_a, decision_b) per row from the row's own date.

    next1 = the next session after t, next2 = the session after next1.
    next1 is an M session if it starts a new calendar week, an F session if next2 is in a later week.
    Decision A: next1 is F and not M. Decision B: next1 is M and not F. (F == M: one-session week, skipped.)
    """
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    if holidays is None:
        yrs = d.astype("datetime64[Y]").astype(np.int64) + 1970
        holidays = _nyse_holidays(np.arange(yrs.min() - 1, yrs.max() + 2))
    n1 = np.busday_offset(d, 1, roll="forward", holidays=holidays)
    n2 = np.busday_offset(n1, 1, roll="forward", holidays=holidays)
    is_m = _week_id(n1) != _week_id(d)
    is_f = _week_id(n2) != _week_id(n1)
    return is_f & ~is_m, is_m & ~is_f


# ----------------------------------------------------------------------------------------------
# Cross-section: eligibility, betas, IVOL, legs
# ----------------------------------------------------------------------------------------------

def _inputs(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0036 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ret_co = np.asarray(data.ret_co)
    ret_oc = np.asarray(data.ret_oc)
    close = np.asarray(data.close)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0         # close-to-close total return
    has_px = np.isfinite(close)
    prev_px = np.zeros_like(has_px)
    prev_px[1:] = has_px[:-1]
    valid = listed & has_px & prev_px & np.isfinite(r)
    return uni, tradable, valid, r


def _legs(t, L, q, uni, tradable, valid, r):
    """Return (S_cols, H_cols, a, b) for decision row t, or None if no book can be formed."""
    cand = np.flatnonzero(uni[t] & tradable[t])
    if cand.size < 2:
        return None
    lo = max(0, t - L + 1)
    cnt = valid[lo:t + 1][:, cand].sum(axis=0)
    elig = cand[cnt >= MIN_VALID_SHARE * L]
    if elig.size < 2:
        return None
    V = valid[lo:t + 1][:, elig]
    R = np.where(V, r[lo:t + 1][:, elig].astype(np.float64), 0.0)
    n_day = V.sum(axis=1)
    m_ok = n_day > 0
    m = np.where(m_ok, R.sum(axis=1) / np.maximum(n_day, 1), 0.0)   # EW market of the eligible set
    W = (V & m_ok[:, None]).astype(np.float64)
    n = W.sum(axis=0)
    mean_r = (W * R).sum(axis=0) / np.maximum(n, 1.0)
    mean_m = (W * m[:, None]).sum(axis=0) / np.maximum(n, 1.0)
    dr = (R - mean_r) * W
    dm = (m[:, None] - mean_m) * W
    var_m = (dm * dm).sum(axis=0)
    cov = (dr * dm).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var_m > 0, cov / var_m, np.nan)
        resid = dr - beta * dm
        ivol = np.sqrt((resid * resid).sum(axis=0) / (n - 1.0))
    ok = np.isfinite(beta) & np.isfinite(ivol) & (n >= 3)
    elig, beta, ivol = elig[ok], beta[ok], ivol[ok]
    K = elig.size
    if K < 2:
        return None
    k = int(np.floor(q * K + 0.5))
    k = min(max(k, 1), K - 1)
    order = np.argsort(-ivol, kind="stable")
    s_idx, h_idx = order[:k], order[k:]
    beta_s = float(beta[s_idx].mean())
    beta_h = float(beta[h_idx].mean())
    if beta_s > 0 and beta_h > 0:
        a = beta_h / (beta_s + beta_h)
        b = beta_s / (beta_s + beta_h)
    else:
        a = b = 0.5
    return elig[s_idx], elig[h_idx], a, b


def target_weights(data, params):
    L = int(params["ivol_lookback_days"])
    q = float(params["leg_quantile"])
    uni, tradable, valid, r = _inputs(data)
    T, N = valid.shape
    w = np.zeros((T, N), dtype=np.float64)
    dec_a, dec_b = _calendar_flags(data.dates)
    for t in np.flatnonzero(dec_a | dec_b):
        legs = _legs(int(t), L, q, uni, tradable, valid, r)
        if legs is None:
            continue
        s_cols, h_cols, a, b = legs
        side = 1.0 if dec_a[t] else -1.0
        w[t, s_cols] = side * a / s_cols.size
        w[t, h_cols] = -side * b / h_cols.size
    return w
