"""H-0065 Quarter-end window-dressing cycle (LIQ-500, dollar-neutral).

Pressure phase: after the close of E (pre_days scheduled XNYS sessions before the quarter's last session L)
go long the top leg_quantile / short the bottom leg_quantile of the residual (beta-adjusted) quarter-to-date
return RQ, hold through the close of L. Release phase: after the close of L re-rank on the full-quarter RQ and
hold the mirror book (long losers, short winners) for post_days sessions, then go flat. Flat otherwise.

Calendar: XNYS sessions = weekdays minus the scheduled NYSE holidays, generated from the holiday rules
(no date literals, never from price rows after t).
"""
import numpy as np

PARAMS = {"leg_quantile": 0.2, "post_days": 5, "pre_days": 10}

BETA_WINDOW = 252          # sessions ending at t
BETA_MIN_PAIRS = 200       # valid (r, m) pairs needed for a beta
QTD_MIN_COVERAGE = 0.8     # share of the quarter's sessions with a return
JUNETEENTH_FIRST_YEAR = 2022


# ----------------------------------------------------------------------------------------------------------
# Scheduled NYSE holiday calendar (rule based)
# ----------------------------------------------------------------------------------------------------------
def _ymd(y, m, d):
    """Integer day numbers (days since the numpy epoch) from integer year, month (may exceed 12), day."""
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _date(y, m, d):
    """datetime64[D] from integer year, month, day."""
    return _ymd(y, m, d).astype("datetime64[D]")


def _weekday(days):
    """Monday = 0 ... Sunday = 6 (day number 0 of the epoch was a Thursday)."""
    return (np.asarray(days, dtype=np.int64) + 3) % 7


def _nth_weekday(y, m, weekday, n):
    first = _ymd(y, m, 1)
    off = (weekday - _weekday(first)) % 7
    return first + off + 7 * (n - 1)


def _last_weekday(y, m, weekday):
    last = _ymd(y, np.asarray(m) + 1, 1) - 1
    off = (_weekday(last) - weekday) % 7
    return last - off


def _easter(y):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
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
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * mm + 114) // 31
    day = (h + l_ - 7 * mm + 114) % 31 + 1
    return _ymd(y, month, day)


def _observed(day):
    """Saturday -> Friday before, Sunday -> Monday after."""
    wd = _weekday(day)
    out = np.where(wd == 6, day + 1, day)
    return np.where(wd == 5, day - 1, out)


def nyse_holidays(first_year, last_year):
    y = np.arange(int(first_year), int(last_year) + 1, dtype=np.int64)
    parts = []
    ny = _ymd(y, 1, 1)                       # New Year: Sunday -> Monday, Saturday -> not observed
    wd = _weekday(ny)
    parts.append(np.where(wd == 6, ny + 1, ny)[wd != 5])
    parts.append(_nth_weekday(y, 1, 0, 3))   # Martin Luther King Jr. Day
    parts.append(_nth_weekday(y, 2, 0, 3))   # Washington's Birthday
    parts.append(_easter(y) - 2)             # Good Friday
    parts.append(_last_weekday(y, 5, 0))     # Memorial Day
    yj = y[y >= JUNETEENTH_FIRST_YEAR]
    if yj.size:
        parts.append(_observed(_ymd(yj, 6, 19)))   # Juneteenth
    parts.append(_observed(_ymd(y, 7, 4)))   # Independence Day
    parts.append(_nth_weekday(y, 9, 0, 1))   # Labor Day
    parts.append(_nth_weekday(y, 11, 3, 4))  # Thanksgiving
    parts.append(_observed(_ymd(y, 12, 25))) # Christmas
    return np.unique(np.concatenate(parts).astype("datetime64[D]"))


def quarter_calendar(dates, pre_days):
    """Per row: quarter id, E and L of the row's quarter, L of the previous quarter and the number of
    scheduled sessions in [L_prev, date) (= 1 on the first session after L_prev)."""
    d = np.asarray(dates).astype("datetime64[D]")
    months = d.astype("datetime64[M]").astype(np.int64)
    qid = months // 3
    years = months // 12 + 1970
    hol = nyse_holidays(int(years.min()) - 1, int(years.max()) + 1)
    first_cur = (qid * 3).astype("datetime64[M]").astype("datetime64[D]")
    first_next = ((qid + 1) * 3).astype("datetime64[M]").astype("datetime64[D]")
    last = np.busday_offset(first_next, -1, roll="forward", holidays=hol)
    last_prev = np.busday_offset(first_cur, -1, roll="forward", holidays=hol)
    entry = np.busday_offset(last, -int(pre_days), roll="forward", holidays=hol)
    k_since = np.busday_count(last_prev, d, holidays=hol)
    return qid, entry, last, last_prev, k_since


# ----------------------------------------------------------------------------------------------------------
# Returns, market, betas, residual QTD returns
# ----------------------------------------------------------------------------------------------------------
def _inputs(data):
    U = data.universe
    if U is None:
        raise ValueError("H-0065 needs the point-in-time LIQ-500 universe")
    U = np.asarray(U, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0                 # close-to-close total return
    valid = listed & np.isfinite(close) & np.isfinite(r)  # the stock has a return that day
    r = np.where(valid, r, 0.0)
    mem = U & valid
    cnt = mem.sum(axis=1)
    m = np.full(r.shape[0], np.nan)
    has = cnt > 0
    m[has] = (r * mem).sum(axis=1)[has] / cnt[has]      # equal-weight market return of members
    elig = U & trad                                     # members tradable on the decision day
    return r, valid, m, elig, U, trad


def residual_qtd(t, r, valid, m, qid):
    """Beta over the 252 sessions ending at t and residual quarter-to-date return RQ[t]; returns (rq, ok)."""
    lo = max(0, t - BETA_WINDOW + 1)
    x = m[lo:t + 1]
    xf = np.isfinite(x)
    xs = np.where(xf, x, 0.0)
    ok = (valid[lo:t + 1] & xf[:, None]).astype(np.float64)
    n = ok.sum(axis=0)
    sx = xs @ ok
    sxx = (xs * xs) @ ok
    yv = r[lo:t + 1] * ok
    sy = yv.sum(axis=0)
    sxy = xs @ yv
    den = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (n * sxy - sx * sy) / den
    beta_ok = (n >= BETA_MIN_PAIRS) & (den > 0) & np.isfinite(beta)
    beta = np.where(beta_ok, beta, 0.0)

    q0 = int(np.searchsorted(qid[:t + 1], qid[t], side="left"))   # first row of the current quarter
    nq = t - q0 + 1
    xq = m[q0:t + 1]
    xqf = np.isfinite(xq)
    xqs = np.where(xqf, xq, 0.0)
    okq = (valid[q0:t + 1] & xqf[:, None]).astype(np.float64)
    cnt = okq.sum(axis=0)
    rq = (r[q0:t + 1] * okq).sum(axis=0) - beta * (xqs @ okq)
    ranked = beta_ok & (cnt >= QTD_MIN_COVERAGE * nq) & np.isfinite(rq)
    return rq, ranked


def leg_book(rq, cand, leg_quantile, sign):
    """sign=+1: long top / short bottom; sign=-1: the mirror. Each leg sums to 0.5 in absolute value."""
    w = np.zeros(rq.shape[0])
    idx = np.flatnonzero(cand)
    n = idx.size
    k = min(int(np.floor(leg_quantile * n + 1e-9)), n // 2)
    if k < 1:
        return w
    order = idx[np.argsort(rq[idx], kind="stable")]
    bottom = order[:k]
    top = order[n - k:]
    w[top] = sign * 0.5 / k
    w[bottom] = -sign * 0.5 / k
    return w


def decision_rows(dates, pre_days):
    d = np.asarray(dates).astype("datetime64[D]")
    qid, entry, last, last_prev, k_since = quarter_calendar(d, pre_days)
    is_L = d == last
    prev = np.empty_like(d)
    if d.size:
        prev[0] = d[0]
        prev[1:] = d[:-1]
    first_after_E = (d >= entry) & ((prev < entry) | (d == entry))   # first data row on/after E
    is_E = first_after_E & (d < last) & ~is_L
    return qid, last, last_prev, k_since, is_E, is_L


# ----------------------------------------------------------------------------------------------------------
def target_weights(data, params):
    q = float(params["leg_quantile"])
    post_days = int(params["post_days"])
    pre_days = int(params["pre_days"])

    dates = np.asarray(data.dates)
    T = dates.shape[0]
    N = len(data.instruments)
    W = np.zeros((T, N))
    if T == 0:
        return W

    r, valid, m, elig, U, trad = _inputs(data)
    qid, last, last_prev, k_since, is_E, is_L = decision_rows(dates, pre_days)

    book = None
    alive = None
    kind = None
    key = None
    for t in range(T):
        if is_L[t]:
            rq, ranked = residual_qtd(t, r, valid, m, qid)
            book = leg_book(rq, ranked & elig[t], q, -1.0)     # release: long losers, short winners
            alive = book != 0
            kind, key = "release", last[t]
        elif is_E[t]:
            rq, ranked = residual_qtd(t, r, valid, m, qid)
            book = leg_book(rq, ranked & elig[t], q, 1.0)      # pressure: long winners, short losers
            alive = book != 0
            kind, key = "pressure", qid[t]
        elif kind == "pressure" and qid[t] == key and dates[t] < last[t]:
            pass                                               # hold through the close of L
        elif kind == "release" and last_prev[t] == key and 1 <= k_since[t] < post_days:
            pass                                               # hold sessions L+1 .. L+post_days
        else:
            book, alive, kind, key = None, None, None, None    # flat
        if book is not None:
            alive &= U[t] & trad[t]          # left the universe / not tradable: closed, leg not re-scaled
            W[t] = np.where(alive, book, 0.0)
    return W
