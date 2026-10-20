"""H-0065 mechanism events: pressure entry after the close of E (top RQ +1, bottom -1) and release entry after
the close of L (top RQ -1, bottom +1). Same calendar, betas, residual QTD returns and leg selection as the
strategy (code duplicated because the diagnostic may only import numpy and the lab API)."""
import numpy as np

BETA_WINDOW = 252
BETA_MIN_PAIRS = 200
QTD_MIN_COVERAGE = 0.8
JUNETEENTH_FIRST_YEAR = 2022


def _ymd(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _weekday(days):
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
    wd = _weekday(day)
    out = np.where(wd == 6, day + 1, day)
    return np.where(wd == 5, day - 1, out)


def nyse_holidays(first_year, last_year):
    y = np.arange(int(first_year), int(last_year) + 1, dtype=np.int64)
    parts = []
    ny = _ymd(y, 1, 1)
    wd = _weekday(ny)
    parts.append(np.where(wd == 6, ny + 1, ny)[wd != 5])
    parts.append(_nth_weekday(y, 1, 0, 3))
    parts.append(_nth_weekday(y, 2, 0, 3))
    parts.append(_easter(y) - 2)
    parts.append(_last_weekday(y, 5, 0))
    yj = y[y >= JUNETEENTH_FIRST_YEAR]
    if yj.size:
        parts.append(_observed(_ymd(yj, 6, 19)))
    parts.append(_observed(_ymd(y, 7, 4)))
    parts.append(_nth_weekday(y, 9, 0, 1))
    parts.append(_nth_weekday(y, 11, 3, 4))
    parts.append(_observed(_ymd(y, 12, 25)))
    return np.unique(np.concatenate(parts).astype("datetime64[D]"))


def quarter_calendar(dates, pre_days):
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
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    valid = listed & np.isfinite(close) & np.isfinite(r)
    r = np.where(valid, r, 0.0)
    mem = U & valid
    cnt = mem.sum(axis=1)
    m = np.full(r.shape[0], np.nan)
    has = cnt > 0
    m[has] = (r * mem).sum(axis=1)[has] / cnt[has]
    elig = U & trad
    return r, valid, m, elig, U, trad


def residual_qtd(t, r, valid, m, qid):
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

    q0 = int(np.searchsorted(qid[:t + 1], qid[t], side="left"))
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
    first_after_E = (d >= entry) & ((prev < entry) | (d == entry))
    is_E = first_after_E & (d < last) & ~is_L
    return qid, last, last_prev, k_since, is_E, is_L


def events(data, params):
    q = float(params["leg_quantile"])
    pre_days = int(params["pre_days"])
    dates = np.asarray(data.dates)
    T = dates.shape[0]
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    if T == 0:
        return mask, side
    r, valid, m, elig, U, trad = _inputs(data)
    qid, last, last_prev, k_since, is_E, is_L = decision_rows(dates, pre_days)
    for t in np.flatnonzero(is_E | is_L):
        rq, ranked = residual_qtd(t, r, valid, m, qid)
        sign = -1.0 if is_L[t] else 1.0
        book = leg_book(rq, ranked & elig[t], q, sign)
        mask[t] = book != 0
        side[t] = np.sign(book)
    return mask, side
