"""H-0036 mechanism events: speculative-leg (top leg_quantile by IVOL) LIQ-500 stocks at
decision A (side +1, before the last session of a calendar week) and decision B (side -1, before the
first session of the next week). Same eligibility, IVOL and calendar code as strategy.py (copied,
because the diagnostic may only import numpy and lab.framework.api)."""
import numpy as np

MIN_VALID_SHARE = 0.8


def _days(x):
    x = np.asarray(x)
    if x.dtype.kind == "M":
        return x.astype("datetime64[D]").astype(np.int64)
    return x.astype(np.int64)


def _ymd_days(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    months = (y - 1970) * 12 + (m - 1)
    return months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (d - 1)


def _dow(days):
    return (_days(days) + 3) % 7


def _week_id(days):
    return (_days(days) + 3) // 7


def _nth_weekday(y, m, wd, n):
    first = _ymd_days(y, m, 1)
    return first + ((wd - _dow(first)) % 7) + 7 * (n - 1)


def _last_weekday(y, m, wd):
    last = _ymd_days(y, m + 1, 1) - 1
    return last - ((_dow(last) - wd) % 7)


def _observed(days, saturday_to_friday=True):
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
    ys = np.asarray(years, dtype=np.int64)
    parts = [
        _observed(_ymd_days(ys, 1, 1), saturday_to_friday=False),
        _nth_weekday(ys[ys >= 1998], 1, 0, 3),
        _nth_weekday(ys, 2, 0, 3),
        _good_friday(ys),
        _last_weekday(ys, 5, 0),
        _observed(_ymd_days(ys[ys >= 2022], 6, 19)),
        _observed(_ymd_days(ys, 7, 4)),
        _nth_weekday(ys, 9, 0, 1),
        _nth_weekday(ys, 11, 3, 4),
        _observed(_ymd_days(ys, 12, 25)),
    ]
    hol = np.concatenate([np.asarray(p, dtype=np.int64).ravel() for p in parts])
    hol = np.unique(hol[_dow(hol) < 5])
    return hol.astype("datetime64[D]")


def _calendar_flags(dates, holidays=None):
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
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    has_px = np.isfinite(close)
    prev_px = np.zeros_like(has_px)
    prev_px[1:] = has_px[:-1]
    valid = listed & has_px & prev_px & np.isfinite(r)
    return uni, tradable, valid, r


def _legs(t, L, q, uni, tradable, valid, r):
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
    m = np.where(m_ok, R.sum(axis=1) / np.maximum(n_day, 1), 0.0)
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


def events(data, params):
    L = int(params["ivol_lookback_days"])
    q = float(params["leg_quantile"])
    uni, tradable, valid, r = _inputs(data)
    T, N = valid.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    dec_a, dec_b = _calendar_flags(data.dates)
    for t in np.flatnonzero(dec_a | dec_b):
        legs = _legs(int(t), L, q, uni, tradable, valid, r)
        if legs is None:
            continue
        s_cols = legs[0]
        mask[t, s_cols] = True                     # members and tradable on t by construction
        side[t, s_cols] = 1.0 if dec_a[t] else -1.0
    return mask, side
