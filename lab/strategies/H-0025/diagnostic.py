"""H-0025 mechanism events: switch decisions of the turn-of-month beta rotation.

Switch A (close of the session before the month's last session): top beta leg +1, bottom beta leg -1.
Switch B (close of session +post_days of the new month): bottom beta leg +1, top beta leg -1.
Calendar, betas, eligibility and legs are identical to strategy.py (code duplicated because the
diagnostic may only import numpy and the lab framework).
"""
import numpy as np

MIN_OBS_CARD = 189
MIN_OBS_RATIO = 0.75
MIN_LEG = 20


def _holidays(dates):
    yrs = dates.astype("datetime64[Y]").astype(np.int64)
    Y = np.arange(yrs.min(), yrs.max() + 2) + 1970
    y_month = (Y - 1970).astype("datetime64[Y]").astype("datetime64[M]")
    a = Y % 19
    b = Y // 100
    c = Y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    x = h + l - 7 * m + 114
    e_month = x // 31
    e_day = x % 31 + 1
    months = lambda n: np.full(Y.shape, n, dtype=np.int64).astype("timedelta64[M]")
    easter = ((y_month + (e_month - 1).astype("timedelta64[M]")).astype("datetime64[D]")
              + (e_day - 1).astype("timedelta64[D]"))
    good_friday = easter - np.full(Y.shape, 2, dtype=np.int64).astype("timedelta64[D]")
    june1 = (y_month + months(5)).astype("datetime64[D]")
    memorial = np.busday_offset(june1, -1, roll="forward", weekmask="Mon")
    nov1 = (y_month + months(10)).astype("datetime64[D]")
    thanksgiving = np.busday_offset(nov1, 3, roll="forward", weekmask="Thu")
    return np.sort(np.concatenate([good_friday, memorial, thanksgiving]))


def _leg_state(dates, post_days):
    T = len(dates)
    mon = dates.astype("datetime64[M]")
    new = np.ones(T, dtype=bool)
    new[1:] = mon[1:] != mon[:-1]
    start = np.maximum.accumulate(np.where(new, np.arange(T), 0))
    k = np.arange(T) - start + 1
    hol = _holidays(dates)
    nxt = np.busday_offset(dates, 1, roll="backward", holidays=hol)
    nxt2 = np.busday_offset(nxt, 1, roll="forward", holidays=hol)
    turn = nxt2.astype("datetime64[M]") != mon
    return np.where(k < post_days, True, np.where(k == post_days, False, turn))


def _switch_rows(dates, post_days):
    s = _leg_state(dates, post_days)
    rows = np.nonzero(s[1:] != s[:-1])[0] + 1
    return [(int(t), bool(s[t])) for t in rows]


def _daily_returns(data):
    r = (1.0 + data.ret_co.astype(np.float64)) * (1.0 + data.ret_oc.astype(np.float64)) - 1.0
    px = np.isfinite(data.close)
    lst = data.listed.astype(bool)
    valid = np.zeros_like(px)
    valid[1:] = px[1:] & px[:-1] & lst[1:] & lst[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, 0.0)
    return r, valid


def _market_proxy(r, valid, universe):
    inc = valid & universe
    n = inc.sum(axis=1)
    s = np.where(inc, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m


def _min_obs(lookback):
    return int(min(MIN_OBS_CARD, int(np.ceil(MIN_OBS_RATIO * lookback))))


def _betas_at(t, cols, r, valid, m, lookback):
    lo = max(0, t - lookback + 1)
    mw = m[lo:t + 1]
    v = valid[lo:t + 1][:, cols] & np.isfinite(mw)[:, None]
    mm = np.where(np.isfinite(mw), mw, 0.0)[:, None]
    rr = r[lo:t + 1][:, cols]
    n = v.sum(axis=0).astype(np.float64)
    ok = n >= _min_obs(lookback)
    nn = np.maximum(n, 1.0)
    mx = np.where(v, mm, 0.0).sum(axis=0) / nn
    my = np.where(v, rr, 0.0).sum(axis=0) / nn
    dm = np.where(v, mm - mx[None, :], 0.0)
    dr = np.where(v, rr - my[None, :], 0.0)
    cov = (dm * dr).sum(axis=0)
    var = (dm * dm).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(ok & (var > 0), cov / np.where(var > 0, var, 1.0), np.nan)
    return beta


def _legs_at(t, data, r, valid, m, lookback, q):
    uni = data.universe[t].astype(bool) & data.tradable[t].astype(bool)
    cand = np.nonzero(uni)[0]
    if cand.size == 0:
        return cand, cand, cand
    beta = _betas_at(t, cand, r, valid, m, lookback)
    good = np.isfinite(beta)
    elig = cand[good]
    b = beta[good]
    ne = elig.size
    if ne == 0:
        return elig, elig, elig
    n_leg = min(max(int(np.floor(q * ne + 0.5)), MIN_LEG), ne)
    order = np.argsort(b, kind="stable")
    bottom = elig[order[:n_leg]]
    order_desc = np.argsort(-b, kind="stable")
    top = elig[order_desc[:n_leg]]
    return elig, top, bottom


def events(data, params):
    lookback = int(params["beta_lookback"])
    q = float(params["leg_quantile"])
    post_days = int(params["post_days"])
    if data.universe is None:
        raise ValueError("H-0025 needs the point-in-time LIQ-500 universe")
    T = len(data.dates)
    N = len(data.instruments)
    universe = data.universe.astype(bool)
    r, valid = _daily_returns(data)
    m = _market_proxy(r, valid, universe)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t, is_a in _switch_rows(data.dates, post_days):
        elig, top, bottom = _legs_at(t, data, r, valid, m, lookback, q)
        if elig.size == 0:
            continue
        both = np.intersect1d(top, bottom)          # only possible with < 2 * 20 eligible names
        top = np.setdiff1d(top, both)
        bottom = np.setdiff1d(bottom, both)
        mask[t, top] = True
        mask[t, bottom] = True
        side[t, top] = 1.0 if is_a else -1.0
        side[t, bottom] = -1.0 if is_a else 1.0
    return mask, side
