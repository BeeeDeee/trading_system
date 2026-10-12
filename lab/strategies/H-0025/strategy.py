"""H-0025 Turn-of-month beta rotation, always invested (LIQ-500, long-only).

Switch A (after the close of the session before the month's last session): equal weights on the
highest-beta leg_quantile of eligible LIQ-500 members. Switch B (after the close of session
+post_days of the new month): equal weights on the lowest-beta leg_quantile. No rebalancing between
switches (NaN rows), except that a held name that leaves the universe or is not tradable is sold
(target 0) and its proceeds stay in cash until the next switch.
"""
import numpy as np

PARAMS = {"beta_lookback": 252, "leg_quantile": 0.2, "post_days": 3}

MIN_OBS_CARD = 189        # card: at least 189 valid daily returns in the beta window
MIN_OBS_RATIO = 0.75      # 189 / 252, used only where the window is shorter than 189 rows
MIN_LEG = 20              # card: at least 20 names per leg


# ----------------------------------------------------------------------------------------------
# XNYS calendar: weekday arithmetic plus the scheduled NYSE holidays that can fall on the last or
# second-to-last weekday of a month (Good Friday, Memorial Day, Thanksgiving). All other scheduled
# holidays fall mid-month or at the start of a month, where the first sessions are counted from
# the actual rows, so they cannot change a switch date.
# ----------------------------------------------------------------------------------------------
def _holidays(dates):
    yrs = dates.astype("datetime64[Y]").astype(np.int64)          # years since 1970
    Y = np.arange(yrs.min(), yrs.max() + 2) + 1970
    y_month = (Y - 1970).astype("datetime64[Y]").astype("datetime64[M]")
    # Easter (anonymous Gregorian computus)
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
    memorial = np.busday_offset(june1, -1, roll="forward", weekmask="Mon")   # last Monday of May
    nov1 = (y_month + months(10)).astype("datetime64[D]")
    thanksgiving = np.busday_offset(nov1, 3, roll="forward", weekmask="Thu")
    return np.sort(np.concatenate([good_friday, memorial, thanksgiving]))


def leg_state(dates, post_days):
    """(T,) bool: True = the high-beta (window) leg should be held after the close of row t.

    Uses only the date of row t and the rows of the current month up to t."""
    T = len(dates)
    mon = dates.astype("datetime64[M]")
    new = np.ones(T, dtype=bool)
    new[1:] = mon[1:] != mon[:-1]
    start = np.maximum.accumulate(np.where(new, np.arange(T), 0))
    k = np.arange(T) - start + 1                      # session number within the month (rows <= t)
    hol = _holidays(dates)
    nxt = np.busday_offset(dates, 1, roll="backward", holidays=hol)   # next session after row t
    nxt2 = np.busday_offset(nxt, 1, roll="forward", holidays=hol)     # the session after that
    # next session is the month's last session (day -1) or already in the next month
    turn = nxt2.astype("datetime64[M]") != mon
    return np.where(k < post_days, True, np.where(k == post_days, False, turn))


def switch_rows(dates, post_days):
    """Rows where the leg state changes: list of (row, is_switch_A)."""
    s = leg_state(dates, post_days)
    rows = np.nonzero(s[1:] != s[:-1])[0] + 1
    return [(int(t), bool(s[t])) for t in rows]


# ----------------------------------------------------------------------------------------------
# Returns, market proxy, betas
# ----------------------------------------------------------------------------------------------
def daily_returns(data):
    r = (1.0 + data.ret_co.astype(np.float64)) * (1.0 + data.ret_oc.astype(np.float64)) - 1.0
    px = np.isfinite(data.close)
    lst = data.listed.astype(bool)
    valid = np.zeros_like(px)
    valid[1:] = px[1:] & px[:-1] & lst[1:] & lst[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, 0.0)
    return r, valid


def market_proxy(r, valid, universe):
    inc = valid & universe
    n = inc.sum(axis=1)
    s = np.where(inc, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m


def min_obs(lookback):
    return int(min(MIN_OBS_CARD, int(np.ceil(MIN_OBS_RATIO * lookback))))


def betas_at(t, cols, r, valid, m, lookback):
    """OLS betas of columns cols over rows t-lookback+1..t, pairwise valid days; NaN if < min_obs."""
    lo = max(0, t - lookback + 1)
    mw = m[lo:t + 1]
    v = valid[lo:t + 1][:, cols] & np.isfinite(mw)[:, None]
    mm = np.where(np.isfinite(mw), mw, 0.0)[:, None]
    rr = r[lo:t + 1][:, cols]
    n = v.sum(axis=0).astype(np.float64)
    ok = n >= min_obs(lookback)
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


def legs_at(t, data, r, valid, m, lookback, q):
    """Eligible columns and their top / bottom beta legs (column index arrays) at row t."""
    uni = data.universe[t].astype(bool) & data.tradable[t].astype(bool)
    cand = np.nonzero(uni)[0]
    if cand.size == 0:
        return cand, cand, cand
    beta = betas_at(t, cand, r, valid, m, lookback)
    good = np.isfinite(beta)
    elig = cand[good]
    b = beta[good]
    ne = elig.size
    if ne == 0:
        return elig, elig, elig
    n_leg = min(max(int(np.floor(q * ne + 0.5)), MIN_LEG), ne)
    order = np.argsort(b, kind="stable")              # ascending beta, ties by column order
    bottom = elig[order[:n_leg]]
    order_desc = np.argsort(-b, kind="stable")        # descending beta, ties by column order
    top = elig[order_desc[:n_leg]]
    return elig, top, bottom


# ----------------------------------------------------------------------------------------------
def target_weights(data, params):
    lookback = int(params["beta_lookback"])
    q = float(params["leg_quantile"])
    post_days = int(params["post_days"])
    if data.universe is None:
        raise ValueError("H-0025 needs the point-in-time LIQ-500 universe")

    T = len(data.dates)
    N = len(data.instruments)
    universe = data.universe.astype(bool)
    tradable = data.tradable.astype(bool)
    r, valid = daily_returns(data)
    m = market_proxy(r, valid, universe)

    W = np.full((T, N), np.nan)
    sw = switch_rows(data.dates, post_days)
    for j, (t, is_a) in enumerate(sw):
        end = sw[j + 1][0] if j + 1 < len(sw) else T
        elig, top, bottom = legs_at(t, data, r, valid, m, lookback, q)
        if elig.size == 0:
            continue                                       # nothing eligible yet: no decision
        leg = top if is_a else bottom
        row = np.zeros(N)
        row[leg] = 1.0 / leg.size
        W[t] = row
        if end > t + 1:
            # between switches: sell held names that leave the universe or become untradable;
            # proceeds stay in cash until the next switch (target 0 from then on), others held.
            seg = slice(t + 1, end)
            out = ~universe[seg][:, leg] | ~tradable[seg][:, leg]
            out = np.logical_or.accumulate(out, axis=0)
            block = np.full((end - t - 1, leg.size), np.nan)
            block[out] = 0.0
            W[seg, leg] = block
    return W
