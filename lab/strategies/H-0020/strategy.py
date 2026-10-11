"""H-0020 Option-expiration hedge-unwind cycle (LIQ-500, beta-neutral).

OE week  = Mon..Fri of the calendar week whose Friday is the third Friday (day 15-21).
POST week = the following calendar week (its Friday has day 22-28).
After the close of row t the strategy looks at the NEXT XNYS trading day (estimated from the date of row t
with weekday arithmetic plus the regular NYSE holiday rules, no future rows):
  next day in an OE week   -> hold the OE book   (row where this starts = D1, groups formed here)
  next day in a POST week  -> hold the POST book (row where this starts = D2, same stocks, sides reversed)
  otherwise                -> flat               (row where this starts = D3)
Weights are written only on rows where that state changes; all other rows are NaN (keep positions).
"""
import numpy as np

PARAMS = {"leg_n": 100, "lookback_days": 63}

COVERAGE = 0.8            # dollar_volume present on >= 80 % of the lookback days
BETA_WINDOW = 252
BETA_MIN_OBS = 126
STOCK_BETA_LO, STOCK_BETA_HI = 0.3, 3.0
LEG_BETA_LO, LEG_BETA_HI = 0.5, 2.0

PH_NONE, PH_OE, PH_POST = 0, 1, 2


# ----------------------------------------------------------------------------- calendar (date arithmetic only)
def _ymdw(x):
    """x: int64 days since the numpy epoch -> year, month, day of month, weekday (Mon=0)."""
    x = np.asarray(x, dtype=np.int64)
    mon = x.astype("datetime64[D]").astype("datetime64[M]")
    mi = mon.astype(np.int64)
    year = mi // 12 + 1970
    month = mi % 12 + 1
    day = x - mon.astype("datetime64[D]").astype(np.int64) + 1
    wd = (x + 3) % 7          # day 0 was a Thursday
    return year, month, day, wd


def _easter(year):
    """Gregorian Easter Sunday (anonymous algorithm) -> month, day."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = (h + ll - 7 * m + 114) % 31 + 1
    return month, day


JUNETEENTH_FROM_YEAR = 2022


def _is_nyse_holiday(x):
    """Regular NYSE full-day holidays (rules, no date list). Unscheduled closures are not known."""
    y, m, d, wd = _ymdw(x)
    em, ed = _easter(y)
    gf_m = np.where(ed > 2, em, 3)
    gf_d = np.where(ed > 2, ed - 2, ed + 29)          # Easter on Apr 1/2 -> Good Friday Mar 30/31
    hol = (m == gf_m) & (d == gf_d)
    hol |= (m == 1) & ((d == 1) | ((d == 2) & (wd == 0)))                       # New Year
    hol |= (m == 1) & (wd == 0) & (d >= 15) & (d <= 21)                         # MLK day
    hol |= (m == 2) & (wd == 0) & (d >= 15) & (d <= 21)                         # Presidents day
    hol |= (m == 5) & (wd == 0) & (d >= 25)                                     # Memorial day
    hol |= (y >= JUNETEENTH_FROM_YEAR) & (m == 6) & (
        (d == 19) | ((d == 18) & (wd == 4)) | ((d == 20) & (wd == 0)))         # Juneteenth
    hol |= (m == 7) & ((d == 4) | ((d == 3) & (wd == 4)) | ((d == 5) & (wd == 0)))   # Independence day
    hol |= (m == 9) & (wd == 0) & (d <= 7)                                      # Labor day
    hol |= (m == 11) & (wd == 3) & (d >= 22) & (d <= 28)                        # Thanksgiving
    hol |= (m == 12) & ((d == 25) | ((d == 24) & (wd == 4)) | ((d == 26) & (wd == 0)))  # Christmas
    return hol


def _next_trading_day(x):
    """Next weekday after x that is not a regular NYSE holiday (int days)."""
    x = np.asarray(x, dtype=np.int64)
    nxt = np.full(x.shape, -1, dtype=np.int64)
    found = np.zeros(x.shape, dtype=bool)
    for k in range(1, 15):
        c = x + k
        ok = (~found) & ((c + 3) % 7 < 5) & (~_is_nyse_holiday(c))
        nxt[ok] = c[ok]
        found |= ok
    return nxt


def _week_phase(x):
    """Phase of the calendar week containing day x, and a cycle id (the OE Friday) for OE/POST weeks."""
    x = np.asarray(x, dtype=np.int64)
    fri = x - (x + 3) % 7 + 4
    _, _, dfri, _ = _ymdw(fri)
    phase = np.full(x.shape, PH_NONE, dtype=np.int64)
    cycle = np.full(x.shape, -1, dtype=np.int64)
    oe = (dfri >= 15) & (dfri <= 21)
    post = (dfri >= 22) & (dfri <= 28)
    phase[oe] = PH_OE
    cycle[oe] = fri[oe]
    phase[post] = PH_POST
    cycle[post] = fri[post] - 7
    return phase, cycle


def schedule(dates):
    """For each row: phase and cycle of the next trading day (point-in-time: uses only the date of row t)."""
    x = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return _week_phase(_next_trading_day(x))


# ----------------------------------------------------------------------------- groups and betas at D1
def _form_groups(t, leg_n, lookback, uni, trad, listed, dv, ret_co, ret_oc):
    lo = max(0, t - lookback + 1)
    win = np.asarray(dv[lo:t + 1], dtype=np.float64)
    has = np.isfinite(win)
    cnt = has.sum(axis=0)
    elig = uni[t] & trad[t] & (cnt >= COVERAGE * lookback)
    idx = np.flatnonzero(elig)
    if idx.size < 2:
        return None
    mean_dv = np.where(has[:, idx], win[:, idx], 0.0).sum(axis=0) / cnt[idx]
    k = min(leg_n, idx.size // 2)
    order = np.lexsort((idx, -mean_dv))           # descending mean dollar volume, ties by column
    pos_high = order[:k]
    pos_low = order[idx.size - k:]

    # betas of the eligible stocks on their equal-weight mean, last 252 rows up to t
    blo = max(0, t - BETA_WINDOW + 1)
    rco = np.asarray(ret_co[blo:t + 1][:, idx], dtype=np.float64)
    roc = np.asarray(ret_oc[blo:t + 1][:, idx], dtype=np.float64)
    R = (1.0 + rco) * (1.0 + roc) - 1.0                 # daily total return
    V = np.asarray(listed[blo:t + 1][:, idx], dtype=bool) & np.isfinite(R)
    Rz = np.where(V, R, 0.0)
    nv = V.sum(axis=1)
    mkt_ok = nv > 0
    mkt = np.where(mkt_ok, Rz.sum(axis=1) / np.maximum(nv, 1), 0.0)
    M = V & mkt_ok[:, None]
    Mf = M.astype(np.float64)
    n = Mf.sum(axis=0)
    sx = (Mf * mkt[:, None]).sum(axis=0)
    sy = (Mf * Rz).sum(axis=0)
    sxx = (Mf * (mkt * mkt)[:, None]).sum(axis=0)
    sxy = (Mf * Rz * mkt[:, None]).sum(axis=0)
    den = n * sxx - sx * sx
    num = n * sxy - sx * sy
    okb = (n >= BETA_MIN_OBS) & (den > 0)
    beta = np.ones(idx.size)
    beta[okb] = num[okb] / den[okb]
    beta = np.clip(beta, STOCK_BETA_LO, STOCK_BETA_HI)

    b_h = float(np.clip(beta[pos_high].mean(), LEG_BETA_LO, LEG_BETA_HI))
    b_l = float(np.clip(beta[pos_low].mean(), LEG_BETA_LO, LEG_BETA_HI))
    g_h = b_l / (b_h + b_l)
    g_l = b_h / (b_h + b_l)
    return idx[pos_high], idx[pos_low], g_h, g_l


def plan(data, params):
    """List of decisions (t, phase, high_cols, low_cols, g_high, g_low); high/low None means flat.

    For PH_OE the book is +g_high/len(high) on high, -g_low/len(low) on low.
    For PH_POST the high/low sets are the D1 groups still tradable (and in the universe) at t,
    and the book is the reverse: -g_high/len(high) on high, +g_low/len(low) on low.
    """
    leg_n = int(params["leg_n"])
    lookback = int(params["lookback_days"])
    if leg_n < 1 or lookback < 1:
        raise ValueError("leg_n and lookback_days must be positive")
    if data.universe is None:
        raise ValueError("H-0020 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = data.listed
    dv = data.dollar_volume
    ret_co = data.ret_co
    ret_oc = data.ret_oc

    phase, cycle = schedule(data.dates)
    T = len(phase)
    out = []
    groups = None        # (cycle, high, low, g_h, g_l) formed at the last D1
    for t in range(T):
        if t > 0 and phase[t] == phase[t - 1] and cycle[t] == cycle[t - 1]:
            continue
        p = int(phase[t])
        if p == PH_OE:
            g = _form_groups(t, leg_n, lookback, uni, trad, listed, dv, ret_co, ret_oc)
            if g is None:
                groups = None
                out.append((t, p, None, None, 0.0, 0.0))
            else:
                groups = (int(cycle[t]), g[0], g[1], g[2], g[3])
                out.append((t, p, g[0], g[1], g[2], g[3]))
        elif p == PH_POST:
            if groups is not None and groups[0] == int(cycle[t]):
                keep = uni[t] & trad[t]
                high = groups[1][keep[groups[1]]]
                low = groups[2][keep[groups[2]]]
                if high.size > 0 and low.size > 0:
                    out.append((t, p, high, low, groups[3], groups[4]))
                    continue
            out.append((t, p, None, None, 0.0, 0.0))
        else:
            out.append((t, p, None, None, 0.0, 0.0))
    return out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    for t, p, high, low, g_h, g_l in plan(data, params):
        row = np.zeros(N)
        if high is not None:
            sign = 1.0 if p == PH_OE else -1.0
            row[high] = sign * g_h / high.size
            row[low] = -sign * g_l / low.size
        w[t] = row
    return w
