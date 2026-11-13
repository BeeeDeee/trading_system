"""H-0138: US-cash-open range breakout on BTC/ETH perps (15m), traded only on high-volatility sessions.

Part 1 (ORB, as H-0137): range = high/low of the 09:30 and 09:45 NY candles; first real close strictly
beyond it among the candles opening 10:00..15:15 NY opens a position (+/- 0.5) filled at the next open;
stop on a real close beyond the opposite edge; time exit at the open of the 16:00 NY candle (decision
after the close of the 15:45 candle). One trade per instrument and session, no re-entry.

Part 2 (volatility state): rv = sqrt(sum of squared 15m log close-to-close returns over the 672 candles
ending at the close of the 09:15 NY candle), real candles only; valid if >= 90 % of the 672 are real.
rank = share of the previous 60 valid rv values strictly below the current one; trade only if
rank >= vol_rank_min.

Clock: times are UTC candle open times; New York = UTC-4 under US daylight saving time (second Sunday
of March to first Sunday of November), UTC-5 otherwise, hand-coded (no zoneinfo).
"""
import numpy as np

PARAMS = {"vol_rank_min": 0.667}

WEIGHT = 0.5                 # fixed weight per instrument while in a position
BAR = 15                     # minutes per candle
RV_CANDLES = 672             # 7 x 24 hours of 15m candles
RV_MIN_REAL_SHARE = 0.9      # fewer real candles than this share -> no trade that session
RANK_SESSIONS = 60           # rank against the previous 60 sessions' rv
OPEN_NY = 9 * 60 + 30        # 09:30 New York, minutes after local midnight
INSTRUMENTS = ("BTCUSDT", "ETHUSDT")


# ----------------------------------------------------------------------------------------------
# calendar helpers (integer day numbers since 1970-01-01, proleptic Gregorian)
# ----------------------------------------------------------------------------------------------
def days_from_civil(y, m, d):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    y = y - (m <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_year(days):
    z = np.asarray(days, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    return y + (m <= 2)


def weekday(days):
    """Monday = 0 ... Sunday = 6 (day 0 of the epoch was a Thursday)."""
    return (np.asarray(days, dtype=np.int64) + 3) % 7


def _first_sunday_on_or_after(days):
    return days + (6 - weekday(days)) % 7


def us_dst(days):
    """True where US daylight saving time is in force on that calendar day (weekdays only matter)."""
    days = np.asarray(days, dtype=np.int64)
    y = civil_year(days)
    start = _first_sunday_on_or_after(days_from_civil(y, 3, 1)) + 7   # second Sunday of March
    end = _first_sunday_on_or_after(days_from_civil(y, 11, 1))        # first Sunday of November
    return (days >= start) & (days < end)


def sessions(m):
    """Session days (Mon-Fri UTC days present in the data) and the UTC minute of their 09:30 NY candle."""
    days = np.unique(m // 1440)
    days = days[weekday(days) < 5]
    offset = np.where(us_dst(days), 240, 300)   # minutes New York is behind UTC
    t930 = days * 1440 + OPEN_NY + offset
    return days, t930


def row_at(m, t):
    """Row index of the candle opening exactly at minute t, -1 if absent."""
    t = np.asarray(t, dtype=np.int64)
    n = len(m)
    if n == 0:
        return np.full(t.shape, -1, dtype=np.int64)
    i = np.searchsorted(m, t, side="left")
    ic = np.minimum(i, n - 1)
    ok = (i < n) & (m[ic] == t)
    return np.where(ok, ic, -1)


# ----------------------------------------------------------------------------------------------
# volatility state
# ----------------------------------------------------------------------------------------------
def realized_vol(m, c, good, t915):
    """rv (S, N) over the 672 candles ending at the close of the 09:15 NY candle, and its validity."""
    T, N = c.shape
    S = len(t915)
    idx = np.where(good, np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)                     # last good row <= t
    prev = np.vstack([np.full((1, N), -1, dtype=last.dtype), last[:-1]])   # last good row < t
    logc = np.log(np.where(good, c, 1.0))
    prev_log = np.take_along_axis(logc, np.maximum(prev, 0), axis=0)
    ret = np.where(good & (prev >= 0), logc - prev_log, 0.0)
    zero = np.zeros((1, N))
    csq = np.vstack([zero, np.cumsum(ret * ret, axis=0)])
    cnt = np.vstack([zero, np.cumsum(good.astype(np.float64), axis=0)])

    rv = np.full((S, N), np.nan)
    valid = np.zeros((S, N), dtype=bool)
    if T == 0 or S == 0:
        return rv, valid
    i915 = row_at(m, t915)
    t_first = t915 - (RV_CANDLES - 1) * BAR                        # open time of the first window candle
    lo = np.searchsorted(m, t_first, side="left")
    complete = (i915 >= 0) & (t_first >= m[0])
    hi = np.where(i915 >= 0, i915 + 1, 0)
    lo = np.where(complete, lo, 0)
    n_real = cnt[hi] - cnt[lo]
    ss = csq[hi] - csq[lo]
    valid = complete[:, None] & (n_real >= RV_MIN_REAL_SHARE * RV_CANDLES)
    rv = np.where(valid, np.sqrt(np.maximum(ss, 0.0)), np.nan)
    return rv, valid


def vol_rank(rv, valid, n=RANK_SESSIONS):
    """Share of the previous n valid rv values strictly below the current one; NaN during warm-up."""
    S, N = rv.shape
    rank = np.full((S, N), np.nan)
    for j in range(N):
        where = np.flatnonzero(valid[:, j])
        vals = rv[where, j]
        L = len(vals)
        if L <= n:
            continue
        q = np.arange(n, L)
        window = vals[q[:, None] - n + np.arange(n)[None, :]]
        rank[where[n:], j] = np.mean(window < vals[n:, None], axis=1)
    return rank


# ----------------------------------------------------------------------------------------------
# opening-range breakout
# ----------------------------------------------------------------------------------------------
def orb_positions(m, h, l, c, good, t930, trade, weight=WEIGHT):
    """Positions (T, N) of the ORB rule on the sessions/instruments where trade[s, j] is True."""
    T, N = c.shape
    pos = np.zeros((T, N))
    if T == 0:
        return pos
    i930 = row_at(m, t930)
    i945 = row_at(m, t930 + BAR)
    sig_lo = np.searchsorted(m, t930 + 2 * BAR, side="left")       # 10:00 NY
    sig_hi = np.searchsorted(m, t930 + 23 * BAR, side="right")     # 15:15 NY inclusive
    hold_end = np.searchsorted(m, t930 + 25 * BAR, side="left")    # 15:45 NY row and later are flat
    for s in np.flatnonzero(trade.any(axis=1)):
        a, b = int(i930[s]), int(i945[s])
        if a < 0 or b < 0:
            continue
        lo_s, hi_s, end_s = int(sig_lo[s]), int(sig_hi[s]), int(hold_end[s])
        for j in range(N):
            if not trade[s, j] or not (good[a, j] and good[b, j]):
                continue
            H = max(h[a, j], h[b, j])
            L = min(l[a, j], l[b, j])
            if not (np.isfinite(H) and np.isfinite(L)):
                continue
            seg = c[lo_s:hi_s, j]
            ok = good[lo_s:hi_s, j]
            brk = ok & ((seg > H) | (seg < L))
            if not brk.any():
                continue
            k = lo_s + int(np.argmax(brk))
            side = 1.0 if c[k, j] > H else -1.0
            end = end_s
            if end > k + 1:
                seg2 = c[k + 1:end, j]
                ok2 = good[k + 1:end, j]
                stop = ok2 & ((seg2 < L) if side > 0 else (seg2 > H))
                if stop.any():
                    end = k + 1 + int(np.argmax(stop))
            if end > k:
                pos[k:end, j] = side * weight
    return pos


def core_positions(m, h, l, c, real, vol_rank_min):
    """Full rule on plain arrays: m int64 UTC minutes (sorted), h/l/c (T, N), real (T, N) bool."""
    m = np.asarray(m, dtype=np.int64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    good = np.asarray(real, dtype=bool) & np.isfinite(c) & (c > 0)
    _, t930 = sessions(m)
    rv, valid = realized_vol(m, c, good, t930 - BAR)               # window ends with the 09:15 NY candle
    rank = vol_rank(rv, valid)
    trade = np.isfinite(rank) & (rank >= vol_rank_min)
    return orb_positions(m, h, l, c, good, t930, trade)


def target_positions(data, params):
    vol_rank_min = float(params["vol_rank_min"])
    cols = [data.col(name) for name in INSTRUMENTS]
    m = np.asarray(data.times).astype(np.int64)
    T = len(m)
    out = np.zeros((T, len(data.instruments)))
    pos = core_positions(m, np.asarray(data.h)[:, cols], np.asarray(data.l)[:, cols],
                         np.asarray(data.c)[:, cols], np.asarray(data.real)[:, cols], vol_rank_min)
    out[:, cols] = pos
    if data.universe is not None:
        out = np.where(np.asarray(data.universe, dtype=bool), out, 0.0)
    return out
