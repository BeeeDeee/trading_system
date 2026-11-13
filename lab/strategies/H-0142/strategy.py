"""H-0142 Altcoins in play at the US open (15m Binance perps, top-20 PIT universe ex BTC/ETH).

Opening range = the two 15m candles opening at 09:30 and 09:45 US Eastern time. A coin is "in play" when its
opening-range quote volume is at least rvol_mult times its own 20-session baseline and at least rvol_mult times
the median coin's relative volume that day. The top_k in-play alts trade the first close beyond the opening
range (candles closing 10:15-15:00 ET), stop at a close beyond the opposite boundary, exit at the 16:00 ET close.
"""
import numpy as np

PARAMS = {"rvol_mult": 2.0, "top_k": 3}

BASELINE_SESSIONS = 20        # fixed by the card (not gridded)
MIN_BASELINE_SESSIONS = 15    # fixed by the card
EXCLUDED = ("BTCUSDT", "ETHUSDT")   # in the median, never traded

# ET minute of day of the candle OPEN time
OR1_OPEN = 9 * 60 + 30        # 09:30-09:45
OR2_OPEN = 9 * 60 + 45        # 09:45-10:00 (selection decision after its close)
ENTRY_FIRST_OPEN = 10 * 60    # candle closing 10:15
ENTRY_LAST_OPEN = 14 * 60 + 45  # candle closing 15:00
EXIT_OPEN = 15 * 60 + 45      # candle closing 16:00: every position is closed


def _days_from_civil(y, m, d):
    """Days since the Unix epoch of the proleptic Gregorian date (y, m, d); y array, m and d scalars."""
    y = np.asarray(y, dtype=np.int64) - (1 if m <= 2 else 0)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = m - 3 if m > 2 else m + 9
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _year_from_days(z):
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    return y + (m <= 2)


def _weekday(days):
    """Monday = 0 ... Sunday = 6 (the Unix epoch day was a Thursday)."""
    return (np.asarray(days, dtype=np.int64) + 3) % 7


def _is_edt(days):
    """US daylight-saving rule on the UTC date: EDT from the second Sunday of March to the first Sunday of
    November. The switch hour (02:00 local) only matters on Sundays, which are never sessions."""
    y = _year_from_days(days)
    mar1 = _days_from_civil(y, 3, 1)
    nov1 = _days_from_civil(y, 11, 1)
    start = mar1 + (6 - _weekday(mar1)) % 7 + 7
    end = nov1 + (6 - _weekday(nov1)) % 7
    return (days >= start) & (days < end)


def et_clock(times):
    """(UTC minutes since epoch, UTC day number, ET minute of day) of each candle open time.
    For the session candles (13:30-21:00 UTC) the ET date equals the UTC date."""
    mins = np.asarray(times).astype("datetime64[m]").astype(np.int64)
    day = np.floor_divide(mins, 1440)
    mod = mins - day * 1440
    offset = np.where(_is_edt(day), 240, 300)
    return mins, day, mod - offset


def _trade(out, j, k0, day, et, d, real, uni, c, hi, lo, w):
    """Walk the candles of session d after the opening range for coin j (rows >= k0)."""
    T = out.shape[0]
    state = 0
    k = k0
    while k < T and day[k] == d and et[k] <= EXIT_OPEN:
        if state != 0:
            if et[k] >= EXIT_OPEN:                 # (b) time exit at the 16:00 ET close
                out[k, j] = 0.0
                return
            if not uni[k, j]:                       # (c) left the PIT universe
                out[k, j] = 0.0
                return
            if real[k, j] and ((state > 0 and c[k, j] < lo) or (state < 0 and c[k, j] > hi)):
                out[k, j] = 0.0                     # (a) stop beyond the opposite boundary
                return
            out[k, j] = state * w                   # hold (also through gap rows, (d))
        else:
            if et[k] > ENTRY_LAST_OPEN:             # breakouts after the 15:00 close are ignored
                return
            if et[k] >= ENTRY_FIRST_OPEN and real[k, j] and np.isfinite(c[k, j]):
                side = 1 if c[k, j] > hi else (-1 if c[k, j] < lo else 0)
                if side != 0:                       # first breakout of the day only
                    if not uni[k, j]:
                        return
                    state = side
                    out[k, j] = state * w
        k += 1


def target_positions(data, params):
    mult = float(params["rvol_mult"])
    top_k = int(params["top_k"])
    c = np.asarray(data.c, dtype=float)
    T, N = c.shape
    out = np.zeros((T, N), dtype=float)
    if data.universe is None:
        raise ValueError("H-0142 needs the crypto_top_n point-in-time universe")
    uni = np.asarray(data.universe, dtype=bool)
    real = np.asarray(data.real, dtype=bool)
    h = np.asarray(data.h, dtype=float)
    lw = np.asarray(data.l, dtype=float)
    qv = np.asarray(data.qv, dtype=float)
    excl = np.zeros(N, dtype=bool)
    for name in EXCLUDED:
        excl[data.col(name)] = True

    mins, day, et = et_clock(data.times)
    i2 = np.flatnonzero((_weekday(day) < 5) & (et == OR2_OPEN))
    if i2.size == 0:
        return out
    i1 = i2 - 1
    has1 = i1 >= 0
    i1 = np.where(has1, i1, 0)
    has1 &= mins[i1] == mins[i2] - 15

    # opening-range volume, only when both candles are real
    valid = real[i1] & real[i2] & has1[:, None]
    ov = np.where(valid, qv[i1] + qv[i2], np.nan)
    valid &= np.isfinite(ov)
    ov = np.where(valid, ov, np.nan)

    # baseline over the previous 20 weekday sessions, valid ones only, at least 15 of them
    S = i2.size
    ovz = np.where(valid, ov, 0.0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(ovz, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(valid.astype(float), axis=0)])
    s = np.arange(S)
    lo_s = np.maximum(s - BASELINE_SESSIONS, 0)
    bsum = cs[s] - cs[lo_s]
    bcnt = cn[s] - cn[lo_s]
    base = bsum / np.maximum(bcnt, 1.0)
    rv_ok = valid & (bcnt >= MIN_BASELINE_SESSIONS) & (base > 0)
    rvol = np.where(rv_ok, ov / np.where(base > 0, base, 1.0), np.nan)

    hi = np.maximum(h[i1], h[i2])
    lo = np.minimum(lw[i1], lw[i2])
    member = uni[i2]
    w = 1.0 / top_k

    for si in range(S):
        ok = member[si] & rv_ok[si]
        if not ok.any():
            continue
        m = float(np.median(rvol[si][ok]))           # market level, BTC/ETH included
        thr = mult * max(1.0, m)
        cand = np.flatnonzero(ok & ~excl & (rvol[si] >= thr)
                              & np.isfinite(hi[si]) & np.isfinite(lo[si]))
        if cand.size == 0:
            continue
        order = np.lexsort((cand, -ov[si, cand], -rvol[si, cand]))   # RVOL desc, then OV desc
        for j in cand[order[:top_k]]:
            _trade(out, int(j), int(i2[si]) + 1, day, et, day[i2[si]], real, uni, c,
                   hi[si, j], lo[si, j], w)
    return out
