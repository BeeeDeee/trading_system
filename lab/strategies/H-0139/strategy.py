"""H-0139: US-cash-open range breakout on BTC/ETH perps after a narrow-range weekday (NR-n), 15m.

Rules (card signal.description):
 (1) Daily UTC bars (00:00-23:45 UTC candles): H = max high, L = min low, R = ln(H/L). Only Monday-Friday
     UTC bars enter the comparison. Session on New York weekday D is eligible when the previous weekday UTC
     bar has R <= min R of the nr_n most recent weekday bars ending with it. A bar with more than 4 non-real
     (gap or missing) candles, or fewer than nr_n weekday bars, makes the session not eligible.
 (2) Opening range: max high / min low of the candles opening 09:30 and 09:45 New York; skip if either
     candle is a gap.
 (3) Entry: first candle opening 10:00..14:45 New York whose close is above OR_H (long) / below OR_L (short).
 (4) Exit: first close beyond the opposite side (no re-entry), else after the 15:45 New York candle closes.
 (5) Weight +-0.5 per instrument, 0 otherwise.
New York time: UTC-4 from the second Sunday of March 02:00 local to the first Sunday of November 02:00 local,
UTC-5 otherwise (explicit US daylight-saving rule, no timezone library).
"""
import numpy as np

PARAMS = {"nr_n": 4}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")
CANDLES_PER_UTC_DAY = 96
MAX_GAP_CANDLES = 4
OR_FIRST_NY_MIN = 9 * 60 + 30        # 09:30 New York
OR_SECOND_NY_MIN = 9 * 60 + 45       # 09:45 New York
ENTRY_FIRST_NY_MIN = 10 * 60         # first entry candle opens 10:00
ENTRY_LAST_NY_MIN = 14 * 60 + 45     # last entry candle opens 14:45 (closes 15:00)
EXIT_NY_MIN = 15 * 60 + 45           # flat after the 15:45 candle closes (filled at the 16:00 open)
SESSION_END_OPEN_NY_MIN = 16 * 60    # the 16:00 New York candle
WEIGHT = 0.5
MIN_PER_DAY = 1440


# ----------------------------------------------------------------------------------------------- calendar
def days_from_civil(y, m, d):
    """Days since the Unix epoch of the proleptic Gregorian date (y, m, d); vectorized integers."""
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


def year_of_day(z):
    """Gregorian year of day number z (days since the Unix epoch)."""
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    return yoe + era * 400 + (m <= 2)


def weekday(day):
    """Monday = 0 ... Sunday = 6 (the Unix epoch day was a Thursday)."""
    return (np.asarray(day, dtype=np.int64) + 3) % 7


def ny_offset_minutes(utc_min):
    """New York offset from UTC in minutes (-240 in daylight-saving time, -300 otherwise).

    DST runs from 02:00 EST (07:00 UTC) on the second Sunday of March to 02:00 EDT (06:00 UTC) on the
    first Sunday of November.
    """
    utc_min = np.asarray(utc_min, dtype=np.int64)
    y = year_of_day(np.floor_divide(utc_min, MIN_PER_DAY))
    mar1 = days_from_civil(y, 3, 1)
    second_sun_mar = mar1 + (6 - weekday(mar1)) % 7 + 7
    nov1 = days_from_civil(y, 11, 1)
    first_sun_nov = nov1 + (6 - weekday(nov1)) % 7
    start = second_sun_mar * MIN_PER_DAY + 7 * 60
    end = first_sun_nov * MIN_PER_DAY + 6 * 60
    return np.where((utc_min >= start) & (utc_min < end), -240, -300)


def time_frame(times):
    """Per-row UTC day, New York day / minute-of-day, on a calendar-day axis starting at `base`."""
    utc_min = np.asarray(times).astype("datetime64[m]").astype(np.int64)
    utc_day = np.floor_divide(utc_min, MIN_PER_DAY)
    local = utc_min + ny_offset_minutes(utc_min)
    ny_day = np.floor_divide(local, MIN_PER_DAY)
    ny_min = local - ny_day * MIN_PER_DAY
    base = int(utc_day.min()) - 1          # New York date is the UTC date or the day before
    n_cal = int(utc_day.max()) - base + 1
    cal_days = base + np.arange(n_cal, dtype=np.int64)
    return {
        "utc_day": utc_day - base,
        "ny_day": ny_day - base,
        "ny_min": ny_min,
        "ny_weekday": weekday(ny_day) < 5,
        "cal_weekday": weekday(cal_days) < 5,
        "n_cal": n_cal,
    }


# ------------------------------------------------------------------------------------- narrow-range flag
def nr_eligibility(frame, h, l, real, nr_n):
    """(eligible, defined) per calendar day of the session (New York date).

    eligible[D]: D is a weekday and the previous weekday UTC bar is the narrowest (ln(H/L), ties count) of
    the nr_n most recent weekday UTC bars ending with it, all with at most MAX_GAP_CANDLES gap candles.
    defined[D]: the comparison could be made (enough complete bars).
    """
    n_cal = frame["n_cal"]
    ud = frame["utc_day"]
    real = np.asarray(real, dtype=bool)
    good = real & np.isfinite(h) & np.isfinite(l)
    hi = np.full(n_cal, -np.inf)
    lo = np.full(n_cal, np.inf)
    np.maximum.at(hi, ud[good], h[good])
    np.minimum.at(lo, ud[good], l[good])
    n_real = np.bincount(ud[real], minlength=n_cal)[:n_cal]
    # gap candles = 96 - real candles (missing rows count as gaps)
    ok = (n_real >= CANDLES_PER_UTC_DAY - MAX_GAP_CANDLES) & np.isfinite(hi) & np.isfinite(lo) & (lo > 0)
    rng = np.full(n_cal, np.nan)
    rng[ok] = np.log(hi[ok] / lo[ok])

    wk = frame["cal_weekday"]
    rw = rng[wk]                            # weekday bars only, in calendar order
    m = rw.size
    flag = np.zeros(m, dtype=bool)
    defined_w = np.zeros(m, dtype=bool)
    if m >= nr_n:
        win = np.stack([rw[j:m - nr_n + 1 + j] for j in range(nr_n)])   # column i = bars i..i+nr_n-1
        fin = np.all(np.isfinite(win), axis=0)
        w = np.where(fin, win, 0.0)
        flag[nr_n - 1:] = fin & (w[-1] <= w.min(axis=0))
        defined_w[nr_n - 1:] = fin
    ordinal = np.cumsum(wk) - 1             # weekday ordinal of each calendar day
    prev = ordinal - 1                      # previous weekday bar (Friday for a Monday)
    sel = wk & (prev >= 0)
    eligible = np.zeros(n_cal, dtype=bool)
    defined = np.zeros(n_cal, dtype=bool)
    eligible[sel] = flag[prev[sel]]
    defined[sel] = defined_w[prev[sel]]
    return eligible, defined


# --------------------------------------------------------------------------------------- opening range
def opening_range(frame, h, l, real):
    """OR_H, OR_L and validity per New York calendar day (both 09:30 and 09:45 candles real)."""
    n_cal = frame["n_cal"]
    nd = frame["ny_day"]
    nm = frame["ny_min"]
    is_or = frame["ny_weekday"] & ((nm == OR_FIRST_NY_MIN) | (nm == OR_SECOND_NY_MIN))
    good = is_or & np.asarray(real, dtype=bool) & np.isfinite(h) & np.isfinite(l)
    orh = np.full(n_cal, -np.inf)
    orl = np.full(n_cal, np.inf)
    np.maximum.at(orh, nd[good], h[good])
    np.minimum.at(orl, nd[good], l[good])
    valid = np.bincount(nd[good], minlength=n_cal)[:n_cal] == 2
    return orh, orl, valid


# ---------------------------------------------------------------------------------------------- trades
def session_trades(frame, c, orh, orl, sessions):
    """Trades of the breakout rule on the sessions flagged in `sessions` (per New York calendar day).

    Returns a list of (entry_row, exit_row, side, session_day): the position is held on rows
    entry_row .. exit_row - 1 (decided at the close of entry_row, filled at the next open) and is zero on
    exit_row (the stop candle, or the 15:45 New York candle for the time exit).
    """
    nm = frame["ny_min"]
    nd = frame["ny_day"]
    win = frame["ny_weekday"] & (nm >= ENTRY_FIRST_NY_MIN) & (nm < EXIT_NY_MIN)
    wrows = np.flatnonzero(win)
    wday = nd[wrows]
    out = []
    for s in np.flatnonzero(sessions):
        a = int(np.searchsorted(wday, s, side="left"))
        b = int(np.searchsorted(wday, s, side="right"))
        if b <= a:
            continue
        rows = wrows[a:b]
        cs = c[rows]
        entry_ok = nm[rows] <= ENTRY_LAST_NY_MIN
        up = (cs > orh[s]) & entry_ok
        dn = (cs < orl[s]) & entry_ok
        trig = up | dn
        if not trig.any():
            continue
        e = int(np.argmax(trig))
        side = 1 if up[e] else -1
        rest = cs[e + 1:]
        stop = (rest < orl[s]) if side > 0 else (rest > orh[s])
        if stop.any():
            x = int(rows[e + 1 + int(np.argmax(stop))])
        else:
            x = int(rows[-1]) + 1           # the 15:45 New York candle: flat at its close
        out.append((int(rows[e]), x, side, int(s)))
    return out


# --------------------------------------------------------------------------------------------- strategy
def target_positions(data, params):
    nr_n = int(params["nr_n"])
    if nr_n < 1:
        raise ValueError("nr_n must be >= 1")
    T = len(data.times)
    N = len(data.instruments)
    pos = np.zeros((T, N), dtype=float)
    cols = [data.col(name) for name in INSTRUMENTS]      # fail loudly if an instrument is missing
    if T == 0:
        return pos
    frame = time_frame(data.times)
    for j in cols:
        h = np.asarray(data.h[:, j], dtype=np.float64)
        l = np.asarray(data.l[:, j], dtype=np.float64)
        c = np.asarray(data.c[:, j], dtype=np.float64)
        real = np.asarray(data.real[:, j], dtype=bool)
        eligible, _ = nr_eligibility(frame, h, l, real, nr_n)
        orh, orl, or_valid = opening_range(frame, h, l, real)
        for entry, exit_row, side, _s in session_trades(frame, c, orh, orl, eligible & or_valid):
            pos[entry:exit_row, j] = side * WEIGHT
    if data.universe is not None:
        pos[~np.asarray(data.universe, dtype=bool)] = 0.0
    return pos
