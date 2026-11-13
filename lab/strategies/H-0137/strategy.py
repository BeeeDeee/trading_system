"""H-0137: US-cash-open range breakout on BTC/ETH perps (15m), exit at the 16:00 New York close.

Clock: the intraday view's `times` are UTC candle OPEN times. New York local time is derived with the
US daylight-saving rule in force since 2007 (the tz database rule for America/New_York): EDT (UTC-4) from
the second Sunday of March 02:00 EST (07:00 UTC) to the first Sunday of November 02:00 EDT (06:00 UTC),
EST (UTC-5) otherwise. `zoneinfo` cannot be imported under the lab's static rules, so the rule is coded
in integer arithmetic (identical to tzdata for 2007 onward).
"""
import numpy as np

PARAMS = {"range_minutes": 30, "stop_opposite": True}

# Control switch (not a tunable parameter): "us_open" (09:30 New York) or "utc_midnight" (00:00 UTC).
ANCHOR = "us_open"

UNIVERSE = ("BTCUSDT", "ETHUSDT")
WEIGHT = 0.5
ANCHOR_LOCAL_MIN = {"us_open": 9 * 60 + 30, "utc_midnight": 0}
ENTRY_CUTOFF_MIN = 5 * 60 + 15     # signal candles must START before anchor + 5h15m (14:45 New York)
EXIT_MIN = 6 * 60 + 15             # decision after the candle starting at anchor + 6h15m (15:45 NY, ends 16:00)


def year_from_days(days):
    """Gregorian year of integer days since the Unix epoch (H. Hinnant's civil_from_days)."""
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    month = np.where(mp < 10, mp + 3, mp - 9)
    return y + (month <= 2)


def _march1_days(y):
    """Days since the Unix epoch of the first day of March of year y (days_from_civil, month 3, day 1)."""
    era = y // 400
    yoe = y - era * 400
    doe = yoe * 365 + yoe // 4 - yoe // 100
    return era * 146097 + doe - 719468


def _first_sunday_on_or_after(days):
    wd = (days + 3) % 7            # 0 = Monday (1970-01-01 was a Thursday)
    return days + (6 - wd) % 7


def ny_utc_offset_minutes(tmin):
    """UTC offset of America/New_York (minutes, -240 or -300) at UTC minutes-since-epoch `tmin`."""
    days = tmin // 1440
    y = year_from_days(days)
    mar1 = _march1_days(y)
    nov1 = mar1 + 245
    dst_start = (_first_sunday_on_or_after(mar1) + 7) * 1440 + 7 * 60
    dst_end = _first_sunday_on_or_after(nov1) * 1440 + 6 * 60
    in_dst = (tmin >= dst_start) & (tmin < dst_end)
    return np.where(in_dst, -240, -300)


def orb_positions(tmin, h, l, c, real, bar_min, range_minutes, stop_opposite, anchor):
    """Core rule on plain arrays. tmin (T,) int64 UTC candle-open minutes; h, l, c, real (T, K).

    Returns (T, K) float positions (+-WEIGHT inside a trade, 0 otherwise)."""
    if anchor not in ANCHOR_LOCAL_MIN:
        raise ValueError("unknown anchor: %r" % (anchor,))
    range_minutes = int(range_minutes)
    if range_minutes <= 0 or range_minutes % bar_min != 0:
        raise ValueError("range_minutes must be a positive multiple of the candle length")
    n_range = range_minutes // bar_min
    stop_opposite = bool(stop_opposite)

    tmin = np.asarray(tmin, dtype=np.int64)
    T = tmin.shape[0]
    K = h.shape[1]
    out = np.zeros((T, K), dtype=float)

    if anchor == "us_open":
        local = tmin + ny_utc_offset_minutes(tmin)
    else:
        local = tmin.copy()
    lday = local // 1440
    ltod = local - lday * 1440
    weekday = (lday + 3) % 7                         # 0 = Monday
    m = ltod - ANCHOR_LOCAL_MIN[anchor]              # minutes since the session anchor (candle start)
    in_sess = (weekday < 5) & (m >= 0) & (m < EXIT_MIN)
    rows = np.nonzero(in_sess)[0]
    if rows.size == 0:
        return out
    sess = lday[rows]
    # rows are in time order, so each session is a contiguous block of `rows`
    brk = np.nonzero(np.diff(sess) != 0)[0] + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [rows.size]))

    for a, b in zip(starts, ends):
        idx = rows[a:b]
        mm = m[idx]
        rng = mm < range_minutes
        if int(rng.sum()) != n_range:                # a range candle is missing from the grid
            continue
        ent_win = (mm >= range_minutes) & (mm < ENTRY_CUTOFF_MIN)
        r_idx = idx[rng]
        for j in range(K):
            if not np.all(real[r_idx, j]):           # a range candle is a gap row: skip the session
                continue
            H = np.max(h[r_idx, j])
            L = np.min(l[r_idx, j])
            cj = c[idx, j]
            rj = real[idx, j]
            brk_long = ent_win & rj & (cj > H)
            brk_short = ent_win & rj & (cj < L)
            hit = np.nonzero(brk_long | brk_short)[0]
            if hit.size == 0:
                continue
            e = hit[0]
            side = 1.0 if brk_long[e] else -1.0
            x = idx.size                             # default: hold through the 15:30 candle; 0 from 15:45 on
            if stop_opposite:
                opp = (cj < L) if side > 0 else (cj > H)
                opp = opp & rj
                opp[: e + 1] = False
                s_hit = np.nonzero(opp)[0]
                if s_hit.size:
                    x = s_hit[0]
            out[idx[e:x], j] = side * WEIGHT
    return out


def target_positions(data, params):
    anchor = params.get("anchor", ANCHOR)
    cols = [data.col(name) for name in UNIVERSE]     # fail loudly if an instrument is missing
    T = data.times.shape[0]
    N = len(data.instruments)
    bar_min = 1440 // int(data.bars_per_day)
    tmin = data.times.astype("datetime64[m]").astype(np.int64)
    h = np.asarray(data.h, dtype=float)[:, cols]
    l = np.asarray(data.l, dtype=float)[:, cols]
    c = np.asarray(data.c, dtype=float)[:, cols]
    real = np.asarray(data.real, dtype=bool)[:, cols]
    pos = orb_positions(tmin, h, l, c, real, bar_min, params["range_minutes"], params["stop_opposite"], anchor)
    w = np.zeros((T, N), dtype=float)
    w[:, cols] = pos
    if data.universe is not None:
        w = np.where(np.asarray(data.universe, dtype=bool), w, 0.0)
    return w
