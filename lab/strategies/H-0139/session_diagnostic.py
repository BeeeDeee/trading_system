"""H-0139 descriptive diagnostic (card signal.description, K-0186): not a gate, not used by the strategy.

session_report(data, params) -> dict, per instrument and pooled, over all New York weekday sessions in `data`
(pass the dev period) with a valid opening range (both 09:30 and 09:45 NY candles real):
  - move_*: mean |ln(open of the 16:00 NY candle / open of the 09:30 NY candle)| and session count, on
    eligible (post-NR-n) sessions and on all other sessions;
  - gross_*_{long,short}: mean gross log return per trade, side * ln(exit fill open / entry fill open), of the
    same entry/exit rule run on eligible and on all other sessions (entry fill = open after the entry candle,
    exit fill = open after the stop candle or the 16:00 NY open), with trade counts.
'other' follows the card literally (every non-eligible session, including those whose flag could not be
computed); n_other_flag_undefined says how many of those there were.
The helper functions are copies of strategy/strategy.py (a separate file, as the diagnostic contract asks).
"""
import numpy as np

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")
CANDLES_PER_UTC_DAY = 96
MAX_GAP_CANDLES = 4
OR_FIRST_NY_MIN = 9 * 60 + 30
OR_SECOND_NY_MIN = 9 * 60 + 45
ENTRY_FIRST_NY_MIN = 10 * 60
ENTRY_LAST_NY_MIN = 14 * 60 + 45
EXIT_NY_MIN = 15 * 60 + 45
SESSION_END_OPEN_NY_MIN = 16 * 60
MIN_PER_DAY = 1440


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


def year_of_day(z):
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    return yoe + era * 400 + (m <= 2)


def weekday(day):
    return (np.asarray(day, dtype=np.int64) + 3) % 7


def ny_offset_minutes(utc_min):
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
    utc_min = np.asarray(times).astype("datetime64[m]").astype(np.int64)
    utc_day = np.floor_divide(utc_min, MIN_PER_DAY)
    local = utc_min + ny_offset_minutes(utc_min)
    ny_day = np.floor_divide(local, MIN_PER_DAY)
    ny_min = local - ny_day * MIN_PER_DAY
    base = int(utc_day.min()) - 1
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


def nr_eligibility(frame, h, l, real, nr_n):
    n_cal = frame["n_cal"]
    ud = frame["utc_day"]
    real = np.asarray(real, dtype=bool)
    good = real & np.isfinite(h) & np.isfinite(l)
    hi = np.full(n_cal, -np.inf)
    lo = np.full(n_cal, np.inf)
    np.maximum.at(hi, ud[good], h[good])
    np.minimum.at(lo, ud[good], l[good])
    n_real = np.bincount(ud[real], minlength=n_cal)[:n_cal]
    ok = (n_real >= CANDLES_PER_UTC_DAY - MAX_GAP_CANDLES) & np.isfinite(hi) & np.isfinite(lo) & (lo > 0)
    rng = np.full(n_cal, np.nan)
    rng[ok] = np.log(hi[ok] / lo[ok])
    wk = frame["cal_weekday"]
    rw = rng[wk]
    m = rw.size
    flag = np.zeros(m, dtype=bool)
    defined_w = np.zeros(m, dtype=bool)
    if m >= nr_n:
        win = np.stack([rw[j:m - nr_n + 1 + j] for j in range(nr_n)])
        fin = np.all(np.isfinite(win), axis=0)
        w = np.where(fin, win, 0.0)
        flag[nr_n - 1:] = fin & (w[-1] <= w.min(axis=0))
        defined_w[nr_n - 1:] = fin
    ordinal = np.cumsum(wk) - 1
    prev = ordinal - 1
    sel = wk & (prev >= 0)
    eligible = np.zeros(n_cal, dtype=bool)
    defined = np.zeros(n_cal, dtype=bool)
    eligible[sel] = flag[prev[sel]]
    defined[sel] = defined_w[prev[sel]]
    return eligible, defined


def opening_range(frame, h, l, real):
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


def session_trades(frame, c, orh, orl, sessions):
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
            x = int(rows[-1]) + 1
        out.append((int(rows[e]), x, side, int(s)))
    return out


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def _summary(acc):
    out = {}
    for group in ("eligible", "other"):
        out["move_" + group + "_mean"] = _mean(acc["move_" + group])
        out["move_" + group + "_n"] = len(acc["move_" + group])
        for side in ("long", "short"):
            key = "gross_" + group + "_" + side
            out[key + "_mean"] = _mean(acc[key])
            out[key + "_n"] = len(acc[key])
        both = acc["gross_" + group + "_long"] + acc["gross_" + group + "_short"]
        out["gross_" + group + "_all_mean"] = _mean(both)
        out["gross_" + group + "_all_n"] = len(both)
    out["n_other_flag_undefined"] = acc["n_other_flag_undefined"]
    return out


def _empty_acc():
    acc = {"n_other_flag_undefined": 0}
    for group in ("eligible", "other"):
        acc["move_" + group] = []
        for side in ("long", "short"):
            acc["gross_" + group + "_" + side] = []
    return acc


def session_report(data, params):
    nr_n = int(params["nr_n"])
    T = len(data.times)
    frame = time_frame(data.times)
    nd = frame["ny_day"]
    nm = frame["ny_min"]
    wkd = frame["ny_weekday"]
    n_cal = frame["n_cal"]
    pooled = _empty_acc()
    report = {}
    for name in INSTRUMENTS:
        j = data.col(name)
        o = np.asarray(data.o[:, j], dtype=np.float64)
        h = np.asarray(data.h[:, j], dtype=np.float64)
        l = np.asarray(data.l[:, j], dtype=np.float64)
        c = np.asarray(data.c[:, j], dtype=np.float64)
        real = np.asarray(data.real[:, j], dtype=bool)
        eligible, defined = nr_eligibility(frame, h, l, real, nr_n)
        orh, orl, or_valid = opening_range(frame, h, l, real)
        acc = _empty_acc()

        open_start = np.full(n_cal, np.nan)
        open_end = np.full(n_cal, np.nan)
        r0 = np.flatnonzero(wkd & (nm == OR_FIRST_NY_MIN))
        r1 = np.flatnonzero(wkd & (nm == SESSION_END_OPEN_NY_MIN))
        open_start[nd[r0]] = o[r0]
        open_end[nd[r1]] = o[r1]
        ok = or_valid & np.isfinite(open_start) & np.isfinite(open_end) & (open_start > 0) & (open_end > 0)
        for s in np.flatnonzero(ok):
            group = "eligible" if eligible[s] else "other"
            acc["move_" + group].append(abs(float(np.log(open_end[s] / open_start[s]))))
        acc["n_other_flag_undefined"] = int(np.sum(or_valid & ~eligible & ~defined))

        for entry, exit_row, side, s in session_trades(frame, c, orh, orl, or_valid):
            fe, fx = entry + 1, exit_row + 1
            if fx >= T or not (np.isfinite(o[fe]) and np.isfinite(o[fx]) and o[fe] > 0 and o[fx] > 0):
                continue
            group = "eligible" if eligible[s] else "other"
            key = "gross_" + group + ("_long" if side > 0 else "_short")
            acc[key].append(side * float(np.log(o[fx] / o[fe])))

        report[name] = _summary(acc)
        for key, value in acc.items():
            pooled[key] = pooled[key] + value
    report["pooled"] = _summary(pooled)
    report["nr_n"] = nr_n
    return report
