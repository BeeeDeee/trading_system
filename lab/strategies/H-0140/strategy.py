"""H-0140: US-open range breakout on BTC/ETH perps aligned with the pre-session move (15m).

Sessions are Monday-Friday New York dates. New York local time is converted to UTC with the
US daylight-saving rule (in force since 2007): EDT (UTC-4) from the second Sunday of March to
the first Sunday of November, EST (UTC-5) otherwise. The rule is computed from integer day
numbers (no date literals); at the session hours used here (09:30-16:00 ET) the 02:00 local
switch time never matters, so the offset is a function of the New York date alone.

Per instrument and session:
  r_pre  = ln(open of the 09:30 ET candle / open of the 00:00 UTC candle of the same UTC date)
  s      = sample std of r_pre over the previous 20 weekday sessions (>= 15 valid), z = r_pre / s
  OR     = max(high), min(low) of the 09:30 and 09:45 ET candles
  scan   = candles opening 10:00 .. 15:30 ET (closing 10:15 .. 15:45 ET); the first real candle
           closing above OR_high (d = +1) or below OR_low (d = -1) is the only breakout
  filter = side_mode 'with': d*z > k; 'against': d*z < -k; 'all': no filter (z must still exist)
  weight = d * 0.5 from the decision after the breakout candle up to (excluding) the decision
           after the candle that closes at 16:00 ET (opens 15:45 ET), where it is set to 0.
"""
import numpy as np

PARAMS = {"k": 0.25}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")
SIDE_MODES = ("with", "against", "all")

MIN_PER_DAY = 1440
OPEN_0930 = 570          # 09:30 local, minutes after local midnight
BAR = 15                 # candle length in minutes
SCAN_START = 30          # first scanned candle opens at 10:00 ET (30 min after 09:30)
N_SCAN = 23              # candles opening 10:00 .. 15:30 ET (closing 10:15 .. 15:45 ET)
EXIT_OFFSET = 375        # candle opening 15:45 ET (closing 16:00 ET) carries the exit decision
LOOKBACK = 20            # previous weekday sessions for the scale of r_pre
MIN_VALID = 15           # minimum valid r_pre values in the lookback
WEIGHT = 0.5             # per-instrument weight


def _first_sunday_on_or_after(day):
    # day numbers since 1970-01-01 (a Thursday); weekday Monday=0 .. Sunday=6 is (day + 3) % 7
    return day + (6 - (day + 3) % 7) % 7


def ny_utc_offset_minutes(day):
    """Minutes to add to New York local time to get UTC, for New York date(s) `day` (days since epoch)."""
    day = np.asarray(day, dtype=np.int64)
    years = day.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)   # years since 1970
    mar1 = (years * 12 + 2).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    nov1 = (years * 12 + 10).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    dst_start = _first_sunday_on_or_after(mar1) + 7      # second Sunday of March
    dst_end = _first_sunday_on_or_after(nov1)            # first Sunday of November
    dst = (day >= dst_start) & (day < dst_end)
    return np.where(dst, 240, 300).astype(np.int64)


def _lookup(tm, target):
    """Row index of the candle opening exactly at UTC minute `target` and whether it exists."""
    T = tm.shape[0]
    idx = np.searchsorted(tm, target)
    idxc = np.minimum(idx, T - 1)
    ok = (idx < T) & (tm[idxc] == target)
    return idxc, ok


def _trailing_scale(r_pre):
    """Sample std of the previous LOOKBACK session values (today excluded), NaN if < MIN_VALID valid."""
    S = r_pre.shape[0]
    lag = np.arange(S)[:, None] - np.arange(1, LOOKBACK + 1)[None, :]
    vals = np.where(lag >= 0, r_pre[np.clip(lag, 0, None)], np.nan)
    fin = np.isfinite(vals)
    cnt = fin.sum(axis=1)
    mean = np.where(fin, vals, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    dev = np.where(fin, vals - mean[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(cnt - 1, 1)
    s = np.sqrt(var)
    return np.where((cnt >= MIN_VALID) & (s > 0), s, np.nan)


def target_positions(data, params, side_mode="with"):
    if side_mode not in SIDE_MODES:
        raise ValueError("side_mode must be one of " + ", ".join(SIDE_MODES))
    k = float(params["k"])

    tm = np.asarray(data.times).astype("datetime64[m]").astype(np.int64)   # UTC open minute
    T = tm.shape[0]
    N = len(data.instruments)
    out = np.zeros((T, N), dtype=np.float64)
    if T == 0:
        return out

    # sessions: every Monday-Friday New York date covered by the data
    days = np.arange(tm[0] // MIN_PER_DAY, tm[-1] // MIN_PER_DAY + 1, dtype=np.int64)
    days = days[(days + 3) % 7 < 5]
    S = days.shape[0]
    if S == 0:
        return out
    off = ny_utc_offset_minutes(days)
    t930 = days * MIN_PER_DAY + OPEN_0930 + off          # UTC open of the 09:30 ET candle
    t0 = days * MIN_PER_DAY                              # 00:00 UTC of the same UTC date

    r0, ok0 = _lookup(tm, t0)
    r930, ok930 = _lookup(tm, t930)
    r945, ok945 = _lookup(tm, t930 + BAR)
    scan_t = t930[:, None] + SCAN_START + BAR * np.arange(N_SCAN)[None, :]
    rs, oks = _lookup(tm, scan_t)                        # (S, N_SCAN)
    end_row = np.searchsorted(tm, t930 + EXIT_OFFSET)    # first row at/after the 15:45 ET candle
    ar = np.arange(S)

    O = np.asarray(data.o)
    H = np.asarray(data.h)
    L = np.asarray(data.l)
    C = np.asarray(data.c)
    REAL = np.asarray(data.real)

    for name in INSTRUMENTS:
        j = data.col(name)
        o = O[:, j].astype(np.float64)
        h = H[:, j].astype(np.float64)
        lo = L[:, j].astype(np.float64)
        c = C[:, j].astype(np.float64)
        real = REAL[:, j].astype(bool)

        # (2) pre-session move: needs real 00:00 UTC and 09:30 ET candles
        o0 = o[r0]
        o930 = o[r930]
        ok_pre = (ok0 & ok930 & real[r0] & real[r930] & np.isfinite(o0) & np.isfinite(o930)
                  & (o0 > 0) & (o930 > 0))
        r_pre = np.full(S, np.nan)
        r_pre[ok_pre] = np.log(o930[ok_pre] / o0[ok_pre])
        s = _trailing_scale(r_pre)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = r_pre / s

        # (1) opening range: both candles must be real
        orh = np.maximum(h[r930], h[r945])
        orl = np.minimum(lo[r930], lo[r945])
        ok_or = (ok930 & ok945 & real[r930] & real[r945] & np.isfinite(orh) & np.isfinite(orl))

        # (3) first breakout among real scan candles, in time order
        cs = np.where(oks, c[rs], np.nan)
        rl = oks & real[rs]
        with np.errstate(invalid="ignore"):
            above = rl & (cs > orh[:, None])
            below = rl & (cs < orl[:, None])
        brk = above | below
        has = brk.any(axis=1)
        first = np.argmax(brk, axis=1)
        d = np.where(above[ar, first], 1.0, -1.0)
        brow = rs[ar, first]

        # (4) alignment filter (same code path for the three modes)
        dz = d * z
        valid_z = np.isfinite(dz)
        with np.errstate(invalid="ignore"):
            if side_mode == "with":
                align = valid_z & (dz > k)
            elif side_mode == "against":
                align = valid_z & (dz < -k)
            else:
                align = valid_z
        trade = has & ok_pre & ok_or & align

        # (5)/(6) hold d*0.5 from the breakout decision row until the exit decision row (weight 0)
        delta = np.zeros(T + 1, dtype=np.float64)
        st = brow[trade]
        en = end_row[trade]
        w = WEIGHT * d[trade]
        np.add.at(delta, st, w)
        np.add.at(delta, en, -w)
        out[:, j] = np.cumsum(delta)[:T]

    out = np.round(out, 12)
    uni = data.universe
    if uni is not None:
        out = np.where(np.asarray(uni, dtype=bool), out, 0.0)
    return out
