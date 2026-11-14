"""H-0150 VWAP Wave setup 4: VWAP hold after a return into the UTC-session value area (BTC/ETH perps, 5m).

Per instrument, independently: session VWAP and +-1 sigma volume-weighted bands anchored at 00:00 UTC,
a re-entry into value, the first VWAP test after it (within 24 candles), a confirmation close beyond the
test candle within confirm_bars candles, entry at the next open with weight +-0.5, exit on a close at or
beyond the fixed stop or target, or at the session end (00:00 UTC open of the next session).
"""
import numpy as np

PARAMS = {"confirm_bars": 3, "target": "band", "tol_atr": 0.1}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")   # the card's universe
WEIGHT = 0.5                           # per instrument, gross <= 1
BAND_K = 1.0                           # value area = V +- 1.0 sigma
WARMUP_BARS = 12                       # no events in the first 12 candles of a session (00:00-00:55)
LATE_BARS = 12                         # no new signals in the last 12 candles (open 23:00-23:55)
CONTEXT_BARS = 24                      # re-entry r with b-24 <= r <= b-1
ATR_N = 14                             # Wilder ATR on 5m candles
STOP_ATR = 0.25                        # stop = extreme of b..s -/+ 0.25 ATR_s
MIN_PER_DAY = 1440

REASON_STOP, REASON_TARGET, REASON_SESSION, REASON_OPEN = "stop", "target", "session_end", "open"


def session_bands(minutes, h, l, c, qv, real, bar_min):
    """Session VWAP V, upper/lower bands U/L and a 'bands defined' mask for one instrument.

    minutes: int64 minutes since epoch of each candle's open (UTC). Weights w = qv / tp, tp = (h+l+c)/3,
    0 on gap rows. Cumulative sums restart at each UTC day. Bands are undefined in the warm-up.
    """
    T = len(c)
    tp = (h + l + c) / 3.0
    ok = real & np.isfinite(tp) & np.isfinite(qv) & (tp > 0) & (qv >= 0)
    tps = np.where(ok, tp, 1.0)
    w = np.where(ok, qv / tps, 0.0)
    wtp = np.where(ok, w * tps, 0.0)
    wtp2 = np.where(ok, wtp * tps, 0.0)
    day = minutes // MIN_PER_DAY
    idx = np.arange(T)
    new = np.ones(T, dtype=bool)
    new[1:] = day[1:] != day[:-1]
    start = np.maximum.accumulate(np.where(new, idx, 0))
    out = []
    for x in (w, wtp, wtp2):
        cs = np.cumsum(x)
        prev = np.where(start > 0, cs[np.maximum(start - 1, 0)], 0.0)
        out.append(cs - prev)
    sw, swtp, swtp2 = out
    pos = sw > 0
    swp = np.where(pos, sw, 1.0)
    V = np.where(pos, swtp / swp, np.nan)
    var = np.where(pos, swtp2 / swp - V * V, np.nan)
    sig = np.sqrt(np.maximum(var, 0.0))
    U = V + BAND_K * sig
    L = V - BAND_K * sig
    bar = (minutes % MIN_PER_DAY) // bar_min
    defined = pos & (bar >= WARMUP_BARS) & np.isfinite(V)
    V = np.where(defined, V, np.nan)
    U = np.where(defined, U, np.nan)
    L = np.where(defined, L, np.nan)
    return V, U, L, defined


def wilder_atr(h, l, c, real, n):
    """Wilder ATR through candle t (seed = mean of the first n true ranges, then atr += (tr - atr) / n).

    Gap rows (real = false) do not update the ATR; they carry the last value."""
    T = len(c)
    prev_c = np.full(T, np.nan)
    prev_c[1:] = c[:-1]
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    okl = (real & np.isfinite(tr)).tolist()
    trl = tr.tolist()
    out = [float("nan")] * T
    cnt = 0
    acc = 0.0
    val = float("nan")
    for t in range(T):
        if okl[t]:
            if cnt < n:
                acc += trl[t]
                cnt += 1
                if cnt == n:
                    val = acc / n
            else:
                val = val + (trl[t] - val) / n
        out[t] = val
    return np.array(out, dtype=float)


def find_trades(minutes, h, l, c, real, V, U, L, defined, atr, member, confirm_bars, target, tol_atr, bar_min):
    """All trades of one instrument. Returns a list of dicts with the signal row s (entry at the open of s+1),
    the exit decision row e (exit at the open of e+1; e = T if still open), side, test row b, stop, target,
    reason and ATR_s / c[s]."""
    T = len(c)
    K = int(confirm_bars)
    tol = float(tol_atr)
    use_band = _target_is_band(target)
    day = minutes // MIN_PER_DAY
    mod = minutes % MIN_PER_DAY
    bars_per_session = MIN_PER_DAY // bar_min
    bar = mod // bar_min
    late = bar >= bars_per_session - LATE_BARS
    last_bar = bar >= bars_per_session - 1
    idx = np.arange(T)

    # (2) re-entry into value at r: c[r-1] outside [L,U]_{r-1}, c[r] inside [L,U]_r, same session, both defined
    inside = defined & (c >= L) & (c <= U)
    outside = defined & ((c > U) | (c < L))
    prev_ok = np.zeros(T, dtype=bool)
    prev_ok[1:] = defined[:-1] & defined[1:] & (day[1:] == day[:-1])
    reentry = np.zeros(T, dtype=bool)
    reentry[1:] = prev_ok[1:] & outside[:-1] & inside[1:] & real[1:]

    last_r = np.maximum.accumulate(np.where(reentry, idx, -1))
    lr_prev = np.full(T, -1)
    lr_prev[1:] = last_r[:-1]                      # last re-entry at or before b-1
    ctx = (lr_prev >= 0) & (idx - lr_prev <= CONTEXT_BARS) & (day[np.maximum(lr_prev, 0)] == day)

    # (3) test candles
    cp = np.full(T, np.nan)
    Vp = np.full(T, np.nan)
    Up = np.full(T, np.nan)
    Lp = np.full(T, np.nan)
    cp[1:] = c[:-1]
    Vp[1:] = V[:-1]
    Up[1:] = U[:-1]
    Lp[1:] = L[:-1]
    base = real & defined & prev_ok & ctx & member
    long_b = base & (Vp < cp) & (cp <= Up) & (l <= V + tol * atr) & (V < c) & (c <= U)
    short_b = base & (Lp <= cp) & (cp < Vp) & (h >= V - tol * atr) & (L <= c) & (c < V)

    cl, hl, ll, Vl = c.tolist(), h.tolist(), l.tolist(), V.tolist()
    reall, defl, dayl = real.tolist(), defined.tolist(), day.tolist()

    # (4) entry signals
    cands = {}
    for side, bs in ((1, np.nonzero(long_b)[0]), (-1, np.nonzero(short_b)[0])):
        for b in bs.tolist():
            s = -1
            if K == 0:
                s = b
            else:
                lvl = hl[b] if side > 0 else ll[b]
                for k in range(b + 1, min(b + K, T - 1) + 1):
                    if dayl[k] != dayl[b] or not defl[k]:
                        break
                    if (side > 0 and cl[k] < Vl[k]) or (side < 0 and cl[k] > Vl[k]):
                        break                       # a close on the wrong side of V cancels the setup
                    if reall[k] and ((side > 0 and cl[k] > lvl) or (side < 0 and cl[k] < lvl)):
                        s = k
                        break
            if s < 0 or late[s] or not member[s] or not real[s] or not defined[s]:
                continue
            if not np.isfinite(atr[s]):
                continue
            if side > 0:
                stop = float(np.nanmin(l[b:s + 1])) - STOP_ATR * atr[s]
            else:
                stop = float(np.nanmax(h[b:s + 1])) + STOP_ATR * atr[s]
            R = abs(c[s] - stop)
            if use_band:
                tgt = U[s] if side > 0 else L[s]
                if (side > 0 and c[s] >= tgt) or (side < 0 and c[s] <= tgt):
                    continue                        # target already behind the entry
            else:
                tgt = c[s] + 2.0 * R if side > 0 else c[s] - 2.0 * R
            cands.setdefault(s, []).append((b, side, stop, float(tgt)))

    # (5) one position at a time; a new trade needs a test candle after the last exit candle
    trades = []
    last_exit = -1
    for s in sorted(cands):
        if s <= last_exit:
            continue
        elig = [x for x in cands[s] if x[0] > last_exit]
        if not elig:
            continue
        if len(set(x[1] for x in elig)) > 1:
            continue                                # long and short on the same candle: none
        b, side, stop, tgt = max(elig)              # same side, several tests: the most recent test candle
        seg = slice(s + 1, min(s + 1 + bars_per_session + 1, T))
        cs = c[seg]
        if side > 0:
            hit_stop = cs <= stop
            hit_tgt = cs >= tgt
        else:
            hit_stop = cs >= stop
            hit_tgt = cs <= tgt
        hit_end = last_bar[seg] | (day[seg] != day[s])
        anyhit = hit_stop | hit_tgt | hit_end
        if anyhit.any():
            j = int(np.argmax(anyhit))
            e = s + 1 + j
            reason = REASON_STOP if hit_stop[j] else (REASON_TARGET if hit_tgt[j] else REASON_SESSION)
        else:
            e = T
            reason = REASON_OPEN
        trades.append({"signal_row": s, "exit_row": e, "side": side, "test_row": b, "stop": stop,
                       "target": tgt, "reason": reason, "atr_over_price": float(atr[s] / c[s])})
        last_exit = e
    return trades


def _target_is_band(target):
    t = str(target)
    if t == "band":
        return True
    if t == "2R":
        return False
    raise ValueError("unknown target: " + t)


def _instrument_trades(data, params, name):
    j = data.col(name)
    minutes = data.times.astype(np.int64)
    bar_min = MIN_PER_DAY // int(data.bars_per_day)
    h = np.asarray(data.h[:, j], dtype=float)
    l = np.asarray(data.l[:, j], dtype=float)
    c = np.asarray(data.c[:, j], dtype=float)
    qv = np.asarray(data.qv[:, j], dtype=float)
    real = np.asarray(data.real[:, j], dtype=bool)
    if data.universe is None:
        member = np.ones(len(c), dtype=bool)
    else:
        member = np.asarray(data.universe[:, j], dtype=bool)
    V, U, L, defined = session_bands(minutes, h, l, c, qv, real, bar_min)
    atr = wilder_atr(h, l, c, real, ATR_N)
    return j, member, find_trades(minutes, h, l, c, real, V, U, L, defined, atr, member,
                                  params["confirm_bars"], params["target"], params["tol_atr"], bar_min)


def trade_log(data, params):
    """Descriptive log for the owner's report (side, signal/exit rows and times, reason, ATR_s/c[s]);
    returns come from the engine's trade list."""
    rows = []
    for name in INSTRUMENTS:
        _, _, trades = _instrument_trades(data, params, name)
        for tr in trades:
            d = dict(tr)
            d["instrument"] = name
            d["signal_time"] = data.times[tr["signal_row"]]
            d["exit_time"] = data.times[tr["exit_row"]] if tr["exit_row"] < len(data.times) else None
            rows.append(d)
    return rows


def target_positions(data, params):
    T = len(data.times)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)
    for name in INSTRUMENTS:
        j, member, trades = _instrument_trades(data, params, name)
        for tr in trades:
            w[tr["signal_row"]:tr["exit_row"], j] = WEIGHT * tr["side"]
        w[~member, j] = 0.0
    return w
