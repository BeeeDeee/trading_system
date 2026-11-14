"""H-0149 VWAP Wave setup 3: return to value after a failed excursion outside the UTC-session VWAP bands.

BTCUSDT / ETHUSDT perps, 5m candles. Each instrument is handled independently. Row t of the output is the
target position decided after candle t closed (filled by the engine at the open of t+1).

The short setup is implemented as the exact mirror of the long setup by running the long logic on negated
prices (c' = -c, h' = -l, l' = -h, V' = -V, same sigma and ATR), so both sides share one code path.
"""
import numpy as np

PARAMS = {"entry_mode": "conservative", "min_excursion_sigma": 0.5, "target": "vwap"}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")
WEIGHT = 0.5                 # per instrument; two instruments -> gross <= 1
BAND_K = 1.0                 # U/L = V +/- 1.0 sigma
FIRST_VALID_CANDLE = 24      # bands valid from the 24th candle of the session (closes at 02:00 UTC on 5m)
MIN_RUN = 3                  # excursion: at least 3 consecutive closes outside the band
TOL = 0.25                   # pullback tolerance in sigma
STOP_ATR = 0.25              # stop buffer in ATR
ATR_N = 14                   # simple mean of the true range over 14 candles
TRIGGER_WINDOW = 6           # b < s <= b + 6
EXPIRY = 24                  # no signal by r + 24 -> setup expires
LAST_SIGNAL_CLOSE_MIN = 22 * 60   # no signal whose candle closes after 22:00 UTC
DAY_MIN = 1440


def calendar(times, bars_per_day):
    """Session (UTC day) id, band validity by time, 22:00 signal cut-off, last candle of session, row link."""
    minutes = np.asarray(times).astype("datetime64[m]").astype(np.int64)
    bar_min = DAY_MIN // int(bars_per_day)
    day = minutes // DAY_MIN
    mod = minutes - day * DAY_MIN
    pos_in_session = mod // bar_min                       # 0-based candle number inside the session
    valid_time = pos_in_session >= FIRST_VALID_CANDLE - 1
    signal_time_ok = mod + bar_min <= LAST_SIGNAL_CLOSE_MIN
    last = mod + bar_min >= DAY_MIN
    T = minutes.shape[0]
    link = np.zeros(T, dtype=bool)                         # row t directly follows row t-1 in the same session
    if T > 1:
        link[1:] = (day[1:] == day[:-1]) & ((minutes[1:] - minutes[:-1]) == bar_min)
    return day, valid_time, signal_time_ok, last, link


def session_bands(h, l, c, qv, real, day):
    """Session VWAP V_t and volume-weighted std sigma_t of the typical price, from the session's first candle
    through t. Weight w = qv / tp (base volume). Gap candles (not real) carry no weight."""
    T = c.shape[0]
    V = np.full(T, np.nan)
    S = np.full(T, np.nan)
    if T == 0:
        return V, S
    tp = (h + l + c) / 3.0
    with np.errstate(invalid="ignore", divide="ignore"):
        good = real & np.isfinite(tp) & np.isfinite(qv) & (qv > 0) & (tp > 0)
    tp_safe = np.where(good, tp, 1.0)
    w = np.where(good, qv / tp_safe, 0.0)
    brk = np.ones(T, dtype=bool)
    brk[1:] = day[1:] != day[:-1]
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], T)
    for a, b in zip(starts.tolist(), ends.tolist()):
        g = np.flatnonzero(good[a:b])
        if g.size == 0:
            continue
        anchor = tp[a + int(g[0])]                         # first priced candle of the session (point-in-time)
        d = np.where(good[a:b], tp[a:b] - anchor, 0.0)
        ws = np.cumsum(w[a:b])
        s1 = np.cumsum(w[a:b] * d)
        s2 = np.cumsum(w[a:b] * d * d)
        pos = ws > 0
        m1 = np.full(b - a, np.nan)
        m2 = np.full(b - a, np.nan)
        np.divide(s1, ws, out=m1, where=pos)
        np.divide(s2, ws, out=m2, where=pos)
        var = np.maximum(m2 - m1 * m1, 0.0)
        V[a:b] = anchor + m1
        S[a:b] = np.sqrt(var)
    return V, S


def atr_simple(h, l, c, n):
    """Simple mean of the true range over the last n candles (rolls across sessions); NaN until full."""
    T = c.shape[0]
    pc = np.full(T, np.nan)
    if T > 1:
        pc[1:] = c[:-1]
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    acc = np.zeros(T)
    for k in range(n):
        sh = np.full(T, np.nan)
        if k < T:
            sh[k:] = tr[:T - k]
        acc = acc + sh
    return acc / n


def long_signals(h, l, c, V, sig, atr, ok, link, day, signal_time_ok, x, aggressive, opposite):
    """Long setups of one instrument. Returns a list of (s, stop, target)."""
    T = c.shape[0]
    out = []
    if T < 2:
        return out
    L = V - BAND_K * sig
    U = V + BAND_K * sig
    with np.errstate(invalid="ignore"):
        below = ok & (c < L)
        ext = below & (l <= L - x * sig)
        at_or_above = ok & (c >= L)
    idx = np.arange(T)
    prev_below = np.zeros(T, dtype=bool)
    prev_below[1:] = below[:-1]
    start = below & ~(prev_below & link)
    last_start = np.maximum.accumulate(np.where(start, idx, -1))
    runlen = np.where(below, idx - last_start + 1, 0)
    cumext = np.cumsum(ext.astype(np.int64))
    ls = np.maximum(last_start, 0)
    ext_in_run = below & ((cumext - cumext[ls] + ext[ls].astype(np.int64)) > 0)
    ret = np.zeros(T, dtype=bool)
    ret[1:] = at_or_above[1:] & link[1:] & (runlen[:-1] >= MIN_RUN) & ext_in_run[:-1]

    for r in np.flatnonzero(ret).tolist():
        if aggressive:
            s = r
            lo = l[r]
        else:
            s = -1
            b = -1
            acc = 1                                        # closes >= L since the return, including r
            for j in range(r + 1, min(r + EXPIRY, T - 1) + 1):
                if day[j] != day[r]:
                    break                                  # session end cancels the armed setup
                if not ok[j]:
                    continue                               # gap candle: neither acceptance nor cancel
                if c[j] < L[j]:
                    break                                  # close back below L cancels the setup
                if b >= 0 and j <= b + TRIGGER_WINDOW and c[j] > h[b]:
                    s = j
                    break
                if acc >= 2 and l[j] <= L[j] + TOL * sig[j]:
                    b = j                                  # newest pullback candle
                acc += 1
            if s < 0:
                continue
            lo = np.min(l[b:s + 1])
        if not signal_time_ok[s]:
            continue
        stop = lo - STOP_ATR * atr[s]
        tgt = U[s] if opposite else V[s]
        if not (np.isfinite(stop) and np.isfinite(tgt)):
            continue
        if c[s] >= tgt or c[s] <= stop:
            continue                                       # already at/through target or stop: no trade
        out.append((s, float(stop), float(tgt)))
    return out


def book_positions(h, l, c, V, sig, atr, ok, real, link, day, signal_time_ok, last, x, aggressive, opposite,
                   bars_per_day):
    """Position path (+/-WEIGHT or 0) of one instrument from precomputed bands."""
    T = c.shape[0]
    pos = np.zeros(T)
    if T == 0:
        return pos
    sides = (
        (1.0, h, l, c, V),                    # long, original prices
        (-1.0, -l, -h, -c, -V),               # short = long logic on negated prices
    )
    cands = []
    for rank, (sign, hh, ll, cc, vv) in enumerate(sides):
        for s, stop, tgt in long_signals(hh, ll, cc, vv, sig, atr, ok, link, day, signal_time_ok, x,
                                         aggressive, opposite):
            cands.append((s, rank, sign, stop, tgt))
    cands.sort(key=lambda z: (z[0], z[1]))
    busy = -1                                  # exit decision row of the open trade
    span = int(bars_per_day) + 1
    for s, rank, sign, stop, tgt in cands:
        if s <= busy:
            continue                           # position open: new setups are ignored
        hh, ll = sides[rank][1], sides[rank][2]
        end = min(T, s + 1 + span)
        seg = slice(s + 1, end)
        with np.errstate(invalid="ignore"):
            hit = (real[seg] & ((ll[seg] <= stop) | (hh[seg] >= tgt))) | last[seg] | (day[seg] != day[s])
        nz = np.flatnonzero(hit)
        k = s + 1 + int(nz[0]) if nz.size else end
        pos[s:k] = sign * WEIGHT
        busy = k
    return pos


def instrument_positions(h, l, c, qv, real, cal, x, aggressive, opposite, bars_per_day):
    day, valid_time, signal_time_ok, last, link = cal
    V, sig = session_bands(h, l, c, qv, real, day)
    atr = atr_simple(h, l, c, ATR_N)
    ok = (real & valid_time & np.isfinite(V) & np.isfinite(sig)
          & np.isfinite(c) & np.isfinite(h) & np.isfinite(l))
    return book_positions(h, l, c, V, sig, atr, ok, real, link, day, signal_time_ok, last, x, aggressive,
                          opposite, bars_per_day)


def target_positions(data, params):
    mode = str(params["entry_mode"])
    if mode not in ("conservative", "aggressive"):
        raise ValueError("unknown entry_mode")
    tmode = str(params["target"])
    if tmode not in ("vwap", "opposite_band"):
        raise ValueError("unknown target")
    x = float(params["min_excursion_sigma"])
    aggressive = mode == "aggressive"
    opposite = tmode == "opposite_band"

    T = len(data.times)
    N = len(data.instruments)
    out = np.zeros((T, N))
    cal = calendar(data.times, data.bars_per_day)
    for name in INSTRUMENTS:
        j = data.col(name)
        h = np.asarray(data.h[:, j], dtype=np.float64)
        l = np.asarray(data.l[:, j], dtype=np.float64)
        c = np.asarray(data.c[:, j], dtype=np.float64)
        qv = np.asarray(data.qv[:, j], dtype=np.float64)
        real = np.asarray(data.real[:, j], dtype=bool) & np.isfinite(c)
        out[:, j] = instrument_positions(h, l, c, qv, real, cal, x, aggressive, opposite, data.bars_per_day)
    if data.universe is not None:
        out[~np.asarray(data.universe, dtype=bool)] = 0.0
    return out
