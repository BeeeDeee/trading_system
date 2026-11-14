"""H-0148 VWAP Wave setup 2: fade a rejected test of the UTC-session VWAP value-area edge (BTC/ETH perps, 5m).

Decision after the close of each 5m candle (row t), fill at the open of row t + 1 (engine).
All state is built from rows <= t only: session-cumulative VWAP bands, an ATR over the last 14 real
candles, test / confirmation logic looking only backwards from the signal candle, and an event loop
that walks forward in time.
"""
import numpy as np

PARAMS = {"confirm_candles": 3, "target_frac": 0.5}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")

BARS_PER_DAY = 288        # 5m candles per UTC session (card: index 0 = 00:00, 287 = 23:55)
BAND_MULT = 1.0           # fixed constant of the card (not a parameter)
TOL_ATR = 0.25            # test tolerance and stop buffer, in ATR
ATR_N = 14                # ATR = mean true range of the last 14 real candles
CONTEXT_N = 6             # closes b-6 .. b-1 inside their own bands
FIRST_TEST_IDX = 36       # first three hours only build the value area
LAST_SIGNAL_IDX = 264     # no test candle / signal candle after 22:00
LAST_IDX = 287            # last candle of the session: flat at 00:00 UTC
MIN_MOVE = 0.0025         # cost filter: move to target at least 0.25 %
WEIGHT = 0.5              # position size per instrument


# ----------------------------------------------------------------------------------------------- clock
def session_clock(times, bars_per_day):
    """Session id (UTC day number) and candle index inside the session for each row."""
    tm = np.asarray(times).astype(np.int64)            # minutes since epoch (datetime64[m])
    sess = tm // 1440
    idx = ((tm - sess * 1440) * bars_per_day) // 1440
    return sess, idx


def _session_bounds(sess):
    T = sess.shape[0]
    if T == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    new = np.ones(T, bool)
    new[1:] = sess[1:] != sess[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], T)
    return starts, ends


# ----------------------------------------------------------------------------------------------- bands
def session_bands(h, l, c, qv, real, sess):
    """Developing session VWAP and volume-weighted std of the typical price, candles 0..j of the session.

    Weight = qv on real candles with qv > 0, else 0. Returns V, sigma (NaN until the session has weight).
    Sums are taken relative to the first weighted typical price of the session (numerical precision only;
    the result is mathematically identical and uses no later row).
    """
    T = c.shape[0]
    tp = (h + l + c) / 3.0
    w = np.where(real & np.isfinite(qv) & (qv > 0) & np.isfinite(tp), qv, 0.0)
    V = np.full(T, np.nan)
    sig = np.full(T, np.nan)
    starts, ends = _session_bounds(sess)
    for a, b in zip(starts, ends):
        ww = w[a:b]
        pos = ww > 0
        if not pos.any():
            continue
        ref = tp[a + int(np.argmax(pos))]
        dd = np.where(pos, tp[a:b] - ref, 0.0)
        s0 = np.cumsum(ww)
        s1 = np.cumsum(ww * dd)
        s2 = np.cumsum(ww * dd * dd)
        ok = s0 > 0
        den = np.where(ok, s0, 1.0)
        m1 = s1 / den
        var = np.maximum(0.0, s2 / den - m1 * m1)
        V[a:b] = np.where(ok, ref + m1, np.nan)
        sig[a:b] = np.where(ok, np.sqrt(var), np.nan)
    return V, sig


def atr_real(h, l, c, real, n):
    """Mean true range max(h, c_prev) - min(l, c_prev) over the last n real candles up to row t.

    c_prev is the previous row's close (gap rows repeat the previous close). May span sessions.
    """
    T = c.shape[0]
    out = np.full(T, np.nan)
    if T == 0:
        return out
    cp = np.full(T, np.nan)
    cp[1:] = c[:-1]
    tr = np.maximum(h, cp) - np.minimum(l, cp)          # NaN if any input is NaN
    ri = np.flatnonzero(real)
    if ri.size < n:
        return out
    x = tr[ri]
    bad = ~np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(bad, 0.0, x))))
    cb = np.concatenate(([0], np.cumsum(bad)))
    m = (cs[n:] - cs[:-n]) / n
    m[(cb[n:] - cb[:-n]) > 0] = np.nan
    atr_r = np.full(ri.size, np.nan)
    atr_r[n - 1:] = m
    last = np.searchsorted(ri, np.arange(T), side="right") - 1
    out = np.where(last >= 0, atr_r[np.maximum(last, 0)], np.nan)
    return out


# ----------------------------------------------------------------------------------------------- setups
def _signals(side, test, h, l, c, L, U, atr, real, sess, idx, member, n_conf, frac):
    """Signals of one side. Returns arrays (s, b, target, stop) of setups whose signal passed the cost filter."""
    T = c.shape[0]
    bs = np.flatnonzero(test)
    if bs.size == 0:
        e = np.zeros(0, np.int64)
        f = np.zeros(0)
        return e, e, f, f
    if n_conf == 0:
        sig = bs.copy()
        ext_run = (l[bs] if side > 0 else h[bs]).copy()
    else:
        ext = h[bs] if side > 0 else l[bs]                   # the test candle's extreme to close beyond
        ext_run = (l[bs] if side > 0 else h[bs]).copy()      # running min low (long) / max high (short) b..s
        sig = np.full(bs.size, -1, np.int64)
        alive = np.ones(bs.size, bool)
        for m in range(1, n_conf + 1):
            r = bs + m
            act = alive & (sig < 0)
            inb = r < T
            alive &= inb | ~act                              # window runs past the data: nothing yet
            act &= inb
            rr = np.minimum(r, T - 1)
            if side > 0:
                ext_run = np.where(act, np.fmin(ext_run, l[rr]), ext_run)
                cancel = c[rr] < L[rr]
                trig = c[rr] > ext
            else:
                ext_run = np.where(act, np.fmax(ext_run, h[rr]), ext_run)
                cancel = c[rr] > U[rr]
                trig = c[rr] < ext
            cancel = cancel | (sess[rr] != sess[bs])         # window crosses the session end
            alive &= ~(act & cancel)
            hit = act & ~cancel & trig & real[rr]            # a gap candle is never the signal
            good = hit & (idx[rr] <= LAST_SIGNAL_IDX)
            sig[good] = rr[good]
            alive &= ~(hit & ~good)                          # first close beyond came too late
    keep = sig >= 0
    s = sig[keep]
    b = bs[keep]
    ext_run = ext_run[keep]
    keep2 = real[s] & member[s] & (idx[s] <= LAST_SIGNAL_IDX) & np.isfinite(atr[s])
    s, b, ext_run = s[keep2], b[keep2], ext_run[keep2]
    width = U[s] - L[s]
    if side > 0:
        target = L[s] + frac * width
        stop = ext_run - TOL_ATR * atr[s]
        move = (target - c[s]) / c[s]
    else:
        target = U[s] - frac * width
        stop = ext_run + TOL_ATR * atr[s]
        move = (c[s] - target) / c[s]
    ok = np.isfinite(target) & np.isfinite(stop) & (move >= MIN_MOVE)
    return s[ok], b[ok], target[ok], stop[ok]


def book(events, h, l, sess, idx, member, T):
    """Walk the events forward in time. events: list of (s, b, side, target, stop).

    One position at a time; at most one long and one short entry per session; signals while a position is held
    (including on its exit candle) are ignored; if a long and a short signal fire on the same candle while flat,
    both are skipped (ambiguous). Exit at the first close k >= s+1 with stop or target touched, the last candle
    of the session, a new session, or the instrument leaving the universe.
    """
    pos = np.zeros(T)
    if not events:
        return pos
    events = sorted(events, key=lambda e: (e[0], e[2], e[1]))
    busy_until = -1
    used = set()
    i = 0
    ne = len(events)
    while i < ne:
        s = events[i][0]
        j = i
        while j < ne and events[j][0] == s:
            j += 1
        group = events[i:j]
        i = j
        if s <= busy_until:
            continue
        cands = {}
        for ev in group:                                     # earliest test candle per side
            side = ev[2]
            if (sess[s], side) in used:
                continue
            if side not in cands or ev[1] < cands[side][1]:
                cands[side] = ev
        if len(cands) != 1:
            continue
        _, _, side, target, stop = list(cands.values())[0]
        k0 = s + 1
        lim = min(T, s + 1 + BARS_PER_DAY)
        hh = h[k0:lim]
        ll = l[k0:lim]
        if side > 0:
            hit = (ll <= stop) | (hh >= target)
        else:
            hit = (hh >= stop) | (ll <= target)
        end = (idx[k0:lim] >= LAST_IDX) | (sess[k0:lim] != sess[s]) | ~member[k0:lim]
        ex = np.flatnonzero(hit | end)
        k = k0 + int(ex[0]) if ex.size else T
        pos[s:k] = side * WEIGHT
        busy_until = k
        used.add((sess[s], side))
    return pos


def positions_one(h, l, c, qv, real, member, sess, idx, n_conf, frac):
    T = c.shape[0]
    V, sg = session_bands(h, l, c, qv, real, sess)
    L = V - BAND_MULT * sg
    U = V + BAND_MULT * sg
    atr = atr_real(h, l, c, real, ATR_N)

    inside = real & (L <= c) & (c <= U)
    ctx = np.zeros(T, bool)
    if T > CONTEXT_N:
        ci = np.concatenate(([0], np.cumsum(inside)))
        cnt = ci[CONTEXT_N:T] - ci[0:T - CONTEXT_N]
        ctx[CONTEXT_N:] = (cnt == CONTEXT_N) & (sess[:-CONTEXT_N] == sess[CONTEXT_N:])
    base = (ctx & real & member & (idx >= FIRST_TEST_IDX) & (idx <= LAST_SIGNAL_IDX)
            & np.isfinite(atr) & np.isfinite(L))
    long_test = base & (l <= L + TOL_ATR * atr) & (c >= L)
    short_test = base & (h >= U - TOL_ATR * atr) & (c <= U)

    events = []
    for side, test in ((1, long_test), (-1, short_test)):
        s, b, tg, st = _signals(side, test, h, l, c, L, U, atr, real, sess, idx, member, n_conf, frac)
        for q in range(s.size):
            events.append((int(s[q]), int(b[q]), side, float(tg[q]), float(st[q])))
    return book(events, h, l, sess, idx, member, T)


def target_positions(data, params):
    n_conf = int(params["confirm_candles"])
    frac = float(params["target_frac"])
    if int(data.bars_per_day) != BARS_PER_DAY:
        raise ValueError("H-0148 is specified on 5m candles (288 per UTC session)")
    T = len(data.times)
    N = len(data.instruments)
    out = np.zeros((T, N))
    sess, idx = session_clock(data.times, BARS_PER_DAY)
    for name in INSTRUMENTS:
        j = data.col(name)
        h = np.asarray(data.h[:, j], np.float64)
        l = np.asarray(data.l[:, j], np.float64)
        c = np.asarray(data.c[:, j], np.float64)
        qv = np.asarray(data.qv[:, j], np.float64)
        real = np.asarray(data.real[:, j], bool)
        if data.universe is None:
            member = np.ones(T, bool)
        else:
            member = np.asarray(data.universe[:, j], bool)
        out[:, j] = positions_one(h, l, c, qv, real, member, sess, idx, n_conf, frac)
    return out
