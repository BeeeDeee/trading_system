"""H-0147 VWAP Wave setup 1: acceptance outside the UTC-session VWAP band, buy (sell) the first held pullback
to the old band edge, hold to session end (or a 2R target in the exit_mode grid neighbor).

BTCUSDT and ETHUSDT perps, 5m candles, each instrument handled independently.
Decision after the close of candle t (rows <= t only), fill at the open of candle t + 1 (engine).
"""
import numpy as np

PARAMS = {"entry_mode": "backtest", "exit_mode": "session_end", "k": 1.5}

INSTRUMENTS = ("BTCUSDT", "ETHUSDT")
WEIGHT = 0.5              # 0.5 of equity per instrument -> gross <= 1 with both
BAR_MIN = 5               # candle length in minutes
DAY_MIN = 1440
WARMUP_END = 60           # candles opening before 01:00 UTC (the first 12 of the session) only feed V and sigma
CUTOFF_CLOSE = 1320       # no entry signal on candles closing after 22:00 UTC
LAST_OPEN = 1435          # the 23:55 UTC candle: after its close every position is closed (00:00 open)
ATR_N = 14                # Wilder ATR on 5m candles, runs across sessions
TOL_ATR = 0.25            # tol = 0.25 * ATR
STOP_ATR = 0.25           # stop buffer 0.25 * ATR
ACCEPT_COUNT = 3          # consecutive closes beyond the band
ACCEPT_EXT = 0.5          # or one close (k + 0.5) sigma from V
BT_WINDOW = 6             # candles after the backtest candle in which the entry must come


def _seg_cumsum(v, first_idx, sid):
    """Cumulative sum restarting at every session start (prefix-only, so point-in-time)."""
    cs = np.cumsum(v)
    before = (cs - v)[first_idx]
    return cs - before[sid]


def session_bands(times_min, h, l, c, qv, real, k):
    """Session (UTC day) VWAP V, volume-weighted sigma of the typical price around V, and U/L = V +- k sigma.

    tp = (h + l + c) / 3, bv = qv / tp; V = sum(qv) / sum(bv) = sum(bv tp) / sum(bv);
    sigma^2 = sum(bv (tp - V)^2) / sum(bv). Non-real candles contribute nothing.
    """
    times_min = np.asarray(times_min, dtype=np.int64)
    n = times_min.shape[0]
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    qv = np.asarray(qv, dtype=np.float64)
    if n == 0:
        e = np.zeros(0)
        return e, e, e, e
    day = times_min // DAY_MIN
    tp = (h + l + c) / 3.0
    valid = np.asarray(real, dtype=bool) & np.isfinite(tp) & np.isfinite(qv) & (tp > 0) & (qv >= 0)
    tp0 = np.where(valid, tp, 0.0)
    bv = np.where(valid, qv / np.where(valid, tp, 1.0), 0.0)
    start = np.ones(n, dtype=bool)
    start[1:] = day[1:] != day[:-1]
    sid = np.cumsum(start) - 1
    first_idx = np.flatnonzero(start)
    # centre tp on the session's first close (known at that row) for numerical precision only
    ref_first = c[first_idx]
    ref_first = np.where(np.isfinite(ref_first), ref_first, 0.0)
    ref = ref_first[sid]
    x = np.where(valid, tp0 - ref, 0.0)
    s0 = _seg_cumsum(bv, first_idx, sid)
    s1 = _seg_cumsum(bv * x, first_idx, sid)
    s2 = _seg_cumsum(bv * x * x, first_idx, sid)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = np.where(s0 > 0, s1 / np.where(s0 > 0, s0, 1.0), np.nan)
        m2 = np.where(s0 > 0, s2 / np.where(s0 > 0, s0, 1.0), np.nan)
        V = ref + m1
        sig = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    U = V + k * sig
    L = V - k * sig
    return V, sig, U, L


def simulate_instrument(times_min, o, h, l, c, qv, real, tradable, member, params):
    """Positions (+0.5 / -0.5 / 0) of one instrument; row t uses rows <= t only."""
    entry_mode = str(params["entry_mode"])
    exit_mode = str(params["exit_mode"])
    k = float(params["k"])
    if entry_mode not in ("backtest", "aggressive"):
        raise ValueError("unknown entry_mode " + entry_mode)
    if exit_mode not in ("session_end", "target_2r"):
        raise ValueError("unknown exit_mode " + exit_mode)
    aggressive = entry_mode == "aggressive"
    use_target = exit_mode == "target_2r"

    times_min = np.asarray(times_min, dtype=np.int64)
    T = times_min.shape[0]
    out = np.zeros(T, dtype=np.float64)
    if T == 0:
        return out
    V, sig, U, L = session_bands(times_min, h, l, c, qv, real, k)
    ext = k + ACCEPT_EXT

    day_l = (times_min // DAY_MIN).tolist()
    mod_l = (times_min % DAY_MIN).tolist()
    o_l = np.asarray(o, dtype=np.float64).tolist()
    h_l = np.asarray(h, dtype=np.float64).tolist()
    l_l = np.asarray(l, dtype=np.float64).tolist()
    c_l = np.asarray(c, dtype=np.float64).tolist()
    real_l = np.asarray(real, dtype=bool).tolist()
    trad_l = np.asarray(tradable, dtype=bool).tolist()
    mem_l = np.asarray(member, dtype=bool).tolist()
    V_l = V.tolist()
    sig_l = sig.tolist()
    U_l = U.tolist()
    L_l = L.tolist()
    nan = float("nan")

    # ATR state (Wilder, real candles only, across sessions)
    atr = nan
    n_tr = 0
    tr_sum = 0.0
    prev_close = nan

    cur_day = None
    pos = 0
    pending = False
    stop = nan
    fill = nan
    target = nan

    for t in range(T):
        d = day_l[t]
        mod = mod_l[t]
        if d != cur_day:
            cur_day = d
            # all setup state resets at 00:00 UTC
            used_long = False
            used_short = False
            acc_l = False
            cnt_l = 0
            b_l = False
            age_l = 0
            hi_b_l = nan
            lo_run_l = nan
            acc_s = False
            cnt_s = 0
            b_s = False
            age_s = 0
            lo_b_s = nan
            hi_run_s = nan
            prev_in_session = False
            pc = nan
            pU = nan
            pL = nan
            if pos != 0:
                # no position across a session boundary (only reached if the 23:55 row is missing)
                pos = 0
                pending = False

        # the fill price of a pending entry is the first tradable open after the signal
        if pos != 0 and pending and trad_l[t]:
            fill = o_l[t]
            pending = False
            r = abs(fill - stop)
            target = fill + 2.0 * r if pos > 0 else fill - 2.0 * r

        if not real_l[t]:
            # gap candle: never creates, updates or triggers a signal; the session end still applies
            if pos != 0 and mod == LAST_OPEN:
                pos = 0
                pending = False
            out[t] = WEIGHT * pos
            continue

        ht = h_l[t]
        lt = l_l[t]
        ct = c_l[t]
        # Wilder ATR including candle t
        if prev_close == prev_close:
            tr = max(ht - lt, abs(ht - prev_close), abs(lt - prev_close))
        else:
            tr = ht - lt
        prev_close = ct
        if tr == tr:
            if n_tr < ATR_N:
                n_tr += 1
                tr_sum += tr
                if n_tr == ATR_N:
                    atr = tr_sum / ATR_N
            else:
                atr = (atr * (ATR_N - 1) + tr) / ATR_N

        # ---- exits (evaluated after the close, filled at the next open) ----
        was_open = pos != 0
        if pos != 0:
            ex = False
            if pos > 0:
                if lt <= stop:
                    ex = True
                elif not pending and fill <= stop:
                    ex = True
                elif use_target and not pending and ht >= target:
                    ex = True
            else:
                if ht >= stop:
                    ex = True
                elif not pending and fill >= stop:
                    ex = True
                elif use_target and not pending and lt <= target:
                    ex = True
            if mod == LAST_OPEN:
                ex = True
            if not mem_l[t]:
                ex = True
            if ex:
                pos = 0
                pending = False

        Vt = V_l[t]
        st = sig_l[t]
        Ut = U_l[t]
        Lt = L_l[t]

        sig_long = False
        sig_short = False
        stop_long = nan
        stop_short = nan

        if mod >= WARMUP_END:
            if aggressive:
                if prev_in_session:
                    if ct > Ut and pc <= pU:
                        sig_long = True
                        stop_long = lt - STOP_ATR * atr
                    if ct < Lt and pc >= pL:
                        sig_short = True
                        stop_short = ht + STOP_ATR * atr
            else:
                tol = TOL_ATR * atr
                # ---- long setup ----
                if ct < Vt:
                    acc_l = False
                    b_l = False
                    cnt_l = 0
                else:
                    cnt_l = cnt_l + 1 if ct > Ut else 0
                    fired = False
                    if b_l:
                        age_l += 1
                        if lt < lo_run_l:
                            lo_run_l = lt
                        if ct > hi_b_l:
                            fired = True
                            sig_long = True
                            stop_long = lo_run_l - STOP_ATR * atr
                            b_l = False
                        elif age_l >= BT_WINDOW:
                            b_l = False
                    if acc_l and not fired and lt <= Ut + tol and ct >= Ut - tol:
                        b_l = True
                        age_l = 0
                        hi_b_l = ht
                        lo_run_l = lt
                    if not acc_l and (cnt_l >= ACCEPT_COUNT or ct >= Vt + ext * st):
                        acc_l = True
                # ---- short setup (mirror) ----
                if ct > Vt:
                    acc_s = False
                    b_s = False
                    cnt_s = 0
                else:
                    cnt_s = cnt_s + 1 if ct < Lt else 0
                    fired = False
                    if b_s:
                        age_s += 1
                        if ht > hi_run_s:
                            hi_run_s = ht
                        if ct < lo_b_s:
                            fired = True
                            sig_short = True
                            stop_short = hi_run_s + STOP_ATR * atr
                            b_s = False
                        elif age_s >= BT_WINDOW:
                            b_s = False
                    if acc_s and not fired and ht >= Lt - tol and ct <= Lt + tol:
                        b_s = True
                        age_s = 0
                        lo_b_s = lt
                        hi_run_s = ht
                    if not acc_s and (cnt_s >= ACCEPT_COUNT or ct <= Vt - ext * st):
                        acc_s = True

        # remember the previous real candle of the session (aggressive cross)
        prev_in_session = True
        pc = ct
        pU = Ut
        pL = Lt

        # ---- entries ----
        can_enter = (not was_open) and pos == 0 and mod + BAR_MIN <= CUTOFF_CLOSE and mem_l[t]
        if can_enter and not (sig_long and sig_short):
            if sig_long and not used_long and stop_long == stop_long:
                pos = 1
                pending = True
                stop = stop_long
                used_long = True
            elif sig_short and not used_short and stop_short == stop_short:
                pos = -1
                pending = True
                stop = stop_short
                used_short = True

        out[t] = WEIGHT * pos
    return out


def target_positions(data, params):
    T = data.times.shape[0]
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=np.float64)
    if np.datetime_data(data.times.dtype)[0] != "m":
        raise ValueError("expected data.times in datetime64[m]")
    times_min = data.times.astype(np.int64)
    uni = data.universe
    for name in INSTRUMENTS:
        j = data.col(name)
        member = np.ones(T, dtype=bool) if uni is None else np.asarray(uni[:, j], dtype=bool)
        w[:, j] = simulate_instrument(
            times_min, data.o[:, j], data.h[:, j], data.l[:, j], data.c[:, j], data.qv[:, j],
            data.real[:, j], data.tradable[:, j], member, params,
        )
    return w
