"""H-0089 Absorbed taker flow on Binance perps (15m candles, crypto_top_n universe).

After the close of candle t (fills at the open of t+1):
1. Hourly window t-3..t: QV = sum qv, TB = sum tbq, I = 2*TB/QV - 1, R = c_t / c_(t-4) - 1.
2. Baseline over the 2880 candles ending at t-4 (rows t-2883..t-4): mu_I, sd_I (population sd) of the
   valid hourly I values and QVbar = mean of the valid hourly QV values; needs >= 2400 real candles in
   those rows and all 4 window candles real with qv > 0. z = (I - mu_I) / sd_I.
3. Event: |z| >= imbalance_z, QV >= 2 * QVbar, sign(z) * R <= 0. Side = -sign(z).
4. Five slots of 0.20; ranked by |z| when slots are short; no pyramiding / reversal.
5. Exit after hold_candles candles, or earlier at the first decision where the symbol is out of the
   point-in-time universe or its candle is not real.
"""
import numpy as np

PARAMS = {"hold_candles": 24, "imbalance_z": 2.5}

WINDOW = 4            # one hour of 15-minute candles
BASELINE = 2880       # 30 days of 15-minute candles
MIN_REAL = 2400       # real candles required in the baseline
VOL_MULT = 2.0        # QV >= 2 * QVbar (fixed in the card)
SLOTS = 5
SLOT_WEIGHT = 0.20


def _prefix(x):
    """Prefix sums with a leading zero row: P[k] = sum of rows 0..k-1 (point-in-time)."""
    p = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=x.dtype)
    np.cumsum(x, axis=0, out=p[1:])
    return p


def _window_sum(x, n, lag):
    """Sum of x over rows t-lag-n+1 .. t-lag; valid[t] False where the window starts before row 0."""
    T = x.shape[0]
    p = _prefix(x)
    out = np.zeros_like(p[1:])
    valid = np.zeros(T, dtype=bool)
    t0 = n + lag - 1
    if t0 < T:
        t = np.arange(t0, T)
        out[t0:] = p[t + 1 - lag] - p[t + 1 - lag - n]
        valid[t0:] = True
    return out, valid


def signals(data, params):
    """Return (event mask, side, |z|) as (T, N) arrays, all from rows <= t."""
    k = float(params["imbalance_z"])
    c = np.asarray(data.c, dtype=np.float64)
    qv = np.asarray(data.qv, dtype=np.float64)
    tbq = np.asarray(data.tbq, dtype=np.float64)
    real = np.asarray(data.real, dtype=bool)
    uni = data.universe
    if uni is None:
        raise ValueError("H-0089 needs the point-in-time crypto_top_n universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    T, N = c.shape

    # 1. hourly window t-3..t
    good = real & np.isfinite(qv) & (qv > 0) & np.isfinite(tbq)
    qv0 = np.where(good, qv, 0.0)
    tb0 = np.where(good, tbq, 0.0)
    QV, ok_q = _window_sum(qv0, WINDOW, 0)
    TB, _ = _window_sum(tb0, WINDOW, 0)
    ngood, _ = _window_sum(good.astype(np.int64), WINDOW, 0)
    win_ok = (ngood == WINDOW) & ok_q[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(win_ok, 2.0 * TB / np.where(win_ok, QV, 1.0) - 1.0, 0.0)

    c_lag = np.full_like(c, np.nan)
    c_lag[WINDOW:] = c[:-WINDOW]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = c / c_lag - 1.0
    R_ok = np.isfinite(R) & np.isfinite(c_lag) & (c_lag > 0)

    # 2. baseline: the 2880 candles ending at t-4 (no overlap with the window t-3..t)
    lag = WINDOW
    Iv = np.where(win_ok, I, 0.0)
    nI, ok_b = _window_sum(win_ok.astype(np.int64), BASELINE, lag)
    sI, _ = _window_sum(Iv, BASELINE, lag)
    sI2, _ = _window_sum(Iv * Iv, BASELINE, lag)
    sQV, _ = _window_sum(np.where(win_ok, QV, 0.0), BASELINE, lag)
    nreal, _ = _window_sum(real.astype(np.int64), BASELINE, lag)
    base_ok = ok_b[:, None] & (nreal >= MIN_REAL) & (nI >= 2)
    n_safe = np.where(base_ok, nI, 1).astype(np.float64)
    mu = sI / n_safe
    var = np.maximum(sI2 / n_safe - mu * mu, 0.0)
    sd = np.sqrt(var)
    QVbar = sQV / n_safe
    base_ok &= sd > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(base_ok & win_ok, (I - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    s = np.sign(z)

    # 3. event
    event = (
        win_ok
        & base_ok
        & R_ok
        & uni
        & (np.abs(z) >= k)
        & (s != 0)
        & (QV >= VOL_MULT * QVbar)
        & (s * np.where(R_ok, R, 0.0) <= 0.0)
    )
    side = -s
    return event, side, np.abs(z)


def target_positions(data, params):
    H = int(params["hold_candles"])
    if H < 1:
        raise ValueError("hold_candles must be >= 1")
    event, side, absz = signals(data, params)
    real = np.asarray(data.real, dtype=bool)
    uni = np.asarray(data.universe, dtype=bool)
    T, N = event.shape
    bad = ~(real & uni)   # early-exit condition (K-0051)

    pos = np.zeros((T, N), dtype=np.float64)
    open_close = {}       # column -> decision row at which the position is closed
    rows = np.nonzero(event.any(axis=1))[0]
    for t in rows:
        t = int(t)
        # free slots whose close decision is at or before t
        for j in [j for j, cl in open_close.items() if cl <= t]:
            del open_close[j]
        free = SLOTS - len(open_close)
        if free <= 0:
            continue
        cand = np.nonzero(event[t])[0]
        cand = np.array([j for j in cand if int(j) not in open_close], dtype=np.int64)
        if cand.size == 0:
            continue
        order = np.argsort(-absz[t, cand], kind="stable")   # largest |z| first, ties by column
        for j in cand[order][:free]:
            j = int(j)
            end = min(t + H, T)                       # scheduled close decision
            seg = bad[t + 1:end, j]
            if seg.any():
                end = t + 1 + int(np.argmax(seg))     # first decision out of universe / not real
            pos[t:end, j] = SLOT_WEIGHT * side[t, j]
            if end < T:
                pos[end, j] = 0.0
            open_close[j] = end
    return pos
