"""H-0035: intraday liquidation-cascade rebound on Binance perps (15m candles).

After the close of candle t, an eligible instrument fires an event when its 1-hour (4-candle) log return is an
extreme z-score move (|z| >= z_threshold, z = r_w / (2 * sigma_15m)), on >= 3x normal quote volume, with the
taker flow on the moving side (taker-buy share < 0.5 for a drop, > 0.5 for a rise). The strategy fades it with a
fixed 1/20 slot (+1/20 after a drop, -1/20 after a rise), entered at the open of t+1 and closed at the open of
t+1+hold_candles. One position per instrument at a time; no stops, no targets, no re-normalization.
"""
import numpy as np

PARAMS = {"hold_candles": 8, "z_threshold": 4}

# Constants fixed by design (not on the grid)
W = 4                 # 1-hour window in 15m candles
REF = 672             # 7-day reference window in 15m candles
VOL_MULT = 3.0        # volume multiple
TAKER_SPLIT = 0.5     # taker-buy share split
SLOT = 1.0 / 20.0     # fixed position size


def _shift(x, k):
    """x shifted down by k rows (row t holds x[t-k]); first k rows NaN. k >= 0."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k == 0:
        out[:] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _trailing_sum(x, n):
    """Sum of x over rows t-n+1..t; NaN unless all n values are finite. Point-in-time (prefix sums)."""
    T = x.shape[0]
    valid = np.isfinite(x)
    xz = np.where(valid, x, 0.0)
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cs[1:] = np.cumsum(xz, axis=0)
    cnt = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cnt[1:] = np.cumsum(valid, axis=0)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if T >= n:
        s = cs[n:] - cs[:-n]
        c = cnt[n:] - cnt[:-n]
        out[n - 1:] = np.where(c == n, s, np.nan)
    return out


def _features(data):
    c = np.asarray(data.c, dtype=np.float64)
    c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    qv = np.asarray(data.qv, dtype=np.float64)
    tbq = np.asarray(data.tbq, dtype=np.float64)
    logc = np.log(c)

    # (1) 1-hour log return ln(c[t] / c[t-4])
    r_w = logc - _shift(logc, W)

    # (2) sigma: std (ddof=0) of the 672 15-minute log returns ending at candle t-4
    lr = logc - _shift(logc, 1)
    s1 = _trailing_sum(lr, REF)
    s2 = _trailing_sum(lr * lr, REF)
    mean = s1 / REF
    var = np.maximum(s2 / REF - mean * mean, 0.0)
    sigma = _shift(np.sqrt(var), W)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, r_w / (sigma * 2.0), np.nan)

    # (3) volume ratio: qv over t-3..t / (4 x mean qv per candle over the 672 candles ending at t-4)
    qv_w = _trailing_sum(qv, W)
    qv_ref_mean = _shift(_trailing_sum(qv, REF) / REF, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(qv_ref_mean > 0, qv_w / (W * qv_ref_mean), np.nan)

    # (4) taker-buy share over t-3..t
    tbq_w = _trailing_sum(tbq, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        taker_share = np.where(qv_w > 0, tbq_w / qv_w, np.nan)
    return z, vol_ratio, taker_share


def _eligible(data):
    real = np.asarray(data.real, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    universe = np.asarray(data.universe, dtype=bool)
    real_count = np.cumsum(real, axis=0)                      # real candles up to and including t
    last4_real = _trailing_sum(real.astype(np.float64), W) == W  # candles t-3..t all real
    return universe & listed & (real_count >= REF) & last4_real


def events(data, params):
    """(T, N) int8: +1 long-cascade event (go long), -1 short-squeeze event (go short), 0 none."""
    zt = float(params["z_threshold"])
    z, vol_ratio, taker_share = _features(data)
    elig = _eligible(data)
    with np.errstate(invalid="ignore"):
        big_vol = vol_ratio >= VOL_MULT
        long_ev = elig & (z <= -zt) & big_vol & (taker_share < TAKER_SPLIT)
        short_ev = elig & (z >= zt) & big_vol & (taker_share > TAKER_SPLIT)
    ev = np.zeros(z.shape, dtype=np.int8)
    ev[long_ev] = 1
    ev[short_ev] = -1
    return ev


def target_positions(data, params):
    hold = int(params["hold_candles"])
    ev = events(data, params)
    universe = np.asarray(data.universe, dtype=bool)
    T, N = ev.shape
    pos = np.zeros((T, N), dtype=np.float64)
    for j in range(N):
        idx = np.flatnonzero(ev[:, j])
        busy_until = -1          # last decision row on which the instrument still holds a position
        for t in idx:
            if t <= busy_until:
                continue         # already in a position: ignore (no extension, reversal or pyramiding)
            # decided after close t, filled at open t+1; target 0 from row t+hold -> exit at open t+1+hold
            end = min(t + hold, T)
            # Engine constraint (G0): positions may be held only while the instrument is a universe member.
            # If it leaves the universe at row k inside the hold, the target goes to 0 at row k (exit at open k+1).
            out = np.flatnonzero(~universe[t:end, j])
            if out.size:
                end = t + int(out[0])
            pos[t:end, j] = SLOT * float(ev[t, j])
            busy_until = end
    return pos
