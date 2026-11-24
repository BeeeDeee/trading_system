"""H-0187 mechanism_test events: first daily close outside a quiet range_days close range,
volume-confirmed, coin-specific vs BTCUSDT.P; side +1 up-breakout, -1 down-breakout.
Same event code as strategy.py (copied); every event is marked, also on coins that already
hold a slot in the book (the event definition does not depend on the book)."""
import numpy as np

BTC = "BTCUSDT.P"
QUIET_LOOKBACK = 90
QUIET_PCTL = 0.25
VOL_LOOKBACK = 20
UNIVERSE_N = 30
UNIVERSE_VOL_DAYS = 30


def _swv(x, window, axis=0):
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)


def _fallback_universe(close, dollar_volume, n=UNIVERSE_N, days=UNIVERSE_VOL_DAYS):
    T, N = close.shape
    v = np.where(np.isfinite(dollar_volume), dollar_volume, 0.0).astype(np.float64)
    cs = np.cumsum(v, axis=0)
    prev = np.zeros_like(cs)
    if T > days:
        prev[days:] = cs[:-days]
    vol = cs - prev
    ok = np.isfinite(close) & (vol > 0)
    key = np.where(ok, vol, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return ok & (rank < n)


def _events(data, params, universe):
    rd = int(params["range_days"])
    close = np.asarray(data.close, dtype=np.float64)
    qv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = close.shape
    btc = data.col(BTC)
    side = np.zeros((T, N), dtype=np.float64)
    if T < rd + 1:
        return side

    hi = np.full((T, N), np.nan)
    lo = np.full((T, N), np.nan)
    win = _swv(close, rd, axis=0)
    with np.errstate(invalid="ignore"):
        mx = win.max(axis=-1)
        mn = win.min(axis=-1)
    hi[rd:] = mx[:-1]
    lo[rd:] = mn[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.log(hi / lo)
    R[~np.isfinite(R)] = np.nan

    quiet = np.zeros((T, N), dtype=bool)
    volok = np.zeros((T, N), dtype=bool)
    cols = np.flatnonzero(universe.any(axis=0))
    k = QUIET_PCTL * (QUIET_LOOKBACK - 1)
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, QUIET_LOOKBACK - 1)
    frac = k - k0
    m0 = VOL_LOOKBACK // 2 - 1
    m1 = VOL_LOOKBACK // 2
    for j in cols:
        if T > QUIET_LOOKBACK:
            Rj = R[:, j]
            w = _swv(Rj, QUIET_LOOKBACK)
            bad = np.isnan(w).any(axis=1)
            p = np.partition(np.where(np.isnan(w), np.inf, w), [k0, k1], axis=1)
            with np.errstate(invalid="ignore"):
                q = p[:, k0] + frac * (p[:, k1] - p[:, k0])
            q[bad] = np.nan
            qt = np.full(T, np.nan)
            qt[QUIET_LOOKBACK:] = q[:-1]
            with np.errstate(invalid="ignore"):
                quiet[:, j] = np.isfinite(Rj) & np.isfinite(qt) & (Rj <= qt)
        if T > VOL_LOOKBACK:
            vj = qv[:, j]
            w = _swv(vj, VOL_LOOKBACK)
            bad = np.isnan(w).any(axis=1)
            p = np.partition(np.where(np.isnan(w), np.inf, w), [m0, m1], axis=1)
            med = 0.5 * (p[:, m0] + p[:, m1])
            med[bad] = np.nan
            mt = np.full(T, np.nan)
            mt[VOL_LOOKBACK:] = med[:-1]
            with np.errstate(invalid="ignore"):
                volok[:, j] = np.isfinite(vj) & np.isfinite(mt) & (vj >= mt)

    ret = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = close[1:] / close[:-1] - 1.0
    rel = ret - ret[:, [btc]]

    valid = np.isfinite(close) & np.isfinite(qv) & np.isfinite(rel)
    base = valid & quiet & volok & universe & np.asarray(data.tradable, dtype=bool)
    base[:, btc] = False
    with np.errstate(invalid="ignore"):
        up = base & (close > hi) & (rel > 0)
        dn = base & (close < lo) & (rel < 0)
    side[up] = 1.0
    side[dn] = -1.0
    return side


def events(data, params):
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
    else:
        uni = _fallback_universe(np.asarray(data.close, dtype=np.float64),
                                 np.asarray(data.dollar_volume, dtype=np.float64))
    s = _events(data, params, uni)
    mask = s != 0
    side = np.where(mask, s, 1.0)
    return mask, side
