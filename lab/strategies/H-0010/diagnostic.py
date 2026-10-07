"""H-0010 mechanism events (same detection and schedule as strategy.py).

Pre-ex (+1): after the close of t, t+1 = E - W (first session of the W-session window before the
predicted ex-day E) for a regular payer at t.
Post-ex (-1): day t is a detected ex-day of a regular payer (regularity evaluated at t, detection included).
Only where the stock is a universe member and tradable on row t.
"""
import numpy as np

LOOKBACK = 252
MIN_COUNT = 3
MAX_COUNT = 5
MIN_GAP = 55
MAX_GAP = 71
D_MIN = 0.001
D_MAX = 0.10
SPLIT_LOG = 0.25
BLOCK = 256


def _detect_ex_days(close, rco, roc, listed):
    T, n = close.shape
    det = np.zeros((T, n), dtype=bool)
    if T < 2:
        return det
    c0 = close[:-1]
    c1 = close[1:]
    r_co = rco[1:]
    r_oc = roc[1:]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ok = (np.isfinite(c0) & np.isfinite(c1) & (c0 > 0) & (c1 > 0)
              & np.isfinite(r_co) & np.isfinite(r_oc) & listed[:-1] & listed[1:])
        ratio = np.where(ok, c1 / np.where(ok, c0, 1.0), 1.0)
        ok &= np.abs(np.log(ratio)) < SPLIT_LOG
        gross = (1.0 + r_co) * (1.0 + r_oc)
        d = gross / ratio - 1.0
        det[1:] = ok & (d >= D_MIN) & (d <= D_MAX)
    return det


def _schedule_block(det):
    T, n = det.shape
    idx = np.arange(T, dtype=np.int64)[:, None]
    last = np.where(det, idx, -1)
    np.maximum.accumulate(last, axis=0, out=last)
    last_before = np.empty_like(last)
    last_before[0] = -1
    last_before[1:] = last[:-1]
    prev = np.take_along_axis(last_before, np.maximum(last, 0), axis=0)
    prev[last < 0] = -1
    cs = np.cumsum(det, axis=0, dtype=np.int32)
    cnt = cs.copy()
    if T > LOOKBACK:
        cnt[LOOKBACK:] -= cs[:-LOOKBACK]
    gap = last - prev
    reg = ((cnt >= MIN_COUNT) & (cnt <= MAX_COUNT) & (last >= 0) & (prev >= 0)
           & (gap >= MIN_GAP) & (gap <= MAX_GAP))
    E = last + gap
    return det, reg, E


def events(data, params):
    W = int(params["window_days"])
    if data.universe is None:
        raise ValueError("H-0010 needs the point-in-time LIQ-500 universe (data.universe is None)")
    close = np.asarray(data.close)
    rco = np.asarray(data.ret_co)
    roc = np.asarray(data.ret_oc)
    f8 = np.float64
    listed = np.asarray(data.listed, dtype=bool)
    eligible = np.asarray(data.universe, dtype=bool) & np.asarray(data.tradable, dtype=bool)
    T, N = close.shape
    pre = np.zeros((T, N), dtype=bool)
    post = np.zeros((T, N), dtype=bool)
    t1 = np.arange(T, dtype=np.int64)[:, None] + 1
    for a in range(0, N, BLOCK):
        b = min(N, a + BLOCK)
        det, reg, E = _schedule_block(_detect_ex_days(close[:, a:b].astype(f8), rco[:, a:b].astype(f8),
                                                     roc[:, a:b].astype(f8), listed[:, a:b]))
        pre[:, a:b] = reg & (t1 == E - W)
        post[:, a:b] = reg & det
    pre &= eligible
    post &= eligible
    mask = pre | post
    side = np.where(post, -1.0, 1.0)
    return mask, side
