"""H-0010 Ex-dividend price pressure (LIQ-500, dollar neutral, daily decisions).

Rules of card.signal.description and where they live:
(1) distribution detection          -> _detect_ex_days
(2) regular quarterly payer / E     -> _schedule_block
(3) long set (pre-ex window)        -> _sets_block (long_c)
(4) short set (post-ex window)      -> _sets_block (short_c)
(5) weights, thin-leg market hedge  -> target_weights
All quantities on row t use rows 0..t only; row t is executed at the open of t+1.
"""
import numpy as np

PARAMS = {"window_days": 5}

LOOKBACK = 252          # trading days, inclusive of t
MIN_COUNT = 3           # detected ex-days in the lookback
MAX_COUNT = 5
MIN_GAP = 55            # trading days between the last two detected ex-days
MAX_GAP = 71
D_MIN = 0.001           # implied cash distribution bounds
D_MAX = 0.10
SPLIT_LOG = 0.25        # |log(close_u(t)/close_u(t-1))| must be below this
MIN_LEG = 5             # a leg with fewer names is replaced by the market basket
LEG_WEIGHT = 0.5
BLOCK = 256             # columns per block (memory)


def _detect_ex_days(close, rco, roc, listed):
    """(T, n) bool: day t is a detected ex-day (needs bars on t-1 and t)."""
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
        gross = (1.0 + r_co) * (1.0 + r_oc)          # 1 + R(t)
        d = gross / ratio - 1.0                       # (1+R) * c(t-1)/c(t) - 1
        det[1:] = ok & (d >= D_MIN) & (d <= D_MAX)
    return det


def _schedule_block(det):
    """Regular-payer flag, last detected ex-day and predicted next ex-day E for each (t, j)."""
    T, n = det.shape
    idx = np.arange(T, dtype=np.int64)[:, None]
    last = np.where(det, idx, -1)
    np.maximum.accumulate(last, axis=0, out=last)            # last detected ex-day <= t
    last_before = np.empty_like(last)
    last_before[0] = -1
    last_before[1:] = last[:-1]                               # last detected ex-day <= t-1
    prev = np.take_along_axis(last_before, np.maximum(last, 0), axis=0)   # the one before `last`
    prev[last < 0] = -1
    cs = np.cumsum(det, axis=0, dtype=np.int32)
    cnt = cs.copy()
    if T > LOOKBACK:
        cnt[LOOKBACK:] -= cs[:-LOOKBACK]                      # detections in rows t-251..t
    gap = last - prev
    reg = ((cnt >= MIN_COUNT) & (cnt <= MAX_COUNT) & (last >= 0) & (prev >= 0)
           & (gap >= MIN_GAP) & (gap <= MAX_GAP))
    E = last + gap
    return reg, last, E


def _sets_block(close, rco, roc, listed, W):
    det = _detect_ex_days(close, rco, roc, listed)
    reg, last, E = _schedule_block(det)
    T = det.shape[0]
    t1 = np.arange(T, dtype=np.int64)[:, None] + 1            # the next session t+1
    # (3) pre-ex window: E - W <= t+1 <= E - 1, predicted from the latest detection (so none after it)
    long_c = reg & (t1 >= E - W) & (t1 <= E - 1)
    # (4) post-ex window: 1 <= t+1-x <= W, regular payer evaluated at x (detection at x included)
    s = t1 - last
    reg_at_x = np.take_along_axis(reg, np.maximum(last, 0), axis=0) & (last >= 0)
    short_c = reg_at_x & (s >= 1) & (s <= W)
    return long_c, short_c


def compute_sets(data, W):
    """Candidate long/short sets (T, N) bool, before universe/tradability filtering."""
    close = np.asarray(data.close)
    rco = np.asarray(data.ret_co)
    roc = np.asarray(data.ret_oc)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = close.shape
    long_c = np.zeros((T, N), dtype=bool)
    short_c = np.zeros((T, N), dtype=bool)
    f8 = np.float64
    for a in range(0, N, BLOCK):
        b = min(N, a + BLOCK)
        lc, sc = _sets_block(close[:, a:b].astype(f8), rco[:, a:b].astype(f8), roc[:, a:b].astype(f8),
                             listed[:, a:b], W)
        long_c[:, a:b] = lc
        short_c[:, a:b] = sc
    return long_c, short_c


def target_weights(data, params):
    W = int(params["window_days"])
    if W < 1:
        raise ValueError("window_days must be >= 1")
    if data.universe is None:
        raise ValueError("H-0010 needs the point-in-time LIQ-500 universe (data.universe is None)")
    if data.close is None:
        raise ValueError("H-0010 needs unadjusted closes (data.close is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    eligible = universe & tradable

    long_c, short_c = compute_sets(data, W)
    long_s = long_c & eligible
    short_s = short_c & eligible
    basket = eligible & ~long_s & ~short_s

    nL = long_s.sum(axis=1)
    nS = short_s.sum(axis=1)
    nB = basket.sum(axis=1)

    full_L = nL >= MIN_LEG
    full_S = nS >= MIN_LEG
    # a thin leg is replaced by the equal-weighted basket of the other eligible members
    use_basket_L = ~full_L & full_S
    use_basket_S = full_L & ~full_S
    # no basket available (degenerate) -> flat rather than an unhedged leg
    active = (full_L & full_S) | ((use_basket_L | use_basket_S) & (nB > 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        wL = np.where(full_L, LEG_WEIGHT / np.maximum(nL, 1), 0.0)
        wS = np.where(full_S, LEG_WEIGHT / np.maximum(nS, 1), 0.0)
        wB = LEG_WEIGHT / np.maximum(nB, 1)

    wBsigned = np.where(use_basket_L, wB, 0.0) - np.where(use_basket_S, wB, 0.0)
    w = np.where(long_s, wL[:, None], 0.0)
    w -= np.where(short_s, wS[:, None], 0.0)
    w += np.where(basket, wBsigned[:, None], 0.0)
    w[~active] = 0.0
    return w
