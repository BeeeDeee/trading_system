"""H-0063 stop-cascade breakout from compressed ranges on Binance perps (15m).

Decision after the close of candle t, fill at the open of t+1 (engine).
Rules (card signal.description):
 (1) HI/LO = max(h)/min(l) over candles t-W..t-1, R = (HI-LO)/c[t-1], W = range_hours * candles/hour
 (2) CR = R[t] / median(R[t-L..t-1]), L = 30 days of candles; compressed if CR <= compression_ratio
 (3) long candidate c[t] > HI, short candidate c[t] < LO
 (4) qv[t] >= 2 * mean(qv[t-W..t-1]); tbq/qv >= 0.55 long, <= 0.45 short
 (5) skip if |log(c[t]/c[t-4])| > 4 * std of non-overlapping 1h log returns over the past 30 days
 (6) cooldown: no entry if the instrument had an entry in the previous W candles
 entry +-0.25, at most 4 open, ranked by qv[t]/mean qv, extra candidates dropped.
 exits: time stop hold_hours * candles/hour, failed break through the entry range midpoint,
 leaving the PIT universe.
"""
import numpy as np

PARAMS = {"compression_ratio": 0.6, "hold_hours": 6, "range_hours": 24}

WEIGHT = 0.25          # fixed by the card
MAX_POS = 4            # fixed by the card (gross <= 1)
VOL_MULT = 2.0         # fixed flow confirmation
LONG_SHARE = 0.55
SHORT_SHARE = 0.45
EXCL_SIGMA = 4.0       # fixed H-0035 exclusion
LOOKBACK_DAYS = 30
CHUNK = 1024


def _shift(x, k):
    """x shifted down by k >= 0 rows (row t holds x[t-k]); NaN on the first k rows."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    T = x.shape[0]
    if k < T:
        out[k:] = x[:T - k]
    return out


def _rolling_extreme(x, w, fn):
    """Trailing max/min over rows t-w+1..t (NaN if fewer than w rows or a NaN inside).

    Van Herk / Gil-Werman blocks: the suffix part only covers rows <= t, so it is causal."""
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if w > T:
        return out
    nb = -(-T // w)
    pad = np.full((nb * w, N), np.nan, dtype=np.float64)
    pad[:T] = x
    blocks = pad.reshape(nb, w, N)
    pre = fn.accumulate(blocks, axis=1).reshape(nb * w, N)
    suf = fn.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape(nb * w, N)
    t = np.arange(w - 1, T)
    out[w - 1:] = fn(suf[t - w + 1], pre[t])
    return out


def _rolling_sum_prev(x, w):
    """Sum over rows t-w..t-1 (excluding t); NaN for t < w."""
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cs[1:] = np.cumsum(x, axis=0, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if w < T:
        out[w:] = cs[w:T] - cs[0:T - w]
    return out


def _gather(x, rows, cols, offsets):
    """x[rows - offsets, cols] as (n, len(offsets)); NaN where the row index is negative."""
    idx = rows[:, None] - offsets[None, :]
    valid = idx >= 0
    g = x[np.where(valid, idx, 0), cols[:, None]].astype(np.float64)
    g[~valid] = np.nan
    return g


def target_positions(data, params):
    bpd = int(data.bars_per_day)
    bph = bpd // 24                                   # candles per hour (4 on 15m)
    W = int(round(float(params["range_hours"]) * bph))
    H = int(round(float(params["hold_hours"]) * bph))
    ratio = float(params["compression_ratio"])
    L = LOOKBACK_DAYS * bpd                           # 2880 candles on 15m

    if data.universe is None:
        raise ValueError("H-0063 needs the point-in-time crypto_top_n universe")
    uni = np.asarray(data.universe, dtype=bool)
    h = np.asarray(data.h, dtype=np.float64)
    l = np.asarray(data.l, dtype=np.float64)
    c = np.asarray(data.c, dtype=np.float64)
    qv = np.asarray(data.qv, dtype=np.float64)
    tbq = np.asarray(data.tbq, dtype=np.float64)
    real = np.asarray(data.real).astype(bool)
    T, N = c.shape
    P = np.zeros((T, N), dtype=np.float64)
    if T <= L:
        return P

    # history: first real candle at least L candles before t (30 days)
    seen = np.maximum.accumulate(real, axis=0)
    hist_ok = np.zeros((T, N), dtype=bool)
    hist_ok[L:] = seen[:T - L]

    # no gap row inside t-W..t-1, and t itself a real candle
    n_gap = _rolling_sum_prev((~real).astype(np.float64), W)
    ok = real & uni & hist_ok & (n_gap == 0)
    del n_gap, seen, hist_ok

    # (4) flow confirmation
    qv_fin = np.isfinite(qv)
    qv_cnt = _rolling_sum_prev(qv_fin.astype(np.float64), W)
    qv_mean = _rolling_sum_prev(np.where(qv_fin, qv, 0.0), W) / W
    with np.errstate(divide="ignore", invalid="ignore"):
        good_mean = (qv_cnt == W) & (qv_mean > 0)
        vol_ratio = np.where(good_mean, qv / np.where(good_mean, qv_mean, 1.0), np.nan)
        good_qv = qv_fin & (qv > 0)
        share = np.where(good_qv, tbq / np.where(good_qv, qv, 1.0), np.nan)
    del qv_cnt, qv_mean, good_mean, good_qv, qv_fin
    ok &= vol_ratio >= VOL_MULT

    # (1) prior range over t-W..t-1 and (3) break
    HI = _shift(_rolling_extreme(h, W, np.maximum), 1)
    LO = _shift(_rolling_extreme(l, W, np.minimum), 1)
    cprev = _shift(c, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(cprev > 0, (HI - LO) / np.where(cprev > 0, cprev, 1.0), np.nan)
    del cprev
    is_long = (c > HI) & (share >= LONG_SHARE)
    is_short = (c < LO) & (share <= SHORT_SHARE)
    ok &= is_long | is_short
    del share

    rows, cols = np.nonzero(ok)            # row-major: sorted by row, then column
    del ok
    if rows.size == 0:
        return P

    # (5) exclusion of already-extended moves: phase-aligned non-overlapping 1h returns
    c4 = _shift(c, bph)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (c > 0) & (c4 > 0)
        r1h = np.where(pos, np.log(np.where(pos, c, 1.0) / np.where(pos, c4, 1.0)), np.nan)
    del c4, pos
    n_h = L // bph                                    # 720 hourly returns in 30 days
    h_off = bph * np.arange(1, n_h + 1)
    keep = np.zeros(rows.size, dtype=bool)
    for a in range(0, rows.size, CHUNK):
        rr, cc = rows[a:a + CHUNK], cols[a:a + CHUNK]
        g = _gather(r1h, rr, cc, h_off)
        fin = np.isfinite(g)
        cnt = fin.sum(axis=1)
        g0 = np.where(fin, g, 0.0)
        mean = g0.sum(axis=1) / np.maximum(cnt, 1)
        var = (np.where(fin, g - mean[:, None], 0.0) ** 2).sum(axis=1) / np.maximum(cnt, 1)
        sigma = np.where(cnt >= 2, np.sqrt(var), np.nan)
        cur = r1h[rr, cc]
        keep[a:a + CHUNK] = np.isfinite(cur) & np.isfinite(sigma) & (np.abs(cur) <= EXCL_SIGMA * sigma)
    rows, cols = rows[keep], cols[keep]
    del r1h
    if rows.size == 0:
        return P

    # (2) compression against the median of R over the previous 30 days (t-L..t-1)
    m_off = np.arange(1, L + 1)
    keep = np.zeros(rows.size, dtype=bool)
    for a in range(0, rows.size, CHUNK):
        rr, cc = rows[a:a + CHUNK], cols[a:a + CHUNK]
        g = _gather(R, rr, cc, m_off)
        has = np.isfinite(g).any(axis=1)
        med = np.full(rr.size, np.nan)
        if has.any():
            med[has] = np.nanmedian(g[has], axis=1)
        cur = R[rr, cc]
        with np.errstate(divide="ignore", invalid="ignore"):
            good = np.isfinite(cur) & np.isfinite(med) & (med > 0)
            cr = np.where(good, cur / np.where(good, med, 1.0), np.nan)
        keep[a:a + CHUNK] = good & (cr <= ratio)
    rows, cols = rows[keep], cols[keep]
    if rows.size == 0:
        return P

    side = np.where(is_long[rows, cols], 1.0, -1.0)
    rank_key = vol_ratio[rows, cols]
    mid = 0.5 * (HI[rows, cols] + LO[rows, cols])

    # (6) cooldown, slots, exits: sequential over candidate rows
    last_entry = np.full(N, -(10 ** 12), dtype=np.int64)
    exit_row = np.full(N, -1, dtype=np.int64)     # position is non-zero on rows entry..exit_row-1
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    ends = np.r_[starts[1:], rows.size]
    for s0, s1 in zip(starts, ends):
        t = int(rows[s0])
        n_open = int((exit_row > t).sum())
        free = MAX_POS - n_open
        if free <= 0:
            continue
        idx = np.arange(s0, s1)
        js = cols[idx]
        elig = (exit_row[js] <= t) & (t - last_entry[js] > W)
        idx = idx[elig]
        if idx.size == 0:
            continue
        order = np.argsort(-rank_key[idx], kind="stable")
        for k in idx[order[:free]]:
            j = int(cols[k])
            w = WEIGHT * side[k]
            stop = min(t + H, T - 1)
            seg_c = c[t + 1:stop + 1, j]
            if side[k] > 0:
                fail = seg_c < mid[k]
            else:
                fail = seg_c > mid[k]
            fail = fail | ~uni[t + 1:stop + 1, j]
            if fail.any():
                e = t + 1 + int(np.argmax(fail))
            else:
                e = t + H
            P[t:min(e, T), j] = w
            exit_row[j] = e
            last_entry[j] = t
    return P
