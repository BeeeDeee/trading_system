"""H-0076 mechanism event: coin placed in the long leg (+1) or short leg (-1) of the cohort formed after the
close of day t, exactly as in signal steps 1-3 (same code as strategy.legs, copied)."""
import numpy as np

UPBIT = "upbit_krw_1d"
PERP_SUFFIX = "USDT.P"
MIN_VALUE_KRW = 1.0e9
MAX_ABS_REL = 0.20
MIN_ELIGIBLE = 10
UNIVERSE_N = 30
VOLUME_DAYS = 30


def top_n_universe(data, n=UNIVERSE_N, days=VOLUME_DAYS):
    """Same as strategy.top_n_universe (copied): data.universe if passed, else PIT top-n by 30-day volume."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=float)
    P = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = dv.shape
    dv0 = np.where(np.isfinite(dv) & (dv > 0), dv, 0.0)
    vol = np.zeros((T, N))
    for lag in range(min(days, T)):               # lags >= T add nothing (short truncated views)
        vol[lag:] += dv0[: T - lag]
    cand = listed & np.isfinite(P) & (P > 0) & (vol > 0)
    key = np.where(cand, vol, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty((T, N), dtype=np.int64)
    np.put_along_axis(rank, order, np.arange(N)[None, :].repeat(T, axis=0), axis=1)
    return cand & (rank < n)


def perp_base(name):
    if not name.endswith(PERP_SUFFIX):
        return None
    base = name[: len(name) - len(PERP_SUFFIX)]
    if len(base) == 0:
        return None
    nd = 0
    while nd < len(base) and base[nd].isdigit():
        nd += 1
    if nd >= 2:
        return None
    return base


def nanmedian_last(x):
    fin = np.isfinite(x)
    cnt = fin.sum(axis=-1)
    s = np.sort(np.where(fin, x, np.nan), axis=-1)
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.maximum(cnt // 2, 0)
    vlo = np.take_along_axis(s, lo[..., None], axis=-1)[..., 0]
    vhi = np.take_along_axis(s, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (vlo + vhi)
    med = np.where(cnt > 0, med, np.nan)
    return med, cnt


def legs(data, params):
    B = int(params["baseline_days"])
    n_leg = int(params["n_leg"])
    uni = top_n_universe(data)
    T = len(data.dates)
    N = len(data.instruments)
    P = np.asarray(data.close, dtype=float)
    basis = np.asarray(data.extras["basis"], dtype=float)
    prefix = UPBIT + "."
    if not any(str(k).startswith(prefix) for k in data.series):
        raise KeyError("no " + UPBIT + " series delivered")

    cols = []
    U = np.full((T, N), np.nan)
    V = np.full((T, N), np.nan)
    for j, name in enumerate(data.instruments):
        base = perp_base(name)
        if base is None:
            continue
        key_c = prefix + base + ".close"
        if key_c not in data.series:
            continue
        cols.append(j)
        U[:, j] = np.asarray(data.series[key_c], dtype=float)
        V[:, j] = np.asarray(data.series[prefix + base + ".value_krw"], dtype=float)

    if len(cols) == 0 and any(str(nm).endswith(PERP_SUFFIX) for nm in data.instruments):
        raise KeyError("perp ids present but none matches an " + UPBIT + " base")
    ok = uni & np.isfinite(P) & (P > 0)
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)
    if len(cols) == 0:
        return long_m, short_m, ok
    cols = np.asarray(cols, dtype=int)

    with np.errstate(invalid="ignore", divide="ignore"):
        onepb = 1.0 + basis
        S = np.where(np.isfinite(basis) & (onepb > 0), P / onepb, P)
        valid = ok & np.isfinite(S) & (S > 0) & np.isfinite(U) & (U > 0) & np.isfinite(V) & (V >= MIN_VALUE_KRW)
        k = np.where(valid, np.log(np.where(valid, U, 1.0)) - np.log(np.where(valid, S, 1.0)), np.nan)
    kc = k[:, cols]
    m, _ = nanmedian_last(kc)
    r = kc - m[:, None]
    r = np.where(np.abs(r) > MAX_ABS_REL, np.nan, r)

    M = len(cols)
    padded = np.vstack([np.full((B, M), np.nan), r])
    win = np.lib.stride_tricks.sliding_window_view(padded, B, axis=0)[:T]
    base_med, base_cnt = nanmedian_last(win)
    a = np.where(3 * base_cnt >= 2 * B, r - base_med, np.nan)

    elig = np.isfinite(a)
    cnt = elig.sum(axis=1)
    order = np.argsort(np.where(elig, a, np.inf), axis=1, kind="stable")
    rows = np.nonzero(cnt >= MIN_ELIGIBLE)[0]
    if len(rows):
        lo_idx = order[rows, :n_leg]
        hi_pos = cnt[rows][:, None] - n_leg + np.arange(n_leg)[None, :]
        hi_idx = np.take_along_axis(order[rows], hi_pos, axis=1)
        lc = np.zeros((len(rows), M), dtype=bool)
        sc = np.zeros((len(rows), M), dtype=bool)
        np.put_along_axis(lc, hi_idx, True, axis=1)
        np.put_along_axis(sc, lo_idx, True, axis=1)
        sub_l = np.zeros((T, M), dtype=bool)
        sub_s = np.zeros((T, M), dtype=bool)
        sub_l[rows] = lc
        sub_s[rows] = sc
        long_m[:, cols] = sub_l
        short_m[:, cols] = sub_s
    return long_m, short_m, ok


def events(data, params):
    long_m, short_m, ok = legs(data, params)
    uni = top_n_universe(data)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (long_m | short_m) & uni & tradable
    side = np.where(long_m, 1.0, np.where(short_m, -1.0, 0.0))
    return mask, side
