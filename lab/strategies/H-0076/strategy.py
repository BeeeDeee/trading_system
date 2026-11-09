"""H-0076 Korean-exchange premium transmission (Binance perps, Upbit KRW signal).

Rows are UTC days; row t is decided after the 00:00 UTC close of day t and executed at the open of t+1.
"""
import numpy as np

PARAMS = {"baseline_days": 30, "hold_days": 3, "n_leg": 3}

UPBIT = "upbit_krw_1d"
PERP_SUFFIX = "USDT.P"
MIN_VALUE_KRW = 1.0e9      # V_j >= 1,000,000,000 KRW
MAX_ABS_REL = 0.20         # |r_j| > 0.20 dropped for the day
MIN_ELIGIBLE = 10          # fewer eligible coins: no cohort
UNIVERSE_N = 30            # card universe: crypto_top_n, n = 30
VOLUME_DAYS = 30           # top-n by 30-day quote volume (card data_requirements)


def top_n_universe(data, n=UNIVERSE_N, days=VOLUME_DAYS):
    """Card universe. Uses data.universe when the gate passes it; otherwise (the gate has passed None on daily
    perp data, K-0054 / K-0118) rebuilds the point-in-time top-n by trailing 30-day quote volume: on row t,
    the n perps with the largest sum of dollar_volume over rows t-29 .. t (missing volume counts as no
    volume), among perps listed with a finite close on row t and positive 30-day volume. Rows <= t only."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=float)
    P = np.asarray(data.close, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    T, N = dv.shape
    dv0 = np.where(np.isfinite(dv) & (dv > 0), dv, 0.0)
    vol = np.zeros((T, N))
    for lag in range(min(days, T)):               # exact trailing sum (no cumsum drift); lags >= T add nothing
        vol[lag:] += dv0[: T - lag]
    cand = listed & np.isfinite(P) & (P > 0) & (vol > 0)
    key = np.where(cand, vol, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")   # descending volume, ties by column order
    rank = np.empty((T, N), dtype=np.int64)
    np.put_along_axis(rank, order, np.arange(N)[None, :].repeat(T, axis=0), axis=1)
    return cand & (rank < n)


def perp_base(name):
    """Base asset of a perp id <BASE>USDT.P; None if not of that form or multiplier-prefixed (1000SHIB)."""
    if not name.endswith(PERP_SUFFIX):
        return None
    base = name[: len(name) - len(PERP_SUFFIX)]
    if len(base) == 0:
        return None
    nd = 0
    while nd < len(base) and base[nd].isdigit():
        nd += 1
    if nd >= 2:            # numeric multiplier prefix (1000SHIB, 10000LADYS, 1000000MOG); 1INCH keeps nd == 1
        return None
    return base


def nanmedian_last(x):
    """Median over the last axis ignoring NaN (NaN where no finite value), and the count of finite values."""
    fin = np.isfinite(x)
    cnt = fin.sum(axis=-1)
    s = np.sort(np.where(fin, x, np.nan), axis=-1)          # NaN sort to the end
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.maximum(cnt // 2, 0)
    vlo = np.take_along_axis(s, lo[..., None], axis=-1)[..., 0]
    vhi = np.take_along_axis(s, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (vlo + vhi)
    med = np.where(cnt > 0, med, np.nan)
    return med, cnt


def legs(data, params):
    """Steps 1-3 of the card. Returns (long, short, ok): (T, N) bool masks of the legs formed after the close
    of day t, and ok = in the universe with perp data on day t."""
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

    # match perps to Upbit bases (exact base-asset match; coins without an Upbit series are never eligible)
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

    # Step 1: relative premium
    with np.errstate(invalid="ignore", divide="ignore"):
        onepb = 1.0 + basis
        S = np.where(np.isfinite(basis) & (onepb > 0), P / onepb, P)      # missing basis: S = P
        valid = ok & np.isfinite(S) & (S > 0) & np.isfinite(U) & (U > 0) & np.isfinite(V) & (V >= MIN_VALUE_KRW)
        k = np.where(valid, np.log(np.where(valid, U, 1.0)) - np.log(np.where(valid, S, 1.0)), np.nan)
    kc = k[:, cols]
    m, _ = nanmedian_last(kc)                                  # cross-sectional median over the valid coins
    r = kc - m[:, None]
    r = np.where(np.abs(r) > MAX_ABS_REL, np.nan, r)           # dropped for the day (NaN stays NaN)

    # Step 2: abnormal premium vs the median of r over days t-B .. t-1, needs >= 2/3 of B valid days
    M = len(cols)
    padded = np.vstack([np.full((B, M), np.nan), r])
    win = np.lib.stride_tricks.sliding_window_view(padded, B, axis=0)[:T]   # (T, M, B): rows t-B .. t-1
    base_med, base_cnt = nanmedian_last(win)
    a = np.where(3 * base_cnt >= 2 * B, r - base_med, np.nan)

    # Step 3: legs on days with >= MIN_ELIGIBLE eligible coins
    elig = np.isfinite(a)
    cnt = elig.sum(axis=1)
    order = np.argsort(np.where(elig, a, np.inf), axis=1, kind="stable")   # ascending, ineligible last
    rows = np.nonzero(cnt >= MIN_ELIGIBLE)[0]
    if len(rows):
        lo_idx = order[rows, :n_leg]                                         # n_leg lowest a
        hi_pos = cnt[rows][:, None] - n_leg + np.arange(n_leg)[None, :]
        hi_idx = np.take_along_axis(order[rows], hi_pos, axis=1)             # n_leg highest a
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


def target_weights(data, params):
    H = int(params["hold_days"])
    n_leg = int(params["n_leg"])
    long_m, short_m, ok = legs(data, params)
    T, N = long_m.shape
    # Step 4: each day's cohort has gross 1/H, +-0.5/(H*n_leg) per coin
    c = (long_m.astype(float) - short_m.astype(float)) * (0.5 / (H * n_leg))
    # run[t, j] = number of consecutive days ending at t on which j is in the universe with perp data
    run = np.zeros((T, N), dtype=np.int64)
    prev = np.zeros(N, dtype=np.int64)
    for t in range(T):
        prev = np.where(ok[t], prev + 1, 0)
        run[t] = prev
    w = np.zeros((T, N))
    for lag in range(min(H, T)):                   # cohorts older than the view do not exist
        sh = np.zeros((T, N))
        sh[lag:] = c[: T - lag]
        # cohort formed at t-lag is live on row t; a coin that left the universe / lost perp data on any day
        # since then is closed and not replaced
        w += np.where(run >= lag + 1, sh, 0.0)
    return w
