"""H-0174 Korean retail pump fade.

After the 00:00 UTC close of day t: a top-50 Binance perp (not BTC/ETH) whose base X has an Upbit KRW
series and a Binance spot pair XUSDT is an event when
    U = Upbit value_krw(t) / median(value_krw, t-30..t-1)          >= k
    K = U / B, B = spot quote_volume(t) / median(qv, t-30..t-1)    >= k
    perp return of day t > cross-sectional median perp return of the universe on day t
    no event of the coin in rows t-hold_days..t-1
(both medians need >= 20 valid days, Upbit median >= 1e9 KRW).
Each event opens a slot at the open of t+1: short the coin 0.15, long 0.075 BTCUSDT.P + 0.075 ETHUSDT.P,
held for hold_days bars (weights on decision rows t..t+hold_days-1, zero at row t+hold_days, i.e. closed at
the open of t+1+hold_days). At most 3 slots; excess events ranked by K (largest first). A slot closes at the
next decision if the coin leaves the universe or its perp has no price.
"""
import numpy as np

PARAMS = {"hold_days": 5, "k": 2.0}

UNIVERSE_N = 50            # card universe crypto_top_n, n = 50 (fallback ranking only)
WINDOW = 30                # median window t-30..t-1
MIN_VALID = 20             # valid days needed in the window
UPBIT_FLOOR_KRW = 1.0e9    # fixed liquidity floor on the Upbit median
SLOT_W = 0.15              # short leg per slot
MAX_SLOTS = 3
HEDGE_A = "BTCUSDT.P"
HEDGE_B = "ETHUSDT.P"
HEDGE_BASES = ("BTC", "ETH")
UPBIT_PREFIX = "upbit_krw_1d."
UPBIT_SUFFIX = ".value_krw"
SPOT_PREFIX = "binance_spot_1d."
SPOT_SUFFIX = ".quote_volume"
QUOTE = "USDT"
PERP_SUFFIX = ".P"


def _clean_volume(v):
    """Valid positive values; a value identical to the previous row is treated as a carried-forward gap."""
    v = np.asarray(v, dtype=float).copy()
    rep = np.zeros(v.shape[0], dtype=bool)
    if v.shape[0] > 1:
        rep[1:] = v[1:] == v[:-1]
    bad = ~np.isfinite(v) | (v <= 0) | rep
    v[bad] = np.nan
    return v


def _lagged_median(v, window, min_valid):
    """Median of v over rows t-window..t-1 (row t excluded); NaN if fewer than min_valid finite values."""
    T = v.shape[0]
    pad = np.concatenate([np.full(window, np.nan), v])
    win = np.lib.stride_tricks.sliding_window_view(pad, window)[:T]
    cnt = np.sum(np.isfinite(win), axis=1)
    s = np.sort(win, axis=1)                       # NaN sorted last
    lo = np.clip((cnt - 1) // 2, 0, window - 1)
    hi = np.clip(cnt // 2, 0, window - 1)
    rows = np.arange(T)
    med = 0.5 * (s[rows, lo] + s[rows, hi])
    med[cnt < min_valid] = np.nan
    return med


def _row_nanmedian(x):
    """Row-wise median ignoring NaN; NaN for empty rows (no warnings)."""
    T, N = x.shape
    cnt = np.sum(np.isfinite(x), axis=1)
    s = np.sort(x, axis=1)
    lo = np.clip((cnt - 1) // 2, 0, max(N - 1, 0))
    hi = np.clip(cnt // 2, 0, max(N - 1, 0))
    rows = np.arange(T)
    med = 0.5 * (s[rows, lo] + s[rows, hi])
    med[cnt == 0] = np.nan
    return med


def _perp_universe(data):
    perp = np.array([ac == "crypto_perp" for ac in data.asset_class], dtype=bool)
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool) & perp[None, :]
    # Fallback (K-0131): PIT top-n perps by trailing 30-row quote volume (rows t-29..t), listed on t.
    T, N = data.close.shape
    dv = np.where(np.isfinite(data.dollar_volume), data.dollar_volume, 0.0).astype(float)
    dv[:, ~perp] = 0.0
    cs = np.vstack([np.zeros((1, N)), np.cumsum(dv, axis=0)])
    end = np.arange(1, T + 1)
    start = np.maximum(end - WINDOW, 0)
    vol = cs[end] - cs[start]
    elig = np.asarray(data.listed, dtype=bool) & perp[None, :] & (vol > 0)
    score = np.where(elig, vol, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N), (T, N)).copy(), axis=1)
    return elig & (ranks < UNIVERSE_N)


def _events(data, params):
    """Mechanism events. Returns (ev (T, M) bool, K (T, M), perp_cols (M,), uni (T, N))."""
    k = float(params["k"])
    hold = int(params["hold_days"])
    T = len(data.dates)
    names = {nm: i for i, nm in enumerate(data.instruments)}
    uni = _perp_universe(data)

    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    tradable = np.asarray(data.tradable, dtype=bool)
    ok = listed & np.isfinite(close)
    valid_r = np.zeros_like(ok)
    valid_r[1:] = ok[1:] & ok[:-1]
    r = (1.0 + np.asarray(data.ret_co, dtype=float)) * (1.0 + np.asarray(data.ret_oc, dtype=float)) - 1.0
    r = np.where(valid_r, r, np.nan)
    umed = _row_nanmedian(np.where(uni, r, np.nan))

    qv_src = data.extras["quote_volume"] if "quote_volume" in data.extras else data.dollar_volume

    perp_cols, spot_qv, upbit_keys = [], [], []
    n_perp_matched = 0
    for key in sorted(data.series.keys()):
        if not (key.startswith(UPBIT_PREFIX) and key.endswith(UPBIT_SUFFIX)):
            continue
        base = key[len(UPBIT_PREFIX):len(key) - len(UPBIT_SUFFIX)]
        if base in HEDGE_BASES:
            continue
        pj = names.get(base + QUOTE + PERP_SUFFIX)
        if pj is None:
            continue
        n_perp_matched += 1
        sj = names.get(base + QUOTE)
        if sj is not None and data.asset_class[sj] != "crypto_perp":
            q = qv_src[:, sj]                      # spot pair XUSDT delivered as a (signal-only) column
        else:
            q = None                               # or delivered as a series of binance_spot_1d
            for sk in (SPOT_PREFIX + base + QUOTE + SPOT_SUFFIX, SPOT_PREFIX + base + SPOT_SUFFIX):
                if sk in data.series:
                    q = data.series[sk]
                    break
            if q is None:
                continue
        perp_cols.append(pj)
        spot_qv.append(q)
        upbit_keys.append(key)
    if n_perp_matched > 0 and not perp_cols:
        spot_names = [nm for nm, ac in zip(data.instruments, data.asset_class) if ac != "crypto_perp"][:10]
        raise ValueError("H-0174: %d perps match an Upbit base but no Binance spot XUSDT column was found "
                         "(non-perp instruments: %s)" % (n_perp_matched, spot_names))

    M = len(perp_cols)
    cond = np.zeros((T, M), dtype=bool)
    Kmat = np.full((T, M), np.nan)
    for m in range(M):
        j = perp_cols[m]
        V = _clean_volume(data.series[upbit_keys[m]])
        Q = _clean_volume(spot_qv[m])
        mV = _lagged_median(V, WINDOW, MIN_VALID)
        mQ = _lagged_median(Q, WINDOW, MIN_VALID)
        with np.errstate(invalid="ignore", divide="ignore"):
            U = V / mV
            B = Q / mQ
            K = U / B
        good = np.isfinite(U) & np.isfinite(B) & (B > 0) & np.isfinite(K) & (mV >= UPBIT_FLOOR_KRW)
        c = good & (U >= k) & (K >= k)
        c &= valid_r[:, j] & np.isfinite(umed) & (np.nan_to_num(r[:, j], nan=-np.inf) > np.nan_to_num(umed, nan=np.inf))
        c &= uni[:, j] & tradable[:, j]
        cond[:, m] = c
        Kmat[:, m] = np.where(good, K, np.nan)

    # no re-trigger: an event needs no event of the same coin in rows t-hold..t-1
    ev = np.zeros((T, M), dtype=bool)
    last = np.full(M, -10 ** 9)
    for t in range(T):
        e = cond[t] & (t - last > hold)
        ev[t] = e
        last[e] = t
    return ev, Kmat, np.array(perp_cols, dtype=int), uni


def target_weights(data, params):
    hold = int(params["hold_days"])
    T, N = len(data.dates), len(data.instruments)
    ib = data.col(HEDGE_A)
    ie = data.col(HEDGE_B)
    ev, Kmat, perp_cols, uni = _events(data, params)
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    alive = listed & np.isfinite(close)

    w = np.zeros((T, N), dtype=float)
    slots = []                                     # (m, open_row)
    for t in range(T):
        keep = []
        for (m, s) in slots:
            j = perp_cols[m]
            if t >= s + hold:                      # held rows s..s+hold-1
                continue
            if not uni[t, j] or not alive[t, j]:   # left the universe / lost data: close both legs
                continue
            keep.append((m, s))
        slots = keep
        free = MAX_SLOTS - len(slots)
        if free > 0:
            cand = np.flatnonzero(ev[t])
            if cand.size:
                kv = Kmat[t, cand]
                order = np.lexsort((cand, -kv))    # largest K first, ties by column order
                for m in cand[order][:free]:
                    slots.append((int(m), t))
        for (m, s) in slots:
            w[t, perp_cols[m]] -= SLOT_W
            w[t, ib] += SLOT_W / 2.0
            w[t, ie] += SLOT_W / 2.0
    return w
