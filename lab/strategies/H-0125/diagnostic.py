"""H-0125 mechanism event: Korea-specific Upbit KRW volume surge with a Binance up-move (side -1).

Same event code as strategy.py (copied: diagnostic.py is a separate file). An event of coin j on day t:
U >= k, K = U / B >= k, both 30-day medians (t-30..t-1) with >= 20 valid days, Upbit median >= 1e9 KRW,
perp return of day t above the universe's cross-sectional median, coin in the universe and tradable,
no event of the coin in the previous hold_days days. All events are marked (not only those that got a slot).
"""
import numpy as np
from lab.framework.api import total_return_index

HEDGE_A = "BTCUSDT.P"
HEDGE_B = "ETHUSDT.P"
TOP_N = 50
WINDOW = 30
MIN_VALID = 20
FLOOR_KRW = 1.0e9
UPBIT_PREFIX = "upbit_krw_1d."
UPBIT_SUFFIX = ".value_krw"
SPOT_DATASET = "binance_spot_1d."


def _nanmedian_last(x, min_count):
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=-1)
    s = np.sort(x, axis=-1)
    lo = np.clip((n - 1) // 2, 0, x.shape[-1] - 1)
    hi = np.clip(n // 2, 0, x.shape[-1] - 1)
    a = np.take_along_axis(s, lo[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(s, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (a + b)
    med = np.where(n >= max(int(min_count), 1), med, np.nan)
    return med


def _trailing_median(v, window, min_valid):
    T, M = v.shape
    if M == 0:
        return np.full((T, 0), np.nan)
    xp = np.vstack([np.full((window, M), np.nan), v])
    win = np.lib.stride_tricks.sliding_window_view(xp, window, axis=0)[:T]
    return _nanmedian_last(win, min_valid)


def _valid(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


def _base(name):
    s = name[:-2] if name.endswith(".P") else name
    if s.endswith("USDT") and len(s) > 4:
        return s[:-4]
    return None


def _universe(data, perp):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    T, N = len(data.dates), len(data.instruments)
    listed = np.asarray(data.listed, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=float)
    dv = np.where(np.isfinite(dv) & listed & perp[None, :], dv, 0.0)
    cs = np.cumsum(dv, axis=0)
    roll = cs.copy()
    if T > WINDOW:
        roll[WINDOW:] = cs[WINDOW:] - cs[:-WINDOW]
    elig = listed & perp[None, :] & (roll > 0)
    score = np.where(elig, roll, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(N)[None, :].repeat(T, axis=0), axis=1)
    return elig & (rank < TOP_N)


def _spot_qv(data, base, perp_j):
    key = SPOT_DATASET + base + "USDT.quote_volume"
    if data.series is not None and key in data.series:
        return np.asarray(data.series[key], dtype=float)
    extras = data.extras if data.extras is not None else {}
    name = base + "USDT"
    if name in data.instruments:
        j = data.col(name)
        if str(data.asset_class[j]) == "crypto_spot":
            if "quote_volume" in extras:
                return np.asarray(extras["quote_volume"], dtype=float)[:, j]
            return np.asarray(data.dollar_volume, dtype=float)[:, j]
    if "spot_quote_volume" in extras:
        return np.asarray(extras["spot_quote_volume"], dtype=float)[:, perp_j]
    if "quote_volume" in extras:
        return np.asarray(extras["quote_volume"], dtype=float)[:, perp_j]
    return None


def compute_events(data, params):
    k = float(params["k"])
    h = int(params["hold_days"])
    T, N = len(data.dates), len(data.instruments)
    names = tuple(data.instruments)
    perp = np.array([str(a) == "crypto_perp" for a in data.asset_class], dtype=bool)
    jb = data.col(HEDGE_A)
    je = data.col(HEDGE_B)
    if data.series is None:
        raise KeyError("upbit_krw_1d series missing")
    upbit = {}
    for key in data.series:
        if key.startswith(UPBIT_PREFIX) and key.endswith(UPBIT_SUFFIX):
            upbit[key[len(UPBIT_PREFIX):-len(UPBIT_SUFFIX)]] = key
    if not upbit:
        raise KeyError("no upbit_krw_1d.<BASE>.value_krw series in the view")

    uni = _universe(data, perp)
    cols, up, sp = [], [], []
    n_cand = 0
    for j in range(N):
        if not perp[j] or j == jb or j == je:
            continue
        b = _base(names[j])
        if b is None or b not in upbit:
            continue
        n_cand += 1
        qv = _spot_qv(data, b, j)
        if qv is None:
            continue
        cols.append(j)
        up.append(np.asarray(data.series[upbit[b]], dtype=float))
        sp.append(qv)
    if n_cand > 0 and not cols:
        raise KeyError("Upbit-matched perps exist but no Binance spot quote volume was delivered")
    cols = np.array(cols, dtype=int)
    M = len(cols)
    if M == 0:
        return np.zeros((T, 0), bool), np.zeros((T, 0)), cols, uni, jb, je

    vu = _valid(np.stack(up, axis=1))
    vb = _valid(np.stack(sp, axis=1))
    mu = _trailing_median(vu, WINDOW, MIN_VALID)
    mb = _trailing_median(vb, WINDOW, MIN_VALID)
    with np.errstate(divide="ignore", invalid="ignore"):
        U = vu / mu
        B = vb / mb
        K = U / B

    listed = np.asarray(data.listed, dtype=bool)
    idx = np.asarray(total_return_index(data), dtype=float)
    r = np.full((T, N), np.nan)
    if T > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = idx[1:] / idx[:-1] - 1.0
        r[1:] = np.where(listed[1:] & listed[:-1], r[1:], np.nan)
    r = np.where(np.isfinite(r), r, np.nan)
    rm = np.where(uni & perp[None, :], r, np.nan)
    xs_med = _nanmedian_last(rm, 1)

    tradable = np.asarray(data.tradable, dtype=bool)
    rc = r[:, cols]
    with np.errstate(invalid="ignore"):
        raw = (np.isfinite(U) & np.isfinite(K) & np.isfinite(mb) & (mu >= FLOOR_KRW)
               & (U >= k) & (K >= k) & np.isfinite(rc) & (rc > xs_med[:, None])
               & uni[:, cols] & listed[:, cols] & tradable[:, cols])

    event = np.zeros((T, M), dtype=bool)
    last = np.full(M, -(10 ** 9), dtype=np.int64)
    for t in range(T):
        e = raw[t] & ((t - last) > h)
        if e.any():
            event[t] = e
            last[e] = t
    K = np.where(event, K, np.nan)
    return event, K, cols, uni, jb, je


def events(data, params):
    T, N = len(data.dates), len(data.instruments)
    ev, K, cols, uni, jb, je = compute_events(data, params)
    mask = np.zeros((T, N), dtype=bool)
    if len(cols):
        mask[:, cols] = ev
    side = np.where(mask, -1.0, 0.0)
    return mask, side
