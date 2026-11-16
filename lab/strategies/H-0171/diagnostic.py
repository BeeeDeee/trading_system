"""H-0171 mechanism test events: on each shock date with a valid book, the 12 picked coins.

Short leg (3 highest crowd per BTC-beta half) -> side -1, long leg (3 lowest per half) -> side +1.
The selection code is a copy of strategy.py (same rules, same tie-breaks). Marked only where the coin is a
universe member and tradable on the shock row.
"""
import numpy as np

BTC = "BTCUSDT.P"
VIX_KEY = "fred_macro.VIXCLS"
TOP_N = 30
QV_DAYS = 30
BETA_WINDOW = 60
BETA_MIN_PAIRS = 45
N_PER_LEG_HALF = 3
MIN_HALF = 6


def _shift(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _trailing_mean_all_valid(x, n):
    x = np.asarray(x, dtype=float)
    acc = np.zeros(x.shape)
    ok = np.ones(x.shape, dtype=bool)
    for k in range(int(n)):
        s = _shift(x, k)
        fin = np.isfinite(s)
        ok &= fin
        acc += np.where(fin, s, 0.0)
    return np.where(ok, acc / float(n), np.nan)


def _trailing_nansum(x, n):
    x = np.asarray(x, dtype=float)
    acc = np.zeros(x.shape)
    for k in range(int(n)):
        s = _shift(x, k)
        acc += np.where(np.isfinite(s), s, 0.0)
    return acc


def _ffill_1d(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    idx = np.where(np.isfinite(v), np.arange(v.shape[0]), -1)
    idx = np.maximum.accumulate(idx) if v.shape[0] else idx
    return np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)


def _shock_rows(vix, vix_shock):
    v = _ffill_1d(vix)
    prev = np.full(v.shape, np.nan)
    prev[1:] = v[:-1]
    ok = np.isfinite(v) & np.isfinite(prev) & (v > 0) & (prev > 0)
    lc = np.full(v.shape, np.nan)
    lc[ok] = np.log(v[ok] / prev[ok])
    new = ok & (v != prev)
    return new & (lc >= float(vix_shock))


def _cc_returns(data):
    r = (1.0 + np.asarray(data.ret_co, float)) * (1.0 + np.asarray(data.ret_oc, float)) - 1.0
    close = np.asarray(data.close, float)
    has = np.isfinite(close) & np.asarray(data.listed, bool)
    prev = np.zeros(has.shape, dtype=bool)
    prev[1:] = has[:-1]
    return np.where(has & prev & np.isfinite(r), r, np.nan)


def _rolling_beta(r, j_btc, window=BETA_WINDOW, min_pairs=BETA_MIN_PAIRS):
    x = np.broadcast_to(r[:, j_btc : j_btc + 1], r.shape)
    valid = np.isfinite(r) & np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, r, 0.0)
    T = r.shape[0]
    pad = np.zeros((1, r.shape[1]))

    def wsum(a):
        cs = np.concatenate([pad, np.cumsum(a, axis=0)])
        hi = np.arange(1, T + 1)
        lo = np.maximum(hi - window, 0)
        return cs[hi] - cs[lo]

    n = wsum(valid.astype(float))
    sx, sy, sxy, sxx = wsum(xv), wsum(yv), wsum(xv * yv), wsum(xv * xv)
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (n * sxy - sx * sy) / den
    good = (n >= min_pairs) & (den > 1e-18)
    beta = np.where(good, beta, np.nan)
    beta[:, j_btc] = np.where(n[:, j_btc] >= min_pairs, 1.0, np.nan)
    return beta


def _universe(data, qv30):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    listed = np.isfinite(np.asarray(data.close, float)) & np.asarray(data.listed, bool)
    key = np.where(listed & (qv30 > 0), -qv30, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    return (rank < TOP_N) & np.isfinite(key)


def _select(crowd_t, beta_t, qv_t, elig_t):
    cols = np.flatnonzero(elig_t)
    if cols.size < 2 * MIN_HALF:
        return None
    order = cols[np.lexsort((cols, beta_t[cols]))]
    h = cols.size // 2
    halves = (order[:h], order[cols.size - h :])
    if h < MIN_HALF:
        return None
    shorts, longs = [], []
    for half in halves:
        q = np.where(np.isfinite(qv_t[half]), qv_t[half], -np.inf)
        c = crowd_t[half]
        hi = half[np.lexsort((half, -q, -c))]
        lo = half[np.lexsort((half, -q, c))]
        lo = lo[~np.isin(lo, hi[:N_PER_LEG_HALF])]
        shorts.extend(hi[:N_PER_LEG_HALF].tolist())
        longs.extend(lo[:N_PER_LEG_HALF].tolist())
    return np.array(shorts, dtype=int), np.array(longs, dtype=int)


def events(data, params):
    fd = int(params["funding_days"])
    vix = data.series[VIX_KEY]
    funding = np.asarray(data.extras["funding"], dtype=float)
    j_btc = data.col(BTC)
    qv30 = _trailing_nansum(np.asarray(data.dollar_volume, float), QV_DAYS)
    member = _universe(data, qv30)
    crowd = _trailing_mean_all_valid(funding, fd)
    beta = _rolling_beta(_cc_returns(data), j_btc)
    shock = _shock_rows(vix, params["vix_shock"])
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = member.shape
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    for t in np.flatnonzero(shock):
        elig = member[t] & np.isfinite(crowd[t]) & np.isfinite(beta[t])
        sel = _select(crowd[t], beta[t], qv30[t], elig)
        if sel is None:
            continue
        s, l = sel
        side[t, s] = -1.0
        side[t, l] = 1.0
        mask[t, s] = True
        mask[t, l] = True
    mask &= member & tradable
    side = np.where(mask, side, 0.0)
    return mask, side
