"""H-0049 mechanism events: same event definition and side logic as strategy.compute_events (code duplicated so
the diagnostic stands alone)."""
import numpy as np

VOL_WINDOW = 30
FUNDING_WINDOW = 7
PCTL = 33.3
TOP_N = 30


def _hedge_cols(data):
    """BTCUSDT.P / ETHUSDT.P by name (raises if missing); on the lab's synthetic SYN... market only, the two
    alphabetically first instruments stand in (same rule as strategy.hedge_cols)."""
    names = tuple(data.instruments)
    if "BTCUSDT.P" not in names and "ETHUSDT.P" not in names and all(s.startswith("SYN") for s in names):
        stand_in = sorted(names)[:2]
        return data.col(stand_in[0]), data.col(stand_in[1])
    return data.col("BTCUSDT.P"), data.col("ETHUSDT.P")


def _universe(data):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = dv.shape
    fin = np.isfinite(dv)
    c = np.vstack([np.zeros((1, N)), np.cumsum(np.where(fin, dv, 0.0), axis=0)])
    n = np.vstack([np.zeros((1, N)), np.cumsum(fin, axis=0)])
    idx = np.arange(1, T + 1)
    lo = np.maximum(idx - VOL_WINDOW, 0)
    vol30 = (c[idx] - c[lo]) / np.maximum(n[idx] - n[lo], 1)
    vol30 = np.where((n[idx] - n[lo]) > 0, vol30, np.nan)
    vol30 = np.where(np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64)),
                     vol30, np.nan)
    key = np.where(np.isfinite(vol30), vol30, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return (rank < TOP_N) & np.isfinite(vol30)


def events(data, params):
    k_sigma = float(params["k_sigma"])
    close = np.asarray(data.close, dtype=np.float64)
    T, N = close.shape
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    univ = _universe(data)
    funding = np.asarray(data.extras["funding"], dtype=np.float64)
    hedge_cols = _hedge_cols(data)

    has_px = np.isfinite(close) & listed
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.zeros((T, N), dtype=bool)
    valid[1:] = has_px[1:] & has_px[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)

    rz = np.where(valid, r, 0.0)
    c1 = np.vstack([np.zeros((1, N)), np.cumsum(rz, axis=0)])
    c2 = np.vstack([np.zeros((1, N)), np.cumsum(rz * rz, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(valid, axis=0)])
    sigma = np.full((T, N), np.nan)
    if T > VOL_WINDOW:
        t_idx = np.arange(VOL_WINDOW, T)
        s1 = c1[t_idx] - c1[t_idx - VOL_WINDOW]
        s2 = c2[t_idx] - c2[t_idx - VOL_WINDOW]
        cnt = cn[t_idx] - cn[t_idx - VOL_WINDOW]
        var = np.maximum((s2 - s1 * s1 / VOL_WINDOW) / (VOL_WINDOW - 1), 0.0)
        sigma[t_idx] = np.where(cnt == VOL_WINDOW, np.sqrt(var), np.nan)

    eligible = univ & valid & np.isfinite(sigma)

    re = np.where(eligible, r, np.nan)
    pct = np.full(T, np.nan)
    has = eligible.any(axis=1)
    if has.any():
        pct[has] = np.nanpercentile(re[has], PCTL, axis=1)

    cand = eligible & tradable & (re <= pct[:, None]) & (re <= -k_sigma * sigma)
    for hc in hedge_cols:
        cand[:, hc] = False

    lags = np.full((FUNDING_WINDOW, T, N), np.nan)
    for k in range(1, FUNDING_WINDOW + 1):
        if k < T:
            lags[k - 1, k:] = funding[:-k]
    f_ok = np.isfinite(funding) & np.all(np.isfinite(lags), axis=0)
    lag_safe = np.where(np.isfinite(lags), lags, 0.0)
    fmin = np.min(lag_safe, axis=0)
    fmed = np.median(lag_safe, axis=0)
    f_now = np.where(np.isfinite(funding), funding, 0.0)
    flush = (f_now < 0.0) & (f_now < fmin)
    unflushed = f_now >= fmed

    cand &= f_ok
    side = np.zeros((T, N))
    side[cand & flush] = 1.0
    side[cand & unflushed & ~flush] = -1.0

    mask = side != 0.0
    return mask, side
