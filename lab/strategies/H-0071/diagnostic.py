"""H-0071 mechanism events: big moves of top-30 alt perps (not BTC/ETH), side -sign(r) when leverage-led
(abnormal perp/spot volume ratio above the trailing 180-day pooled event median), +sign(r) when spot-led.

Identical rules to strategy.compute_events (the diagnostic may not import the strategy module, so the code is
duplicated; a test checks that both give the same events).
"""
import numpy as np

HEDGE_ID = "BTCUSDT.P"
EXCLUDED_IDS = ("BTCUSDT.P", "ETHUSDT.P")
PERP_SUFFIX = ".P"
PERP_CLASS = "crypto_perp"

SIGMA_WINDOW = 30
SIGMA_MIN_OBS = 25
RATIO_WINDOW = 30
RATIO_MIN_OBS = 20
POOL_WINDOW = 180
POOL_MIN_EVENTS = 30
UNIVERSE_N = 30
UNIVERSE_VOLUME_WINDOW = 30


def _resolve_inputs(data):
    names = tuple(data.instruments)
    n_rows, n_cols = len(data.dates), len(names)
    perp = np.array([str(a) == PERP_CLASS for a in data.asset_class], dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=float)
    perp_qv = np.where(perp[None, :], dv, np.nan)
    spot_qv = np.full((n_rows, n_cols), np.nan)
    real = any(n.endswith(PERP_SUFFIX) for n in names)
    if real:
        index = {n: i for i, n in enumerate(names)}
        found = 0
        for j, n in enumerate(names):
            if perp[j] and n.endswith(PERP_SUFFIX):
                s = index.get(n[: -len(PERP_SUFFIX)])
                if s is not None and not perp[s]:
                    spot_qv[:, j] = dv[:, s]
                    found += 1
        if found == 0:
            raise KeyError("binance_spot_1d quote volume missing: no spot column matches any perp id")
        hedge = data.col(HEDGE_ID)
        excluded = [data.col(x) for x in EXCLUDED_IDS]
    else:
        spot_qv[1:] = dv[:-1]
        spot_qv[:, ~perp] = np.nan
        hedge = None
        excluded = []
    return perp, perp_qv, spot_qv, hedge, excluded


def _universe(data, perp, perp_qv):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool) & perp[None, :]
    q = np.where(np.isfinite(perp_qv), perp_qv, 0.0)
    c = np.cumsum(q, axis=0)
    s = c.copy()
    s[UNIVERSE_VOLUME_WINDOW:] -= c[:-UNIVERSE_VOLUME_WINDOW]
    s[:, ~perp] = 0.0
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    rank[rows, order] = np.arange(s.shape[1])[None, :]
    return (rank < UNIVERSE_N) & (s > 0) & perp[None, :]


def _trailing_sums(x, ok, window):
    xv = np.where(ok, x, 0.0)
    n_rows = x.shape[0]
    z = np.zeros((1, x.shape[1]))
    c0 = np.concatenate([z, np.cumsum(ok.astype(float), axis=0)])
    c1 = np.concatenate([z, np.cumsum(xv, axis=0)])
    c2 = np.concatenate([z, np.cumsum(xv * xv, axis=0)])
    hi = np.arange(n_rows)
    lo = np.maximum(hi - window, 0)
    return c0[hi] - c0[lo], c1[hi] - c1[lo], c2[hi] - c2[lo]


def events(data, params):
    k_sigma = float(params["k_sigma"])
    perp, perp_qv, spot_qv, hedge, excluded = _resolve_inputs(data)
    n_rows, n_cols = perp_qv.shape
    uni = _universe(data, perp, perp_qv)

    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close) & perp[None, :]
    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log1p(ret_co) + np.log1p(ret_oc)
    r_ok = has_bar.copy()
    r_ok[0] = False
    r_ok[1:] &= has_bar[:-1]
    r_ok &= np.isfinite(r)
    cnt, s1, s2 = _trailing_sums(r, r_ok, SIGMA_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1.0)
    sigma_ok = cnt >= SIGMA_MIN_OBS
    sigma = np.sqrt(np.where(sigma_ok, np.maximum(var, 0.0), np.nan))
    sigma_ok &= sigma > 0

    candidate = uni & np.asarray(data.tradable, dtype=bool) & r_ok
    for j in excluded:
        candidate[:, j] = False
    big = np.zeros((n_rows, n_cols), dtype=bool)
    rr = np.where(r_ok, r, 0.0)
    ss = np.where(sigma_ok, sigma, np.inf)
    big[sigma_ok] = (np.abs(rr) >= k_sigma * ss)[sigma_ok]
    big &= candidate

    v_ok = np.isfinite(perp_qv) & np.isfinite(spot_qv) & (perp_qv > 0) & (spot_qv > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(v_ok, np.log(np.where(v_ok, perp_qv, 1.0) / np.where(v_ok, spot_qv, 1.0)), np.nan)
    ev_t, ev_j = np.nonzero(big & v_ok)
    a_vals = np.full(len(ev_t), np.nan)
    for i in range(len(ev_t)):
        t, j = ev_t[i], ev_j[i]
        lo = max(t - RATIO_WINDOW, 0)
        hist = v[lo:t, j]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= RATIO_MIN_OBS:
            a_vals[i] = v[t, j] - np.median(hist)
    keep = np.isfinite(a_vals)
    ev_t, ev_j, a_vals = ev_t[keep], ev_j[keep], a_vals[keep]

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    side = np.zeros((n_rows, n_cols), dtype=float)
    for t in np.unique(ev_t):
        lo = np.searchsorted(ev_t, t - POOL_WINDOW, side="left")
        hi = np.searchsorted(ev_t, t, side="left")
        if hi - lo < POOL_MIN_EVENTS:
            continue
        m = np.median(a_vals[lo:hi])
        sel = np.arange(hi, np.searchsorted(ev_t, t, side="right"))
        for i in sel:
            j = ev_j[i]
            a = a_vals[i]
            if a > m:
                s = -np.sign(r[t, j])
            elif a < m:
                s = np.sign(r[t, j])
            else:
                continue
            if s != 0:
                mask[t, j] = True
                side[t, j] = s
    return mask, side
