"""H-0071 Leverage-led vs spot-led big moves in Binance alts.

After a big close-to-close move (|r| >= k_sigma * trailing 30-day sigma) of a top-30 alt perp, compare the
coin's abnormal perp/spot quote-volume ratio with the pooled median of that quantity over the big-move events
of the previous 180 days: leverage-led (above) -> fade the move, spot-led (below) -> follow it. Each event is a
fixed slot of 0.125 for hold_days days, the net is hedged with BTCUSDT.P, gross capped at 1 by proportional
scaling.

Inputs (daily, one calendar of UTC days):
- perps (binance_perp_1d, ids <SYMBOL>.P, asset_class crypto_perp): ret_co, ret_oc, close, dollar_volume
  (= quote volume in USDT), tradable, universe (PIT top 30).
- spot pairs (binance_spot_1d, id = perp id without '.P'): dollar_volume (= spot quote volume), signal only.
"""
import numpy as np

PARAMS = {"hold_days": 3, "k_sigma": 2.0}

HEDGE_ID = "BTCUSDT.P"
EXCLUDED_IDS = ("BTCUSDT.P", "ETHUSDT.P")
PERP_SUFFIX = ".P"
PERP_CLASS = "crypto_perp"

SLOT_WEIGHT = 0.125
SIGMA_WINDOW = 30          # days t-30..t-1
SIGMA_MIN_OBS = 25
RATIO_WINDOW = 30          # days t-30..t-1
RATIO_MIN_OBS = 20
POOL_WINDOW = 180          # event days t-180..t-1
POOL_MIN_EVENTS = 30
UNIVERSE_N = 30            # only used if the gate supplies no universe (card notes, K-0054)
UNIVERSE_VOLUME_WINDOW = 30


def _resolve_inputs(data):
    """Perp mask, perp and spot quote volume per perp column, hedge column, excluded columns.

    Real data: perps are named <SYMBOL>.P and the spot pair <SYMBOL> is a column of the same view. Missing
    inputs raise. The synthetic market of `lab try` has no '.P' ids, no spot pairs and no BTC; only there a
    stand-in spot volume (the perp's own volume of the previous day) is used and there is no hedge.
    """
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
    # card notes: PIT top 30 perps by trailing 30-day quote volume (rows t-29..t)
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
    """Sums over rows t-window..t-1 (current row excluded): count, sum, sum of squares."""
    xv = np.where(ok, x, 0.0)
    n_rows = x.shape[0]
    z = np.zeros((1, x.shape[1]))
    c0 = np.concatenate([z, np.cumsum(ok.astype(float), axis=0)])
    c1 = np.concatenate([z, np.cumsum(xv, axis=0)])
    c2 = np.concatenate([z, np.cumsum(xv * xv, axis=0)])
    hi = np.arange(n_rows)
    lo = np.maximum(hi - window, 0)
    return c0[hi] - c0[lo], c1[hi] - c1[lo], c2[hi] - c2[lo]


def compute_events(data, params):
    """Event mask (T, N), side (T, N), universe (T, N), hedge column, per the card's rules (1)-(4)."""
    k_sigma = float(params["k_sigma"])
    perp, perp_qv, spot_qv, hedge, excluded = _resolve_inputs(data)
    n_rows, n_cols = perp_qv.shape
    uni = _universe(data, perp, perp_qv)

    # (1) big move: log close-to-close return against the std of t-30..t-1 (>= 25 valid days)
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

    # (2) venue ratio v = log(perp qv / spot qv); a = v - median(v over t-30..t-1), >= 20 valid days
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

    # (3) reference level: median of a over all big-move events with event day in t-180..t-1 (>= 30)
    # (4) side: leverage-led (a > m) -> -sign(r); spot-led (a < m) -> +sign(r); a == m -> no event
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
    return mask, side, uni, hedge


def target_weights(data, params):
    hold = int(params["hold_days"])
    mask, side, uni, hedge = compute_events(data, params)
    n_rows, n_cols = mask.shape
    weights = np.zeros((n_rows, n_cols), dtype=float)
    cur_side = np.zeros(n_cols)
    last_row = np.full(n_cols, -1, dtype=np.int64)
    for t in range(n_rows):
        ev = mask[t]
        if ev.any():
            # (5) a new event opens / replaces the coin's position, hold restarts
            cur_side[ev] = side[t, ev]
            last_row[ev] = t + hold - 1
        # held on rows t..t+hold-1 -> filled at open t+1, closed at open t+1+hold
        expired = last_row < t
        # leaving the universe closes the position at the first decision row outside it (K-0051)
        gone = ~uni[t]
        drop = expired | gone
        cur_side[drop] = 0.0
        last_row[drop] = -1
        w = cur_side * SLOT_WEIGHT
        if hedge is not None:
            net = w.sum()
            w[hedge] = -net if uni[t, hedge] else 0.0
        gross = np.abs(w).sum()
        if gross > 1.0:
            w = w / gross
        weights[t] = w
    return weights
