"""H-0049: flushed vs unflushed crashes in crypto perps.

After the close of day t: among eligible top-30 perps (not BTC/ETH) with a bottom-tercile day-t return that is also
<= -k_sigma * own 30-day vol, go long (+0.10) when day-t funding is negative and a new 7-day low (FLUSH), short
(-0.10) when day-t funding is >= the median of the previous 7 days (UNFLUSHED). Slots last hold_days days, at most
one slot per perp, total slot gross capped at 0.5, hedge -N/2 in BTCUSDT.P and ETHUSDT.P.
"""
import numpy as np

PARAMS = {"hold_days": 3, "k_sigma": 2.0}

SLOT_WEIGHT = 0.10
MAX_SLOT_GROSS = 0.5
VOL_WINDOW = 30
FUNDING_WINDOW = 7
PCTL = 33.3
TOP_N = 30


def hedge_cols(data):
    """Columns of the hedge instruments BTCUSDT.P and ETHUSDT.P, looked up by name (raises if missing).
    Only on the lab's synthetic market (every instrument named SYN..., which has no BTC/ETH) the two
    alphabetically first instruments stand in, so that `lab try` can exercise the code path."""
    names = tuple(data.instruments)
    if "BTCUSDT.P" not in names and "ETHUSDT.P" not in names and all(s.startswith("SYN") for s in names):
        stand_in = sorted(names)[:2]
        return data.col(stand_in[0]), data.col(stand_in[1])
    return data.col("BTCUSDT.P"), data.col("ETHUSDT.P")


def _universe(data):
    """The PIT top-30 universe from the gate; if the gate passes none, the card's own definition
    (top 30 by trailing 30-day quote volume, rows t-29..t)."""
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


def compute_events(data, params):
    """Returns (side, ok): side (T, N) in {+1, -1, 0} for FLUSH / UNFLUSHED / no event on day t, and ok (T, N),
    the perp is a universe member with a price on row t (a slot may be held)."""
    k_sigma = float(params["k_sigma"])
    close = np.asarray(data.close, dtype=np.float64)
    T, N = close.shape
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    univ = _universe(data)
    funding = np.asarray(data.extras["funding"], dtype=np.float64)
    hcols = hedge_cols(data)

    # r(j,t): close-to-close total return of day t; valid when there is a close on t-1 and on t.
    has_px = np.isfinite(close) & listed
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.zeros((T, N), dtype=bool)
    valid[1:] = has_px[1:] & has_px[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)

    # sigma(j,t): sample std (ddof 1) of r over rows t-30..t-1, all 30 must be valid.
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

    # 33.3rd percentile of r(., t) across the eligible universe (BTC/ETH included), linear interpolation.
    re = np.where(eligible, r, np.nan)
    pct = np.full(T, np.nan)
    has = eligible.any(axis=1)
    if has.any():
        pct[has] = np.nanpercentile(re[has], PCTL, axis=1)

    cand = eligible & tradable & (re <= pct[:, None]) & (re <= -k_sigma * sigma)
    for hc in hcols:
        cand[:, hc] = False

    # Funding classification: f(t) and f(t-7..t-1) must all be known.
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

    ok = univ & has_px
    return side, ok


def target_weights(data, params):
    hold = int(params["hold_days"])
    side_ev, ok = compute_events(data, params)
    T, N = side_ev.shape
    univ = _universe(data)
    hb, he = hedge_cols(data)

    w = np.zeros((T, N))
    last_row = np.full(N, -1, dtype=np.int64)   # last decision row on which the slot is held
    slot_side = np.zeros(N)
    for t in range(T):
        active = last_row >= t
        # a slot whose perp left the universe or has no price is closed at this row
        last_row[active & ~ok[t]] = -1
        new = (side_ev[t] != 0.0) & ~(last_row >= t)
        last_row[new] = t + hold - 1
        slot_side[new] = side_ev[t, new]
        active = last_row >= t
        sw = np.where(active, SLOT_WEIGHT * slot_side, 0.0)
        g = np.abs(sw).sum()
        if g > MAX_SLOT_GROSS:
            sw = sw * (MAX_SLOT_GROSS / g)
        net = sw.sum()
        row = sw
        # hedge: -N/2 in each of BTC and ETH (only universe members may be held)
        row[hb] = -net / 2.0 if univ[t, hb] else 0.0
        row[he] = -net / 2.0 if univ[t, he] else 0.0
        w[t] = row
    return w
