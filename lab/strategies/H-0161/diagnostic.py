"""H-0161 mechanism test: an Asia-Pacific country ETF enters the tilt (copy of the strategy pipeline)."""
import numpy as np

HOME = {
    "EWJ": "exchange_holidays_1d.XTKS.closed_next",
    "EWH": "exchange_holidays_1d.XHKG.closed_next",
    "FXI": "exchange_holidays_1d.XHKG.closed_next",
    "EWY": "exchange_holidays_1d.XKRX.closed_next",
    "EWT": "exchange_holidays_1d.XTAI.closed_next",
    "EWS": "exchange_holidays_1d.XSES.closed_next",
    "EWM": "exchange_holidays_1d.XKLS.closed_next",
    "EWA": "exchange_holidays_1d.XASX.closed_next",
    "THD": "exchange_holidays_1d.XBKK.closed_next",
    "EIDO": "exchange_holidays_1d.XIDX.closed_next",
    "EPHE": "exchange_holidays_1d.XPHS.closed_next",
    "ENZL": "exchange_holidays_1d.XNZE.closed_next",
    "INDA": "exchange_holidays_1d.XBOM.closed_next",
}
TILT = tuple(HOME.keys())

MAX_WINDOW = 10
VOL_WINDOW = 63
MIN_VALID = 50
MIN_HISTORY = 63
MAX_HOLD = 5


def _prefix(x):
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def _signal_parts(data):
    T = len(data.dates)
    cols = np.array([data.col(k) for k in TILT], dtype=int)
    spy = data.col("SPY")
    J = len(cols)
    idx = np.arange(T)

    with np.errstate(invalid="ignore", divide="ignore"):
        co = np.log1p(np.asarray(data.ret_co, dtype=np.float64))
        oc = np.log1p(np.asarray(data.ret_oc, dtype=np.float64))
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)

    cols_s = np.concatenate([cols, [spy]])
    co_s = co[:, cols_s]
    oc_s = oc[:, cols_s]
    full = co_s + oc_s
    bad_full = ~np.isfinite(full)
    S_full = _prefix(np.where(bad_full, 0.0, full))
    S_bad = _prefix(bad_full.astype(np.float64))
    bad_oc = ~np.isfinite(oc_s)
    oc_z = np.where(bad_oc, 0.0, oc_s)

    d = oc[:, cols] - oc[:, [spy]]
    valid = np.isfinite(d) & tradable[:, cols] & tradable[:, [spy]]
    dz = np.where(valid, d, 0.0)
    Sv = _prefix(valid.astype(np.float64))
    Sd = _prefix(dz)
    Sq = _prefix(dz * dz)
    lo = np.maximum(idx - VOL_WINDOW, 0)
    cnt = Sv[idx] - Sv[lo]
    sm = Sd[idx] - Sd[lo]
    sq = Sq[idx] - Sq[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sm / cnt
        var = (sq - cnt * mean * mean) / (cnt - 1.0)
    sigma = np.sqrt(np.maximum(var, 0.0))
    sig_ok = (idx[:, None] >= VOL_WINDOW) & (cnt >= MIN_VALID) & (sigma > 0)

    has_px = np.isfinite(close[:, cols])
    Sh = _prefix(has_px.astype(np.float64))
    hist_ok = Sh[idx] >= MIN_HISTORY

    flags = np.full((T, J), np.nan)
    z = np.full((T, J), np.nan)
    cache = {}
    for jj, name in enumerate(TILT):
        key = HOME[name]
        if key not in cache:
            f = np.asarray(data.series[key], dtype=np.float64)
            if f.shape[0] != T:
                raise ValueError("series length mismatch: " + key)
            closed_prev = np.zeros(T, dtype=bool)
            closed_prev[1:] = f[:-1] == 1.0
            anch = np.where(~closed_prev, idx, 0)
            last = np.maximum.accumulate(anch)
            a = np.maximum(last, idx - (MAX_WINDOW - 1))
            a = np.maximum(a, 0)
            cache[key] = (f, a)
        f, a = cache[key]
        flags[:, jj] = f
        n = idx - a + 1

        def window_move(k):
            u = oc_z[a, k] + (S_full[idx + 1, k] - S_full[a + 1, k])
            nbad = bad_oc[a, k].astype(np.float64) + (S_bad[idx + 1, k] - S_bad[a + 1, k])
            return np.where(nbad > 0, np.nan, u)

        u_j = window_move(jj)
        u_spy = window_move(J)
        with np.errstate(invalid="ignore", divide="ignore"):
            zj = (u_j - u_spy) / (sigma[:, jj] * np.sqrt(n))
        ok = sig_ok[:, jj] & hist_ok[:, jj] & np.isfinite(zj)
        z[:, jj] = np.where(ok, zj, np.nan)
    return flags, z


def _entries(data, params):
    thr = float(params["z_threshold"])
    flags, z = _signal_parts(data)
    T = len(data.dates)
    cols = np.array([data.col(k) for k in TILT], dtype=int)
    J = len(cols)
    trad_j = np.asarray(data.tradable, dtype=bool)[:, cols]
    with np.errstate(invalid="ignore"):
        cand = (flags == 1.0) & np.isfinite(z) & (z <= -thr)
    if data.universe is not None:
        cand &= np.asarray(data.universe, dtype=bool)[:, cols]
    stays = flags == 1.0

    entries = np.zeros((T, J), dtype=bool)
    held = np.zeros(J, dtype=bool)
    entry_row = np.full(J, -1, dtype=int)
    pend_in = np.zeros(J, dtype=bool)
    pend_out = np.zeros(J, dtype=bool)
    for t in range(T):
        tr = trad_j[t]
        filled_in = pend_in & tr
        held = held | filled_in
        entry_row = np.where(filled_in, t, entry_row)
        pend_in[:] = False
        pend_out = pend_out & ~(pend_out & tr)
        sessions = t - entry_row + 1
        exit_now = held & ((~stays[t]) | (sessions >= MAX_HOLD))
        held = held & ~exit_now
        pend_out = pend_out | exit_now
        new = cand[t] & ~held & ~pend_out & ~exit_now
        entries[t] = new
        pend_in = new.copy()
    return entries, cols


def events(data, params):
    entries, cols = _entries(data, params)
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    mask[:, cols] = entries
    mask &= np.asarray(data.tradable, dtype=bool)
    if data.universe is not None:
        mask &= np.asarray(data.universe, dtype=bool)
    side = np.where(mask, 1.0, 0.0)
    return mask, side
