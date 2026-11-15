"""H-0161: home-closure down-move tilt on a 60/40 SPY/IEF core.

Decision after the US close of row t, executed at the open of row t+1.
"""
import numpy as np

PARAMS = {"slot_weight": 0.3, "z_threshold": 1.0}

# (1) home exchange map: tilt ETF -> closed_next flag of its home exchange
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

MAX_WINDOW = 10      # unanchored window capped at the last 10 sessions
VOL_WINDOW = 63      # sigma over s = t-63 .. t-1
MIN_VALID = 50       # at least 50 valid residuals
MIN_HISTORY = 63     # sessions of price history before an ETF may enter
MAX_HOLD = 5         # closed at the open after the 5th held session
EQUITY_SLEEVE = 0.60
BOND_WEIGHT = 0.40


def _prefix(x):
    """S[k] = sum of x[0..k-1] along axis 0 (S has T+1 rows)."""
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def signal_parts(data, params):
    """Returns flags (T, J), z (T, J), n (T, J) for the tilt ETFs in TILT order."""
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

    cols_s = np.concatenate([cols, [spy]])               # ETFs + SPY (last)
    co_s = co[:, cols_s]
    oc_s = oc[:, cols_s]
    full = co_s + oc_s                                    # close(s-1) -> close(s)
    bad_full = ~np.isfinite(full)
    S_full = _prefix(np.where(bad_full, 0.0, full))
    S_bad = _prefix(bad_full.astype(np.float64))
    bad_oc = ~np.isfinite(oc_s)
    oc_z = np.where(bad_oc, 0.0, oc_s)

    # (3) sigma_j(t): std of d_j(s) = oc_j(s) - oc_SPY(s), s = t-63..t-1
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

    # (8) at least 63 sessions of price history before t
    has_px = np.isfinite(close[:, cols])
    Sh = _prefix(has_px.astype(np.float64))
    hist_ok = Sh[idx] >= MIN_HISTORY                      # rows 0..t-1

    flags = np.full((T, J), np.nan)
    z = np.full((T, J), np.nan)
    n_out = np.zeros((T, J), dtype=int)
    cache = {}
    for jj, name in enumerate(TILT):
        key = HOME[name]
        if key not in cache:
            f = np.asarray(data.series[key], dtype=np.float64)
            if f.shape[0] != T:
                raise ValueError("series length mismatch: " + key)
            # (2) a = most recent s <= t with closed_next(s-1) = 0 (NaN counts as anchored)
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
        n_out[:, jj] = n

        def window_move(k):
            # ln(1+oc(a)) + sum_{s=a+1..t} [ln(1+co(s)) + ln(1+oc(s))]
            u = oc_z[a, k] + (S_full[idx + 1, k] - S_full[a + 1, k])
            nbad = bad_oc[a, k].astype(np.float64) + (S_bad[idx + 1, k] - S_bad[a + 1, k])
            return np.where(nbad > 0, np.nan, u)

        u_j = window_move(jj)
        u_spy = window_move(J)
        with np.errstate(invalid="ignore", divide="ignore"):
            zj = (u_j - u_spy) / (sigma[:, jj] * np.sqrt(n))
        ok = sig_ok[:, jj] & hist_ok[:, jj] & np.isfinite(zj)
        z[:, jj] = np.where(ok, zj, np.nan)
    return flags, z, n_out


def run_book(data, params):
    """State machine over rows. Returns weights (T, N) and the entry-order mask (T, J)."""
    slot = float(params["slot_weight"])
    thr = float(params["z_threshold"])
    flags, z, _ = signal_parts(data, params)
    T = len(data.dates)
    N = len(data.instruments)
    cols = np.array([data.col(k) for k in TILT], dtype=int)
    spy = data.col("SPY")
    ief = data.col("IEF")
    J = len(cols)
    tradable = np.asarray(data.tradable, dtype=bool)
    trad_j = tradable[:, cols]

    with np.errstate(invalid="ignore"):
        cand = (flags == 1.0) & np.isfinite(z) & (z <= -thr)
    if data.universe is not None:
        cand &= np.asarray(data.universe, dtype=bool)[:, cols]
    stays = flags == 1.0                                   # NaN or 0 -> exit

    W = np.zeros((T, N), dtype=np.float64)
    entries = np.zeros((T, J), dtype=bool)
    held = np.zeros(J, dtype=bool)            # position filled and open
    entry_row = np.full(J, -1, dtype=int)     # row of the entry fill (open)
    pend_in = np.zeros(J, dtype=bool)         # entry ordered at previous row
    pend_out = np.zeros(J, dtype=bool)        # exit ordered, not yet filled
    for t in range(T):
        tr = trad_j[t]
        # resolve orders placed at row t-1 (fills at the open of row t)
        filled_in = pend_in & tr
        held = held | filled_in
        entry_row = np.where(filled_in, t, entry_row)
        pend_in[:] = False                    # unfilled entry: cancelled
        done_out = pend_out & tr
        pend_out = pend_out & ~done_out       # untradable: closed at next tradable open
        # (6) exits at the close of row t
        sessions = t - entry_row + 1
        exit_now = held & ((~stays[t]) | (sessions >= MAX_HOLD))
        held = held & ~exit_now
        pend_out = pend_out | exit_now
        # (5) entries: not held, not still being closed, no pending order
        new = cand[t] & ~held & ~pend_out & ~exit_now
        entries[t] = new
        pend_in = new.copy()
        # (7) weights
        active = held | new
        k = int(active.sum())
        w = slot if k * slot <= EQUITY_SLEEVE else EQUITY_SLEEVE / k
        row = np.zeros(N)
        row[cols[active]] = w
        row[spy] = max(0.0, EQUITY_SLEEVE - w * k)        # clip float rounding at the cap
        row[ief] = BOND_WEIGHT
        W[t] = row
    return W, entries


def target_weights(data, params):
    W, _ = run_book(data, params)
    return W
