"""H-0163: excess ETF-specific implied-vol shocks, trade the ETF toward the feared side, SPY-hedged.

Decision after the close of row t, execution at the open of t+1 (framework convention).
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_entry": 2.0}

# (ETF, vol-index series key); SPY / VIXCLS is the market factor and the hedge, never marked.
PAIRS = (
    ("QQQ", "fred_macro.VXNCLS"),
    ("IWM", "cboe_etf_vol_1d.RVXCLS"),
    ("EEM", "cboe_etf_vol_1d.VXEEMCLS"),
    ("EWZ", "cboe_etf_vol_1d.VXEWZCLS"),
    ("FXI", "cboe_etf_vol_1d.VXFXICLS"),
    ("XLE", "cboe_etf_vol_1d.VXXLECLS"),
    ("GDX", "cboe_etf_vol_1d.VXGDXCLS"),
    ("SLV", "cboe_etf_vol_1d.VXSLVCLS"),
    ("GLD", "fred_macro.GVZCLS"),
    ("USO", "fred_macro.OVXCLS"),
    ("FXE", "cboe_etf_vol_1d.EVZCLS"),
)
MARKET_ETF = "SPY"
MARKET_VOL = "fred_macro.VIXCLS"

WINDOW = 252          # sessions in the beta window and in the factor-regression window
MIN_OBS = 200         # minimum valid observations in a window
MIN_ABS_CORR = 0.10   # feared-side guard
BETA_LO, BETA_HI = 0.0, 1.5
MAX_STALE = 2         # s may be at most 2 sessions before t
SLOT_WEIGHT = 0.30
MAX_SLOTS = 3
MAX_GROSS = 1.0


def _wsum(a, w):
    """Trailing sum over the last w rows (inclusive) along axis 0; partial windows at the start."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def _shift1(a):
    """Row k gets row k-1 (row 0 gets 0): statistics of a window that ends strictly before k."""
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out


def _session_dv(v):
    """Log change of a next_morning vol series, indexed by the session it belongs to.

    Row r of the aligned series holds the value of session r-1. A value is new at row r when it is
    finite, positive and differs from the last earlier value (a repeat is treated as a carried, stale
    value). dv[k] = ln v(k) - ln v(previous value), for session k = r-1; NaN where session k has no
    new value. The last session (value would arrive on the next row) is NaN.
    """
    v = np.asarray(v, dtype=np.float64)
    T = v.shape[0]
    out = np.full(T, np.nan)
    if T < 2:
        return out
    finite = np.isfinite(v) & (v > 0)
    idx = np.where(finite, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    prev_idx = np.full(T, -1)
    prev_idx[1:] = last[:-1]
    prev_val = np.where(prev_idx >= 0, v[np.maximum(prev_idx, 0)], np.nan)
    ok = finite & (prev_idx >= 0) & (v != prev_val)
    lv = np.log(np.where(ok, v, 1.0)) - np.log(np.where(ok, prev_val, 1.0))
    d_row = np.where(ok, lv, np.nan)
    out[:-1] = d_row[1:]
    return out


def _daily_ret(data):
    """Close-to-close total return per row from ret_co and ret_oc; valid only with a close on t and t-1."""
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    r = (1.0 + co) * (1.0 + oc) - 1.0
    has = np.isfinite(close)
    ok = np.zeros_like(has)
    ok[1:] = has[1:] & has[:-1]
    ok &= np.isfinite(r)
    return np.where(ok, r, 0.0), ok


def _events(data, z_entry):
    """Per decision row t and pair p: event flag, side, |eps| and the SPY beta used for the hedge."""
    T = len(data.dates)
    P = len(PAIRS)
    spy = data.col(MARKET_ETF)
    cols = np.array([data.col(etf) for etf, _ in PAIRS], dtype=int)
    vix = np.asarray(data.series[MARKET_VOL], dtype=np.float64)
    dm = _session_dv(vix)                                                   # (T,)
    dv = np.column_stack([_session_dv(data.series[key]) for _, key in PAIRS])  # (T, P)

    R, Rok = _daily_ret(data)
    rs, rs_ok = R[:, spy], Rok[:, spy]
    ri, ri_ok = R[:, cols], Rok[:, cols]

    with np.errstate(divide="ignore", invalid="ignore"):
        # (2) beta of the ETF on SPY over the WINDOW sessions ending s (inclusive), clipped
        mb = ri_ok & rs_ok[:, None]
        xs = np.where(mb, rs[:, None], 0.0)
        ys = np.where(mb, ri, 0.0)
        n = _wsum(mb.astype(np.float64), WINDOW)
        sx, sy = _wsum(xs, WINDOW), _wsum(ys, WINDOW)
        sxx, sxy = _wsum(xs * xs, WINDOW), _wsum(xs * ys, WINDOW)
        vxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        beta = np.where((n >= MIN_OBS) & (vxx > 0), cxy / vxx, np.nan)
        beta = np.clip(beta, BETA_LO, BETA_HI)
        x = np.where(mb & np.isfinite(beta), ri - beta * rs[:, None], np.nan)

        # observation s is usable when both vol changes exist and the residual return exists
        avail = np.isfinite(dv) & np.isfinite(dm)[:, None]
        ok = avail & np.isfinite(x)
        dmb = np.broadcast_to(dm[:, None], dv.shape)
        Y = np.where(ok, dv, 0.0)
        M = np.where(ok, dmb, 0.0)
        X = np.where(ok, x, 0.0)
        Rr = np.where(ok, ri, 0.0)
        one = ok.astype(np.float64)

        # (3) regression window: the WINDOW sessions strictly before s
        def S(a):
            return _shift1(_wsum(a, WINDOW))

        nn = S(one)
        Sm, Sx, Sy, Sr = S(M), S(X), S(Y), S(Rr)
        Smm_c = S(M * M) - Sm * Sm / nn
        Sxx_c = S(X * X) - Sx * Sx / nn
        Smx_c = S(M * X) - Sm * Sx / nn
        Smy_c = S(M * Y) - Sm * Sy / nn
        Sxy_c = S(X * Y) - Sx * Sy / nn
        Syy_c = S(Y * Y) - Sy * Sy / nn
        Srr_c = S(Rr * Rr) - Sr * Sr / nn
        Syr_c = S(Y * Rr) - Sy * Sr / nn
        det = Smm_c * Sxx_c - Smx_c * Smx_c
        good = (nn >= MIN_OBS) & (Smm_c > 0) & (Sxx_c > 0) & (det > 1e-12 * Smm_c * Sxx_c)
        c = (Sxx_c * Smy_c - Smx_c * Sxy_c) / det
        g = (Smm_c * Sxy_c - Smx_c * Smy_c) / det
        a = (Sy - c * Sm - g * Sx) / nn
        sse = Syy_c - c * Smy_c - g * Sxy_c
        sd = np.sqrt(np.maximum(sse, 0.0) / (nn - 3.0))
        corr = Syr_c / np.sqrt(Syy_c * Srr_c)
        good &= (sd > 0) & (Syy_c > 0) & (Srr_c > 0) & np.isfinite(corr)
        eps = np.where(good & ok, (dv - a - c * dmb - g * x) / sd, np.nan)
        fear = np.where(good, np.sign(corr), 0.0)
        fear_ok = good & (np.abs(corr) >= MIN_ABS_CORR)

    # (1) observation day: newest session s <= t-1 with both vol changes; at most MAX_STALE back;
    # each observation is acted on once (at the first decision where it is usable).
    rows = np.arange(T)
    idx = np.where(avail, rows[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0) if T else idx
    s_t = np.full((T, P), -1)
    s_t[1:] = last[:-1]
    s_prev = np.full((T, P), -1)
    s_prev[1:] = s_t[:-1]
    new = (s_t >= 0) & (rows[:, None] - s_t <= MAX_STALE) & (s_t != s_prev)
    sc = np.maximum(s_t, 0)
    pc = np.broadcast_to(np.arange(P)[None, :], (T, P))
    e = eps[sc, pc]
    f = fear[sc, pc]
    fok = fear_ok[sc, pc]
    b = beta[sc, pc]
    with np.errstate(invalid="ignore"):
        fire = new & np.isfinite(e) & fok & (np.abs(e) >= z_entry) & np.isfinite(b)
    # (4) eps >= z: toward the feared side f; eps <= -z: the other way
    side = np.where(fire, np.where(e > 0, f, -f), 0.0)
    fire &= side != 0
    return fire, side, np.where(fire, np.abs(e), 0.0), np.where(fire, b, 0.0), cols, spy


def _has_bar(data):
    has = np.isfinite(np.asarray(data.close, dtype=np.float64)) & np.asarray(data.listed, dtype=bool)
    if data.universe is not None:
        has &= np.asarray(data.universe, dtype=bool)
    return has


def _book(fire, side, abs_eps, beta, cols, spy, has_bar, hold_days, n_instruments):
    """(5) Event slots: ETF side*0.30, SPY -side*beta*0.30, weight rows t..t+hold_days-1."""
    T, P = fire.shape
    W = np.zeros((T, n_instruments))
    slots = []   # (pair, side, beta, first row with zero weight)
    for t in range(T):
        slots = [s for s in slots if s[3] > t]
        busy = {s[0] for s in slots}
        free = MAX_SLOTS - len(slots)
        if free > 0:
            cand = [(-abs_eps[t, p], p) for p in np.flatnonzero(fire[t])
                    if p not in busy and has_bar[t, cols[p]]]
            cand.sort()
            for _, p in cand[:free]:
                slots.append((p, side[t, p], beta[t, p], t + hold_days))
        if not slots:
            continue
        for p, sd, bt, _ in slots:
            W[t, cols[p]] += sd * SLOT_WEIGHT
            W[t, spy] -= sd * bt * SLOT_WEIGHT
        gross = np.abs(W[t]).sum()
        if gross > MAX_GROSS:
            W[t] *= MAX_GROSS / gross
    return W


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    z_entry = float(params["z_entry"])
    fire, side, abs_eps, beta, cols, spy = _events(data, z_entry)
    return _book(fire, side, abs_eps, beta, cols, spy, _has_bar(data), hold_days, len(data.instruments))
