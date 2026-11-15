"""H-0163 mechanism events: same signal code as strategy.py (copied, diagnostic.py is a separate file)."""
import numpy as np

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

WINDOW = 252
MIN_OBS = 200
MIN_ABS_CORR = 0.10
BETA_LO, BETA_HI = 0.0, 1.5
MAX_STALE = 2


def _wsum(a, w):
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def _shift1(a):
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out


def _session_dv(v):
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
    T = len(data.dates)
    P = len(PAIRS)
    spy = data.col(MARKET_ETF)
    cols = np.array([data.col(etf) for etf, _ in PAIRS], dtype=int)
    vix = np.asarray(data.series[MARKET_VOL], dtype=np.float64)
    dm = _session_dv(vix)
    dv = np.column_stack([_session_dv(data.series[key]) for _, key in PAIRS])

    R, Rok = _daily_ret(data)
    rs, rs_ok = R[:, spy], Rok[:, spy]
    ri, ri_ok = R[:, cols], Rok[:, cols]

    with np.errstate(divide="ignore", invalid="ignore"):
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

        avail = np.isfinite(dv) & np.isfinite(dm)[:, None]
        ok = avail & np.isfinite(x)
        dmb = np.broadcast_to(dm[:, None], dv.shape)
        Y = np.where(ok, dv, 0.0)
        M = np.where(ok, dmb, 0.0)
        X = np.where(ok, x, 0.0)
        Rr = np.where(ok, ri, 0.0)
        one = ok.astype(np.float64)

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
    side = np.where(fire, np.where(e > 0, f, -f), 0.0)
    fire &= side != 0
    return fire, side, cols


def events(data, params):
    z_entry = float(params["z_entry"])
    fire, side_p, cols = _events(data, z_entry)
    T, N = len(data.dates), len(data.instruments)
    has = (np.isfinite(np.asarray(data.close, dtype=np.float64))
           & np.asarray(data.listed, dtype=bool) & np.asarray(data.tradable, dtype=bool))
    if data.universe is not None:
        has &= np.asarray(data.universe, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for p in range(len(PAIRS)):
        j = cols[p]
        m = fire[:, p] & has[:, j]
        mask[:, j] = m
        side[:, j] = np.where(m, side_p[:, p], 0.0)
    return mask, np.where(mask, side, 1.0)
