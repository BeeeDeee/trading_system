"""H-0175 mechanism test: first crossing of the 5-session beta-adjusted residual z-score of a closed-end
fund against its anchor ETF to <= -z_entry, eligibility and de-duplication exactly as steps 1-4 of the
signal (same code as strategy.py). Only the funds are marked; anchors are unmarked peers. Side +1."""
import numpy as np

PAIRS = (
    ("ADX", "SPY"), ("GAB", "SPY"), ("USA", "SPY"), ("TY", "SPY"), ("GAM", "SPY"),
    ("RVT", "IWM"), ("RMT", "IWM"),
    ("NAD", "MUB"), ("NEA", "MUB"), ("NUV", "MUB"), ("NZF", "MUB"), ("NVG", "MUB"), ("NMZ", "MUB"),
    ("MQY", "MUB"), ("MYI", "MUB"), ("BFK", "MUB"), ("VKQ", "MUB"), ("PML", "MUB"), ("PMF", "MUB"),
    ("PHK", "HYG"), ("DHY", "HYG"), ("HIX", "HYG"), ("EHI", "HYG"), ("DSU", "HYG"), ("HYT", "HYG"),
    ("PCN", "HYG"), ("PTY", "HYG"),
    ("RQI", "IYR"), ("RNP", "IYR"), ("IGR", "IYR"), ("JRS", "IYR"),
    ("UTF", "XLU"), ("UTG", "XLU"), ("DNP", "XLU"), ("GUT", "XLU"),
    ("MXF", "EWW"), ("TWN", "EWT"), ("KF", "EWY"), ("GF", "EWG"), ("SWZ", "EWL"), ("JOF", "EWJ"),
    ("TDF", "EEM"), ("EMF", "EEM"), ("IFN", "EEM"),
)

BETA_WINDOW = 252
BETA_GAP = 5
BETA_MIN_OBS = 189
BETA_LO, BETA_HI = 0.3, 2.5
FORM = 5
LIQ_WINDOW = 63
LIQ_MIN = 500_000.0


def _wsum(x, w):
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cs[1:] = np.cumsum(x, axis=0)
    lo = np.maximum(np.arange(T) - w + 1, 0)
    return cs[1:] - cs[lo]


def _daily_returns(data):
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    bar = listed & np.isfinite(close)
    valid = np.zeros_like(bar)
    valid[1:] = bar[1:] & bar[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, 0.0), valid


def _signal(data, params):
    hold = int(params["hold_sessions"])
    ze = float(params["z_entry"])
    fc = np.array([data.col(f) for f, _ in PAIRS], dtype=np.int64)
    ac = np.array([data.col(a) for _, a in PAIRS], dtype=np.int64)
    T = len(data.dates)
    P = len(PAIRS)

    r, valid = _daily_returns(data)
    rf, ra = r[:, fc], r[:, ac]
    vf, va = valid[:, fc], valid[:, ac]
    ra_lag = np.zeros_like(ra)
    ra_lag[1:] = ra[:-1]
    va_lag = np.zeros_like(va)
    va_lag[1:] = va[:-1]

    m = vf & va & va_lag
    y = np.where(m, rf, 0.0)
    x1 = np.where(m, ra, 0.0)
    x2 = np.where(m, ra_lag, 0.0)
    W = BETA_WINDOW
    n = _wsum(m.astype(np.float64), W)
    sy, s1, s2 = _wsum(y, W), _wsum(x1, W), _wsum(x2, W)
    s11, s22, s12 = _wsum(x1 * x1, W), _wsum(x2 * x2, W), _wsum(x1 * x2, W)
    s1y, s2y, syy = _wsum(x1 * y, W), _wsum(x2 * y, W), _wsum(y * y, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        nn = np.where(n > 0, n, np.nan)
        c11 = s11 - s1 * s1 / nn
        c22 = s22 - s2 * s2 / nn
        c12 = s12 - s1 * s2 / nn
        c1y = s1y - s1 * sy / nn
        c2y = s2y - s2 * sy / nn
        cyy = syy - sy * sy / nn
        det = c11 * c22 - c12 * c12
        b0 = (c1y * c22 - c2y * c12) / det
        b1 = (c2y * c11 - c1y * c12) / det
        b = np.clip(b0 + b1, BETA_LO, BETA_HI)
        var_u = (cyy - 2.0 * b * c1y + b * b * c11) / (nn - 1.0)
        ok = (n >= BETA_MIN_OBS) & (c11 > 0) & (c22 > 0) & (det > 1e-10 * c11 * c22)
        ok &= np.isfinite(b) & np.isfinite(var_u) & (var_u > 0)
        beta_e = np.where(ok, b, np.nan)
        sig_e = np.where(ok, np.sqrt(np.where(ok, var_u, 1.0)), np.nan)
    beta = np.full((T, P), np.nan)
    sig = np.full((T, P), np.nan)
    if T > BETA_GAP:
        beta[BETA_GAP:] = beta_e[:-BETA_GAP]
        sig[BETA_GAP:] = sig_e[:-BETA_GAP]

    vfa = vf & va
    cnt = _wsum(vfa.astype(np.float64), FORM)
    sum_f = _wsum(np.where(vfa, rf, 0.0), FORM)
    sum_a = _wsum(np.where(vfa, ra, 0.0), FORM)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (sum_f - beta * sum_a) / (sig * np.sqrt(float(FORM)))
    z = np.where(cnt >= FORM, z, np.nan)

    trad = np.asarray(data.tradable, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)[:, fc]
    with np.errstate(invalid="ignore"):
        liquid_day = np.isfinite(dv) & (dv >= LIQ_MIN)
    liq_cnt = _wsum(liquid_day.astype(np.float64), LIQ_WINDOW)
    liq_ok = (liq_cnt >= (LIQ_WINDOW // 2 + 1)) & (np.arange(T) >= LIQ_WINDOW - 1)[:, None]
    elig = trad[:, fc] & trad[:, ac] & liq_ok & (cnt >= FORM) & np.isfinite(z)
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
        elig &= uni[:, fc] & uni[:, ac]

    zprev = np.full((T, P), np.nan)
    zprev[1:] = z[:-1]
    with np.errstate(invalid="ignore"):
        cand = elig & (z <= -ze) & np.isfinite(zprev) & (zprev > -ze)
    ev = np.zeros((T, P), dtype=bool)
    for p in range(P):
        last = None
        for t in np.flatnonzero(cand[:, p]):
            if last is None or t - last > hold:
                ev[t, p] = True
                last = t
    return ev, z, beta, fc, ac


def events(data, params):
    ev, _z, _beta, fc, _ac = _signal(data, params)
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    mask[:, fc] = ev
    return mask, None
