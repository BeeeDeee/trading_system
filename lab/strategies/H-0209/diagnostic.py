"""H-0209 mechanism event: a producer ETF whose formation_days residual to its commodity ETF and SPY (betas from
the trailing 252 sessions) first crosses beyond z_entry residual sds after the close of t.
Side +1 when the producer lagged (z <= -z_entry), -1 when it ran ahead (z >= +z_entry).
The code below is a copy of the signal pipeline in strategy.py (diagnostic.py is a separate file)."""
import numpy as np

PAIRS = (
    ("GDX", "GLD"),
    ("GDXJ", "GLD"),
    ("SIL", "SLV"),
    ("XOP", "USO"),
    ("XLE", "USO"),
    ("FCG", "UNG"),
)
MARKET = "SPY"

BETA_WINDOW = 252
MIN_OBS = 200
MIN_HISTORY = 252


def rolling_sum(x, n):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    n = int(n)
    if x.shape[0] >= n:
        c = np.cumsum(x)
        out[n - 1] = c[n - 1]
        out[n:] = c[n:] - c[:-n]
    return out


def daily_returns(data):
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def has_bar(data):
    close = np.asarray(data.close, dtype=np.float64)
    return np.asarray(data.listed, dtype=bool) & np.asarray(data.tradable, dtype=bool) & np.isfinite(close)


def pair_zscores(y, xc, xm, valid, formation_days, window=BETA_WINDOW, min_obs=MIN_OBS):
    y = np.asarray(y, dtype=np.float64)
    xc = np.asarray(xc, dtype=np.float64)
    xm = np.asarray(xm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    L = int(formation_days)
    v = valid.astype(np.float64)
    y0 = np.where(valid, y, 0.0)
    c0 = np.where(valid, xc, 0.0)
    m0 = np.where(valid, xm, 0.0)

    n = rolling_sum(v, window)
    sy = rolling_sum(y0, window)
    sc = rolling_sum(c0, window)
    sm = rolling_sum(m0, window)
    scc = rolling_sum(c0 * c0, window)
    smm = rolling_sum(m0 * m0, window)
    scm = rolling_sum(c0 * m0, window)
    scy = rolling_sum(c0 * y0, window)
    smy = rolling_sum(m0 * y0, window)
    syy = rolling_sum(y0 * y0, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        vcc = scc - sc * sc / n
        vmm = smm - sm * sm / n
        vcm = scm - sc * sm / n
        vcy = scy - sc * sy / n
        vmy = smy - sm * sy / n
        vyy = syy - sy * sy / n
        det = vcc * vmm - vcm * vcm
        b_c = (vcy * vmm - vmy * vcm) / det
        b_m = (vmy * vcc - vcy * vcm) / det
        ssr = vyy - b_c * vcy - b_m * vmy
        s = np.sqrt(np.maximum(ssr, 0.0) / (n - 3.0))
        ok = (
            np.isfinite(n) & (n >= min_obs)
            & (vcc > 0) & (vmm > 0) & (det > 1e-10 * vcc * vmm)
            & np.isfinite(b_c) & np.isfinite(b_m) & np.isfinite(s) & (s > 0)
        )
        e = rolling_sum(y, L) - b_c * rolling_sum(xc, L) - b_m * rolling_sum(xm, L)
        z = e / (s * np.sqrt(L))
    return np.where(ok & np.isfinite(z), z, np.nan)


def first_crossings(z, z_entry):
    z = np.asarray(z, dtype=np.float64)
    ze = float(z_entry)
    zp = np.full(z.shape, np.nan)
    zp[1:] = z[:-1]
    with np.errstate(invalid="ignore"):
        dn = (z <= -ze) & (zp > -ze)
        up = (z >= ze) & (zp < ze)
    return dn, up


def pair_active(bar_p, bar_c, bar_m, member=None, min_history=MIN_HISTORY):
    act = bar_p & bar_c & bar_m
    for b in (bar_p, bar_c, bar_m):
        act = act & (np.cumsum(b.astype(np.int64)) >= int(min_history))
    if member is not None:
        act = act & member
    return act


def events(data, params):
    L = int(params["formation_days"])
    ze = float(params["z_entry"])
    T = len(data.dates)
    N = len(data.instruments)
    r = daily_returns(data)
    bar = has_bar(data)
    tradable = np.asarray(data.tradable, dtype=bool)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    jm = data.col(MARKET)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for prod, comm in PAIRS:
        jp = data.col(prod)
        jc = data.col(comm)
        valid = bar[:, jp] & bar[:, jc] & bar[:, jm]
        z = pair_zscores(r[:, jp], r[:, jc], r[:, jm], valid, L)
        member = None if uni is None else (uni[:, jp] & uni[:, jc] & uni[:, jm])
        act = pair_active(bar[:, jp], bar[:, jc], bar[:, jm], member)
        z = np.where(act, z, np.nan)
        dn, up = first_crossings(z, ze)
        ev = (dn | up) & tradable[:, jp]
        mask[:, jp] = ev
        side[:, jp] = np.where(dn & ev, 1.0, np.where(up & ev, -1.0, 0.0))
    return mask, side
