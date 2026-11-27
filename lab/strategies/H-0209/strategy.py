"""H-0209: producer equity ETFs vs their own commodity ETF and SPY, residual gap catch-up, hedged 5-day slots.

Rules of the card and where they live:
  pairs / market anchor ............ PAIRS, MARKET
  (1) trailing 252-session OLS ..... pair_zscores (rolling sums over joint-bar sessions, >= 200 required)
  (2) gap e_t and z_t .............. pair_zscores (L-session sum of r_p - b_c r_c - b_m r_SPY, betas at t)
  (3) first crossing entry ......... first_crossings, select_entries (no re-entry while the pair's slot is open)
  (4) slot weights, 5-session hold,
      gross cap 1.0 ................ target_weights
  (5) history / bar requirements ... pair_active
"""
import numpy as np

PARAMS = {"formation_days": 5, "z_entry": 1.5}

PAIRS = (
    ("GDX", "GLD"),
    ("GDXJ", "GLD"),
    ("SIL", "SLV"),
    ("XOP", "USO"),
    ("XLE", "USO"),
    ("FCG", "UNG"),
)
MARKET = "SPY"

BETA_WINDOW = 252      # trailing sessions of the OLS
MIN_OBS = 200          # valid joint sessions required in the window
MIN_HISTORY = 252      # bars of history each instrument of a pair needs
HOLD = 5               # sessions a slot is held (entry at open t+1, exit at open t+6)
SLOT_GROSS = 0.4       # gross of one slot
MAX_GROSS = 1.0        # cap on the summed gross of all open slots


def rolling_sum(x, n):
    """Sum of x over rows t-n+1..t (trailing, inclusive); NaN while fewer than n rows exist."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    n = int(n)
    if x.shape[0] >= n:
        c = np.cumsum(x)
        out[n - 1] = c[n - 1]
        out[n:] = c[n:] - c[:-n]
    return out


def daily_returns(data):
    """Close-to-close total return from ret_co and ret_oc, 0 where it is not finite."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def has_bar(data):
    """An instrument has a bar on row t: listed, tradable on that row and a finite close."""
    close = np.asarray(data.close, dtype=np.float64)
    return np.asarray(data.listed, dtype=bool) & np.asarray(data.tradable, dtype=bool) & np.isfinite(close)


def pair_zscores(y, xc, xm, valid, formation_days, window=BETA_WINDOW, min_obs=MIN_OBS):
    """Rolling OLS y ~ 1 + xc + xm over the trailing `window` rows using only rows where `valid`.

    Returns z, b_c, b_m, s (each (T,)); z is NaN where fewer than `min_obs` valid rows or a degenerate fit.
    The gap is the sum over the last `formation_days` rows of y - b_c xc - b_m xm with the betas of row t.
    """
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
    z = np.where(ok & np.isfinite(z), z, np.nan)
    b_c = np.where(ok, b_c, np.nan)
    b_m = np.where(ok, b_m, np.nan)
    s = np.where(ok, s, np.nan)
    return z, b_c, b_m, s


def first_crossings(z, z_entry):
    """Rows where z first crosses below -z_entry (dn, producer lagged) or above +z_entry (up, ran ahead).

    Requires a finite z on t-1 on the other side of the threshold (NaN comparisons are False).
    """
    z = np.asarray(z, dtype=np.float64)
    ze = float(z_entry)
    zp = np.full(z.shape, np.nan)
    zp[1:] = z[:-1]
    with np.errstate(invalid="ignore"):
        dn = (z <= -ze) & (zp > -ze)
        up = (z >= ze) & (zp < ze)
    return dn, up


def select_entries(dn, up, hold=HOLD):
    """Side per row (+1 long producer, -1 short, 0 none); a pair cannot re-enter while its slot is open.

    A slot entered on decision row t is held on target rows t..t+hold-1 (opens t+1..t+hold) and closes at the
    open of t+hold+1, so the next entry can be decided on row t+hold at the earliest (its first open is the one
    at which the previous slot closes; the two slots of a pair never overlap).
    """
    dn = np.asarray(dn, dtype=bool)
    up = np.asarray(up, dtype=bool)
    side = np.zeros(dn.shape[0])
    next_ok = 0
    for t in np.flatnonzero(dn | up):
        if t < next_ok:
            continue
        side[t] = 1.0 if dn[t] else -1.0
        next_ok = t + int(hold)
    return side


def pair_active(bar_p, bar_c, bar_m, member=None, min_history=MIN_HISTORY):
    """Pair active on row t: each instrument has >= min_history bars up to t and a bar on t (and is a member)."""
    act = bar_p & bar_c & bar_m
    for b in (bar_p, bar_c, bar_m):
        act = act & (np.cumsum(b.astype(np.int64)) >= int(min_history))
    if member is not None:
        act = act & member
    return act


def pair_signals(data, params):
    """Per pair: (producer col, commodity col, z, b_c, b_m, dn, up) with z NaN where the pair is inactive."""
    L = int(params["formation_days"])
    ze = float(params["z_entry"])
    r = daily_returns(data)
    bar = has_bar(data)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    jm = data.col(MARKET)
    out = []
    for prod, comm in PAIRS:
        jp = data.col(prod)
        jc = data.col(comm)
        valid = bar[:, jp] & bar[:, jc] & bar[:, jm]
        z, b_c, b_m, _s = pair_zscores(r[:, jp], r[:, jc], r[:, jm], valid, L)
        member = None if uni is None else (uni[:, jp] & uni[:, jc] & uni[:, jm])
        act = pair_active(bar[:, jp], bar[:, jc], bar[:, jm], member)
        z = np.where(act, z, np.nan)
        dn, up = first_crossings(z, ze)
        out.append((jp, jc, z, b_c, b_m, dn, up))
    return out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    jm = data.col(MARKET)
    W = np.zeros((T, N), dtype=np.float64)
    slot_gross = np.zeros(T, dtype=np.float64)
    for jp, jc, _z, b_c, b_m, dn, up in pair_signals(data, params):
        side = select_entries(dn, up, HOLD)
        for t in np.flatnonzero(side != 0.0):
            bc = b_c[t]
            bm = b_m[t]
            w = SLOT_GROSS / (1.0 + abs(bc) + abs(bm))
            sd = side[t]
            end = min(t + HOLD, T)
            W[t:end, jp] += sd * w
            W[t:end, jc] += -sd * bc * w
            W[t:end, jm] += -sd * bm * w
            slot_gross[t:end] += w * (1.0 + abs(bc) + abs(bm))
    scale = np.where(slot_gross > MAX_GROSS, MAX_GROSS / np.where(slot_gross > 0, slot_gross, 1.0), 1.0)
    W *= scale[:, None]
    return W
