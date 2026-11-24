"""H-0192: earnings news at the 52-week anchor.

S&P 500 cap-weighted core (market-cap proxy = filing-date sf1_arq_marketcap grown by the
stock's own total return since the filing, recomputed on the first session of each month and
left to drift otherwise) plus a 5-session tilt:
  POSITIVE event (8-K Item 2.02 reaction session, z >= z_min, TRI within band of / above the
  prior 252-session high)  -> +tilt_w overweight for rows t..t+4 (open t+1 .. open t+6);
  NEGATIVE event (z <= -z_min, TRI within band of / below the prior 252-session low)
  -> weight 0 for the same rows.
Weights are emitted only on rows where something changes (month start, tilt set, membership);
other rows are NaN (keep positions, let them drift).
All work is done on the columns that are ever S&P 500 members in the view (others stay 0).
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"band": 0.03, "z_min": 2.0}

TILT_W = 0.05        # overweight per positive event (card: 5 % of the book)
TILT_CAP = 0.50      # cap on the sum of open overweights
HOLD = 5             # sessions an event stays open (rows t .. t+4)
VOL_WIN = 63         # residual volatility window t-63 .. t-1
VOL_MIN = 40         # minimum valid residuals in that window
ANCHOR_WIN = 252     # 52-week anchor window t-252 .. t-1
ANCHOR_MIN = 240     # minimum valid sessions in that window


# ----------------------------------------------------------------------------- helpers
def _shift(x, k, fill):
    """Rows moved down by k >= 0 (row t gets row t-k); the first k rows get `fill`."""
    out = np.empty_like(x)
    if k == 0:
        out[:] = x
        return out
    if k >= x.shape[0]:
        out[:] = fill
        return out
    out[:k] = fill
    out[k:] = x[:-k]
    return out


def _ffill_rows(x):
    """Forward-fill NaNs down the rows (point-in-time)."""
    T = x.shape[0]
    rows = np.arange(T)[:, None]
    idx = np.where(np.isfinite(x), rows, -1)
    idx = np.maximum.accumulate(idx, axis=0)
    out = np.take_along_axis(x, np.maximum(idx, 0), axis=0)
    return np.where(idx >= 0, out, np.nan)


def _rolling_max_incl(x, w):
    """Max over rows t-w+1 .. t (x float64, missing = -inf). van Herk / Gil-Werman, O(T*N)."""
    T, N = x.shape
    L = T + w - 1
    nb = -(-L // w)
    pad = np.full((nb * w, N), -np.inf)
    pad[w - 1:w - 1 + T] = x
    blocks = pad.reshape(nb, w, N)
    g = np.maximum.accumulate(blocks, axis=1).reshape(nb * w, N)
    h = np.maximum.accumulate(blocks[:, ::-1, :], axis=1)[:, ::-1, :].reshape(nb * w, N)
    idx = np.arange(T)
    return np.maximum(h[idx], g[idx + w - 1])


def _lagged_window_sum(x, w):
    """Sum of x over rows t-w .. t-1 (x float64 without NaN)."""
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:])
    cs[1:] = np.cumsum(x, axis=0)
    hi = np.arange(T)
    lo = np.maximum(hi - w, 0)
    return cs[hi] - cs[lo]


def _lagged_extreme(xm, valid, w, kind):
    """Max (kind='max') or min of xm over rows t-w .. t-1, with the count of valid rows."""
    if kind == "max":
        src = np.where(valid, xm, -np.inf)
        r = _rolling_max_incl(src, w)
    else:
        src = np.where(valid, -xm, -np.inf)
        r = -_rolling_max_incl(src, w)
    r = _shift(r, 1, np.nan)
    r = np.where(np.isfinite(r), r, np.nan)
    cnt = _lagged_window_sum(valid.astype(np.float64), w)
    return r, cnt


def _prep(data):
    """Inputs restricted to the columns that are ever members in the view."""
    if data.universe is None:
        raise ValueError("H-0192 needs the point-in-time sp500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    ea_all = data.extras["ea_reaction_day"]              # fail loudly if missing
    mc_all = data.extras["sf1_arq_marketcap"]
    cols = np.flatnonzero(uni.any(axis=0))
    member = uni[:, cols]
    ret_co = np.asarray(data.ret_co)[:, cols].astype(np.float64)
    ret_oc = np.asarray(data.ret_oc)[:, cols].astype(np.float64)
    close = np.asarray(data.close)[:, cols].astype(np.float64)
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    ea = np.asarray(ea_all)[:, cols].astype(np.float64)
    mc = np.asarray(mc_all)[:, cols].astype(np.float64)
    # (1) total-return index = cumulative product of (1 + ret_co)(1 + ret_oc)
    g = (1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc))
    tri = np.cumprod(g, axis=0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = listed & np.isfinite(close) & np.isfinite(r) & np.isfinite(tri) & (tri > 0)
    return {"cols": cols, "N": uni.shape[1], "member": member, "r": r, "valid": valid,
            "ea": ea, "mc": mc, "tri": tri}


def _full(p, x, fill):
    out = np.full((x.shape[0], p["N"]), fill, dtype=x.dtype)
    out[:, p["cols"]] = x
    return out


# ----------------------------------------------------------------------------- events
def _events_p(p, band, z_min):
    member, r, valid, ea, tri = p["member"], p["r"], p["valid"], p["ea"], p["tri"]
    with np.errstate(invalid="ignore", divide="ignore"):
        # (2) residual = r - cross-sectional median r of members with data on t
        msk = valid & member
        X = np.where(msk, r, np.nan)
        has = msk.any(axis=1)
        X[~has] = 0.0
        med = np.nanmedian(X, axis=1) if X.shape[1] > 0 else np.zeros(X.shape[0])
        med[~has] = np.nan
        e = np.where(valid, r - med[:, None], np.nan)
        ok = np.isfinite(e)
        e0 = np.where(ok, e, 0.0)
        n = _lagged_window_sum(ok.astype(np.float64), VOL_WIN)
        s1 = _lagged_window_sum(e0, VOL_WIN)
        s2 = _lagged_window_sum(e0 * e0, VOL_WIN)
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        var = np.where(var > 0, var, 0.0)
        sd = np.sqrt(var)
        sd_ok = (n >= VOL_MIN) & (sd > 0)
        z = np.where(sd_ok, e / np.where(sd_ok, sd, 1.0), np.nan)

        # (3) anchor on the total-return index, sessions t-252 .. t-1
        trim = np.where(valid, tri, np.nan)
        H, cnt = _lagged_extreme(trim, valid, ANCHOR_WIN, "max")
        Lo, _ = _lagged_extreme(trim, valid, ANCHOR_WIN, "min")
        anchor_ok = (cnt >= ANCHOR_MIN) & np.isfinite(H) & np.isfinite(Lo)
        near_hi = anchor_ok & (trim >= (1.0 - band) * H)
        near_lo = anchor_ok & (trim <= (1.0 + band) * Lo)

        # (4) events on release-reaction sessions of members with a return on t
        base = (ea == 1.0) & member & valid & np.isfinite(z)
        pos = base & (z >= z_min) & near_hi
        neg = base & (z <= -z_min) & near_lo
    return pos, neg


def _events(data, band, z_min):
    """POSITIVE / NEGATIVE event masks (T, N) decided after the close of row t."""
    p = _prep(data)
    pos, neg = _events_p(p, band, z_min)
    return _full(p, pos, False), _full(p, neg, False), np.asarray(data.universe, dtype=bool)


# ----------------------------------------------------------------------------- core
def _month_starts(data):
    T = len(data.dates)
    ps = np.asarray(period_starts(data.dates, "M"))
    if ps.dtype == bool:
        ms = ps.reshape(-1).copy()
    else:
        ms = np.zeros(T, dtype=bool)
        ms[ps.astype(np.int64)] = True
    if T > 0:
        ms[0] = True        # the first session of the view starts the core
    return ms


def _core_p(p, ms):
    """Drifting core weights c: cap proxy fixed on the first session of each month,
    then moved by each stock's total return (buy and hold), over current members."""
    member, mc, tri = p["member"], p["mc"], p["tri"]
    T, N = member.shape
    tri_ff = _ffill_rows(np.where(np.isfinite(tri) & (tri > 0), tri, np.nan))

    # cap proxy: marketcap frozen at its filing date f, times TRI_t / TRI_f
    mc_ok = np.isfinite(mc) & (mc > 0)
    prev = _shift(mc, 1, np.nan)
    chg = mc_ok & ~(mc == prev)
    rows = np.arange(T)[:, None]
    a = np.maximum.accumulate(np.where(chg, rows, -1), axis=0)
    f = np.maximum(a - 1, 0)            # visible the day after filing -> filing row = a - 1
    tri_f = np.take_along_axis(tri_ff, f, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cap = mc * tri_ff / tri_f
    cap_ok = mc_ok & (a >= 0) & np.isfinite(cap) & (cap > 0)

    R = np.flatnonzero(ms)
    m_mem = member[R]
    m_cap = np.where(m_mem & cap_ok[R], cap[R], np.nan)
    core0 = np.zeros((len(R), N))
    for i in range(len(R)):                      # one row per month: ~200 iterations
        row = m_cap[i]
        fin = np.isfinite(row)
        fill = float(np.median(row[fin])) if fin.any() else 1.0   # no cap at all: equal weights
        core0[i] = np.where(m_mem[i], np.where(fin, row, fill), 0.0)
    tri0 = tri_ff[R]

    pos = np.cumsum(ms) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = tri_ff / tri0[pos]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    raw = np.where(member, core0[pos] * ratio, 0.0)
    S = raw.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(S[:, None] > 0, raw / np.where(S > 0, S, 1.0)[:, None], 0.0)
    return c


def _core(data):
    p = _prep(data)
    ms = _month_starts(data)
    return _full(p, _core_p(p, ms), 0.0), ms


# ----------------------------------------------------------------------------- book
def _hold_state(pos, neg, member):
    ev = pos.astype(np.int8) - neg.astype(np.int8)
    state = np.zeros_like(ev)
    for k in range(HOLD - 1, -1, -1):           # oldest first; the latest event wins
        s = _shift(ev, k, 0)
        state = np.where(s != 0, s, state)
    return np.where(member, state, 0).astype(np.int8)


def _book(c, state, member, tilt_w):
    A = state > 0
    Z = state < 0
    other = member & ~A & ~Z
    nA = A.sum(axis=1).astype(np.float64)
    O_raw = tilt_w * nA
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(O_raw > TILT_CAP, TILT_CAP / np.where(O_raw > 0, O_raw, 1.0), 1.0)
    o = tilt_w * scale                          # overweight per open positive event
    O = o * nA
    C_A = np.where(A, c, 0.0).sum(axis=1)
    C_O = np.where(other, c, 0.0).sum(axis=1)
    Rem = 1.0 - C_A - O                         # left for the non-event names
    wA = np.where(A, c + o[:, None], 0.0)
    tot_A = C_A + O
    fill_others = (Rem >= 0) & (C_O > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.where(fill_others, Rem / np.where(C_O > 0, C_O, 1.0), 0.0)
        a_scale = np.where(fill_others | (tot_A <= 0), 1.0, 1.0 / np.where(tot_A > 0, tot_A, 1.0))
    w = wA * a_scale[:, None] + np.where(other, c * K[:, None], 0.0)
    w = np.where(member, w, 0.0)
    w = np.maximum(w, 0.0)
    tot = w.sum(axis=1)
    over = tot > 1.0
    if over.any():
        w[over] = w[over] / tot[over][:, None] * (1.0 - 1e-12)
    return w


def target_weights(data, params):
    band = float(params["band"])
    z_min = float(params["z_min"])
    tilt_w = float(params.get("tilt_w", TILT_W))   # card: 5 %; 0 gives the core-only control
    p = _prep(data)
    ms = _month_starts(data)
    member = p["member"]
    pos, neg = _events_p(p, band, z_min)
    c = _core_p(p, ms)
    state = _hold_state(pos, neg, member)
    w = _book(c, state, member, tilt_w)

    T = w.shape[0]
    dec = ms.copy()
    if T > 1:
        uni = np.asarray(data.universe, dtype=bool)
        dec[1:] |= (uni[1:] != uni[:-1]).any(axis=1)
        dec[1:] |= (state[1:] != state[:-1]).any(axis=1)
    out = np.full((T, p["N"]), np.nan)
    out[dec] = 0.0
    out[np.ix_(dec, p["cols"])] = w[dec]
    return out
