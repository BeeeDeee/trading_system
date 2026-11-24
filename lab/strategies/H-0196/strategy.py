"""H-0196 - buyback desks as buyers of last resort.

LIQ-500 event-slot book: long no-news idiosyncratic droppers that are active net repurchasers
inside their open trading window, each slot hedged with a beta-scaled short of the 20 most liquid
members, held 5 sessions. Row t is decided after the close of t (rows 0..t only), filled at t+1.
"""
import numpy as np

PARAMS = {"window_close": 35, "z_min": 2.0}

# constants of the card (not grid parameters)
SLOT_WEIGHT = 0.06
MAX_SLOTS = 8
HOLD = 5                 # slot opened on decision row s is held on rows s..s+4, closed by row s+5
RESID_WIN = 63           # residual std over sessions t-63..t-1
RESID_MIN = 50
WINDOW_OPEN = 3          # k >= 3 sessions after the release session
RELEASE_LOOKBACK = 100   # no release in the last 100 sessions -> not eligible
TOP_FRACTION = 1.0 / 3.0
HEDGE_N = 20
DV_WIN = 63              # median dollar volume over sessions t-62..t
BETA_WIN = 252           # beta over sessions t-252..t-1
BETA_MIN = 200
BETA_LO, BETA_HI = 0.5, 1.5
MAX_GROSS = 1.0


def _close_to_close(data):
    """r[t,j] = (1+ret_co)(1+ret_oc)-1, NaN where there is no close on t or t-1."""
    r = np.asarray(data.ret_co, dtype=np.float64) + 1.0
    r *= np.asarray(data.ret_oc, dtype=np.float64) + 1.0
    r -= 1.0
    has_px = np.isfinite(np.asarray(data.close)) & np.asarray(data.listed, dtype=bool)
    valid = has_px.copy()
    valid[0] = False
    valid[1:] &= has_px[:-1]
    valid &= np.isfinite(r)
    r[~valid] = np.nan
    return r


def _residual_z(r, uni):
    """e = r - EW member mean of r on t; z = e / std(e over t-63..t-1, >= 50 valid)."""
    T, N = r.shape
    rv = np.isfinite(r)
    mem = uni & rv
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    mean = np.full(T, np.nan)
    ok = cnt > 0
    mean[ok] = tot[ok] / cnt[ok]
    del rv, mem, tot
    e = r - mean[:, None]                       # NaN where r invalid or no member mean
    ev = np.isfinite(e)
    lo = np.maximum(np.arange(T) - RESID_WIN, 0)
    cs = np.zeros((T + 1, N))
    np.cumsum(np.where(ev, e, 0.0), axis=0, out=cs[1:])
    S = cs[:T] - cs[lo]                         # sum over rows lo..t-1
    cs[1:] = np.where(ev, e * e, 0.0)
    np.cumsum(cs[1:], axis=0, out=cs[1:])
    Q = cs[:T] - cs[lo]                         # sum of squares over rows lo..t-1
    del cs
    cc = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(ev, axis=0, out=cc[1:])
    n = (cc[:T] - cc[lo]).astype(np.float64)
    del cc
    with np.errstate(invalid="ignore", divide="ignore"):
        S *= S
        S /= n
        Q -= S                                  # Q - S^2 / n
        del S
        Q /= n - 1.0                            # sample variance
    np.maximum(Q, 0.0, out=Q)
    Q[n < RESID_MIN] = np.nan
    del n
    np.sqrt(Q, out=Q)                           # s
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(Q) & (Q > 0) & ev
        e /= Q
    del Q
    e[~ok] = np.nan
    return e                                    # z


def _no_news(ea, nk):
    """ea_reaction_day == 0 and nonearn_8k_reaction_day == 0 on t and t-1 (NaN = not eligible)."""
    quiet = (ea == 0) & (nk == 0)              # NaN compares False
    prev = np.zeros_like(quiet)
    prev[1:] = quiet[:-1]
    return quiet & prev


def _open_window(ea, window_close):
    """k = sessions since the latest session with ea == 1; eligible when 3 <= k <= window_close and k <= 100."""
    T, N = ea.shape
    rows = np.arange(T, dtype=np.int32)[:, None]
    last = np.where(ea == 1, rows, np.int32(-1))
    np.maximum.accumulate(last, axis=0, out=last)
    ok = last >= 0
    last -= rows
    last *= -1                                  # k = t - last release row
    ok &= (last >= WINDOW_OPEN) & (last <= window_close) & (last <= RELEASE_LOOKBACK)
    return ok


def _repurchaser(ncf, mcap, uni):
    """b = -ncfcommon / marketcap > 0 and in the top third of b across members with a valid b."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = -np.asarray(ncf, dtype=np.float64)
        b /= mcap
        bv = np.isfinite(b) & np.isfinite(mcap) & (mcap > 0) & uni
    b[~bv] = np.nan
    bm = b
    n = bv.sum(axis=1)
    asc = np.sort(bm, axis=1)                  # valid values first (ascending), NaN at the end
    m = np.ceil(n * TOP_FRACTION).astype(np.int64)  # size of the top third
    pos = np.clip(n - m, 0, bm.shape[1] - 1)
    thr = np.take_along_axis(asc, pos[:, None], axis=1)[:, 0]
    thr = np.where(n > 0, thr, np.inf)
    with np.errstate(invalid="ignore"):
        return bv & (bm > 0) & (bm >= thr[:, None])


def event_table(data, params):
    """(event mask, z) on (T, N): the card's event (1) & (2) & (3) & (4) for members on t."""
    if data.universe is None:
        raise ValueError("card needs the liq_n point-in-time universe")
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"])
    nk = np.asarray(data.extras["nonearn_8k_reaction_day"])
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"])
    mcap = np.asarray(data.extras["sf1_art_marketcap"])
    z_min = float(params["z_min"])
    window_close = int(params["window_close"])

    r = _close_to_close(data)
    z = _residual_z(r, uni)
    with np.errstate(invalid="ignore"):
        ev = np.isfinite(z) & (z <= -z_min)
    ev &= uni
    ev &= tradable
    ev &= _no_news(ea, nk)
    ev &= _open_window(ea, window_close)
    ev &= _repurchaser(ncf, mcap, uni)
    return ev, z, r


def _hedge(t, j, r, dv, uni_t):
    """Top-20 members on t by 63-day median dollar volume (excluding j) and clipped beta of j on their EW return."""
    mem = np.nonzero(uni_t)[0]
    mem = mem[mem != j]
    lo = max(0, t - DV_WIN + 1)
    sub = dv[lo:t + 1][:, mem]
    has = np.isfinite(sub).any(axis=0)
    med = np.full(mem.shape[0], -np.inf)
    if has.any():
        med[has] = np.nanmedian(sub[:, has], axis=0)
    cand = mem[np.isfinite(med)]
    medc = med[np.isfinite(med)]
    order = np.argsort(-medc, kind="stable")     # ties -> lower column index first
    basket = cand[order[:HEDGE_N]]
    beta = 1.0
    blo = max(0, t - BETA_WIN)
    if basket.size > 0 and t - blo >= BETA_MIN:
        rb = r[blo:t][:, basket]
        rbv = np.isfinite(rb)
        cntb = rbv.sum(axis=1)
        x = np.full(t - blo, np.nan)
        okb = cntb > 0
        x[okb] = np.where(rbv, rb, 0.0).sum(axis=1)[okb] / cntb[okb]
        y = r[blo:t, j]
        v = np.isfinite(x) & np.isfinite(y)
        if v.sum() >= BETA_MIN:
            xv = x[v] - x[v].mean()
            yv = y[v] - y[v].mean()
            sxx = float(np.dot(xv, xv))
            if sxx > 0:
                beta = float(np.dot(xv, yv)) / sxx
    beta = min(max(beta, BETA_LO), BETA_HI)
    return basket, beta


def target_weights(data, params):
    ev, z, r = event_table(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume)
    T, N = ev.shape
    W = np.zeros((T, N))
    ev_rows = ev.any(axis=1)
    slots = []                                    # (event column, start row, hedge columns, hedge weight each)
    for t in range(T):
        # close slots after 5 sessions, or when the event stock left the universe
        slots = [s for s in slots if t <= s[1] + HOLD - 1 and uni[t, s[0]]]
        free = MAX_SLOTS - len(slots)
        if free > 0 and ev_rows[t]:
            held = {s[0] for s in slots}
            cand = np.nonzero(ev[t])[0]
            cand = np.array([j for j in cand if j not in held], dtype=np.int64)
            if cand.size:
                order = np.argsort(z[t, cand], kind="stable")   # most negative z first
                for j in cand[order[:free]]:
                    basket, beta = _hedge(t, int(j), r, dv, uni[t])
                    hw = SLOT_WEIGHT * beta / basket.size if basket.size else 0.0
                    slots.append((int(j), t, basket, hw))
        row = W[t]
        ut = uni[t]
        for j, _, basket, hw in slots:
            row[j] += SLOT_WEIGHT
            if basket.size:
                keep = basket[ut[basket]]         # only members may be held
                row[keep] -= hw
        g = np.abs(row).sum()
        if g > MAX_GROSS:
            row *= MAX_GROSS / g
    return W
