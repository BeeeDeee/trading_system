"""H-0009 Disposition-gated news drift (LIQ-500 long/short, daily decisions).

Rules of card.signal.description and where they are implemented:
 (1) P = total-return index (total_return_index), r_t = (1+ret_co)(1+ret_oc)-1    -> target_weights
 (2) event on day t for LIQ-500 members tradable on t:
     VR = dv_t / median(dv over rows t-63..t-1), >= 40 valid days              -> _baseline_median
     AV = VR / cross-sectional median of VR over LIQ-500 on t                   -> _group_median
     event if AV >= volume_ratio; direction = sign(r_t - median_j r_j,t)        -> target_weights
 (3) overhang G = P_{t-1} / R - 1, R = dv-weighted mean of P over the
     ref_window rows ending at t-1, >= ref_window/2 valid days                  -> _overhang
 (4) aligned: long dir>0 & G>0, short dir<0 & G<0; other events misaligned      -> target_weights
 (5) position active on decision rows t..t+hold_days-1 (enter open t+1, exit
     open t+1+hold_days); newest event decides (aligned restarts the clock
     with its direction, misaligned closes)                                     -> _positions
 (6) each side gross 0.5 split equally, capped at 0.10 per name                 -> book_weights
Deviation: the card keeps names that leave LIQ-500 until their exit date; G0 forbids weights
outside the universe, so such a position is closed on the first row the name is not a member.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"volume_ratio": 3, "ref_window": 252, "hold_days": 5}

BASELINE_DAYS = 63      # volume baseline window t-63..t-1 (card rule 2)
BASELINE_MIN = 40       # minimum valid days in the baseline window
SIDE_GROSS = 0.5        # gross per side (card rule 6)
NAME_CAP = 0.10         # cap per name
CHUNK = 32768           # points per gather chunk (memory bound)


def _group_median(groups, values, n_groups):
    """Median of `values` per integer group id (finite values only); NaN for empty groups."""
    out = np.full(n_groups, np.nan)
    keep = np.isfinite(values)
    g = groups[keep]
    v = values[keep]
    if g.size == 0:
        return out
    order = np.lexsort((v, g))
    vs = v[order]
    counts = np.bincount(g, minlength=n_groups)
    starts = np.cumsum(counts) - counts
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    out[has] = 0.5 * (vs[lo] + vs[hi])
    return out


def _baseline_median(dv, rows, cols, window=BASELINE_DAYS, min_count=BASELINE_MIN):
    """Median of valid (finite, > 0) dollar volume over rows t-window..t-1 for each point (t, i).
    NaN when fewer than min_count valid days."""
    out = np.full(rows.shape[0], np.nan)
    offs = np.arange(window)
    for s in range(0, rows.shape[0], CHUNK):
        r = rows[s:s + CHUNK]
        c = cols[s:s + CHUNK]
        idx = r[:, None] - window + offs[None, :]          # rows t-window .. t-1
        inside = idx >= 0
        win = np.asarray(dv[np.maximum(idx, 0), c[:, None]], dtype=np.float64)
        ok = inside & np.isfinite(win) & (win > 0)
        win = np.where(ok, win, np.nan)
        n = ok.sum(axis=1)
        srt = np.sort(win, axis=1)                          # NaNs sort to the end
        k = np.arange(r.shape[0])
        lo = np.maximum((n - 1) // 2, 0)
        hi = np.minimum(n // 2, window - 1)
        med = 0.5 * (srt[k, lo] + srt[k, hi])
        out[s:s + CHUNK] = np.where(n >= min_count, med, np.nan)
    return out


def _overhang(P, dv, rows, cols, ref_window):
    """G = P[t-1] / R - 1 with R the dollar-volume-weighted mean of P over rows t-ref_window..t-1.
    Valid day: dollar volume finite and > 0 and P finite. NaN when fewer than ref_window/2 valid days."""
    out = np.full(rows.shape[0], np.nan)
    offs = np.arange(ref_window)
    step = max(1, (CHUNK * 64) // max(ref_window, 1))
    for s in range(0, rows.shape[0], step):
        r = rows[s:s + step]
        c = cols[s:s + step]
        idx = r[:, None] - ref_window + offs[None, :]      # rows t-ref_window .. t-1
        inside = idx >= 0
        ii = np.maximum(idx, 0)
        w = np.asarray(dv[ii, c[:, None]], dtype=np.float64)
        p = np.asarray(P[ii, c[:, None]], dtype=np.float64)
        ok = inside & np.isfinite(w) & (w > 0) & np.isfinite(p)
        w = np.where(ok, w, 0.0)
        p = np.where(ok, p, 0.0)
        n = ok.sum(axis=1)
        sw = w.sum(axis=1)
        swp = (w * p).sum(axis=1)
        good = (2 * n >= ref_window) & (sw > 0)
        ref = np.where(good, swp / np.where(sw > 0, sw, 1.0), np.nan)
        prev = np.asarray(P[np.maximum(r - 1, 0), c], dtype=np.float64)
        prev = np.where(r >= 1, prev, np.nan)
        g = prev / ref - 1.0
        out[s:s + step] = np.where(good & np.isfinite(prev) & (ref > 0), g, np.nan)
    return out


def book_weights(side):
    """Rule 6: long side 0.5 split equally over active longs, short side 0.5 over active shorts,
    each name capped at 0.10."""
    w = np.zeros(side.shape[0])
    longs = side > 0
    shorts = side < 0
    nl = int(longs.sum())
    ns = int(shorts.sum())
    if nl > 0:
        w[longs] = min(SIDE_GROSS / nl, NAME_CAP)
    if ns > 0:
        w[shorts] = -min(SIDE_GROSS / ns, NAME_CAP)
    return w


def _positions(uni, ev_rows, ev_cols, ev_code, hold_days):
    """Rule 5. ev_code +1 aligned long, -1 aligned short, 0 misaligned; events sorted by row.
    A position opened on decision row t is active on rows t..t+hold_days-1.
    Deviation forced by G0 (only universe members may be held): a held name that is not a
    LIQ-500 member on row t is closed on that row (sold at the next open)."""
    T, N = uni.shape
    W = np.zeros((T, N))
    side = np.zeros(N, dtype=np.int8)
    start = np.zeros(N, dtype=np.int64)
    bounds = np.searchsorted(ev_rows, np.arange(T + 1), side="left")
    for t in range(T):
        a, b = bounds[t], bounds[t + 1]
        if b > a:
            c = ev_cols[a:b]
            side[c] = ev_code[a:b]
            start[c] = t
        expired = (side != 0) & ((t - start >= hold_days) | ~uni[t])
        side[expired] = 0
        if side.any():
            W[t] = book_weights(side)
    return W


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    ref_window = int(params["ref_window"])
    vol_thr = float(params["volume_ratio"])
    if hold_days < 1 or ref_window < 2:
        raise ValueError("invalid params")

    if data.universe is None:
        raise ValueError("H-0009 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    dv = data.dollar_volume
    ret_co = data.ret_co
    ret_oc = data.ret_oc
    T, N = uni.shape

    dv_t = np.asarray(dv, dtype=np.float64)
    valid_t = np.isfinite(dv_t) & (dv_t > 0)
    del dv_t

    # (2) volume ratio for LIQ-500 members tradable on t with an observed dollar volume on t
    rows, cols = np.nonzero(uni & trad & valid_t)
    del valid_t
    base = _baseline_median(dv, rows, cols)
    vr = np.asarray(dv[rows, cols], dtype=np.float64) / base
    cs_vr = _group_median(rows, vr, T)                      # cross-sectional median of VR on t
    av = vr / cs_vr[rows]

    # (1)+(2) market-adjusted close-to-close return; median over LIQ-500 members tradable on t
    mrows, mcols = np.nonzero(uni & trad)
    mret = (1.0 + np.asarray(ret_co[mrows, mcols], dtype=np.float64)) * \
           (1.0 + np.asarray(ret_oc[mrows, mcols], dtype=np.float64)) - 1.0
    mkt = _group_median(mrows, mret, T)
    r_pt = (1.0 + np.asarray(ret_co[rows, cols], dtype=np.float64)) * \
           (1.0 + np.asarray(ret_oc[rows, cols], dtype=np.float64)) - 1.0
    adj = r_pt - mkt[rows]
    direction = np.where(np.isfinite(adj), np.sign(adj), 0.0)

    is_event = np.isfinite(av) & (av >= vol_thr) & (direction != 0)
    er = rows[is_event]
    ec = cols[is_event]
    edir = direction[is_event]

    # (3) capital-gain overhang as of the close of t-1
    P = total_return_index(data)
    G = _overhang(P, dv, er, ec, ref_window)
    has_g = np.isfinite(G)                                  # without G the event is skipped

    # (4) aligned vs misaligned
    code = np.zeros(er.shape[0], dtype=np.int8)
    code[(edir > 0) & (G > 0)] = 1
    code[(edir < 0) & (G < 0)] = -1
    er, ec, code = er[has_g], ec[has_g], code[has_g]
    order = np.argsort(er, kind="stable")

    # (5)+(6) positions and weights, decided every row
    return _positions(uni, er[order], ec[order], code[order], hold_days)
