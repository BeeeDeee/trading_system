"""H-0067 Earnings-release volume visibility premium (LIQ-500, dollar-neutral, fixed 1/30 event slots).

Row t is decided after the close of session t with rows 0..t only; the engine fills at the open of t+1.
"""
import numpy as np

PARAMS = {"hold_days": 5, "leg_quantile": 0.3}

SLOT = 1.0 / 30.0        # fixed event slot (card rule 5)
WIN_FAR = 68             # baseline window t-68 .. t-6 (63 sessions, skip the 5 before the release)
WIN_NEAR = 6
MIN_VALID = 50           # valid sessions required in the baseline window
POOL_SESSIONS = 252      # reference pool t-252 .. t-1
MIN_BIN = 30             # pool candidates required in the candidate's bin
HEDGE_N = 50             # hedge basket: 50 most liquid members by liq_rank
HEDGE_BAND = 0.05        # hedge rebalanced only when |hedge + event net| > 5 % of equity
GROSS_LIMIT = 1.0 - 1e-9


def _window_sum(x, ti, tj):
    """Sum of x[s, j] over s = t-68 .. t-6 (clipped at row 0), gathered at the points (ti, tj)."""
    T, N = x.shape
    c = np.zeros((T + 1, N), dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    a = np.clip(ti - WIN_FAR, 0, None)
    b = np.clip(ti - WIN_NEAR + 1, 0, None)
    b = np.maximum(a, b)
    return c[b, tj] - c[a, tj]


def _dedupe(rt, rj, hold):
    """Keep a raw flag (rt, rj) only if the same stock had no kept candidate in the previous `hold` sessions."""
    keep = np.zeros(rt.shape[0], dtype=bool)
    order = np.lexsort((rt, rj))
    last_j = -1
    last_t = 0
    for k in order:
        j = int(rj[k])
        t = int(rt[k])
        if j != last_j or t - last_t > hold:
            keep[k] = True
            last_j = j
            last_t = t
    return keep


def _sides_by_bin(ct, absz, pos, av, first_row, q):
    """Matching-bin sort (card rule 4). Inputs are the valid candidates sorted by row ct.

    Returns side (+1 long, -1 short, 0 no event) per candidate."""
    n = ct.shape[0]
    side = np.zeros(n, dtype=np.int8)
    if n == 0:
        return side
    days = np.unique(ct)
    day_lo = np.searchsorted(ct, days, side="left")
    day_hi = np.searchsorted(ct, days, side="right")
    pool_lo = np.searchsorted(ct, days - POOL_SESSIONS, side="left")
    for k in range(days.shape[0]):
        d = int(days[k])
        if d - POOL_SESSIONS < first_row:
            continue                                   # pool does not yet cover 252 sessions of 8-K data
        lo, hi = int(pool_lo[k]), int(day_lo[k])       # pool: candidates in rows d-252 .. d-1
        if hi - lo < MIN_BIN:
            continue
        pz = absz[lo:hi]
        cuts = np.quantile(pz, [1.0 / 3.0, 2.0 / 3.0])
        pbin = np.searchsorted(cuts, pz, side="left") + 3 * pos[lo:hi]
        pav = av[lo:hi]
        cache = {}
        for i in range(int(day_lo[k]), int(day_hi[k])):
            b = int(np.searchsorted(cuts, absz[i], side="left") + 3 * pos[i])
            if b not in cache:
                sel = pav[pbin == b]
                if sel.shape[0] < MIN_BIN:
                    cache[b] = None
                else:
                    cache[b] = (np.quantile(sel, 1.0 - q), np.quantile(sel, q))
            th = cache[b]
            if th is None:
                continue
            up = av[i] >= th[0]
            dn = av[i] <= th[1]
            if up and not dn:
                side[i] = 1
            elif dn and not up:
                side[i] = -1
    return side


def event_table(data, params):
    """Events of the card: rows et, columns ej, sides es (+1 / -1), sorted by row."""
    hold = int(params["hold_days"])
    q = float(params["leg_quantile"])
    if data.universe is None:
        raise ValueError("H-0067 needs the LIQ-500 point-in-time universe")
    U = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    T, N = U.shape
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8))

    has_8k = np.isfinite(ea).any(axis=1)
    if not has_8k.any():
        return empty
    first_row = int(np.argmax(has_8k))                 # first session with 8-K data (rows <= t only)

    # (1) event candidates: Item 2.02 reaction session, LIQ-500 member, tradable, no candidate in previous hold
    with np.errstate(invalid="ignore"):
        raw = (ea == 1.0) & U & trad
    rt, rj = np.nonzero(raw)
    keep = _dedupe(rt, rj, hold)
    ct, cj = rt[keep], rj[keep]
    order = np.lexsort((cj, ct))
    ct, cj = ct[order], cj[order]
    if ct.shape[0] == 0:
        return empty

    # session validity: listed with a positive, finite dollar volume
    with np.errstate(invalid="ignore"):
        dv_ok = listed & np.isfinite(dv) & (dv > 0)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    r[~dv_ok] = np.nan
    r[~np.isfinite(r)] = np.nan
    mem = U & np.isfinite(r)
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    e = r - mkt[:, None]                               # market-adjusted daily return
    del r
    e_ok = np.isfinite(e)
    e0 = np.where(e_ok, e, 0.0)

    # (2) reaction size z = e_t / sigma(e over t-68..t-6)
    n_e = _window_sum(e_ok.astype(np.float64), ct, cj)
    s1 = _window_sum(e0, ct, cj)
    s2 = _window_sum(e0 * e0, ct, cj)
    del e0
    e_t = e[ct, cj]
    del e
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.maximum(n_e, 1.0)) / np.maximum(n_e - 1.0, 1.0)
        sigma = np.sqrt(np.maximum(var, 0.0))
        z = e_t / sigma
    ok = (n_e >= MIN_VALID) & (var > 0) & np.isfinite(z)

    # (3) abnormal dollar volume AV = ln dv_t - ln mean(dv over t-68..t-6)
    n_v = _window_sum(dv_ok.astype(np.float64), ct, cj)
    sv = _window_sum(np.where(dv_ok, dv, 0.0), ct, cj)
    dv_t = dv[ct, cj]
    with np.errstate(invalid="ignore", divide="ignore"):
        av = np.log(dv_t) - np.log(sv / np.maximum(n_v, 1.0))
    ok &= (n_v >= MIN_VALID) & dv_ok[ct, cj] & np.isfinite(av)

    # (4) matching bins against the pool of valid candidates in rows t-252 .. t-1
    vt, vj, vz, vav = ct[ok], cj[ok], z[ok], av[ok]
    side = _sides_by_bin(vt, np.abs(vz), (vz >= 0).astype(np.int64), vav, first_row, q)
    sel = side != 0
    return vt[sel], vj[sel], side[sel]


def _max_scale(pieces):
    """Largest s in [0, 1] with c0 + c1 * s <= GROSS_LIMIT for every linear piece (c0, c1), c1 >= 0."""
    s = 1.0
    for c0, c1 in pieces:
        if c1 > 1e-15:
            s = min(s, (GROSS_LIMIT - c0) / c1)
        elif c0 > GROSS_LIMIT:
            s = 0.0
    return max(s, 0.0)


def build_weights(T, N, et, ej, es, U, trad, liq, hold):
    """Card rules 5 and 6: fixed slots held `hold` sessions, gross-capped openings, banded hedge basket."""
    W = np.zeros((T, N), dtype=np.float64)
    new_by_row = {}
    for t, j, s in zip(et.tolist(), ej.tolist(), es.tolist()):
        new_by_row.setdefault(t, []).append((j, s))
    slots = {}      # j -> (open row, weight)
    hedge = {}      # j -> weight
    for t in range(T):
        Ut = U[t]
        Tt = trad[t]
        for j in list(slots):
            t0 = slots[j][0]
            if t - t0 >= hold or not Ut[j] or not Tt[j]:
                del slots[j]
        for j in list(hedge):
            if not Ut[j] or not Tt[j]:
                del hedge[j]
        H = float(sum(hedge.values()))

        new = [(j, s) for (j, s) in new_by_row.get(t, []) if j not in slots]
        if new:
            g_e = float(sum(abs(w) for (_, w) in slots.values()))
            n_e = float(sum(w for (_, w) in slots.values()))
            g_n = SLOT * len(new)
            n_n = SLOT * float(sum(s for (_, s) in new))
            reb = [(g_e + n_e, g_n + n_n), (g_e - n_e, g_n - n_n)]
            scale = _max_scale(reb)
            if abs(H + n_e + scale * n_n) <= HEDGE_BAND:
                # the hedge would be kept as is: it must fit too
                scale = _max_scale(reb + [(g_e + abs(H), g_n)])
            if scale * SLOT > 1e-12:
                for j, s in new:
                    slots[j] = (t, s * SLOT * scale)

        if not slots:
            hedge = {}                                  # no open event slots: the book is in cash
        else:
            net = float(sum(w for (_, w) in slots.values()))
            H = float(sum(hedge.values()))
            if abs(H + net) > HEDGE_BAND:
                liq_t = liq[t]
                with np.errstate(invalid="ignore"):
                    elig = Ut & Tt & np.isfinite(liq_t)
                for j in slots:
                    elig[j] = False
                idx = np.nonzero(elig)[0]
                order = np.lexsort((idx, liq_t[idx]))
                names = idx[order][:HEDGE_N]
                hedge = {}
                if names.shape[0] > 0:
                    hw = -net / names.shape[0]
                    for j in names.tolist():
                        hedge[j] = hw

        row = W[t]
        for j, w in hedge.items():
            row[j] += w
        for j, (_, w) in slots.items():
            row[j] += w
    return W


def target_weights(data, params):
    hold = int(params["hold_days"])
    et, ej, es = event_table(data, params)
    U = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    liq = np.asarray(data.extras["liq_rank"], dtype=np.float64)
    T, N = U.shape
    return build_weights(T, N, et, ej, es, U, trad, liq, hold)
