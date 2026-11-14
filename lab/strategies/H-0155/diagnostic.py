"""H-0155 mechanism event: a stock enters the long leg at a monthly formation date (it was not in the long
leg of the previous formation). Same formation calendar, regression and cell logic as strategy.py (copied,
since diagnostic.py is a separate file). Side +1."""
import numpy as np

VIX_KEY = "fred_macro.VIXCLS"
MIN_VALID_SHARE = 0.8
N_BETA_GROUPS = 3
N_IVOL_GROUPS = 2
LONG_DIV = 5


def formation_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    out = np.zeros(T, dtype=bool)
    if T == 0:
        return out
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    primary = nxt.astype("datetime64[M]") != month
    out |= primary
    if T > 1:
        new_month = month[1:] != month[:-1]
        out[1:] |= new_month & ~primary[:-1]
    return out


def tie_ranks(instruments):
    names = [str(x) for x in instruments]
    order = sorted(range(len(names)), key=lambda j: (len(names[j]), names[j]))
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    return rank


def vix_innovations(vix_aligned):
    s = np.asarray(vix_aligned, dtype=np.float64)
    out = np.full(s.shape[0], np.nan)
    if s.shape[0] > 1:
        out[:-1] = s[1:] - s[:-1]
    return out


def stock_returns(data):
    co = np.asarray(data.ret_co, dtype=np.float32)
    oc = np.asarray(data.ret_oc, dtype=np.float32)
    r = (np.float32(1.0) + co) * (np.float32(1.0) + oc) - np.float32(1.0)
    has = np.isfinite(np.asarray(data.close))
    valid = np.zeros(has.shape, dtype=bool)
    valid[1:] = has[1:] & has[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.float32(0.0)).astype(np.float32)
    return r, valid


def market_factor(r, valid, uni):
    mem = valid & uni
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, np.float32(0.0)).sum(axis=1, dtype=np.float64)
    out = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    out[ok] = tot[ok] / cnt[ok]
    return out


def exposures(r, valid, mkt, dvix, cols, t, est_days):
    k = len(cols)
    lo = max(0, t - est_days)
    if k == 0 or t - lo < 1:
        z = np.zeros(k)
        return z, z, z, np.zeros(k, dtype=bool)
    x1 = mkt[lo:t]
    x2 = dvix[lo:t]
    ok = np.isfinite(x1) & np.isfinite(x2)
    x1 = np.where(ok, x1, 0.0)
    x2 = np.where(ok, x2, 0.0)
    V = valid[lo:t][:, cols] & ok[:, None]
    Vf = V.astype(np.float64)
    Y = np.where(V, r[lo:t][:, cols].astype(np.float64), 0.0)
    n = Vf.sum(axis=0)
    s1 = x1 @ Vf
    s2 = x2 @ Vf
    s11 = (x1 * x1) @ Vf
    s22 = (x2 * x2) @ Vf
    s12 = (x1 * x2) @ Vf
    sy = Y.sum(axis=0)
    s1y = x1 @ Y
    s2y = x2 @ Y
    syy = (Y * Y).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c11 = s11 - s1 * s1 / n
        c22 = s22 - s2 * s2 / n
        c12 = s12 - s1 * s2 / n
        c1y = s1y - s1 * sy / n
        c2y = s2y - s2 * sy / n
        cyy = syy - sy * sy / n
        det = c11 * c22 - c12 * c12
        b1 = (c22 * c1y - c12 * c2y) / det
        b2 = (c11 * c2y - c12 * c1y) / det
        ssr = np.maximum(cyy - b1 * c1y - b2 * c2y, 0.0)
        ivol = np.sqrt(ssr / n)
    elig = (n >= MIN_VALID_SHARE * est_days) & (det > 0) & np.isfinite(b1) & np.isfinite(b2) & np.isfinite(ivol)
    return b1, b2, ivol, elig


def long_leg(b_mkt, ivol, b_vix, tie):
    M = len(b_mkt)
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((tie, b_mkt))
    grp = np.empty(M, dtype=np.int64)
    grp[order] = (N_BETA_GROUPS * np.arange(M)) // M
    picks = []
    for g in range(N_BETA_GROUPS):
        idx = np.nonzero(grp == g)[0]
        m = len(idx)
        if m == 0:
            continue
        o = idx[np.lexsort((tie[idx], ivol[idx]))]
        half = (N_IVOL_GROUPS * np.arange(m)) // m
        for h in range(N_IVOL_GROUPS):
            cell = o[half == h]
            c = len(cell)
            if c == 0:
                continue
            n_long = max(1, c // LONG_DIV)
            ranked = cell[np.lexsort((tie[cell], b_vix[cell]))]
            picks.append(ranked[:n_long])
    if not picks:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(picks))


def formation_long_leg(r, valid, mkt, dvix, uni, tie, t, est_days):
    cols = np.nonzero(uni[t])[0]
    b1, b2, ivol, elig = exposures(r, valid, mkt, dvix, cols, t, est_days)
    cols = cols[elig]
    loc = long_leg(b1[elig], ivol[elig], b2[elig], tie[cols])
    return cols[loc]


def events(data, params):
    est_days = int(params["est_days"])
    if data.universe is None:
        raise ValueError("H-0155 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    r, valid = stock_returns(data)
    mkt = market_factor(r, valid, uni)
    dvix = vix_innovations(data.series[VIX_KEY])
    tie = tie_ranks(data.instruments)
    form = formation_rows(data.dates)
    T, N = uni.shape
    mask = np.zeros((T, N), dtype=bool)
    prev = None
    for t in np.nonzero(form)[0]:
        leg = formation_long_leg(r, valid, mkt, dvix, uni, tie, t, est_days)
        if prev is None:
            if len(leg) > 0:
                prev = leg            # first long leg: no previous formation to compare with, no events
            continue
        new = np.setdiff1d(leg, prev)
        mask[t, new] = True
        prev = leg
    mask &= uni & np.asarray(data.tradable, dtype=bool)
    side = np.where(mask, 1.0, 0.0)
    return mask, side
