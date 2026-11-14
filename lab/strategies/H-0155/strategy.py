"""H-0155: aggregate-volatility risk premium, LIQ-500 long-only, overlapping monthly cohorts.

Formation after the close of the last session of each calendar month t (approximated point-in-time as the
last weekday of the month; when that weekday is an exchange holiday, the first session of the next month).
Per LIQ-500 member at t: OLS over the last est_days sessions ending t-1 of the close-to-close total return on
the equal-weighted LIQ-500 market return and the daily VIX change (dVIX_d = VIXCLS_d - VIXCLS_(d-1)).
Eligible with >= 80 % valid days. Cells: b_mkt terciles x ivol halves; long leg = lowest-b_vix 20 % of each
cell (floor, at least 1). Cohorts of 1 / hold_months each (scaled up during start-up), re-equal-weighted over
remaining members at each formation; a held stock that leaves the universe is zeroed at once (cash to the
next formation).
"""
import numpy as np

PARAMS = {"est_days": 252, "hold_months": 6}

VIX_KEY = "fred_macro.VIXCLS"
MIN_VALID_SHARE = 0.8      # card: at least 80 % of est_days valid days
N_BETA_GROUPS = 3          # card: terciles of b_mkt
N_IVOL_GROUPS = 2          # card: median split of ivol
LONG_DIV = 5               # card: floor(0.2 x cell size) = cell_size // 5, at least 1


# ----------------------------------------------------------------------------------------------- calendar
def formation_rows(dates):
    """Bool (T,): row t is a monthly formation date, from the dates of rows <= t only.

    Primary rule: the next weekday after date t lies in another month (last weekday of the month;
    np.busday_offset ignores exchange holidays). Fallback: if the last weekday was a holiday (so no row
    of that month was flagged), the first session of the next month is the formation date."""
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


# ----------------------------------------------------------------------------------------------- inputs
def tie_ranks(instruments):
    """Rank of each instrument id for tie-breaking (ids E<number>: numeric order via (length, name))."""
    names = [str(x) for x in instruments]
    order = sorted(range(len(names)), key=lambda j: (len(names[j]), names[j]))
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    return rank


def vix_innovations(vix_aligned):
    """dVIX indexed by the stock-return day d.

    The aligned series (clock next_morning) holds at row t the last VIX close known after the close of t,
    i.e. the VIX of session t-1. So VIX_d = S[d+1] and dVIX_d = S[d+1] - S[d]; NaN when either is missing.
    The last row is NaN (its VIX is not known yet). dVIX_d for d <= t-1 uses S rows <= t only."""
    s = np.asarray(vix_aligned, dtype=np.float64)
    out = np.full(s.shape[0], np.nan)
    if s.shape[0] > 1:
        out[:-1] = s[1:] - s[:-1]
    return out


def stock_returns(data):
    """Close-to-close total return r (float32, 0 where invalid) and its validity mask (T, N).

    Valid on row d when the stock has a close on d and on d-1 (so ret_co spans one close-to-close step)."""
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
    """MKT_d: equal-weighted mean return on day d of the stocks that are universe members on day d."""
    mem = valid & uni
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, np.float32(0.0)).sum(axis=1, dtype=np.float64)
    out = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    out[ok] = tot[ok] / cnt[ok]
    return out


# ----------------------------------------------------------------------------------------------- estimation
def exposures(r, valid, mkt, dvix, cols, t, est_days):
    """OLS r = a + b_mkt MKT + b_vix dVIX over rows [t-est_days, t-1] for the columns `cols`.

    Returns b_mkt, b_vix, ivol (std of residuals, ddof 0) and the eligibility mask (>= 80 % of est_days
    valid days where r, MKT and dVIX all exist, non-singular design)."""
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
    """Local indices of the long leg: b_mkt terciles x ivol halves, lowest b_vix 20 % (floor, >= 1) per cell.

    All sorts break ties by the ticker-id rank `tie`."""
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
    """Global column indices of the long leg formed at row t (members at t that are eligible)."""
    cols = np.nonzero(uni[t])[0]
    b1, b2, ivol, elig = exposures(r, valid, mkt, dvix, cols, t, est_days)
    cols = cols[elig]
    loc = long_leg(b1[elig], ivol[elig], b2[elig], tie[cols])
    return cols[loc]


def prepare(data):
    if data.universe is None:
        raise ValueError("H-0155 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    r, valid = stock_returns(data)
    mkt = market_factor(r, valid, uni)
    dvix = vix_innovations(data.series[VIX_KEY])
    tie = tie_ranks(data.instruments)
    form = formation_rows(data.dates)
    return uni, r, valid, mkt, dvix, tie, form


# ----------------------------------------------------------------------------------------------- book
def target_weights(data, params):
    est_days = int(params["est_days"])
    hold = int(params["hold_months"])
    uni, r, valid, mkt, dvix, tie, form = prepare(data)
    T, N = uni.shape
    W = np.full((T, N), np.nan)
    cohorts = []
    decided = []
    for t in np.nonzero(form)[0]:
        leg = formation_long_leg(r, valid, mkt, dvix, uni, tie, t, est_days)
        if not cohorts and len(leg) == 0:
            continue                      # no cohort exists yet: no decision
        cohorts.append(leg)
        live = cohorts[-hold:]
        share = 1.0 / len(live)           # start-up: existing cohorts scaled up equally
        w = np.zeros(N)
        for c in live:
            rem = c[uni[t, c]]
            if len(rem) > 0:
                w[rem] += share / len(rem)
        W[t] = w
        decided.append(t)
    # exits between formations: a held stock leaving the universe is zeroed (its share stays in cash)
    for i, t in enumerate(decided):
        end = decided[i + 1] if i + 1 < len(decided) else T
        if end - t <= 1:
            continue
        hcols = np.nonzero(W[t] > 0)[0]
        if len(hcols) == 0:
            continue
        out = ~uni[t + 1:end][:, hcols]
        W[t + 1:end, hcols] = np.where(out, 0.0, np.nan)
    return W
