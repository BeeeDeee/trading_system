"""H-0116 - Delayed stock response to rate, dollar and credit news, learned.

LIQ-500, weekly beta-balanced long/short in macro-shock weeks, LightGBM on estimated factor exposure x
factor-specific shock, hold until the next weekly decision (about 5 sessions).

Index conventions
- Stock arrays: row t = day t (known after the close of t).
- Macro series arrive aligned with clock next_morning: row t holds the last value stamped on or before t-1.
  The *daily change of day s* (dF[s]) is therefore aligned[s+1] - aligned[s] and is known only at row s+1.
  At decision row t we use dF[s] for s <= t-1 only (values stamped on or before t-1).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"shock_z": 1.5, "top": 0.2}

# fixed by the card (not in signal.params)
HORIZON = 5
RETRAIN_EVERY = 63
GBM_PARAMS = {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}
REG_DAYS = 252          # trailing sessions of every regression
MIN_STOCK_OBS = 200     # valid daily returns a stock needs to be scored
MIN_FACTOR_OBS = 200    # valid (factor change, MARKET) pairs for the factor-on-market regression
SHOCK_DAYS = 10
REALIZED_DAYS = 10
IVOL_DAYS = 63
VIS_DAYS = 63
MIN_NAMES = 50
LEG_MIN, LEG_MAX = 0.35, 0.65
N_FACTORS = 3           # rate, dollar, credit


# ----------------------------------------------------------------------------------------------- helpers
def _as_mask(x, T):
    a = np.asarray(x)
    if a.dtype == bool and a.shape == (T,):
        return a.copy()
    m = np.zeros(T, dtype=bool)
    if a.size:
        m[a.astype(np.int64)] = True
    return m


def _ffill(x):
    """Forward fill (point-in-time); leading NaNs stay NaN."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    ok = np.isfinite(x)
    idx = np.maximum.accumulate(np.where(ok, np.arange(T), 0))
    out = x[idx]
    out[~np.maximum.accumulate(ok)] = np.nan
    return out


def _day_changes(level):
    """dF[s] = change of day s = level known at row s+1 minus level known at row s (NaN for the last row).
    A missing macro day carries no change; the change over a gap lands on the next day with a value."""
    lv = _ffill(level)
    d = np.full(lv.shape[0], np.nan)
    if lv.shape[0] > 1:
        d[:-1] = lv[1:] - lv[:-1]
    return d


def _excl_cumsum(x):
    """C[k] = sum of x[s] for s < k, shape (T+1, ...)."""
    x = np.asarray(x, dtype=np.float64)
    C = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    if x.shape[0]:
        np.cumsum(x, axis=0, out=C[1:])
    return C


def _win_prev(C, w):
    """Sum over rows [t-w, t-1] (window ending the day before t), for t = 0..T-1."""
    T = C.shape[0] - 1
    lo = np.maximum(np.arange(T) - w, 0)
    return C[:T] - C[lo]


def _win_incl(C, w):
    """Sum over rows [t-w+1, t] (window ending t), for t = 0..T-1."""
    T = C.shape[0] - 1
    lo = np.maximum(np.arange(1, T + 1) - w, 0)
    return C[1:] - C[lo]


def _stock_returns(ret_co, ret_oc, listed, close):
    """Daily close-to-close total return and its validity (listed and priced on t-1 and t)."""
    co = np.asarray(ret_co, dtype=np.float64)
    oc = np.asarray(ret_oc, dtype=np.float64)
    r = (1.0 + co) * (1.0 + oc) - 1.0
    lst = np.asarray(listed, dtype=bool)
    px = np.isfinite(np.asarray(close, dtype=np.float64))
    valid = np.zeros(r.shape, dtype=bool)
    if r.shape[0] > 1:
        valid[1:] = lst[1:] & lst[:-1] & px[1:] & px[:-1]
    valid &= np.isfinite(r)
    r = np.where(valid, r, np.nan)
    return r, valid


def _market(r, valid, uni):
    """MARKET[t]: equal-weight mean daily total return of the members of day t (NaN without any)."""
    m = valid & uni
    n = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def _factor_shock(dF, mkt):
    """For each decision row t: OLS (with intercept) of the factor's daily change on MARKET over days
    [t-252, t-1]; shock = sum of residuals over days [t-10, t-1]; z = shock / (residual std x sqrt(10)).
    Returns a, b, shock, z (each (T,), NaN where not computable)."""
    dF = np.asarray(dF, dtype=np.float64)
    mkt = np.asarray(mkt, dtype=np.float64)
    p = np.isfinite(dF) & np.isfinite(mkt)
    x = np.where(p, mkt, 0.0)
    y = np.where(p, dF, 0.0)
    pf = p.astype(np.float64)
    n = _win_prev(_excl_cumsum(pf), REG_DAYS)
    sx = _win_prev(_excl_cumsum(x), REG_DAYS)
    sy = _win_prev(_excl_cumsum(y), REG_DAYS)
    sxx = _win_prev(_excl_cumsum(x * x), REG_DAYS)
    sxy = _win_prev(_excl_cumsum(x * y), REG_DAYS)
    syy = _win_prev(_excl_cumsum(y * y), REG_DAYS)
    n10 = _win_prev(_excl_cumsum(pf), SHOCK_DAYS)
    sx10 = _win_prev(_excl_cumsum(x), SHOCK_DAYS)
    sy10 = _win_prev(_excl_cumsum(y), SHOCK_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1.0)
        cxx = sxx - sx * sx / nn
        cxy = sxy - sx * sy / nn
        cyy = syy - sy * sy / nn
        ok = (n >= MIN_FACTOR_OBS) & (cxx > 0)
        b = np.where(ok, cxy / np.where(cxx > 0, cxx, 1.0), np.nan)
        a = np.where(ok, (sy - b * sx) / nn, np.nan)
        ssr = np.maximum(cyy - b * cxy, 0.0)
        sd = np.sqrt(ssr / np.maximum(n - 1.0, 1.0))
        shock = np.where(ok, sy10 - a * n10 - b * sx10, np.nan)
        z = np.where(ok & (sd > 0), shock / (sd * np.sqrt(SHOCK_DAYS)), np.nan)
    return a, b, shock, z


def _exposures(r, valid, mkt, dF, a, b, month_mask):
    """On the first session m of each month: per stock OLS of the daily total return on MARKET and the three
    market-residual factor changes over days [m-252, m-1] (factor residuals from the factor-on-market regression
    estimated at row m, i.e. over the same 252 days). Needs >= 200 valid returns. Betas apply to rows of that
    month. ivol[t] = std of the stock's residual return over days [t-63, t-1], with the betas in force at t.
    Returns beta (4, T, N) float32 [mkt, rate, dollar, credit] and ivol (T, N) float32."""
    T, N = r.shape
    betas = [np.full((T, N), np.nan, dtype=np.float32) for _ in range(1 + N_FACTORS)]
    ivol = np.full((T, N), np.nan, dtype=np.float32)
    starts = np.flatnonzero(month_mask)
    ends = np.append(starts[1:], T)
    K = 2 + N_FACTORS
    for m, e in zip(starts, ends):
        m = int(m)
        e = int(e)
        if m < REG_DAYS:
            continue
        am = a[:, m]
        bm = b[:, m]
        if not (np.all(np.isfinite(am)) and np.all(np.isfinite(bm))):
            continue
        s0 = m - REG_DAYS
        mk = mkt[s0:m]
        X = np.empty((REG_DAYS, K))
        X[:, 0] = 1.0
        X[:, 1] = mk
        for f in range(N_FACTORS):
            X[:, 2 + f] = dF[f, s0:m] - am[f] - bm[f] * mk
        rowok = np.isfinite(X).all(axis=1)
        X0 = np.where(rowok[:, None], X, 0.0)
        Wm = valid[s0:m] & rowok[:, None]
        cnt = Wm.sum(axis=0)
        ok = np.flatnonzero(cnt >= MIN_STOCK_OBS)
        if ok.size == 0:
            continue
        Wo = Wm[:, ok]
        XX = (X0[:, :, None] * X0[:, None, :]).reshape(REG_DAYS, K * K)
        XtX = (Wo.T.astype(np.float64) @ XX).reshape(ok.size, K, K)
        R = np.where(Wo, r[s0:m][:, ok], 0.0)
        Xty = R.T @ X0
        coef = np.einsum("kij,kj->ki", np.linalg.pinv(XtX), Xty)
        for i in range(1 + N_FACTORS):
            betas[i][m:e, ok] = coef[:, 1 + i][None, :].astype(np.float32)
        # residual returns of days [m-63, e-2] with this month's betas (elementwise, row-wise deterministic)
        d0 = m - IVOL_DAYS
        d1 = e - 1
        if d1 <= d0:
            continue
        mkd = mkt[d0:d1]
        fit = mkd[:, None] * coef[:, 1][None, :]
        rowok_d = np.isfinite(mkd)
        for f in range(N_FACTORS):
            ef = dF[f, d0:d1] - am[f] - bm[f] * mkd
            rowok_d &= np.isfinite(ef)
            fit = fit + ef[:, None] * coef[:, 2 + f][None, :]
        Vd = valid[d0:d1][:, ok] & rowok_d[:, None]
        res = np.where(Vd, r[d0:d1][:, ok] - fit, 0.0)
        C1 = _excl_cumsum(res)
        C2 = _excl_cumsum(res * res)
        Cn = _excl_cumsum(Vd.astype(np.float64))
        hi = np.arange(m, e) - d0
        lo = hi - IVOL_DAYS
        n = Cn[hi] - Cn[lo]
        s1 = C1[hi] - C1[lo]
        s2 = C2[hi] - C2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
            sd = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
        ivol[m:e, ok] = sd.astype(np.float32)
    return betas, ivol


def _compute(r, valid, uni, dv, levels, month_mask):
    """All point-in-time inputs of the model and the book.
    levels: (rate_level, dollar_level, credit_level), each (T,) aligned macro series (rows hold values <= t-1)."""
    T, N = r.shape
    uni = np.asarray(uni, dtype=bool)
    mkt = _market(r, valid, uni)
    dF = np.vstack([_day_changes(lv) for lv in levels])            # (3, T)
    a = np.full((N_FACTORS, T), np.nan)
    b = np.full((N_FACTORS, T), np.nan)
    shock = np.full((N_FACTORS, T), np.nan)
    z = np.full((N_FACTORS, T), np.nan)
    for f in range(N_FACTORS):
        a[f], b[f], shock[f], z[f] = _factor_shock(dF[f], mkt)
    betas, ivol = _exposures(r, valid, mkt, dF, a, b, month_mask)
    beta_mkt = betas[0]
    implied = [(betas[1 + f] * shock[f][:, None].astype(np.float32)) for f in range(N_FACTORS)]
    # realized: 10-session total return ending t minus beta_mkt x MARKET return over the same sessions
    rr = np.where(valid, r, 0.0)
    okr = valid & (rr > -1.0)
    lr = np.where(okr, np.log1p(np.where(okr, rr, 0.0)), 0.0)
    del rr
    s10 = _win_incl(_excl_cumsum(lr), REALIZED_DAYS)
    n10 = _win_incl(_excl_cumsum(okr.astype(np.float64)), REALIZED_DAYS)
    stock10 = np.where(n10 >= REALIZED_DAYS, np.expm1(s10), np.nan)
    del lr, okr, s10, n10
    mok = np.isfinite(mkt) & (mkt > -1.0)
    ml = np.where(mok, np.log1p(np.where(mok, mkt, 0.0)), 0.0)
    ms = _win_incl(_excl_cumsum(ml), REALIZED_DAYS)
    mn = _win_incl(_excl_cumsum(mok.astype(np.float64)), REALIZED_DAYS)
    mkt10 = np.where(mn >= REALIZED_DAYS, np.expm1(ms), np.nan)
    realized = (stock10 - beta_mkt.astype(np.float64) * mkt10[:, None]).astype(np.float32)
    del stock10
    # visibility: log of mean dollar volume over the 63 sessions ending t
    dvv = np.asarray(dv, dtype=np.float64)
    okd = np.isfinite(dvv)
    sd_ = _win_incl(_excl_cumsum(np.where(okd, dvv, 0.0)), VIS_DAYS)
    nd_ = _win_incl(_excl_cumsum(okd.astype(np.float64)), VIS_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dv = np.where(nd_ > 0, sd_ / np.maximum(nd_, 1.0), np.nan)
        visibility = np.where(mean_dv > 0, np.log(np.where(mean_dv > 0, mean_dv, 1.0)), np.nan).astype(np.float32)
    del sd_, nd_, mean_dv, dvv, okd
    implied_sum = implied[0] + implied[1] + implied[2]
    feats = [implied[0], implied[1], implied[2], realized, ivol, visibility]
    return {"feats": feats, "z": z, "shock": shock, "beta_mkt": beta_mkt, "betas": betas, "implied_sum": implied_sum,
            "mkt": mkt, "dF": dF}


def _gate(z, shock_z):
    """Gate on for row t when max over factors of |z_f| >= shock_z (factors without a valid z ignored)."""
    az = np.where(np.isfinite(z), np.abs(z), -np.inf)
    return az.max(axis=0) >= float(shock_z)


def _legs(score, uni, beta_mkt, gate, week_rows, top, min_names=MIN_NAMES):
    """For each weekly decision row: (long columns, short columns); empty when the gate is off, fewer than
    min_names scored members, or no model scores yet."""
    out = {}
    empty = np.zeros(0, dtype=np.int64)
    for d in week_rows:
        d = int(d)
        if not gate[d]:
            out[d] = (empty, empty)
            continue
        cand = np.flatnonzero(uni[d] & np.isfinite(score[d]) & np.isfinite(beta_mkt[d]))
        n = cand.size
        k = min(int(np.floor(float(top) * n + 1e-9)), n // 2)
        if n < min_names or k < 1:
            out[d] = (empty, empty)
            continue
        order = np.argsort(-np.asarray(score[d, cand], dtype=np.float64), kind="stable")
        out[d] = (cand[order[:k]], cand[order[n - k:]])
    return out


def _book(legs, beta_mkt, T, N):
    """Weekly rows: long leg L/k each, short leg -S/k each, L + S = 1, L x mean beta(long) = S x mean
    beta(short), L clipped to [0.35, 0.65]. Other rows NaN (keep positions)."""
    W = np.full((T, N), np.nan)
    for d, (lo, sh) in legs.items():
        W[d] = 0.0
        if lo.size == 0:
            continue
        bl = float(np.mean(beta_mkt[d, lo]))
        bs = float(np.mean(beta_mkt[d, sh]))
        den = bl + bs
        L = bs / den if abs(den) > 1e-12 else 0.5
        L = min(max(L, LEG_MIN), LEG_MAX)
        S = 1.0 - L
        W[d, lo] = L / lo.size
        W[d, sh] = -S / sh.size
    return W


def _close_exits(W, uni, week_mask):
    """A held name that leaves the universe is closed on the first row where it is not a member."""
    T = W.shape[0]
    last = np.maximum.accumulate(np.where(week_mask, np.arange(T), -1)) if T else np.zeros(0, dtype=np.int64)
    has = last >= 0
    held = np.zeros(W.shape)
    held[has] = W[last[has]]
    held = np.nan_to_num(held, nan=0.0)
    close = (~week_mask)[:, None] & (held != 0) & ~uni
    W = W.copy()
    W[close] = 0.0
    return W


def _events_from_legs(legs, week_rows, implied_sum, uni, tradable, T, N):
    """Mechanism events (identical copy in diagnostic.py, kept here so the tests can reach it): a stock that
    enters the long (short) leg, not in that leg at the previous weekly decision, with a positive (negative)
    unstandardized implied sum; only members tradable on the event row."""
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    prev_long = np.zeros(0, dtype=np.int64)
    prev_short = np.zeros(0, dtype=np.int64)
    for d in week_rows:
        d = int(d)
        lo, sh = legs[d]
        new_lo = np.setdiff1d(lo, prev_long)
        new_sh = np.setdiff1d(sh, prev_short)
        el = new_lo[implied_sum[d, new_lo] > 0]
        es = new_sh[implied_sum[d, new_sh] < 0]
        mask[d, el] = True
        side[d, el] = 1.0
        mask[d, es] = True
        side[d, es] = -1.0
        prev_long, prev_short = lo, sh
    mask &= uni & tradable
    side = np.where(mask, side, 0.0)
    return mask, side


def _inputs(data):
    T = len(data.dates)
    if data.universe is None:
        raise ValueError("H-0116 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    r, valid = _stock_returns(data.ret_co, data.ret_oc, data.listed, data.close)
    s = data.series
    dgs10 = np.asarray(s["fred_macro.DGS10"], dtype=np.float64)
    eur = np.asarray(s["fred_macro.DEXUSEU"], dtype=np.float64)
    baa = np.asarray(s["fred_moody_yields_1d.DBAA"], dtype=np.float64)
    aaa = np.asarray(s["fred_moody_yields_1d.DAAA"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        dollar = -np.log(np.where(eur > 0, eur, np.nan))       # DOLLAR = minus log change of DEXUSEU
    credit = _ffill(baa) - _ffill(aaa)                          # CREDIT = change of DBAA - DAAA
    levels = (dgs10, dollar, credit)
    month_mask = _as_mask(period_starts(data.dates, "M"), T)
    week_mask = _as_mask(period_starts(data.dates, "W"), T)
    return uni, r, valid, levels, month_mask, week_mask


def _scores(data, feats, gate, uni):
    zs = []
    for i in range(len(feats)):                                 # z-score one by one, free the raw feature (RAM)
        zs.append(cs_zscore(feats[i], uni))
        feats[i] = None
    X = stack(*zs)
    del zs
    y = np.array(cs_rank(forward_return(data, HORIZON), uni), dtype=np.float32, copy=True)
    y[~gate] = np.nan                                           # training rows: member-days with the gate on
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=dict(GBM_PARAMS),
                         retrain_every=RETRAIN_EVERY, universe=uni, train_days=None, seed=0)
    return np.asarray(score, dtype=np.float64)


def target_weights(data, params):
    uni, r, valid, levels, month_mask, week_mask = _inputs(data)
    T, N = r.shape
    c = _compute(r, valid, uni, data.dollar_volume, levels, month_mask)
    del r, valid
    beta_mkt = c["beta_mkt"]
    gate = _gate(c["z"], params["shock_z"])
    feats = c["feats"]
    c = None                                                    # free betas / implied_sum before the model
    score = _scores(data, feats, gate, uni)
    legs = _legs(score, uni, beta_mkt, gate, np.flatnonzero(week_mask), params["top"])
    W = _book(legs, beta_mkt, T, N)
    return _close_exits(W, uni, week_mask)
