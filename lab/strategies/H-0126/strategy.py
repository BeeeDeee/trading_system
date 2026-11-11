"""H-0126 Credit-market news into levered equities, learned.

LIQ-500, weekly decisions (first session of each week). On credit-news dates (|credit-shock z| >= z_news) a
LightGBM score (walk-forward, framework) on six balance-sheet / credit features ranks the members; within
terciles of 252-session market beta the top tenth is long (+0.5 shared) and the bottom tenth short (-0.5 shared),
held to the next weekly decision. Other decision dates are flat.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "z_news": 1.0}

HORIZON = 5            # ml.horizon_days: label = next open to the open 5 sessions later
SHOCK_WIN = 5          # sessions of the spread / yield / market change
REG_WIN = 252          # trailing overlapping windows in the shock regression
STALE_MAX = 5          # a series without a new value for more than 5 sessions: no shock
VOL_WIN = 63           # equity volatility window
BETA_WIN = 252         # beta window
BETA_MIN = 200         # minimum valid days for beta
MIN_NEWS_DATES = 40    # credit-news dates with realized labels before the first score
MIN_SCORED = 60        # fewer scored members on a news date: flat
RETRAIN_EVERY = 252    # ml.retrain_every_days
COV_LO, COV_HI = -5.0, 30.0
LEG_FRAC = 0.10
MIN_LEG = 3
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}


# ----------------------------------------------------------------------------------------------- helpers
def decision_mask(dates):
    """Bool (T,) first trading session of each week (point-in-time)."""
    ps = np.asarray(period_starts(dates, "W"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.copy()
    m = np.zeros(T, dtype=bool)
    m[ps.astype(np.int64)] = True
    return m


def daily_returns(data):
    """(T, N) float32 close-to-close total return, NaN where the stock is not listed."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    r = r.astype(np.float32)
    r[~np.asarray(data.listed, dtype=bool)] = np.nan
    return r


def market_return(r, uni):
    """(T,) equal-weight mean daily total return of the members of each day (NaN if none)."""
    m = uni & np.isfinite(r)
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1, dtype=np.float64)
    out = np.full(len(cnt), np.nan)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


def rows_since_change(x):
    """(T,) number of rows since the series last took a new value (large before the first value)."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    idx = np.arange(T)
    prev = np.concatenate([[np.nan], x[:-1]])
    changed = np.isfinite(x) & ~(x == prev)
    last = np.maximum.accumulate(np.where(changed, idx, -1)) if T else idx
    return np.where(last >= 0, idx - last, 10 ** 9)


def shock_inputs(baa, y10, rm, k=SHOCK_WIN):
    """Per row u: dS, dY (change over the k rows ending u; row u holds the day-(u-1) value under the
    next_morning clock) and M (compounded market return of sessions u-k..u-1)."""
    baa = np.asarray(baa, dtype=np.float64)
    y10 = np.asarray(y10, dtype=np.float64)
    T = len(rm)
    S = baa - y10
    dS = np.full(T, np.nan)
    dY = np.full(T, np.nan)
    M = np.full(T, np.nan)
    if T > k:
        dS[k:] = S[k:] - S[:-k]
        dY[k:] = y10[k:] - y10[:-k]
        g = 1.0 + np.asarray(rm, dtype=np.float64)
        prod = np.ones(T - k)
        for i in range(1, k + 1):            # sessions u-1 .. u-k
            prod = prod * g[k - i:T - i]
        M[k:] = prod - 1.0
    return dS, dY, M


def credit_shock_z(baa, y10, rm, rows, win=REG_WIN, k=SHOCK_WIN, stale_max=STALE_MAX):
    """(T,) z of today's residual of dS on [1, M, dY] over the trailing `win` windows (ending t-1), only on
    `rows`; NaN where history is short or a series is stale."""
    T = len(rm)
    dS, dY, M = shock_inputs(baa, y10, rm, k)
    fresh = (rows_since_change(baa) <= stale_max) & (rows_since_change(y10) <= stale_max)
    z = np.full(T, np.nan)
    for t in np.flatnonzero(rows):
        if t < win - 1 or not fresh[t]:
            continue
        sl = slice(t - win + 1, t + 1)
        yv, mv, dv = dS[sl], M[sl], dY[sl]
        if not (np.isfinite(yv).all() and np.isfinite(mv).all() and np.isfinite(dv).all()):
            continue
        A = np.column_stack([np.ones(win), mv, dv])
        coef = np.linalg.lstsq(A, yv, rcond=None)[0]
        res = yv - A @ coef
        sd = res.std()
        if sd > 1e-12:
            z[t] = res[-1] / sd
    return z


def last_change_rows(x):
    """(T, N) int32: row of the latest new value of each column at or before t (-1 if none)."""
    x = np.asarray(x)
    T, N = x.shape
    out = np.full((T, N), -1, dtype=np.int32)
    if T == 0:
        return out
    fin = np.isfinite(x)
    out[0] = np.where(fin[0], 0, -1)
    for t in range(1, T):
        ch = fin[t] & ~(x[t] == x[t - 1])
        out[t] = np.where(ch, t, out[t - 1])
    return out


def coverage(ebit, intexp):
    """TTM EBIT / TTM interest expense in [-5, 30]; interest expense missing or <= 0 gives 30."""
    ebit = np.asarray(ebit, dtype=np.float64)
    ie = np.asarray(intexp, dtype=np.float64)
    no_ie = ~np.isfinite(ie) | (ie <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.clip(ebit / np.where(no_ie, 1.0, ie), COV_LO, COV_HI)
    return np.where(no_ie, COV_HI, c)


def stock_features(t, r, rm, g, mc, mc_anchor, debt, ebit, intexp, z_t, members):
    """Raw features (lev, cov, vol, signed lev, response, beta) of row t, (6, N) float64; a stock with any
    missing feature (or not a member) is NaN in all six."""
    N = r.shape[1]
    nanv = np.full(N, np.nan)
    # (f) beta over the last BETA_WIN sessions, at least BETA_MIN valid days
    lo = max(0, t - BETA_WIN + 1)
    Y = r[lo:t + 1].astype(np.float64)
    x = rm[lo:t + 1]
    valid = np.isfinite(Y) & np.isfinite(x)[:, None]
    n = valid.sum(axis=0)
    xm = np.where(valid, x[:, None], 0.0)
    ym = np.where(valid, Y, 0.0)
    nn = np.maximum(n, 1)
    sx, sy = xm.sum(0), ym.sum(0)
    var = (xm * xm).sum(0) - sx * sx / nn
    cov = (xm * ym).sum(0) - sx * sy / nn
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where((n >= BETA_MIN) & (var > 0), cov / var, np.nan)
    # (c) volatility of the last VOL_WIN daily total returns
    if t >= VOL_WIN - 1:
        V = r[t - VOL_WIN + 1:t + 1].astype(np.float64)
        okv = np.isfinite(V).all(axis=0)
        vol = np.where(okv, np.where(np.isfinite(V), V, 0.0).std(axis=0), np.nan)
    else:
        vol = nanv
    # (e) own response over sessions t-5..t minus beta x market over the same sessions
    if t >= SHOCK_WIN:
        R = r[t - SHOCK_WIN:t + 1].astype(np.float64)
        own = np.prod(1.0 + R, axis=0) - 1.0
        mk = np.prod(1.0 + rm[t - SHOCK_WIN:t + 1]) - 1.0
        resp = own - beta * mk
    else:
        resp = nanv
    # (a) market leverage: debt / (debt + E), E = filing marketcap grown by total return since the filing
    a = mc_anchor[t].astype(np.int64)
    mct = np.asarray(mc[t], dtype=np.float64)
    okm = (a >= 0) & np.isfinite(mct)
    E = np.full(N, np.nan)
    if okm.any():
        span = int(t - a[okm].min()) + 1
        rc = np.cumprod(g[t - span + 1:t + 1][::-1].astype(np.float64), axis=0)   # rc[k] = prod g[t-k..t]
        kk = np.where(okm, t - a, 0)
        E = np.where(okm, mct * rc[kk, np.arange(N)], np.nan)
    d = np.asarray(debt[t], dtype=np.float64)
    d = np.where(np.isfinite(d), d, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.where(np.isfinite(E) & (E > 0), d / (d + E), np.nan)
    # (b) interest coverage
    cv = coverage(ebit[t], intexp[t])
    # (d) signed leverage
    slev = lev * z_t
    F = np.vstack([lev, cv, vol, slev, resp, beta])
    ok = members & np.isfinite(F).all(axis=0)
    F[:, ~ok] = np.nan
    return F


def ready_mask(has_rows, h, min_dates):
    """(T,) True once at least `min_dates` news dates with training rows have realized labels (d + h + 1 <= t)."""
    T = len(has_rows)
    cum = np.cumsum(has_rows)
    realized = np.zeros(T, dtype=np.int64)
    if T > h + 1:
        realized[h + 1:] = cum[:T - h - 1]
    return realized >= min_dates


def pipeline(data, params):
    """Score (T, N), raw beta (T, N), decision mask (T,), news mask (T,)."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0126 needs the liq_n point-in-time universe")
    uni = np.asarray(uni, dtype=bool)
    T, N = uni.shape
    baa = data.series["fred_moody_yields_1d.DBAA"]
    y10 = data.series["fred_macro.DGS10"]
    debt = data.extras["sf1_arq_debt"]
    mc = data.extras["sf1_arq_marketcap"]
    ebit = data.extras["sf1_art_ebit"]
    intexp = data.extras["sf1_art_intexp"]
    z_news = float(params["z_news"])

    r = daily_returns(data)
    g = np.nan_to_num((1.0 + np.asarray(data.ret_co, dtype=np.float32)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float32)),
                      nan=1.0)
    rm = market_return(r, uni)
    dec = decision_mask(data.dates)
    z = credit_shock_z(baa, y10, rm, dec)
    news = dec & np.isfinite(z) & (np.abs(z) >= z_news)

    mc_anchor = last_change_rows(mc)
    feats = [np.full((T, N), np.nan, dtype=np.float32) for _ in range(6)]
    beta_raw = np.full((T, N), np.nan, dtype=np.float32)
    has_rows = np.zeros(T, dtype=bool)
    for t in np.flatnonzero(news):
        F = stock_features(t, r, rm, g, mc, mc_anchor, debt, ebit, intexp, z[t], uni[t])
        for i in range(6):
            feats[i][t] = F[i]
        beta_raw[t] = F[5]
        has_rows[t] = np.isfinite(F[0]).any()
    del mc_anchor, g

    X = stack(*[cs_zscore(f, uni) for f in feats])
    del feats
    h = HORIZON
    y = cs_rank(forward_return(data, h), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = np.asarray(walk_forward(X, y, horizon=h, model="gbm", params=gbm, retrain_every=RETRAIN_EVERY,
                                    universe=uni), dtype=np.float64)
    del X, y

    keep = (news & ready_mask(has_rows, h, MIN_NEWS_DATES))[:, None] & uni
    score = np.where(keep, score, np.nan)
    return score, beta_raw, dec, news


def legs(score_row, beta_row, member_row):
    """Long and short column indices of one news date (empty if fewer than MIN_SCORED scored members)."""
    ok = np.isfinite(score_row) & np.isfinite(beta_row) & member_row
    cols = np.flatnonzero(ok)
    empty = np.zeros(0, dtype=np.int64)
    if len(cols) < MIN_SCORED:
        return empty, empty
    order = cols[np.argsort(beta_row[cols], kind="stable")]
    longs, shorts = [], []
    for grp in np.array_split(order, 3):
        k = int(np.floor(len(grp) * LEG_FRAC + 1e-9))
        if k < MIN_LEG:
            continue
        o = grp[np.argsort(score_row[grp], kind="stable")]
        shorts.append(o[:k])
        longs.append(o[len(o) - k:])
    if not longs:
        return empty, empty
    return np.concatenate(longs), np.concatenate(shorts)


def book(score, beta_raw, dec, news, uni):
    """(T, N) weights: decision rows flat unless news (legs), positions closed when they leave the universe."""
    T, N = score.shape
    W = np.full((T, N), np.nan)
    W[dec] = 0.0
    dec_rows = np.flatnonzero(dec)
    for i, t in enumerate(dec_rows):
        if not news[t]:
            continue
        lo, sh = legs(score[t], beta_raw[t], uni[t])
        if len(lo) == 0:
            continue
        W[t, lo] = 0.5 / len(lo)
        W[t, sh] = -0.5 / len(sh)
        held = np.zeros(N, dtype=bool)
        held[lo] = True
        held[sh] = True
        nd = dec_rows[i + 1] if i + 1 < len(dec_rows) else T
        for s in range(t + 1, nd):
            out = held & ~uni[s]
            if out.any():
                W[s, out] = 0.0
                held &= ~out
    return W


def target_weights(data, params):
    uni = np.asarray(data.universe, dtype=bool)
    score, beta_raw, dec, news = pipeline(data, params)
    return book(score, beta_raw, dec, news, uni)
