"""H-0203 mechanism_test events: on a weekly decision date whose call_surge is in the top third of its own
trailing 252-session distribution, a member that newly enters the long leg (+1) or the short leg (-1) of the
gbm score book. Feature, walk-forward and book code copied from strategy.py (separate file by contract)."""
import numpy as np

from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

HORIZON = 5
RETRAIN_EVERY = 63
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 10000
TOP = 0.2
BOTTOM = 0.2
MIN_NAMES = 50
BETA_WIN, BETA_MIN = 252, 200
IVOL_WIN, IVOL_MIN = 21, 18
RET_WIN = 5
DV_SHORT, DV_LONG = 21, 252
CBOE_SHORT, CBOE_LONG = 5, 126
CBOE_MAX_AGE = 5
MKT_WIN = 5
SURGE_WIN = 252           # trailing distribution of call_surge (sessions, all must be valid)
SURGE_Q = 2.0 / 3.0       # top third


def rolling_sum(x, w):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if x.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def daily_returns(ret_co, ret_oc, close, listed):
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    px = np.isfinite(np.asarray(close, dtype=np.float64)) & np.asarray(listed, dtype=bool)
    valid = px.copy()
    valid[0] = False
    valid[1:] &= px[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def market_mean(r, valid, uni):
    mv = valid & np.asarray(uni, dtype=bool)
    n = mv.sum(axis=1)
    s = np.where(mv, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m


def beta_ivol(r, valid, m):
    v = valid & np.isfinite(m)[:, None]
    rv = np.where(v, r, 0.0)
    mv = np.where(v, m[:, None], 0.0)
    vf = v.astype(np.float64)

    n = rolling_sum(vf, BETA_WIN)
    sr = rolling_sum(rv, BETA_WIN)
    sm = rolling_sum(mv, BETA_WIN)
    srm = rolling_sum(rv * mv, BETA_WIN)
    smm = rolling_sum(mv * mv, BETA_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * smm - sm * sm
        beta = (n * srm - sr * sm) / den
    beta = np.where((n >= BETA_MIN) & (den > 1e-18), beta, np.nan)
    del sr, sm, srm, smm, den

    n2 = rolling_sum(vf, IVOL_WIN)
    sr2 = rolling_sum(rv, IVOL_WIN)
    sm2 = rolling_sum(mv, IVOL_WIN)
    srr2 = rolling_sum(rv * rv, IVOL_WIN)
    srm2 = rolling_sum(rv * mv, IVOL_WIN)
    smm2 = rolling_sum(mv * mv, IVOL_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        cxx = srr2 - sr2 * sr2 / n2
        cxy = srm2 - sr2 * sm2 / n2
        cyy = smm2 - sm2 * sm2 / n2
        var_e = (cxx - 2.0 * beta * cxy + beta * beta * cyy) / (n2 - 1.0)
    ivol = np.sqrt(np.maximum(var_e, 0.0))
    ivol = np.where((n2 >= IVOL_MIN) & np.isfinite(beta) & np.isfinite(var_e), ivol, np.nan)
    return beta, ivol


def compounded(r, w):
    r = np.asarray(r, dtype=np.float64)
    ok = np.isfinite(r)
    lg = np.where(ok, np.log(np.maximum(1.0 + np.where(ok, r, 0.0), 1e-12)), 0.0)
    n = rolling_sum(ok.astype(np.float64), w)
    s = rolling_sum(lg, w)
    return np.where(n >= w, np.expm1(s), np.nan)


def dv_ratio(dv):
    dv = np.asarray(dv, dtype=np.float64)
    ok = np.isfinite(dv)
    n1 = rolling_sum(ok.astype(np.float64), DV_SHORT)
    s1 = rolling_sum(dv, DV_SHORT)
    n2 = rolling_sum(ok.astype(np.float64), DV_LONG)
    s2 = rolling_sum(dv, DV_LONG)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = s1 / n1
        m2 = s2 / n2
        out = np.log(m1 / m2)
    good = (n1 >= IVOL_MIN) & (n2 >= BETA_MIN) & (m1 > 0) & (m2 > 0) & np.isfinite(out)
    return np.where(good, out, np.nan)


def cboe_features(call, pc):
    call = np.asarray(call, dtype=np.float64)
    pc = np.asarray(pc, dtype=np.float64)
    T = call.shape[0]
    fin = np.isfinite(call) & np.isfinite(pc)
    new = fin.copy()
    if T > 1:
        changed = (call[1:] != call[:-1]) | (pc[1:] != pc[:-1]) | ~fin[:-1]
        new[1:] &= changed
    idx = np.flatnonzero(new)
    surge = np.full(T, np.nan)
    dev = np.full(T, np.nan)
    if idx.size == 0:
        return surge, dev
    oc = call[idx]
    op = pc[idx]
    oc_s = rolling_sum(oc, CBOE_SHORT) / CBOE_SHORT
    oc_l = rolling_sum(oc, CBOE_LONG) / CBOE_LONG
    op_s = rolling_sum(op, CBOE_SHORT) / CBOE_SHORT
    op_l = rolling_sum(op, CBOE_LONG) / CBOE_LONG
    k = np.arange(idx.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        o_surge = np.where((k >= CBOE_LONG - 1) & (oc_s > 0) & (oc_l > 0), np.log(oc_s / oc_l), np.nan)
    o_dev = np.where(k >= CBOE_LONG - 1, op_s - op_l, np.nan)
    pos = np.cumsum(new) - 1
    has = pos >= 0
    p = np.maximum(pos, 0)
    age = np.arange(T) - idx[p] + 1
    fresh = has & (age <= CBOE_MAX_AGE)
    surge = np.where(fresh, o_surge[p], np.nan)
    dev = np.where(fresh, o_dev[p], np.nan)
    return surge, dev


def weekly_days(dates):
    d = np.asarray(dates, dtype="datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    return (nxt - d).astype(np.int64) > 1


def features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0203 needs the liq_n universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    r, valid = daily_returns(data.ret_co, data.ret_oc, data.close, data.listed)
    m = market_mean(r, valid, uni)
    beta, ivol = beta_ivol(r, valid, m)
    ret5 = compounded(r, RET_WIN)
    dvr = dv_ratio(data.dollar_volume)
    surge, dev = cboe_features(data.series["cboe_putcall_1d.equity_call"],
                               data.series["cboe_putcall_1d.equity_pc"])
    mkt5 = compounded(m, MKT_WIN)

    def member(x):
        return np.where(uni, x, np.nan)

    def date_level(x):
        return np.where(uni, np.asarray(x, dtype=np.float64)[:, None], np.nan).astype(np.float32)

    X = stack(
        cs_zscore(member(beta), uni),
        cs_zscore(member(ivol), uni),
        cs_zscore(member(ret5), uni),
        cs_zscore(member(dvr), uni),
        date_level(surge),
        date_level(dev),
        date_level(mkt5),
    )
    return X, uni, surge


def surge_top_third(surge):
    """call_surge[t] at or above the 2/3 quantile of call_surge over rows t-251..t (all 252 valid)."""
    s = np.asarray(surge, dtype=np.float64)
    T = s.shape[0]
    out = np.zeros(T, dtype=bool)
    for t in range(SURGE_WIN - 1, T):
        win = s[t - SURGE_WIN + 1:t + 1]
        if np.isfinite(win).all():
            out[t] = s[t] >= np.quantile(win, SURGE_Q)
    return out


def new_entrants(w, days):
    """+1 where w > 0 on a decision day and the stock was not long at the previous decision day; -1 likewise
    for the short leg. Before the first decision day everybody counts as flat."""
    T, N = w.shape
    side = np.zeros((T, N))
    prev_long = np.zeros(N, dtype=bool)
    prev_short = np.zeros(N, dtype=bool)
    for t in np.flatnonzero(days):
        lg = w[t] > 0
        sh = w[t] < 0
        side[t, lg & ~prev_long] = 1.0
        side[t, sh & ~prev_short] = -1.0
        prev_long, prev_short = lg, sh
    return side


def events(data, params):
    X, uni, surge = features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(
        X, y, horizon=HORIZON, model="gbm",
        params={"n_estimators": int(params["n_estimators"]), "num_leaves": int(params["num_leaves"]),
                "learning_rate": LEARNING_RATE, "min_child_samples": MIN_CHILD_SAMPLES},
        retrain_every=RETRAIN_EVERY, universe=uni,
    )
    score = np.where(uni, score, np.nan)
    w = score_weights(score, uni, top=TOP, bottom=BOTTOM, min_names=MIN_NAMES)
    w = np.where(np.isfinite(w) & uni, w, 0.0)
    days = weekly_days(data.dates)
    side = new_entrants(w, days)
    hot = days & surge_top_third(surge)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = hot[:, None] & (side != 0) & uni & tradable
    side = np.where(mask, side, 1.0)
    return mask, side
