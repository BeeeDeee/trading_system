"""H-0203: aggregate call-buying waves x stock speculativeness, LightGBM 5-day score, LIQ-500 weekly
dollar-neutral book.

Features (card ml.features, in this order):
  stock-level, z-scored across members of date t (cs_zscore):
    beta_252  OLS slope of daily total return on the equal-weight member mean, 252 sessions, >= 200 valid
    ivol_21   std of the residual r - beta_252(t) * m over 21 sessions, >= 18 valid
    ret_5     compounded total return over the last 5 sessions
    dv_ratio  log(mean dollar volume 21 sessions / mean dollar volume 252 sessions)
  date-level, raw (same value for every member):
    call_surge log(mean equity_call over last 5 Cboe obs / mean over last 126 Cboe obs)
    pc_dev     mean equity_pc over last 5 Cboe obs - mean over last 126 Cboe obs
    mkt_ret_5  compounded equal-weight member mean return over the last 5 sessions
Label: forward_return(data, 5), ranked across members (cs_rank). Model gbm, expanding walk-forward, refit 63.
Book: weekly (Fridays), long top 20 % / short bottom 20 %, 0.5 / 0.5, flat with fewer than 50 scored members.
"""
import numpy as np

from lab.framework.api import only_on
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"n_estimators": 100, "num_leaves": 15}

# fixed settings of the card (not in signal.params)
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
CBOE_MAX_AGE = 5          # sessions; value stamped more than 5 sessions before t -> missing
MKT_WIN = 5


# ----------------------------------------------------------------------------------------------- helpers

def rolling_sum(x, w):
    """Trailing sum over rows t-w+1..t (NaN treated as 0); works for views shorter than w."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if x.shape[0] > w:
        out[w:] = c[w:] - c[:-w]
    return out


def daily_returns(ret_co, ret_oc, close, listed):
    """Close-to-close total return (1+ret_co)(1+ret_oc)-1, valid only where a price exists today and yesterday."""
    r = (1.0 + np.asarray(ret_co, dtype=np.float64)) * (1.0 + np.asarray(ret_oc, dtype=np.float64)) - 1.0
    px = np.isfinite(np.asarray(close, dtype=np.float64)) & np.asarray(listed, dtype=bool)
    valid = px.copy()
    valid[0] = False
    valid[1:] &= px[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def market_mean(r, valid, uni):
    """Equal-weight mean return of the members of each day (members with a valid return)."""
    mv = valid & np.asarray(uni, dtype=bool)
    n = mv.sum(axis=1)
    s = np.where(mv, r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m


def beta_ivol(r, valid, m):
    """beta_252 (OLS slope with intercept, >= 200 valid of 252) and ivol_21 (std, ddof 1, of r - beta_t * m
    over the last 21 sessions, >= 18 valid), both at each row t from rows <= t."""
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
    """Compounded return over the last w rows, NaN unless all w returns are valid. Works on 1-D or 2-D."""
    r = np.asarray(r, dtype=np.float64)
    ok = np.isfinite(r)
    lg = np.where(ok, np.log(np.maximum(1.0 + np.where(ok, r, 0.0), 1e-12)), 0.0)
    n = rolling_sum(ok.astype(np.float64), w)
    s = rolling_sum(lg, w)
    return np.where(n >= w, np.expm1(s), np.nan)


def dv_ratio(dv):
    """log(mean dollar volume of the last 21 sessions / mean of the last 252), means over sessions with a value;
    at least 18 of 21 and 200 of 252 values (the same minimums ivol_21 and beta_252 already impose)."""
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
    """call_surge and pc_dev per row from the aligned (forward-filled, already next_morning-lagged) Cboe series.
    A new Cboe session is a row where either value changes; means run over the last 5 / 126 such sessions;
    missing when the latest value is more than CBOE_MAX_AGE sessions old (age = rows since it appeared + 1)."""
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
    pos = np.cumsum(new) - 1                       # index of the latest Cboe session at or before row t
    has = pos >= 0
    p = np.maximum(pos, 0)
    age = np.arange(T) - idx[p] + 1
    fresh = has & (age <= CBOE_MAX_AGE)
    surge = np.where(fresh, o_surge[p], np.nan)
    dev = np.where(fresh, o_dev[p], np.nan)
    return surge, dev


def weekly_days(dates):
    """Last session of the calendar week, from the date of row t alone: t is a decision day when the next
    weekday (np.busday_offset, exchange holidays ignored) falls in a later week, i.e. t is a Friday."""
    d = np.asarray(dates, dtype="datetime64[D]")
    nxt = np.busday_offset(d, 1, roll="forward")
    return (nxt - d).astype(np.int64) > 1


def features(data):
    """Returns X (T, N, 7) and the universe mask. All columns are kept; non-members masked per row (K-0134)."""
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
    return X, uni


def model_score(data, params):
    X, uni = features(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    score = walk_forward(
        X, y, horizon=HORIZON, model="gbm",
        params={"n_estimators": int(params["n_estimators"]), "num_leaves": int(params["num_leaves"]),
                "learning_rate": LEARNING_RATE, "min_child_samples": MIN_CHILD_SAMPLES},
        retrain_every=RETRAIN_EVERY, universe=uni,
    )
    return np.where(uni, score, np.nan), uni


def book(score, uni):
    w = score_weights(score, uni, top=TOP, bottom=BOTTOM, min_names=MIN_NAMES)
    w = np.where(np.isfinite(w) & uni, w, 0.0)       # unscored / non-member on a decision date = flat
    return w


# --- mechanism-test helpers (copied verbatim into diagnostic.py, kept here so the tests can import them)

SURGE_WIN = 252           # trailing distribution of call_surge (sessions, all must be valid)
SURGE_Q = 2.0 / 3.0       # top third


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


def target_weights(data, params):
    score, uni = model_score(data, params)
    w = book(score, uni)
    return only_on(weekly_days(data.dates), w)
