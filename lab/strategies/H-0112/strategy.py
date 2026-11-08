"""H-0112: index-basket pressure in low-turnover S&P 500 members (learned, dollar-neutral 5-day cohorts).

Row t is decided after the close of t using rows 0..t only; the engine executes at the open of t+1.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"flow_sigma": 1.5, "n_estimators": 100, "top": 0.2}

# fixed by the card (not in signal.params)
Z_WINDOW = 63            # sessions t-63..t-1 for the std of m and the mean of member dollar volume
DV_WINDOW = 63           # sessions t-62..t for the mean dollar volume (inverse turnover)
DV_MIN_OBS = 32          # at least half the window of finite dollar volumes (see summary)
BETA_WINDOW = 252        # sessions t-251..t
BETA_MIN_PAIRS = 126
PRIOR_DAYS = 5           # own compounded return over t-5..t-1
HORIZON = 5              # ml.horizon_days, also the cohort hold in sessions
RETRAIN_EVERY = 126      # ml.retrain_every_days
TRAIN_SIGMA = 1.0        # training rows: |z_t| >= 1.0 (lowest flow_sigma grid value)
LEG_GROSS = 0.25         # per leg and cohort
MIN_SCORED = 100         # fewer scored members on a flow day: no cohort
MAX_GROSS = 1.0


def _window_sum(a, w):
    """Sum over rows t-w+1..t (truncated at the start); a has no NaN."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if w < a.shape[0]:
        out[w:] -= c[:-w]
    return out


def _prior(a, fill):
    """Value of row t-1 at row t (row 0 gets fill)."""
    out = np.empty_like(a)
    out[0] = fill
    out[1:] = a[:-1]
    return out


def _shift(a, k, fill):
    out = np.full_like(a, fill)
    if k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


def daily_returns(data):
    """Close-to-close total return r_t = (1+ret_co)(1+ret_oc)-1, NaN where the stock is not listed."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    return np.where(np.asarray(data.listed, dtype=bool), r, np.nan)


def market_series(data, r, uni):
    """m_t (cap-weighted member return), z_t (m_t / std of m over t-63..t-1), vshock_t."""
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    ok = uni & np.isfinite(mc) & (mc > 0) & np.isfinite(r)
    num = np.where(ok, mc * np.where(ok, r, 0.0), 0.0).sum(axis=1)
    den = np.where(ok, mc, 0.0).sum(axis=1)
    m = np.full(den.shape, np.nan)
    np.divide(num, den, out=m, where=den > 0)

    # z_t: std (ddof=1) of m over the Z_WINDOW sessions before t, all of them finite
    fin = np.isfinite(m)
    m0 = np.where(fin, m, 0.0)
    n = _prior(_window_sum(fin.astype(np.float64), Z_WINDOW), 0.0)
    s1 = _prior(_window_sum(m0, Z_WINDOW), 0.0)
    s2 = _prior(_window_sum(m0 * m0, Z_WINDOW), 0.0)
    full = n >= Z_WINDOW
    nn = np.where(full, n, 2.0)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    z = np.full(m.shape, np.nan)
    np.divide(m, sd, out=z, where=full & (sd > 0) & fin)

    # vshock_t: log(S_t / mean of S over t-63..t-1), S = members' total dollar volume
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    okv = uni & np.isfinite(dv)
    S = np.where(okv, dv, 0.0).sum(axis=1)
    has = okv.any(axis=1) & (S > 0)
    S0 = np.where(has, S, 0.0)
    ns = _prior(_window_sum(has.astype(np.float64), Z_WINDOW), 0.0)
    ss = _prior(_window_sum(S0, Z_WINDOW), 0.0)
    vshock = np.full(S.shape, np.nan)
    good = has & (ns >= Z_WINDOW) & (ss > 0)
    ratio = np.ones(S.shape)
    np.divide(S0 * np.where(ns > 0, ns, 1.0), np.where(ss > 0, ss, 1.0), out=ratio, where=good)
    vshock[good] = np.log(ratio[good])
    return m, z, vshock


def inverse_turnover(data):
    """log(sf1_arq_marketcap / mean dollar_volume over t-62..t)."""
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    fin = np.isfinite(dv)
    n = _window_sum(fin.astype(np.float64), DV_WINDOW)
    s = _window_sum(np.where(fin, dv, 0.0), DV_WINDOW)
    mean = np.full(dv.shape, np.nan)
    np.divide(s, n, out=mean, where=n >= DV_MIN_OBS)
    ok = np.isfinite(mc) & (mc > 0) & np.isfinite(mean) & (mean > 0)
    out = np.full(dv.shape, np.nan)
    out[ok] = np.log(mc[ok] / mean[ok])
    return out


def rolling_beta(r, m, chunk=256):
    """OLS slope of r on m over t-251..t, at least BETA_MIN_PAIRS valid pairs (column chunks bound memory)."""
    out = np.full(r.shape, np.nan)
    for j in range(0, r.shape[1], chunk):
        out[:, j:j + chunk] = _beta_block(r[:, j:j + chunk], m)
    return out


def _beta_block(r, m):
    valid = np.isfinite(r) & np.isfinite(m)[:, None]
    x = np.where(valid, np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
    y = np.where(valid, r, 0.0)
    n = _window_sum(valid.astype(np.float64), BETA_WINDOW)
    sx = _window_sum(x, BETA_WINDOW)
    sxx = _window_sum(x * x, BETA_WINDOW)
    sxy = _window_sum(x * y, BETA_WINDOW)
    del x
    sy = _window_sum(y, BETA_WINDOW)
    del y
    enough = n >= BETA_MIN_PAIRS
    nn = np.where(enough, n, 1.0)
    cov = sxy - sx * sy / nn
    var = sxx - sx * sx / nn
    beta = np.full(r.shape, np.nan)
    np.divide(cov, var, out=beta, where=enough & (var > 0))
    return beta


def prior_return(r):
    """Own compounded return over t-PRIOR_DAYS..t-1 (excludes t); NaN if any day is missing."""
    p = np.ones_like(r)
    for k in range(1, PRIOR_DAYS + 1):
        p = p * (1.0 + _shift(r, k, np.nan))
    return p - 1.0


def build_features(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0112 needs the sp500 point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    r = daily_returns(data)
    m, z, vshock = market_series(data, r, uni)
    T, N = r.shape
    f1 = cs_zscore(r.astype(np.float32), uni)
    f2 = cs_zscore(inverse_turnover(data).astype(np.float32), uni)
    f3 = cs_zscore(rolling_beta(r, m).astype(np.float32), uni)
    f4 = cs_zscore(prior_return(r).astype(np.float32), uni)
    f5 = np.broadcast_to(z.astype(np.float32)[:, None], (T, N)).copy()
    f6 = np.broadcast_to(vshock.astype(np.float32)[:, None], (T, N)).copy()
    X = stack(f1, f2, f3, f4, f5, f6)
    return X, z, uni


def train_rows(z):
    """Point-in-time row mask of training dates: |z_t| >= TRAIN_SIGMA."""
    return np.isfinite(z) & (np.abs(np.where(np.isfinite(z), z, 0.0)) >= TRAIN_SIGMA)


def gbm_params(params):
    return {"n_estimators": int(params["n_estimators"]), "num_leaves": 15,
            "learning_rate": 0.05, "min_child_samples": 200}


def flow_days(z, flow_sigma):
    return np.isfinite(z) & (np.abs(np.where(np.isfinite(z), z, 0.0)) >= float(flow_sigma))


def build_book(score, uni, z, flow_sigma, top):
    """Overlapping 5-session dollar-neutral cohorts opened on flow days, capped at gross 1."""
    uni = np.asarray(uni, dtype=bool)
    flow = flow_days(z, flow_sigma)
    sw = score_weights(score, uni, top=float(top), bottom=float(top), min_names=MIN_SCORED)
    sw = np.nan_to_num(np.asarray(sw, dtype=np.float64), nan=0.0)
    # score_weights puts 0.5 gross on each leg; the card wants LEG_GROSS per leg
    cohort = np.where(flow[:, None], sw * (LEG_GROSS / 0.5), 0.0)
    w = np.zeros_like(cohort)
    for k in range(HORIZON):          # cohort of row t is held on target rows t..t+HORIZON-1
        w += _shift(cohort, k, 0.0)
    w[~uni] = 0.0                     # a name that left the index is closed
    gross = np.abs(w).sum(axis=1)
    scale = np.where(gross > MAX_GROSS, MAX_GROSS / np.where(gross > 0, gross, 1.0), 1.0)
    # full target rows every day: targets change only on flow days, cohort exits and index exits,
    # and the engine trades only when targets change
    return w * scale[:, None]


def target_weights(data, params):
    X, z, uni = build_features(data)
    tr = train_rows(z)
    score = walk_forward(
        X,
        cs_rank(np.where(tr[:, None], forward_return(data, HORIZON), np.nan), uni),
        horizon=HORIZON, model="gbm", params=gbm_params(params),
        retrain_every=RETRAIN_EVERY, universe=uni,
    )
    return build_book(score, uni, z, params["flow_sigma"], params["top"])
