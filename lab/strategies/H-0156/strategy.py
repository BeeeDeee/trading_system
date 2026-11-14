"""H-0156 Global dollar funding cycle into EM equities.

12-month change of the dollar against the seven DM majors (FRED noon rates) sets a monthly signal
s_t (+1 when the dollar weakened, -1 when it strengthened); the position p_t is the mean of s over the
last hold_months monthly decision dates (cohorts). The book is long p_t in an equal-weighted basket of
EM country ETFs and short SPY with a trailing weekly OLS beta hedge, gross |p_t|.
"""
import numpy as np
from lab.framework.api import period_starts, total_return_index

PARAMS = {"lookback_months": 12, "hold_months": 12}

EM_ETFS = ("EWZ", "EWW", "EWY", "EWT", "EWM", "EZA", "FXI", "ECH", "TUR", "THD",
           "EIDO", "EPOL", "EPHE", "EPU", "INDA")
HEDGE = "SPY"
PEERS = ("DIA", "QQQ", "IWM", "MDY")          # never traded (weight 0)

# x_k = log(foreign units per USD): +log for quotes in foreign per USD, -log for USD per foreign
FX_FOREIGN_PER_USD = ("fred_macro.DEXJPUS", "fred_macro.DEXCAUS", "fred_macro.DEXSZUS",
                      "fred_macro.DEXSDUS")
FX_USD_PER_FOREIGN = ("fred_macro.DEXUSEU", "fred_macro.DEXUSUK", "fred_macro.DEXUSAL")

SESSIONS_PER_MONTH = 21
MIN_CURRENCIES = 5
MIN_PRIOR_PRICED = 21
BETA_BLOCK = 5
BETA_WINDOW = 260
BETA_MIN_OBS = 40
BETA_CLIP = (0.5, 2.0)


def _ffill(x):
    """Forward fill a 1-d float array (point-in-time: uses only earlier rows)."""
    x = np.asarray(x, dtype=float)
    idx = np.where(np.isfinite(x), np.arange(x.shape[0]), -1)
    idx = np.maximum.accumulate(idx)
    out = x[np.maximum(idx, 0)]
    out[idx < 0] = np.nan
    return out


def _log_dollar(v, invert):
    v = _ffill(v)
    x = np.full(v.shape, np.nan)
    pos = np.isfinite(v) & (v > 0)
    x[pos] = np.log(v[pos])
    return -x if invert else x


def dollar_change(data, lookback_months):
    """D_t: equal-weighted mean over currencies of x_k(t) - x_k(t - lookback_months*21 sessions).

    NaN where fewer than MIN_CURRENCIES currencies have both values."""
    T = len(data.dates)
    lag = int(lookback_months) * SESSIONS_PER_MONTH
    rows = []
    for key in FX_FOREIGN_PER_USD:
        rows.append(_log_dollar(data.series[key], False))
    for key in FX_USD_PER_FOREIGN:
        rows.append(_log_dollar(data.series[key], True))
    diffs = np.full((len(rows), T), np.nan)
    for k, x in enumerate(rows):
        if 0 < lag < T:
            diffs[k, lag:] = x[lag:] - x[:T - lag]
    ok = np.isfinite(diffs)
    n = ok.sum(axis=0)
    total = np.where(ok, diffs, 0.0).sum(axis=0)
    D = np.full(T, np.nan)
    good = n >= MIN_CURRENCIES
    D[good] = total[good] / n[good]
    return D


def monthly_signal(D):
    """+1 if D < 0 (dollar weakened), -1 if D > 0, 0 if undefined or exactly 0."""
    D = np.asarray(D, dtype=float)
    s = np.zeros(D.shape)
    s[D < 0] = 1.0
    s[D > 0] = -1.0
    return s


def cohort_position(s_dec, hold_months):
    """p over decision dates: mean of s over the last hold_months decision dates (missing = 0)."""
    H = int(hold_months)
    cs = np.concatenate([[0.0], np.cumsum(s_dec)])
    i = np.arange(1, len(s_dec) + 1)
    return (cs[i] - cs[np.maximum(i - H, 0)]) / H


def em_eligible(data, em_cols):
    """(T, n_em) bool: a price on the row and at least 21 earlier rows with prices."""
    has = np.isfinite(np.asarray(data.close, dtype=float)[:, em_cols])
    prior = np.cumsum(has, axis=0) - has
    return has & (prior >= MIN_PRIOR_PRICED)


def basket_beta(tri, priced, t, basket_cols, spy_col):
    """OLS beta of the EM basket's non-overlapping 5-session returns on SPY's, trailing 260 sessions."""
    n_blocks = BETA_WINDOW // BETA_BLOCK
    ends = t - BETA_BLOCK * np.arange(n_blocks)
    starts = ends - BETA_BLOCK
    keep = starts >= 0
    ends, starts = ends[keep], starts[keep]
    if ends.size < BETA_MIN_OBS or len(basket_cols) == 0:
        return 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = tri[ends, spy_col] / tri[starts, spy_col] - 1.0
        ok_s = priced[ends, spy_col] & priced[starts, spy_col] & np.isfinite(rs)
        re = tri[np.ix_(ends, basket_cols)] / tri[np.ix_(starts, basket_cols)] - 1.0
    ok_e = priced[np.ix_(ends, basket_cols)] & priced[np.ix_(starts, basket_cols)] & np.isfinite(re)
    cnt = ok_e.sum(axis=1)
    rb = np.where(ok_e, re, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    valid = ok_s & (cnt > 0)
    if valid.sum() < BETA_MIN_OBS:
        return 1.0
    x = rs[valid]
    y = rb[valid]
    xd = x - x.mean()
    var = float((xd * xd).sum())
    if not var > 0:
        return 1.0
    b = float((xd * (y - y.mean())).sum()) / var
    return float(np.clip(b, BETA_CLIP[0], BETA_CLIP[1]))


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    em_cols = np.array([data.col(name) for name in EM_ETFS], dtype=int)
    spy = data.col(HEDGE)
    for name in PEERS:
        data.col(name)                        # fail loudly if a declared instrument is missing

    D = dollar_change(data, params["lookback_months"])
    s = monthly_signal(D)

    dec = np.asarray(period_starts(data.dates, "M"), dtype=bool)
    dec_idx = np.flatnonzero(dec)
    p_dec = cohort_position(s[dec_idx], params["hold_months"])

    elig = em_eligible(data, em_cols)
    tri = np.asarray(total_return_index(data), dtype=float)
    priced = np.isfinite(np.asarray(data.close, dtype=float))

    W = np.full((T, N), np.nan)
    for i, t in enumerate(dec_idx):
        row = np.zeros(N)
        p = float(p_dec[i])
        members = em_cols[elig[t]]
        if p != 0.0 and members.size > 0:
            b = basket_beta(tri, priced, t, members, spy)
            row[members] = p / (1.0 + b) / members.size
            row[spy] = -p * b / (1.0 + b)
        W[t] = row
    return W
