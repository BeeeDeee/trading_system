"""H-0180: network adoption not yet in the price.

Binance spot top-50, long-only, monthly overlapping cohorts of the coins whose
30-day active-address growth over L days most exceeds what their own L-day price
run implies (rank regression residual).
"""
import math

import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 182, "lookback_days": 180}

AA_PREFIX = "coinmetrics_adractcnt_1d."
# card data_requirements[coinmetrics_adractcnt_1d].fields: the only keys a spot pair may map to
AA_KEYS = (
    "1INCHUSDT.P", "AAVEUSDT.P", "ADAUSDT.P", "ALGOUSDT.P", "ALPHAUSDT.P", "BALUSDT.P", "BATUSDT.P",
    "BCHUSDT.P", "BSVUSDT.P", "BTCUSDT.P", "BTGUSDT.P", "COMPUSDT.P", "CROUSDT.P", "CRVUSDT.P",
    "CVCUSDT.P", "DASHUSDT.P", "DCRUSDT.P", "DGBUSDT.P", "DOGEUSDT.P", "DOTUSDT.P", "DRGNUSDT.P",
    "ELFUSDT.P", "EOSUSDT.P", "ETCUSDT.P", "ETHUSDT.P", "FTTUSDT.P", "FUNUSDT.P", "GASUSDT.P",
    "GNOUSDT.P", "GNTUSDT.P", "HEDGUSDT.P", "HTUSDT.P", "ICPUSDT.P", "KCSUSDT.P", "KNCUSDT.P",
    "LDOUSDT.P", "LENDUSDT.P", "LINKUSDT.P", "LPTUSDT.P", "LTCUSDT.P", "MAIDUSDT.P", "MANAUSDT.P",
    "MKRUSDT.P", "NEOUSDT.P", "NXMUSDT.P", "OMGUSDT.P", "PAYUSDT.P", "PERPUSDT.P", "POLYUSDT.P",
    "POWRUSDT.P", "PPTUSDT.P", "QASHUSDT.P", "QNTUSDT.P", "RENUSDT.P", "SNTUSDT.P", "SNXUSDT.P",
    "SRMUSDT.P",
)
TOP_N = 50            # card universe: crypto_top_n, n = 50 (fallback only)
VOL_WINDOW = 30       # 30-day quote volume for the fallback universe
WIN = 30              # address-mean window (days)
MIN_OBS = 25          # non-missing address values required per window
SEASON_DAYS = 365     # address series must start >= 365 days before t
MIN_ELIGIBLE = 8      # fewer eligible coins: no cohort that month
DAYS_PER_MONTH = 30.44


def n_cohorts(hold_days):
    """K = hold_days / 30.44 rounded half up (126 -> 4, 182 -> 6, 273 -> 9)."""
    return max(1, int(math.floor(float(hold_days) / DAYS_PER_MONTH + 0.5)))


def aa_key(name):
    """Spot pair <ID>USDT maps to the Coin Metrics key <ID>USDT.P."""
    return name if name.endswith(".P") else name + ".P"


def address_matrix(data):
    """(T, N) active-address counts; NaN where unknown or where the pair has no listed key.

    Every key of the card's list is indexed directly (a missing one raises). A pair maps to
    <ID>USDT -> <ID>USDT.P; a pair whose key is not in the card's list is ineligible (all NaN),
    even if the dataset happens to serve such a key.
    """
    series = {k: data.series[AA_PREFIX + k] for k in AA_KEYS}
    T, N = len(data.dates), len(data.instruments)
    aa = np.full((T, N), np.nan)
    for j, name in enumerate(data.instruments):
        k = aa_key(name)
        if k in series:
            aa[:, j] = np.asarray(series[k], dtype=float).reshape(-1)
    return aa


def universe_matrix(data):
    """data.universe if the gate passes it; else the card's definition, top 50 by 30-day quote volume."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    dv = np.asarray(data.dollar_volume, dtype=float)
    T, N = dv.shape
    v = np.where(np.isfinite(dv) & (dv > 0), dv, 0.0)
    c = np.vstack([np.zeros((1, N)), np.cumsum(v, axis=0)])
    lo = np.maximum(np.arange(T) - VOL_WINDOW + 1, 0)
    s = c[np.arange(T) + 1] - c[lo]
    listed = np.asarray(data.listed, dtype=bool)
    s = np.where(listed & (s > 0), s, -1.0)
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return (rank < TOP_N) & (s > 0)


def avg_rank01(x):
    """Cross-sectional ranks with ties averaged, scaled to [0, 1]."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        k = i
        while k + 1 < n and xs[k + 1] == xs[i]:
            k += 1
        ranks[order[i:k + 1]] = 0.5 * (i + k)     # 0-based average rank
        i = k + 1
    return ranks / (n - 1) if n > 1 else np.zeros(n)


def ols_residual(y, x):
    """Residual of y on x with intercept (OLS)."""
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = np.sum((x - xm) * (y - ym)) / sxx if sxx > 0 else 0.0
    a = ym - b * xm
    return y - a - b * x


def formation_signal(data, params):
    """Per formation row: (t, formed, eligible cols, residuals, long cols, short cols)."""
    L = int(params["lookback_days"])
    dates = data.dates
    T, N = len(dates), len(data.instruments)
    uni = universe_matrix(data)
    aa = address_matrix(data)

    close = np.asarray(data.close, dtype=float)
    cvalid = np.isfinite(close) & (close > 0)
    ccount = np.vstack([np.zeros((1, N)), np.cumsum(cvalid, axis=0)])   # ccount[t+1] = valid closes in 0..t

    avalid = np.isfinite(aa)
    a0 = np.where(avalid, aa, 0.0)
    acnt = np.vstack([np.zeros((1, N)), np.cumsum(avalid, axis=0)])
    asum = np.vstack([np.zeros((1, N)), np.cumsum(a0, axis=0)])
    has_any = avalid.any(axis=0)
    first = np.where(has_any, np.argmax(avalid, axis=0), 0)

    form = np.flatnonzero(np.asarray(period_starts(dates, "M"), dtype=bool))
    out = []
    cols = np.arange(N)
    for t in form:
        rec = {"t": int(t), "formed": False, "elig": np.zeros(0, int), "e": np.zeros(0),
               "long": np.zeros(0, int), "short": np.zeros(0, int)}
        out.append(rec)
        if t - L - WIN < 0:
            continue
        # address windows: rows t-30..t-1 and t-L-30..t-L-1 (values stamped day t-1 or earlier)
        r_lo, r_hi = t - WIN, t           # half-open [r_lo, r_hi)
        o_lo, o_hi = t - L - WIN, t - L
        n_rec = acnt[r_hi] - acnt[r_lo]
        n_old = acnt[o_hi] - acnt[o_lo]
        m_rec = np.where(n_rec > 0, (asum[r_hi] - asum[r_lo]) / np.maximum(n_rec, 1), np.nan)
        m_old = np.where(n_old > 0, (asum[o_hi] - asum[o_lo]) / np.maximum(n_old, 1), np.nan)
        # seasoning: first address value at least 365 days before t
        age = (dates[t] - dates[first]).astype(np.int64)
        seasoned = has_any & (first <= t) & (age >= SEASON_DAYS)
        ok = (uni[t]
              & (ccount[t + 1] >= L + WIN)
              & cvalid[t] & cvalid[t - L]
              & seasoned
              & (n_rec >= MIN_OBS) & (n_old >= MIN_OBS)
              & (m_rec > 0) & (m_old > 0))
        elig = cols[ok]
        rec["elig"] = elig
        if len(elig) < MIN_ELIGIBLE:
            continue
        g = np.log(m_rec[elig]) - np.log(m_old[elig])
        r = np.log(close[t, elig] / close[t - L, elig])
        e = ols_residual(avg_rank01(g), avg_rank01(r))
        k = int(math.ceil(len(elig) / 3.0))
        top = np.lexsort((elig, -e))[:k]
        bot = np.lexsort((elig, e))[:k]
        rec.update(formed=True, e=e, long=elig[top], short=elig[bot])
    return out, uni


def target_weights(data, params):
    T, N = len(data.dates), len(data.instruments)
    K = n_cohorts(params["hold_days"])
    recs, uni = formation_signal(data, params)
    W = np.full((T, N), np.nan)
    cohorts = []
    for m, rec in enumerate(recs):
        t = rec["t"]
        cw = np.zeros(N)
        if rec["formed"]:
            cw[rec["long"]] = 1.0 / K / len(rec["long"])
        cohorts.append(cw)
        target = np.sum(cohorts[max(0, m - K + 1):m + 1], axis=0)
        target[~uni[t]] = 0.0              # hold only members (cohort share of a non-member stays cash)
        W[t] = target
        # until the next formation date: weights drift; a held coin leaving the universe is sold
        t_end = recs[m + 1]["t"] if m + 1 < len(recs) else T
        if t_end > t + 1:
            held = target > 0
            out_rows = held[None, :] & ~uni[t + 1:t_end]
            seg = W[t + 1:t_end]
            seg[out_rows] = 0.0
    return W
