"""H-0180 mechanism test: top third (+1) and bottom third (-1) of the residual adoption
growth among the eligible coins, on monthly formation dates only.

The signal code is a verbatim copy of strategy.py (diagnostic.py is a separate file).
"""
import math

import numpy as np
from lab.framework.api import period_starts

AA_PREFIX = "coinmetrics_adractcnt_1d."
PRESENCE_KEY = AA_PREFIX + "BTCUSDT.P"
TOP_N = 50
VOL_WINDOW = 30
WIN = 30
MIN_OBS = 25
SEASON_DAYS = 365
MIN_ELIGIBLE = 8


def aa_key(name):
    return name if name.endswith(".P") else name + ".P"


def address_matrix(data):
    if PRESENCE_KEY not in data.series:
        raise KeyError("dataset coinmetrics_adractcnt_1d missing from data.series")
    T, N = len(data.dates), len(data.instruments)
    aa = np.full((T, N), np.nan)
    for j, name in enumerate(data.instruments):
        key = AA_PREFIX + aa_key(name)
        if key in data.series:
            aa[:, j] = np.asarray(data.series[key], dtype=float).reshape(-1)
    return aa


def universe_matrix(data):
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
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        k = i
        while k + 1 < n and xs[k + 1] == xs[i]:
            k += 1
        ranks[order[i:k + 1]] = 0.5 * (i + k)
        i = k + 1
    return ranks / (n - 1) if n > 1 else np.zeros(n)


def ols_residual(y, x):
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = np.sum((x - xm) * (y - ym)) / sxx if sxx > 0 else 0.0
    a = ym - b * xm
    return y - a - b * x


def formation_signal(data, params):
    L = int(params["lookback_days"])
    dates = data.dates
    T, N = len(dates), len(data.instruments)
    uni = universe_matrix(data)
    aa = address_matrix(data)

    close = np.asarray(data.close, dtype=float)
    cvalid = np.isfinite(close) & (close > 0)
    ccount = np.vstack([np.zeros((1, N)), np.cumsum(cvalid, axis=0)])

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
        r_lo, r_hi = t - WIN, t
        o_lo, o_hi = t - L - WIN, t - L
        n_rec = acnt[r_hi] - acnt[r_lo]
        n_old = acnt[o_hi] - acnt[o_lo]
        m_rec = np.where(n_rec > 0, (asum[r_hi] - asum[r_lo]) / np.maximum(n_rec, 1), np.nan)
        m_old = np.where(n_old > 0, (asum[o_hi] - asum[o_lo]) / np.maximum(n_old, 1), np.nan)
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


def events(data, params):
    T, N = len(data.dates), len(data.instruments)
    recs, uni = formation_signal(data, params)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    for rec in recs:
        if not rec["formed"]:
            continue
        t = rec["t"]
        mask[t, rec["long"]] = True
        side[t, rec["long"]] = 1.0
        mask[t, rec["short"]] = True
        side[t, rec["short"]] = -1.0
    mask &= uni & tradable
    side = np.where(mask, side, 0.0)
    return mask, side
