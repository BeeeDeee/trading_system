"""H-0085 mechanism test events: the cohort picks of each weekly formation.

Same eligibility, supply lag, beta split and ranking as strategy.py (code copied, separate file).
Side -1 for the highest-growth coins of each beta half, +1 for the lowest-growth coins.
"""
import numpy as np
from lab.framework.api import total_return_index

SUPPLY_PREFIX = "coinmetrics_supply_1d."
BETA_DAYS = 91
MIN_RETURNS = 91
JUMP_LIMIT = 0.20
MIN_ELIGIBLE = 12


def _supply_matrix(data, lookback_days):
    T, N = len(data.dates), len(data.instruments)
    keys = [k for k in data.series if str(k).startswith(SUPPLY_PREFIX)]
    if len(keys) == 0:
        raise KeyError("coinmetrics_supply_1d series are missing from data.series")
    any_value = False
    for k in keys:
        if np.isfinite(np.asarray(data.series[k], dtype=float)).any():
            any_value = True
            break
    if not any_value and T > lookback_days + 1:
        raise ValueError("coinmetrics_supply_1d series are entirely empty")
    S = np.full((T, N), np.nan)
    for j, name in enumerate(data.instruments):
        key = SUPPLY_PREFIX + name
        if key in data.series:
            S[:, j] = np.asarray(data.series[key], dtype=float)
    S[~(S > 0)] = np.nan
    return S


def _ffill(x):
    T = x.shape[0]
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(T)[:, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)
    out = np.take_along_axis(np.where(ok, x, np.nan), np.maximum(idx, 0), axis=0)
    out[idx < 0] = np.nan
    return out


def _weekday(dates):
    d = dates.astype("datetime64[D]").astype(np.int64)
    return (d + 3) % 7


def _prepare(data, params):
    L = int(params["lookback_days"])
    T, N = len(data.dates), len(data.instruments)
    uni = data.universe
    if uni is None:
        raise ValueError("H-0085 needs the crypto_top_n point-in-time universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close) & (close > 0)
    tri = total_return_index(data)
    ltri = np.log(np.where(np.isfinite(tri) & (tri > 0), tri, np.nan))
    r = np.full((T, N), np.nan)
    r[1:] = ltri[1:] - ltri[:-1]
    r_ok = np.zeros((T, N), dtype=bool)
    r_ok[1:] = has_bar[1:] & has_bar[:-1]
    r = np.where(r_ok & np.isfinite(r), r, np.nan)
    cnt = np.cumsum(np.vstack([np.zeros((1, N)), np.isfinite(r).astype(float)]), axis=0)
    S = _supply_matrix(data, L)
    lnS = np.log(S)
    lnS_ff = _ffill(lnS)
    d = np.full((T, N), np.nan)
    d[1:] = lnS_ff[1:] - lnS_ff[:-1]
    jump = (np.abs(np.nan_to_num(d, nan=0.0)) > JUMP_LIMIT).astype(float)
    jcum = np.cumsum(np.vstack([np.zeros((1, N)), jump]), axis=0)
    order_id = np.argsort(np.array([str(x) for x in data.instruments]), kind="stable")
    id_rank = np.empty(N, dtype=np.int64)
    id_rank[order_id] = np.arange(N)
    return dict(L=L, T=T, N=N, uni=uni, has_bar=has_bar, r=r, cnt=cnt, lnS=lnS,
                jcum=jcum, id_rank=id_rank)


def _form_cohort(P, t):
    L, N = P["L"], P["N"]
    if t - 1 - L < 0 or t - BETA_DAYS + 1 < 1:
        return None
    n_ret = P["cnt"][t + 1] - P["cnt"][t + 1 - BETA_DAYS]
    ok = P["uni"][t] & P["has_bar"][t] & (n_ret >= MIN_RETURNS)
    s1 = P["lnS"][t - 1]
    s0 = P["lnS"][t - 1 - L]
    ok &= np.isfinite(s1) & np.isfinite(s0)
    n_jump = P["jcum"][t] - P["jcum"][t - L]
    ok &= n_jump == 0
    elig = np.flatnonzero(ok)
    n = elig.size
    if n < MIN_ELIGIBLE:
        return None
    g = (s1[elig] - s0[elig]) * 365.0 / L
    R = P["r"][t - BETA_DAYS + 1:t + 1][:, elig]
    m = R.mean(axis=1)
    mc = m - m.mean()
    var = float(np.dot(mc, mc))
    if var > 0:
        beta = (mc @ (R - R.mean(axis=0))) / var
    else:
        beta = np.zeros(n)
    ids = P["id_rank"][elig]
    ob = np.lexsort((ids, beta))
    n_low = n // 2
    halves = (elig[ob[:n_low]], elig[ob[n_low:]])
    g_full = np.full(N, np.nan)
    g_full[elig] = g
    longs, shorts = [], []
    for h in halves:
        k = max(1, h.size // 3)
        og = h[np.lexsort((P["id_rank"][h], g_full[h]))]
        longs.append(og[:k])
        shorts.append(og[-k:])
    return np.concatenate(longs), np.concatenate(shorts)


def events(data, params):
    P = _prepare(data, params)
    T, N = P["T"], P["N"]
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    tradable = np.asarray(data.tradable, dtype=bool)
    for t in np.flatnonzero(_weekday(data.dates) == 6):
        f = _form_cohort(P, t)
        if f is None:
            continue
        lo, sh = f
        mask[t, lo] = True
        side[t, lo] = 1.0
        mask[t, sh] = True
        side[t, sh] = -1.0
    mask &= P["uni"] & tradable
    side[~mask] = 0.0
    return mask, side
