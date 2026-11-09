"""H-0085 Token supply dilution, beta-matched, overlapping 26-week cohorts.

Weekly formation after the close of each Sunday (UTC). Eligible coins (point-in-time top-40
members) need 91 finite daily close-to-close log returns in the last 91 days, a supply value
(one-day lag) at t-1 and t-1-lookback_days, and no one-day |dln supply| > 0.20 in the window.
Split at the median 91-day beta to the equal-weight eligible mean; within each half long the
floor(n_half/3) lowest supply growth, short the floor(n_half/3) highest. Long leg +0.5, short
leg -0.5 (equal weights). Each cohort is held hold_weeks weeks with weight 1/hold_weeks.
If the gate passes no universe (data.universe is None), the PIT top 40 is rebuilt from the
trailing 30-row dollar (quote) volume, as the catalog defines crypto_top_n.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"hold_weeks": 26, "lookback_days": 182}

SUPPLY_PREFIX = "coinmetrics_supply_1d."
BETA_DAYS = 91
MIN_RETURNS = 91
JUMP_LIMIT = 0.20
MIN_ELIGIBLE = 12
TOP_N = 40       # card universe: crypto_top_n, n = 40 (fallback only)
VOL_DAYS = 30    # catalog: PIT top-n by 30-day quote volume (fallback only)


def _supply_matrix(data, lookback_days):
    """(T, N) supply in native units aligned to data.instruments; NaN where unknown.

    Raises if the supply dataset is absent, or entirely empty on a view long enough for a
    formation (a view shorter than lookback_days + 2 rows cannot form a cohort anyway; this
    keeps the truncation test on very short views from crashing)."""
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
            v = np.asarray(data.series[key], dtype=float)
            S[:, j] = v
    S[~(S > 0)] = np.nan
    return S


def _ffill(x):
    """Point-in-time forward fill along axis 0."""
    T = x.shape[0]
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(T)[:, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)
    out = np.take_along_axis(np.where(ok, x, np.nan), np.maximum(idx, 0), axis=0)
    out[idx < 0] = np.nan
    return out


def _weekday(dates):
    """Monday = 0 ... Sunday = 6, from the row's own date."""
    d = dates.astype("datetime64[D]").astype(np.int64)
    return (d + 3) % 7


def _week_number(dates):
    d = dates.astype("datetime64[D]").astype(np.int64)
    return (d + 3) // 7


def _id_rank(instruments):
    order_id = np.argsort(np.array([str(x) for x in instruments]), kind="stable")
    id_rank = np.empty(len(instruments), dtype=np.int64)
    id_rank[order_id] = np.arange(len(instruments))
    return id_rank


def fallback_top_n(data, has_bar, n=TOP_N, days=VOL_DAYS):
    """Point-in-time top-n by trailing `days`-row quote (dollar) volume, used only when the
    gate passes no crypto_top_n universe (data.universe is None, K-0054 / K-0118).
    Row t uses dollar_volume of rows t-days+1..t (missing = 0); a member needs a bar on row t
    and positive volume; ties by instrument id."""
    T, N = has_bar.shape
    dv = np.asarray(data.dollar_volume, dtype=float)
    dv = np.where(np.isfinite(dv) & (dv > 0), dv, 0.0)
    c = np.cumsum(np.vstack([np.zeros((1, N)), dv]), axis=0)
    lo = np.maximum(np.arange(T) + 1 - days, 0)
    vol = c[1:] - c[lo]
    ok = has_bar & (vol > 0)
    key = np.where(ok, vol, -np.inf)
    perm = np.argsort(_id_rank(data.instruments), kind="stable")   # columns in id order
    order = np.argsort(-key[:, perm], axis=1, kind="stable")       # desc volume, ties by id
    rank = np.empty((T, N), dtype=np.int64)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(N), (T, N)), axis=1)
    uni = np.zeros((T, N), dtype=bool)
    uni[:, perm] = rank < n
    return uni & ok


def prepare(data, params):
    """Shared inputs for the formation (used by strategy and diagnostic)."""
    L = int(params["lookback_days"])
    T, N = len(data.dates), len(data.instruments)
    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close) & (close > 0)
    uni = data.universe
    if uni is None:
        uni = fallback_top_n(data, has_bar)
    uni = np.asarray(uni, dtype=bool)
    # daily close-to-close log return (via the total-return index, identical for perps)
    tri = total_return_index(data)
    ltri = np.log(np.where(np.isfinite(tri) & (tri > 0), tri, np.nan))
    r = np.full((T, N), np.nan)
    r[1:] = ltri[1:] - ltri[:-1]
    r_ok = np.zeros((T, N), dtype=bool)
    r_ok[1:] = has_bar[1:] & has_bar[:-1]
    r = np.where(r_ok & np.isfinite(r), r, np.nan)
    cnt = np.cumsum(np.vstack([np.zeros((1, N)), np.isfinite(r).astype(float)]), axis=0)  # cnt[t+1] = sum rows<=t

    S = _supply_matrix(data, L)
    lnS = np.log(S)
    lnS_ff = _ffill(lnS)
    d = np.full((T, N), np.nan)
    d[1:] = lnS_ff[1:] - lnS_ff[:-1]
    jump = (np.abs(np.nan_to_num(d, nan=0.0)) > JUMP_LIMIT).astype(float)
    jcum = np.cumsum(np.vstack([np.zeros((1, N)), jump]), axis=0)  # jcum[t+1] = sum rows<=t

    id_rank = _id_rank(data.instruments)

    return dict(L=L, T=T, N=N, uni=uni, has_bar=has_bar, r=r, cnt=cnt, lnS=lnS,
                jcum=jcum, id_rank=id_rank)


def form_cohort(P, t):
    """Return (long_idx, short_idx) of the cohort formed after the close of row t, or None."""
    L, N = P["L"], P["N"]
    if t - 1 - L < 0 or t - BETA_DAYS + 1 < 1:
        return None
    # (a) 91 daily returns over rows t-90..t
    n_ret = P["cnt"][t + 1] - P["cnt"][t + 1 - BETA_DAYS]
    ok = P["uni"][t] & P["has_bar"][t] & (n_ret >= MIN_RETURNS)
    # (b) supply at t-1 and t-1-L (one-day lag)
    s1 = P["lnS"][t - 1]
    s0 = P["lnS"][t - 1 - L]
    ok &= np.isfinite(s1) & np.isfinite(s0)
    # (c) no |dln S| > 0.20 for the L daily changes between t-1-L and t-1 (rows t-L..t-1)
    n_jump = P["jcum"][t] - P["jcum"][t - L]
    ok &= n_jump == 0
    elig = np.flatnonzero(ok)
    n = elig.size
    if n < MIN_ELIGIBLE:
        return None
    g = (s1[elig] - s0[elig]) * 365.0 / L
    R = P["r"][t - BETA_DAYS + 1:t + 1][:, elig]          # (91, n), all finite by (a)
    m = R.mean(axis=1)
    mc = m - m.mean()
    var = float(np.dot(mc, mc))
    if var > 0:
        beta = (mc @ (R - R.mean(axis=0))) / var
    else:
        beta = np.zeros(n)
    ids = P["id_rank"][elig]
    ob = np.lexsort((ids, beta))                          # ascending beta, ties by id
    n_low = n // 2                                        # odd coin goes to the high half
    halves = (elig[ob[:n_low]], elig[ob[n_low:]])
    g_full = np.full(N, np.nan)
    g_full[elig] = g
    longs, shorts = [], []
    for h in halves:
        k = max(1, h.size // 3)
        og = h[np.lexsort((P["id_rank"][h], g_full[h]))]  # ascending g, ties by id
        longs.append(og[:k])
        shorts.append(og[-k:])
    return np.concatenate(longs), np.concatenate(shorts)


def target_weights(data, params):
    H = int(params["hold_weeks"])
    P = prepare(data, params)
    T, N = P["T"], P["N"]
    W = np.full((T, N), np.nan)
    wd = _weekday(data.dates)
    wk = _week_number(data.dates)
    alive = P["uni"] & P["has_bar"]
    cohorts = []                                          # list of [week_number, weight vector]
    for t in np.flatnonzero(wd == 6):
        # drop expired cohorts (held hold_weeks weeks)
        cohorts = [c for c in cohorts if wk[t] - c[0] < H]
        # exit rule: coins that left the universe or have no bar go to zero in that cohort
        for c in cohorts:
            c[1][~alive[t]] = 0.0
        f = form_cohort(P, t)
        if f is not None:
            lo, sh = f
            w = np.zeros(N)
            w[lo] = 0.5 / lo.size
            w[sh] = -0.5 / sh.size
            cohorts.append([wk[t], w])
        row = np.zeros(N)
        for c in cohorts:
            row += c[1]
        W[t] = row / H
    return W
