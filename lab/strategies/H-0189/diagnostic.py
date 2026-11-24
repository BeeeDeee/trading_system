"""H-0189 mechanism events: on predicted peak-week decision dates, top beta quintile +1, bottom -1.

Same pipeline as strategy.py (copied: diagnostic.py is a separate file).
"""
import numpy as np

LAG_DAYS = 364
MIN_OBS_SHARE = 0.8
N_QUANTILES = 5
N_WEEKS = 52


def _day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def _decision_rows(dates):
    d = _day_numbers(dates)
    return np.flatnonzero((d + 3) % 7 == 4)


def _prepare(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0189 needs the point-in-time LIQ-500 universe (data.universe)")
    uni = np.asarray(uni, dtype=bool)
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    obs = np.asarray(data.listed, dtype=bool) & np.isfinite(close) & np.isfinite(r)
    r = np.where(obs, r, 0.0)
    mm = uni & obs
    cnt = mm.sum(axis=1)
    tot = np.where(mm, r, 0.0).sum(axis=1)
    mkt = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    mkt[ok] = tot[ok] / cnt[ok]
    flag = np.asarray(data.extras["ea_reaction_day"])
    return uni, r, obs, mkt, flag


def _betas(r, obs, mkt, t, cols, lookback):
    lo = max(0, t - lookback + 1)
    x = mkt[lo:t + 1][:, None]
    y = r[lo:t + 1][:, cols]
    m = obs[lo:t + 1][:, cols] & np.isfinite(x)
    n = m.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        xs = np.where(m, x, 0.0)
        ys = np.where(m, y, 0.0)
        mx = xs.sum(axis=0) / n
        my = ys.sum(axis=0) / n
        dx = np.where(m, x - mx, 0.0)
        dy = np.where(m, y - my, 0.0)
        var = (dx * dx).sum(axis=0)
        cov = (dx * dy).sum(axis=0)
        b = cov / var
    b[(n < MIN_OBS_SHARE * lookback) | ~(var > 0)] = np.nan
    return b


def _intensity(flag, d, t, cols):
    lo_d = d[t] + 1 - LAG_DAYS
    hi_d = d[t] + 7 - LAG_DAYS
    a = int(np.searchsorted(d, lo_d, side="left"))
    b = int(np.searchsorted(d, hi_d, side="right"))
    if b <= a or cols.size == 0:
        return np.nan
    f = np.asarray(flag[a:b][:, cols], dtype=np.float64)
    elig = np.isfinite(f).any(axis=0)
    ne = int(elig.sum())
    if ne == 0:
        return np.nan
    return float((f[:, elig] == 1.0).sum()) / ne


def _core(data, params):
    lookback = int(params["beta_lookback"])
    pf = float(params["peak_fraction"])
    uni, r, obs, mkt, flag = _prepare(data)
    d = _day_numbers(data.dates)
    dec = _decision_rows(data.dates)
    S = np.full(dec.size, np.nan)
    out = []
    for i, t in enumerate(dec):
        cols = np.flatnonzero(uni[t])
        S[i] = _intensity(flag, d, t, cols)
        peak = False
        if np.isfinite(S[i]):
            hist = S[:i + 1]
            hist = hist[np.isfinite(hist)]
            if hist.size >= N_WEEKS:
                peak = bool(S[i] >= np.quantile(hist[-N_WEEKS:], 1.0 - pf))
        b = _betas(r, obs, mkt, t, cols, lookback)
        v = np.isfinite(b)
        bc = cols[v]
        bv = b[v]
        k = bc.size // N_QUANTILES
        order = np.lexsort((bc, bv))
        bottom = bc[order[:k]]
        top = bc[order[bc.size - k:]] if k > 0 else bc[:0]
        out.append((int(t), peak, top, bottom))
    return out, uni


def events(data, params):
    out, uni = _core(data, params)
    T, N = uni.shape
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t, peak, top, bottom in out:
        if not peak:
            continue
        ok = uni[t] & tradable[t]
        top = top[ok[top]]
        bottom = bottom[ok[bottom]]
        mask[t, top] = True
        side[t, top] = 1.0
        mask[t, bottom] = True
        side[t, bottom] = -1.0
    return mask, side
