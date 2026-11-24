"""H-0189 Earnings-season beta premium (LIQ-500, long-only, weekly).

Decision after the close of each Friday session t (weekday arithmetic on date(t) only):
  (1) market proxy = equal-weighted mean daily total return of the LIQ-500 members of each day;
  (2) beta = OLS slope of the stock's daily total return on the proxy over the last beta_lookback
      rows up to t, at least 80 % of beta_lookback non-missing days, otherwise no beta;
  (3) S_t = reaction sessions (ea_reaction_day == 1) on sessions dated in
      [date(t) + 1 - 364, date(t) + 7 - 364], summed over the members at t whose flag is not NaN
      there, divided by the number of such members;
  (4) peak week if S_t >= the (1 - peak_fraction) empirical quantile of the last 52 weekly S values
      (inclusive); off-peak while fewer than 52 values exist;
  (5) equal weights in the top beta quintile (peak) or bottom beta quintile (off-peak).
"""
import numpy as np

PARAMS = {"beta_lookback": 252, "peak_fraction": 0.33}

LAG_DAYS = 364          # one year earlier = 52 weeks, keeps the weekday
MIN_OBS_SHARE = 0.8     # at least 80 % non-missing days in the beta window
N_QUANTILES = 5         # quintiles
N_WEEKS = 52            # weekly S values in the peak ranking window


def day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def decision_rows(dates):
    """Rows whose date is a Friday (day number 0 of the epoch is a Thursday; Monday = 0)."""
    d = day_numbers(dates)
    return np.flatnonzero((d + 3) % 7 == 4)


def prepare(data):
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


def betas(r, obs, mkt, t, cols, lookback):
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


def intensity(flag, d, t, cols):
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


def core(data, params):
    """Per decision row: (t, peak, top-quintile columns, bottom-quintile columns)."""
    lookback = int(params["beta_lookback"])
    pf = float(params["peak_fraction"])
    uni, r, obs, mkt, flag = prepare(data)
    d = day_numbers(data.dates)
    dec = decision_rows(data.dates)
    S = np.full(dec.size, np.nan)
    out = []
    for i, t in enumerate(dec):
        cols = np.flatnonzero(uni[t])
        S[i] = intensity(flag, d, t, cols)
        peak = False
        if np.isfinite(S[i]):
            hist = S[:i + 1]
            hist = hist[np.isfinite(hist)]
            if hist.size >= N_WEEKS:
                peak = bool(S[i] >= np.quantile(hist[-N_WEEKS:], 1.0 - pf))
        b = betas(r, obs, mkt, t, cols, lookback)
        v = np.isfinite(b)
        bc = cols[v]
        bv = b[v]
        k = bc.size // N_QUANTILES
        order = np.lexsort((bc, bv))          # ascending beta, ties by column
        bottom = bc[order[:k]]
        top = bc[order[bc.size - k:]] if k > 0 else bc[:0]
        out.append((int(t), peak, top, bottom))
    return out, uni, obs


def target_weights(data, params):
    out, uni, obs = core(data, params)
    T, N = uni.shape
    W = np.full((T, N), np.nan)
    dec = {t: (peak, top, bottom) for t, peak, top, bottom in out}
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        if t in dec:
            peak, top, bottom = dec[t]
            sel = top if peak else bottom
            sel = sel[obs[t, sel]]            # skip stocks without a price at t
            row = np.zeros(N)
            if sel.size:
                row[sel] = 1.0 / sel.size
            W[t] = row
            held = row > 0
        else:
            gone = held & ~uni[t]             # held stock left the universe mid-week: sell it
            if gone.any():
                W[t, gone] = 0.0
                held &= ~gone
    return W
