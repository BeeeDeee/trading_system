"""H-0201 Quarterly index share-update pressure, rule-based (S&P 500, dollar-neutral).

Rules (card signal.description) and where they live:
  * quarterly rebalance session R = third Friday of Mar/Jun/Sep/Dec (weekday arithmetic)   -> _calendar
  * decision date d = W sessions before R (weekday count approximation)                     -> _calendar
  * implied share count S = sf1_arq_marketcap / sf1_arq_price of the same filing,
    filing = first usable day on which either field changes, filing session f = row before  -> _filings
  * S_new at close of d, S_ref at close of the session 5 sessions before previous R         -> _pending_g
  * g missing: same filing, missing / non-positive S, |g| >= 5 %, split guard > 0.15        -> _pending_g
  * eligible: member at d, tradable at d, g not missing                                     -> _legs
  * legs: top 50 g > 0.25 %, bottom 50 g < -0.25 %, skip if < 10 on either side             -> _legs
  * weights +0.5/n_long, -0.5/n_short at d, held through close of R, flat at open R+1;
    a name leaving the universe / not tradable / not listed is closed for the rest of the
    window (not redistributed)                                                              -> target_weights
"""
import numpy as np

PARAMS = {"window_sessions": 20}

# fixed by construction in the card (not gridded)
LEG_SIZE = 50
MIN_LEG = 10
G_FLOOR = 0.0025
G_CAP = 0.05
SPLIT_TOL = 0.15
REF_OFFSET = 5


def _third_friday(month_idx):
    """month_idx: int64 months since the epoch -> datetime64[D] of that month's third Friday."""
    first = np.asarray(month_idx, dtype=np.int64).astype("datetime64[M]").astype("datetime64[D]")
    return np.busday_offset(first, 2, roll="forward", weekmask="Fri")


def _calendar(dates, window):
    """Point-in-time quarterly calendar.

    Returns (Fn, Fprev, cnt, decision_rows):
      Fn[t]    third Friday of the next quarterly rebalance month on or after date t
      Fprev[t] third Friday of the quarter before that
      cnt[t]   weekdays in [date_t, Fn[t]) (R is the cnt-th weekday after t)
      decision rows: first row of a quarter with 1 <= cnt <= window whose previous row had
      cnt > window (or belonged to another quarter); row 0 is never a decision row (unknown
      whether the true decision session lay before the data).
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    if T == 0:
        e = np.zeros(0, dtype="datetime64[D]")
        return e, e, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    m = dates.astype("datetime64[M]").astype(np.int64)
    qm = m + (2 - (m % 12) % 3)
    f1 = _third_friday(qm)
    use1 = f1 >= dates
    fm = np.where(use1, qm, qm + 3)
    fn = np.where(use1, f1, _third_friday(qm + 3))
    fprev = _third_friday(fm - 3)
    cnt = np.busday_count(dates, fn).astype(np.int64)
    in_win = (cnt >= 1) & (cnt <= window)
    prev_out = np.ones(T, dtype=bool)
    prev_out[0] = False
    prev_out[1:] = (fm[:-1] != fm[1:]) | (cnt[:-1] > window)
    dec = np.flatnonzero(in_win & prev_out)
    return fn, fprev, cnt, dec


def _filings(mc, pr):
    """First usable row of the latest filing at each row (-1 before any), and the finite mask.

    A new filing is the first row on which marketcap or price is finite and differs from the
    last finite pair (or the first finite pair of the stock).
    """
    T, N = mc.shape
    fin = np.isfinite(mc) & np.isfinite(pr)
    rows = np.arange(T, dtype=np.int64)[:, None]
    last_fin = np.maximum.accumulate(np.where(fin, rows, -1), axis=0)
    prev = np.full((T, N), -1, dtype=np.int64)
    if T > 1:
        prev[1:] = last_fin[:-1]
    pc = np.clip(prev, 0, None)
    mc_prev = np.take_along_axis(mc, pc, axis=0)
    pr_prev = np.take_along_axis(pr, pc, axis=0)
    new = fin & ((prev < 0) | (mc != mc_prev) | (pr != pr_prev))
    fid = np.maximum.accumulate(np.where(new, rows, -1), axis=0)
    return fid, fin


def _inputs(data):
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    pr = np.asarray(data.extras["sf1_arq_price"], dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    T = mc.shape[0]
    rows = np.arange(T, dtype=np.int64)[:, None]
    fid, fin = _filings(mc, pr)
    okc = np.isfinite(close) & (close > 0)
    cidx = np.maximum.accumulate(np.where(okc, rows, -1), axis=0)
    rco = np.clip(np.where(np.isfinite(rco), rco, 0.0), -0.999999, None)
    roc = np.clip(np.where(np.isfinite(roc), roc, 0.0), -0.999999, None)
    ltr = np.cumsum(np.log1p(rco) + np.log1p(roc), axis=0)   # log total-return index at each close
    return mc, pr, close, fid, fin, cidx, ltr


def _pending_g(inp, d, r):
    """g = ln(S_new / S_ref) per instrument, NaN where the card says missing."""
    mc, pr, close, fid, fin, cidx, ltr = inp
    N = mc.shape[1]
    cols = np.arange(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_new = mc[d] / pr[d]
        s_ref = mc[r] / pr[r]
        fn = fid[d]
        fr = fid[r]
        ok = fin[d] & fin[r] & (fn >= 0) & (fr >= 0) & (fn != fr)
        ok &= np.isfinite(s_new) & np.isfinite(s_ref) & (s_new > 0) & (s_ref > 0)
        f_new = fn - 1                      # filing session = session before first usable day
        f_ref = fr - 1
        ok &= (f_ref >= 0) & (f_new >= 0)
        fnc = np.clip(f_new, 0, None)
        frc = np.clip(f_ref, 0, None)
        cn = cidx[fnc, cols]
        cr = cidx[frc, cols]
        ok &= (cn >= 0) & (cr >= 0)
        p_new = close[np.clip(cn, 0, None), cols]
        p_ref = close[np.clip(cr, 0, None), cols]
        price_lr = np.log(p_new / p_ref)
        tr_lr = ltr[fnc, cols] - ltr[frc, cols]
        split = np.abs(price_lr - tr_lr) > SPLIT_TOL
        ok &= np.isfinite(price_lr) & np.isfinite(tr_lr) & ~split
        g = np.log(s_new / s_ref)
        ok &= np.isfinite(g) & (np.abs(g) < G_CAP)
    return np.where(ok, g, np.nan)


def _legs(data, params):
    """List of (d, e, long_cols, short_cols): decision row d, window rows d..e-1 (row e = R or later is flat)."""
    window = int(params["window_sessions"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0201 needs the point-in-time sp500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    T = dates.shape[0]
    fn, fprev, cnt, dec = _calendar(dates, window)
    out = []
    if dec.size == 0:
        return out
    inp = _inputs(data)
    for d in dec:
        d = int(d)
        # window rows: d .. e-1, same quarter, strictly before the third Friday
        e = d + 1
        while e < T and fn[e] == fn[d] and cnt[e] >= 1:
            e += 1
        # previous quarter's rebalance session: last row on or before its third Friday
        r_prev = int(np.searchsorted(dates, fprev[d], side="right")) - 1
        r = r_prev - REF_OFFSET
        if r_prev < 0 or r < 0:
            continue
        g = _pending_g(inp, d, r)
        elig = uni[d] & trad[d] & np.isfinite(g)
        gl = np.where(elig & (g > G_FLOOR), g, np.nan)
        gs = np.where(elig & (g < -G_FLOOR), g, np.nan)
        lc = np.flatnonzero(np.isfinite(gl))
        sc = np.flatnonzero(np.isfinite(gs))
        if lc.size < MIN_LEG or sc.size < MIN_LEG:
            continue
        lc = lc[np.argsort(-gl[lc], kind="stable")[:LEG_SIZE]]
        sc = sc[np.argsort(gs[sc], kind="stable")[:LEG_SIZE]]
        out.append((d, e, lc, sc))
    return out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=np.float64)
    legs = _legs(data, params)
    if not legs:
        return w
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    ok = uni & trad & listed
    for d, e, lc, sc in legs:
        wd = np.zeros(N, dtype=np.float64)
        wd[lc] = 0.5 / lc.size
        wd[sc] = -0.5 / sc.size
        alive = np.ones((e - d, N), dtype=bool)
        if e - d > 1:
            alive[1:] = np.logical_and.accumulate(ok[d + 1:e], axis=0)
        w[d:e] = wd[None, :] * alive
    return w
