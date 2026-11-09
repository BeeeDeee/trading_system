"""H-0118 Tax-deadline loss-selling pressure and release, learned.

LIQ-500, LightGBM (framework walk_forward) on five z-scored stock features (tax-year return,
fund-year return, volume-weighted overhang, 21-session return, 63-session volatility) plus two
unscaled date-level clocks (sessions to the next / since the last tax deadline). Weekly book of
5-day cohorts, only on week starts inside a tax window; long top `leg`, short bottom `leg`.
"""
import numpy as np
from lab.framework.api import period_starts, total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"leg": 0.1, "n_estimators": 100}

HORIZON = 5            # label horizon (sessions) = ml.horizon_days
RETRAIN_EVERY = 252    # ml.retrain_every_days
CLOCK_CAP = 63         # clocks capped at 63 sessions
WINDOW = 15            # tax window: clock_to <= 15 or 1 <= clock_since <= 15
HOLD = 5               # positions of the last in-window decision closed 5 sessions after opening
MIN_NAMES = 50         # fewer scored members on a decision date: flat
HIST = 252             # overhang window t-251..t, also the minimum history
RET_N = 21
VOL_N = 63
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}


# ----------------------------------------------------------------------------- calendar

def _mon(k):
    return np.array(k).astype("timedelta64[M]")


def _day(k):
    return np.array(k).astype("timedelta64[D]")


def _weekday(d):
    """Monday = 0 ... Sunday = 6 (day 0 of the epoch was a Thursday)."""
    return (d.astype("datetime64[D]").astype(np.int64) + 3) % 7


def _holidays(dates):
    """Regular XNYS holidays that can fall between a session and the next tax deadline within
    the 63-session cap: Labor Day (1st Monday of September), Thanksgiving (4th Thursday of
    November), Christmas (Dec 25, Saturday -> Friday, Sunday -> Monday). Built by rule from
    the years present in `dates` (no date literals)."""
    yrs = np.unique(dates.astype("datetime64[Y]"))
    m0 = yrs.astype("datetime64[M]")
    sep1 = (m0 + _mon(8)).astype("datetime64[D]")
    labor = np.busday_offset(sep1, 0, roll="forward", weekmask="Mon")
    nov1 = (m0 + _mon(10)).astype("datetime64[D]")
    thanks = np.busday_offset(nov1, 3, roll="forward", weekmask="Thu")
    dec25 = (m0 + _mon(11)).astype("datetime64[D]") + _day(24)
    wd = _weekday(dec25)
    shift = np.where(wd == 5, -1, np.where(wd == 6, 1, 0)).astype("timedelta64[D]")
    xmas = dec25 + shift
    return np.sort(np.concatenate([labor, thanks, xmas]))


def _last_strict(flag):
    """Index of the most recent row i < t with flag[i] (-1 if none)."""
    T = flag.shape[0]
    a = np.where(flag, np.arange(T), -1)
    out = np.full(T, -1, dtype=np.int64)
    if T > 1:
        out[1:] = np.maximum.accumulate(a)[:-1]
    return out


def deadline_clocks(dates):
    """Date-level clocks, point-in-time.

    Returns clock_to, clock_since (float, capped at 63, NaN when unknown), last_fund, last_ind
    (row index of the most recent October / December deadline strictly before t, -1 if none).

    * clock_to: XNYS sessions in (t, D_next], D_next = the next deadline (last session of
      October or December) by the calendar rule: last weekday of the month that is not a rule
      holiday. Future sessions are counted with weekday arithmetic minus the rule holidays
      (unscheduled closures cannot be known in advance).
    * Past deadlines are read from the data: the last row of a calendar month 10 / 12, which is
      known once a row of the next month exists (row i is the month's last row iff row i+1 is
      in a later month, and i+1 <= t). clock_since = t - that row, exact session count; 0 when
      t itself is a deadline by the calendar rule.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = dates.shape[0]
    hol = _holidays(dates)
    y0 = dates.astype("datetime64[Y]").astype("datetime64[M]")
    oct31 = (y0 + _mon(10)).astype("datetime64[D]") - _day(1)
    dec31 = (y0 + _mon(12)).astype("datetime64[D]") - _day(1)
    d_fund = np.busday_offset(oct31, 0, roll="backward", holidays=hol)
    d_ind = np.busday_offset(dec31, 0, roll="backward", holidays=hol)
    nxt = np.where(dates <= d_fund, d_fund, d_ind)
    cnt = np.busday_count(dates + _day(1), nxt + _day(1), holidays=hol)
    clock_to = np.minimum(np.maximum(cnt, 0), CLOCK_CAP).astype(np.float64)
    is_dl = (dates == d_fund) | (dates == d_ind)

    mon_id = dates.astype("datetime64[M]").astype(np.int64)
    month = mon_id % 12 + 1
    ends = np.zeros(T, dtype=bool)
    if T > 1:
        ends[:-1] = mon_id[1:] != mon_id[:-1]
    last_fund = _last_strict(ends & (month == 10))
    last_ind = _last_strict(ends & (month == 12))
    last_any = np.maximum(last_fund, last_ind)
    idx = np.arange(T)
    since = np.where(last_any >= 0, idx - last_any, np.nan).astype(np.float64)
    since = np.where(is_dl, 0.0, since)
    clock_since = np.minimum(since, CLOCK_CAP)
    return clock_to, clock_since, last_fund, last_ind


def in_tax_window(clock_to, clock_since):
    with np.errstate(invalid="ignore"):
        return (clock_to <= WINDOW) | ((clock_since >= 1) & (clock_since <= WINDOW))


# ----------------------------------------------------------------------------- features

def _csum(x):
    """Prefix sums with a leading zero row: s[k] = sum(x[:k])."""
    out = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=out[1:])
    return out


def _window_sum(cs, n, T):
    """Sum over rows t-n+1..t from prefix sums; NaN for t < n-1."""
    out = np.full((T,) + cs.shape[1:], np.nan)
    if T >= n:
        out[n - 1:] = cs[n:] - cs[:T - n + 1]
    return out


def stock_features(tri, dv, listed, last_fund, last_ind):
    """The five stock features (T, N) float64, NaN where unknown or < 252 sessions of history."""
    tri = np.asarray(tri, dtype=np.float64)
    T, N = tri.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.asarray(listed, dtype=bool) & np.isfinite(tri) & (tri > 0)
        hist = _window_sum(_csum(ok.astype(np.float64)), HIST, T)
        full = hist >= HIST - 0.5                     # listed with a price on all of t-251..t
        trz = np.where(ok, tri, 0.0)

        # (3) overhang = log(TRI_t / R_t), R_t = dollar-volume-weighted mean TRI over t-251..t
        dv = np.asarray(dv, dtype=np.float64)
        dvz = np.where(ok & np.isfinite(dv) & (dv > 0), dv, 0.0)
        num = _window_sum(_csum(dvz * trz), HIST, T)
        den = _window_sum(_csum(dvz), HIST, T)
        del dvz
        ref = np.where(den > 0, num / den, np.nan)
        del num, den
        overhang = np.log(tri / ref)
        del ref

        # (4) ret_21 = TRI_t / TRI_{t-21} - 1
        ret21 = np.full((T, N), np.nan)
        if T > RET_N:
            ret21[RET_N:] = tri[RET_N:] / tri[:-RET_N] - 1.0

        # (5) vol_63 = std of daily total returns over t-62..t
        r = np.zeros((T, N))
        if T > 1:
            r[1:] = np.where(ok[1:] & ok[:-1], trz[1:] / np.where(ok[:-1], trz[:-1], 1.0) - 1.0, 0.0)
        s1 = _window_sum(_csum(r), VOL_N, T)
        s2 = _window_sum(_csum(r * r), VOL_N, T)
        del r
        var = (s2 - s1 * s1 / VOL_N) / (VOL_N - 1)
        vol63 = np.sqrt(np.maximum(var, 0.0))
        del s1, s2, var

        # (1), (2) return since the most recent December / October deadline strictly before t
        def since_ref(last):
            has = last >= 0
            li = np.where(has, last, 0)
            base = np.where(has[:, None] & ok[li], trz[li], np.nan)
            return tri / base - 1.0

        tax_year = since_ref(last_ind)
        fund_year = since_ref(last_fund)

    feats = [tax_year, fund_year, overhang, ret21, vol63]
    good = full.copy()
    for f in feats:
        good &= np.isfinite(f)
    return [np.where(good, f, np.nan) for f in feats]


def stock_features_blocked(tri, dv, listed, last_fund, last_ind, block=256):
    """stock_features over column blocks (each stock's features use only its own column), stored
    as float32 to bound memory on the ~2 400-column LIQ-500 panel."""
    tri = np.asarray(tri)
    dv = np.asarray(dv)
    listed = np.asarray(listed)
    T, N = tri.shape
    out = [np.full((T, N), np.nan, dtype=np.float32) for _ in range(5)]
    for j0 in range(0, N, block):
        j1 = min(N, j0 + block)
        fb = stock_features(tri[:, j0:j1], dv[:, j0:j1], listed[:, j0:j1], last_fund, last_ind)
        for k in range(5):
            out[k][:, j0:j1] = fb[k]
        del fb
    return out


def model_inputs(data):
    """Features X (T, N, 7), clocks and universe, exactly as in the card."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0118 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    clock_to, clock_since, last_fund, last_ind = deadline_clocks(data.dates)
    tri = total_return_index(data)
    feats = stock_features_blocked(tri, data.dollar_volume, data.listed, last_fund, last_ind)
    del tri
    T, N = uni.shape
    vec = uni.copy()
    for f in feats:
        vec &= np.isfinite(f)
    zs = []
    while feats:
        f = feats.pop(0)
        f[~vec] = np.nan                     # z-score among members with a full vector
        z = np.asarray(cs_zscore(f, uni), dtype=np.float32)
        del f
        z = np.array(z, dtype=np.float32, copy=True)
        z[~vec] = np.nan
        zs.append(z)
    ct = np.broadcast_to(clock_to.astype(np.float32)[:, None], (T, N))
    cs = np.broadcast_to(clock_since.astype(np.float32)[:, None], (T, N))
    X = stack(*zs, ct, cs)
    del zs
    return X, uni, clock_to, clock_since


def model_scores(data, params):
    X, uni, clock_to, clock_since = model_inputs(data)
    h = HORIZON
    y = cs_rank(forward_return(data, h), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=h, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), uni, clock_to, clock_since


# ----------------------------------------------------------------------------- book

def _mask(days, T):
    d = np.asarray(days)
    if d.dtype == bool and d.shape == (T,):
        return d.copy()
    m = np.zeros(T, dtype=bool)
    m[d.astype(np.int64)] = True
    return m


def decision_rows(dates, clock_to, clock_since):
    """First session of each week whose date lies inside a tax window."""
    T = len(dates)
    ws = _mask(period_starts(dates, "W"), T)
    return ws & in_tax_window(clock_to, clock_since)


def leg_weights(score, uni, dec, leg):
    """(K, N) equal-weight legs on the decision rows (0.5 gross per side), zeros = flat."""
    idx = np.flatnonzero(dec)
    if idx.size == 0:
        return idx, np.zeros((0, score.shape[1]))
    sw = np.asarray(score_weights(score[idx], uni[idx], top=leg, bottom=leg, min_names=MIN_NAMES),
                    dtype=np.float64)
    sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
    sw[~uni[idx]] = 0.0
    return idx, sw


def build_book(T, N, idx, sw, uni):
    """NaN = keep. New book on each decision row; if no new decision follows, close everything
    5 sessions after the decision (open of t+6); a held stock leaving the universe is sold."""
    out = np.full((T, N), np.nan)
    held = np.zeros(N)
    pos = {int(r): k for k, r in enumerate(idx)}
    last = -10 ** 9
    for r in range(T):
        k = pos.get(r)
        if k is not None:
            row = sw[k].copy()
            out[r] = row
            held = row
            last = r
            continue
        if not held.any():
            continue
        if r - last == HOLD:
            out[r] = 0.0
            held = np.zeros(N)
            continue
        drop = (held != 0) & ~uni[r]
        if drop.any():
            out[r, drop] = 0.0
            held = held.copy()
            held[drop] = 0.0
    return out


def target_weights(data, params):
    leg = float(params["leg"])
    score, uni, clock_to, clock_since = model_scores(data, params)
    T, N = score.shape
    dec = decision_rows(data.dates, clock_to, clock_since)
    idx, sw = leg_weights(score, uni, dec, leg)
    return build_book(T, N, idx, sw, uni)
