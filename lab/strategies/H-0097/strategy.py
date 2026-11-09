"""H-0097: filing-day earnings quality vs headline reaction, learned (LightGBM), LIQ-500 long-only weekly top fraction.

Point-in-time: row t uses rows 0..t only; weights of row t are executed at the open of t+1.
"""
import numpy as np
from lab.framework.api import only_on, total_return_index
from lab.framework.ml import cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"num_leaves": 15, "top": 0.2}

# fixed by the card (not in signal.params, so not varied by G2)
HORIZON = 5                 # label: open t+1 -> open t+6
RETRAIN_EVERY = 63          # ml.retrain_every_days
TRAIN_DAYS = 756            # rolling training window (rows)
REACTION_WINDOW = 126       # latest 8-K Item 2.02 session within the last 126 sessions
VOL_WINDOW = 63             # 63-session standard deviation of daily total returns
EVENT_CAP = 126             # F5 cap and fixed scale
MIN_SCORED = 50           # fewer scored members on a decision date: keep the previous book
WINSOR_LO, WINSOR_HI = 1.0, 99.0
GBM_FIXED = {"n_estimators": 100, "learning_rate": 0.05, "min_child_samples": 1000,
             "subsample": 0.8, "colsample": 1.0}

SF1_FIELDS = ("sf1_arq_netinc", "sf1_arq_ncfo", "sf1_arq_assets")


# ----------------------------------------------------------------------------------------------- helpers

def daily_returns(data):
    """Close-to-close total return of each row and a mask of rows where it is a real return
    (a price on t and on t-1)."""
    ret = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    has_px = np.isfinite(np.asarray(data.close, dtype=np.float64))
    valid = np.zeros_like(has_px)
    valid[1:] = has_px[1:] & has_px[:-1]
    ret = np.where(valid, ret, 0.0)
    return ret, valid


def trailing_std_full(ret, valid, n):
    """Sample std (ddof 1) over the n rows ending at t; NaN unless all n rows hold a valid return."""
    T, N = ret.shape
    x = np.where(valid, ret, 0.0)
    c1 = np.zeros((T + 1, N))
    c2 = np.zeros((T + 1, N))
    cn = np.zeros((T + 1, N))
    np.cumsum(x, axis=0, out=c1[1:])
    np.cumsum(x * x, axis=0, out=c2[1:])
    np.cumsum(valid.astype(np.float64), axis=0, out=cn[1:])
    out = np.full((T, N), np.nan)
    if T < n:
        return out
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    cnt = cn[n:] - cn[:-n]
    var = (s2 - s1 * s1 / n) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    out[n - 1:] = np.where(cnt > n - 0.5, sd, np.nan)
    return out


def last_true_index(mask):
    """Row index of the latest True up to and including t (per column), -1 if none."""
    T = mask.shape[0]
    rows = np.arange(T)[:, None]
    idx = np.where(mask, rows, -1)
    return np.maximum.accumulate(idx, axis=0)


def filing_events(netinc, ncfo, assets):
    """First session on which any sf1_arq field differs from the previous session or becomes non-missing."""
    ev = np.zeros(netinc.shape, dtype=bool)
    for x in (netinc, ncfo, assets):
        x = np.asarray(x, dtype=np.float64)
        cur = np.isfinite(x[1:])
        prev = np.isfinite(x[:-1])
        changed = cur & (~prev | (x[1:] != np.where(prev, x[:-1], 0.0)))
        ev[1:] |= changed
    return ev


def sessions_since_filing(netinc, ncfo, assets):
    """Sessions from the latest filing-usable session to t, capped at EVENT_CAP. Before the first observed
    filing event the count is known only once t >= EVENT_CAP (then the filing is at least that old)."""
    ev = filing_events(netinc, ncfo, assets)
    T, N = ev.shape
    last = last_true_index(ev)
    rows = np.arange(T, dtype=np.float64)[:, None]
    cnt = np.where(last >= 0, rows - last, np.where(rows >= EVENT_CAP, float(EVENT_CAP), np.nan))
    return np.minimum(cnt, float(EVENT_CAP))


def winsorize_rows(x, scored, lo=WINSOR_LO, hi=WINSOR_HI):
    """Clip each row at its lo/hi percentile across the scored members (NaN elsewhere)."""
    x = np.where(scored, x, np.nan)
    out = np.full(x.shape, np.nan)
    rows = np.flatnonzero(scored.any(axis=1))
    if rows.size:
        q = np.nanpercentile(x[rows], [lo, hi], axis=1)
        out[rows] = np.clip(x[rows], q[0][:, None], q[1][:, None])
    return out


def compute_features(data):
    """Raw features F1..F6, the scored mask and the sessions-since-filing count (all (T, N))."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0097 needs the point-in-time liq_n universe")
    uni = np.asarray(uni, dtype=bool)
    netinc = np.asarray(data.extras["sf1_arq_netinc"], dtype=np.float64)
    ncfo = np.asarray(data.extras["sf1_arq_ncfo"], dtype=np.float64)
    assets = np.asarray(data.extras["sf1_arq_assets"], dtype=np.float64)
    flag = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = uni.shape

    ret, valid = daily_returns(data)
    sd = trailing_std_full(ret, valid, VOL_WINDOW)                    # ending at t
    sd_prev = np.full((T, N), np.nan)
    sd_prev[1:] = sd[:-1]                                             # ending at t-1

    # cross-sectional median and equal-weight mean of members' daily returns
    mem_ret = np.where(uni & valid, ret, np.nan)
    has_mem = (uni & valid).any(axis=1)
    med = np.full(T, np.nan)
    ewm = np.zeros(T)
    if has_mem.any():
        med[has_mem] = np.nanmedian(mem_ret[has_mem], axis=1)
        ewm[has_mem] = np.nanmean(mem_ret[has_mem], axis=1)
    del mem_ret
    ew_index = np.cumprod(1.0 + ewm)                                  # (T,)

    # standardized release-session reaction on every row (used at the latest flagged session r)
    react = np.where(valid, (ret - med[:, None]) / sd_prev, np.nan)

    last_r = last_true_index(flag == 1.0)
    rows = np.arange(T)[:, None]
    age = rows - last_r
    in_win = (last_r >= 0) & (age < REACTION_WINDOW)
    r_safe = np.where(last_r >= 0, last_r, 0)
    cols = np.arange(N)[None, :]

    f3 = np.where(in_win, react[r_safe, cols], np.nan)

    tri = np.asarray(total_return_index(data), dtype=np.float64)
    stock_cum = tri / tri[r_safe, cols] - 1.0
    mkt_cum = ew_index[:, None] / ew_index[r_safe] - 1.0
    denom = sd_prev[r_safe, cols] * np.sqrt(np.maximum(1.0, age.astype(np.float64)))
    f4 = np.where(in_win, (stock_cum - mkt_cum) / denom, np.nan)
    f4 = np.where(in_win & (age == 0) & np.isfinite(f3), 0.0, f4)
    del stock_cum, mkt_cum, denom, tri, react

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = (netinc - ncfo) / assets
        f2 = ncfo / assets
    count = sessions_since_filing(netinc, ncfo, assets)
    f5 = count / float(EVENT_CAP)
    f6 = sd

    feats = (f1, f2, f3, f4, f5, f6)
    scored = uni & (assets > 0)
    for f in feats:
        scored &= np.isfinite(f)
    return feats, scored, count


def build_X(feats, scored):
    """F1-F4 and F6 winsorized at 1/99 and z-scored across scored members; F5 on its fixed scale."""
    f1, f2, f3, f4, f5, f6 = feats
    z = [cs_zscore(winsorize_rows(f, scored), scored) for f in (f1, f2, f3, f4)]
    z6 = cs_zscore(winsorize_rows(f6, scored), scored)
    f5s = np.where(scored, f5, np.nan)
    return stack(z[0], z[1], z[2], z[3], f5s, z6)


def model_scores(data, params):
    feats, scored, count = compute_features(data)
    X = build_X(feats, scored)
    del feats
    uni = np.asarray(data.universe, dtype=bool)
    gbm = dict(GBM_FIXED)
    gbm["num_leaves"] = int(params["num_leaves"])
    score = walk_forward(X, forward_return(data, HORIZON), horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni, train_days=TRAIN_DAYS)
    score = np.where(uni & scored, np.asarray(score, dtype=np.float64), np.nan)
    return score, scored, count


def weekly_decision_days(dates):
    """After the close of the last trading session of each calendar week. Approximation: a Friday row
    (weekday arithmetic, holidays unknown); in a week without a Friday session (holiday Friday) the
    decision falls on the first session of the next week."""
    d = np.asarray(dates).astype(np.int64)      # days since 1970-01-01 (a Thursday)
    weekday = (d + 3) % 7                       # Monday = 0
    week = (d + 3) // 7
    friday = weekday == 4
    dec = friday.copy()
    T = d.shape[0]
    if T > 1:
        new_week = np.zeros(T, dtype=bool)
        new_week[1:] = week[1:] != week[:-1]
        prev_friday = np.zeros(T, dtype=bool)
        prev_friday[1:] = friday[:-1]
        dec |= new_week & ~prev_friday
    return dec


def book_from_scores(score, uni, dates, top):
    """Weekly equal-weight top fraction of scored members; < MIN_SCORED scored members keeps the book;
    a stock that leaves the universe between decisions is sold (weight 0) on the first row it is out."""
    uni = np.asarray(uni, dtype=bool)
    T, N = score.shape
    w = np.asarray(score_weights(score, uni, top=float(top), bottom=0.0, min_names=MIN_SCORED), dtype=np.float64)
    n_scored = (np.isfinite(score) & uni).sum(axis=1)
    w[n_scored < MIN_SCORED] = np.nan
    dec = weekly_decision_days(dates)
    w = np.asarray(only_on(dec, w), dtype=np.float64)

    # every decision row is emitted: the new book, or the kept book (flat before the first fit) when fewer
    # than MIN_SCORED members are scored; between decisions only universe exits are emitted
    out = np.full((T, N), np.nan)
    book = np.zeros(N)
    for t in range(T):
        row = w[t]
        decided = dec[t] and np.isfinite(row).any()
        target = np.where(np.isfinite(row), row, book) if decided else book
        target = np.where(uni[t], target, 0.0)
        if dec[t] or np.any(target != book):
            out[t] = target
            book = target
    return out


def target_weights(data, params):
    score, _, _ = model_scores(data, params)
    return book_from_scores(score, data.universe, data.dates, params["top"])
