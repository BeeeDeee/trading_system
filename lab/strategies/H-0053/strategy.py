"""H-0053: return seasonality net of seasonal reversal, LIQ-500 dollar-neutral, monthly.

Score for target month m: mean relative return of the stock in the same calendar month over the past
Y years (lags 12, 24, .., 12Y) minus its mean relative return over the other-month lags 13..12Y that are
not multiples of 12. Long the top leg_quantile, short the bottom leg_quantile (+0.5 / -0.5, equal weight),
decided after the close of the last session of month m-1, held until the next month-end decision.
"""
import numpy as np

PARAMS = {"leg_quantile": 0.2, "years_back": 5}

MIN_SESSIONS = 15        # a stock-month counts only with a return on >= 15 sessions
MIN_OTHER_NUM = 3        # at least 75 % (3/4) of the other-month lags must exist
MIN_OTHER_DEN = 4
MIN_PER_LEG = 10         # fewer names per leg: hold nothing that month
LEG_GROSS = 0.5          # +0.5 long, -0.5 short


def month_ids(dates):
    """Integer calendar-month id of each date (months since the epoch)."""
    return np.asarray(dates).astype("datetime64[M]").astype(np.int64)


def decision_rows(dates):
    """Month-end decision rows and the calendar month each decision targets.

    Row t is the last session of its month if the next weekday after its date (np.busday_offset,
    weekdays only, no exchange holidays) falls in another month. If an exchange holiday sits on the
    last weekday of a month (e.g. Good Friday on Mar 29/30, Memorial Day on May 31), no row of that
    month qualifies; then the first session of the next month is the decision row (one session late).
    Uses only the dates of rows <= t.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    T = len(dates)
    if T == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    mid = month_ids(dates)
    nxt = np.busday_offset(dates, 1, roll="forward")
    is_me = month_ids(nxt) != mid
    starts = np.flatnonzero(np.r_[True, mid[1:] != mid[:-1]])
    block_has_me = np.logical_or.reduceat(is_me, starts)
    fallback = np.zeros(T, dtype=bool)
    if len(starts) > 1:
        fallback[starts[1:]] = ~block_has_me[:-1]
    rows = np.flatnonzero(is_me | fallback)
    target = np.where(is_me[rows], mid[rows] + 1, mid[rows])
    return rows, target


def monthly_relative_returns(dates, ret_co, ret_oc, valid):
    """(first_month_id, R): R[k, i] relative return of stock i in month first_month_id + k.

    Monthly return = product over the month's sessions of (1+ret_co)(1+ret_oc) - 1, using only sessions
    where valid (the stock has a return); NaN unless at least MIN_SESSIONS valid sessions. Relative =
    minus the equal-weighted mean over all stocks that count in that month. Months absent from the
    data are NaN rows.
    """
    mid = month_ids(dates)
    starts = np.flatnonzero(np.r_[True, mid[1:] != mid[:-1]])
    g = np.asarray(ret_co, dtype=np.float64) + 1.0
    g *= np.asarray(ret_oc, dtype=np.float64) + 1.0
    ok = np.asarray(valid, dtype=bool) & np.isfinite(g)
    g[~ok] = 1.0
    np.maximum(g, 1e-300, out=g)          # a -100 % session compounds to a -100 % month
    np.log(g, out=g)
    sums = np.add.reduceat(g, starts, axis=0)
    del g
    counts = np.add.reduceat(ok.astype(np.int32), starts, axis=0)
    mon = np.expm1(sums)
    mon[counts < MIN_SESSIONS] = np.nan
    have = np.isfinite(mon)
    n_have = have.sum(axis=1)
    tot = np.where(have, mon, 0.0).sum(axis=1)
    mean = np.full(len(starts), np.nan)
    np.divide(tot, n_have, out=mean, where=n_have > 0)
    rel = mon - mean[:, None]
    first = int(mid[starts[0]])
    R = np.full((int(mid[starts[-1]]) - first + 1, rel.shape[1]), np.nan)
    R[mid[starts] - first] = rel
    return first, R


def _lag_rows(R, first, target, lags):
    idx = target - np.asarray(lags, dtype=np.int64) - first
    out = np.full((len(idx), R.shape[1]), np.nan)
    inside = (idx >= 0) & (idx < R.shape[0])
    out[inside] = R[idx[inside]]
    return out


def seasonal_score(R, first, target, years_back):
    """score(i) = mean r(i, m-12k), k=1..Y  -  mean r(i, m-L), L=13..12Y, L % 12 != 0.

    NaN unless all Y same-month lags exist and at least 75 % of the other-month lags exist.
    """
    Y = int(years_back)
    ann_lags = [12 * k for k in range(1, Y + 1)]
    oth_lags = [L for L in range(13, 12 * Y + 1) if L % 12 != 0]
    ann = _lag_rows(R, first, target, ann_lags)
    ann_ok = np.isfinite(ann).all(axis=0)
    ann_mean = np.where(ann_ok, np.where(ann_ok, ann, 0.0).sum(axis=0) / Y, np.nan)
    if len(oth_lags) == 0:
        return np.full(R.shape[1], np.nan)
    oth = _lag_rows(R, first, target, oth_lags)
    have = np.isfinite(oth)
    n_have = have.sum(axis=0)
    oth_ok = n_have * MIN_OTHER_DEN >= MIN_OTHER_NUM * len(oth_lags)
    oth_mean = np.full(R.shape[1], np.nan)
    np.divide(np.where(have, oth, 0.0).sum(axis=0), n_have, out=oth_mean, where=oth_ok)
    score = ann_mean - oth_mean
    score[~(ann_ok & oth_ok)] = np.nan
    return score


def book_row(score, eligible, leg_quantile):
    """Equal-weight +0.5 top / -0.5 bottom leg_quantile among eligible names with a score."""
    w = np.zeros(len(score))
    elig = np.flatnonzero(eligible & np.isfinite(score))
    n = int(np.floor(leg_quantile * len(elig) + 1e-9))
    if n < MIN_PER_LEG:
        return w
    order = elig[np.argsort(score[elig], kind="stable")]
    w[order[-n:]] = LEG_GROSS / n
    w[order[:n]] = -LEG_GROSS / n
    return w


def _inputs(data):
    if data.universe is None:
        raise ValueError("H-0053 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    return universe, tradable, listed


def decision_books(data, params):
    """Yields (row, weights) for every month-end decision row."""
    q = float(params["leg_quantile"])
    Y = int(params["years_back"])
    universe, tradable, listed = _inputs(data)
    first, R = monthly_relative_returns(data.dates, data.ret_co, data.ret_oc, listed & tradable)
    rows, targets = decision_rows(data.dates)
    out = []
    for t, m in zip(rows, targets):
        score = seasonal_score(R, first, int(m), Y)
        out.append((int(t), book_row(score, universe[t] & tradable[t], q)))
    return out


def target_weights(data, params):
    T, N = len(data.dates), len(data.instruments)
    universe, tradable, _ = _inputs(data)
    W = np.full((T, N), np.nan)
    books = decision_books(data, params)
    for j, (t, w) in enumerate(books):
        W[t] = w
        end = books[j + 1][0] if j + 1 < len(books) else T
        held = w != 0
        if end <= t + 1 or not held.any():
            continue
        # between decisions: a held stock that leaves the universe or is not tradable is closed
        # (target 0 from that row until the next decision); everything else keeps its position (NaN)
        gone = ~(universe[t + 1:end] & tradable[t + 1:end]) & held[None, :]
        gone = np.logical_or.accumulate(gone, axis=0)
        seg = W[t + 1:end]
        seg[gone] = 0.0
    return W
