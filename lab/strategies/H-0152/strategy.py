"""H-0152: lumpy equity issuers shorted for hold_days, LIQ-500 beta-neutral hedge.

EVENT (1)-(7) as in card.signal.description; BOOK monthly (first session of each month),
short leg = active issuers still in LIQ-500, long leg = other members, beta-neutral sizing.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 252, "issuance_min": 0.1}

SHARES_FIELD = "sf1_arq_sharesbas"
MAX_PREV_AGE = 150          # (2) previous value changed less than 150 sessions before t
RATIO_CAP = 3.0             # (3) upper cap
SPLIT_RATIOS = (1.5, 2.0, 3.0)
SPLIT_TOL = 0.03            # (4) +/- 3 % (relative) around each split ratio
RET_WINDOW = 252            # (5) trailing total return window
RET_FLOOR = -0.5            # (5) trailing return must be above -50 %
MIN_VALID = 200             # (5) and betas: at least 200 valid returns
BETA_WINDOW = 252
BETA_CLIP = (0.3, 2.5)
NAME_CAP = 0.05             # short-leg cap per name


def _daily_returns(data):
    """Daily close-to-close total return and validity mask (stock has a price that day)."""
    r = (1.0 + np.nan_to_num(np.asarray(data.ret_co, dtype=float))) * \
        (1.0 + np.nan_to_num(np.asarray(data.ret_oc, dtype=float))) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=float))
    r = np.where(valid, r, 0.0)
    return r, valid


def _window_sum(x, n):
    """Trailing sum over rows t-n+1..t (fewer rows at the start), point-in-time."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[n:] = c[n:] - c[:-n]
    return out


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0152 needs the point-in-time LIQ-500 universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def compute_events(data, params):
    """(T, N) bool: EVENT (1)-(7) after the close of row t, rows <= t only."""
    hold = int(params["hold_days"])
    imin = float(params["issuance_min"])
    x = np.asarray(data.extras[SHARES_FIELD], dtype=float)
    T, N = x.shape
    uni = _universe(data)
    tradable = np.asarray(data.tradable, dtype=bool)
    rows = np.arange(T)

    prev = np.full_like(x, np.nan)
    prev[1:] = x[:-1]
    # (1) a new value on t, (2) the previous value present and positive
    with np.errstate(invalid="ignore", divide="ignore"):
        cand = np.isfinite(x) & np.isfinite(prev) & (prev > 0) & (x != prev)
        ratio = np.where(cand, x / np.where(prev > 0, prev, 1.0), np.nan)
    # (2) previous value changed less than MAX_PREV_AGE sessions before t
    change = np.ones_like(cand)
    change[1:] = ~(x[1:] == x[:-1])          # NaN transitions count as a change (new run)
    run_start = np.maximum.accumulate(np.where(change, rows[:, None], 0), axis=0)
    prev_start = np.zeros_like(run_start)
    prev_start[1:] = run_start[:-1]
    cand &= (rows[:, None] - prev_start) < MAX_PREV_AGE
    # (3) 1 + issuance_min <= ratio <= 3.0
    with np.errstate(invalid="ignore"):
        cand &= (ratio >= 1.0 + imin) & (ratio <= RATIO_CAP)
        # (4) not within +/- 3 % of a split ratio
        for k in SPLIT_RATIOS:
            cand &= ~(np.abs(ratio / k - 1.0) <= SPLIT_TOL)
    # (5) trailing 252-session total return above -50 %, at least 200 valid returns
    r, valid = _daily_returns(data)
    logr = np.log(np.maximum(1.0 + r, 1e-12))
    n_valid = _window_sum(valid.astype(float), RET_WINDOW)
    tret = np.expm1(_window_sum(logr, RET_WINDOW))
    cand &= (n_valid >= MIN_VALID) & (tret > RET_FLOOR)
    # (6) LIQ-500 member and tradable at t
    cand &= uni & tradable
    # (7) no event in the previous hold_days sessions (sequential per stock)
    events = np.zeros((T, N), dtype=bool)
    tt, jj = np.nonzero(cand)
    last = {}
    for t, j in zip(tt.tolist(), jj.tolist()):   # np.nonzero is row-major: t ascending
        lj = last.get(j)
        if lj is None or t - lj > hold:
            events[t, j] = True
            last[j] = t
    return events


def _betas(r, valid, m, t, cols):
    """OLS slope of stock returns on the member mean over rows t-251..t; 1.0 if < 200 valid days."""
    a = max(0, t - BETA_WINDOW + 1)
    mx = m[a:t + 1]
    V = valid[a:t + 1][:, cols] & np.isfinite(mx)[:, None]
    X = np.where(V, np.nan_to_num(mx)[:, None], 0.0)
    Y = np.where(V, r[a:t + 1][:, cols], 0.0)
    n = V.sum(axis=0).astype(float)
    nn = np.maximum(n, 1.0)
    xbar = X.sum(axis=0) / nn
    ybar = Y.sum(axis=0) / nn
    dx = np.where(V, X - xbar, 0.0)
    dy = np.where(V, Y - ybar, 0.0)
    cov = (dx * dy).sum(axis=0)
    var = (dx * dx).sum(axis=0)
    ok = (n >= MIN_VALID) & (var > 0)
    beta = np.where(ok, cov / np.where(var > 0, var, 1.0), 1.0)
    return np.clip(beta, BETA_CLIP[0], BETA_CLIP[1])


def _rebalance_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.copy()
    mask = np.zeros(T, dtype=bool)
    mask[ps.astype(int)] = True
    return mask


def target_weights(data, params):
    hold = int(params["hold_days"])
    uni = _universe(data)
    T, N = uni.shape
    events = compute_events(data, params)
    r, valid = _daily_returns(data)

    # equal-weight mean daily return of LIQ-500 members (valid returns only)
    mv = uni & valid
    cnt = mv.sum(axis=1)
    m = np.where(cnt > 0, np.where(mv, r, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)

    rows = np.arange(T)
    last_ev = np.maximum.accumulate(np.where(events, rows[:, None], -10 ** 9), axis=0)

    reb = _rebalance_mask(data.dates)
    w = np.full((T, N), np.nan)
    reb_rows = np.nonzero(reb)[0]
    for i, t in enumerate(reb_rows.tolist()):
        row = np.zeros(N)
        active = (t - last_ev[t]) < hold            # event in the last hold_days sessions
        short = active & uni[t]
        longs = uni[t] & ~active
        ns, nl = int(short.sum()), int(longs.sum())
        if ns > 0 and nl > 0:
            sc = np.nonzero(short)[0]
            lc = np.nonzero(longs)[0]
            b_s = float(_betas(r, valid, m, t, sc).mean())
            b_l = float(_betas(r, valid, m, t, lc).mean())
            s = b_l / (b_s + b_l)
            l = b_s / (b_s + b_l)
            if s / ns > NAME_CAP:
                s = NAME_CAP * ns
                l = s * b_s / b_l
            row[sc] = -s / ns
            row[lc] = l / nl
        w[t] = row
        # between rebalances: close held names at the first row outside the universe (K-0051)
        end = reb_rows[i + 1] if i + 1 < len(reb_rows) else T
        if end > t + 1:
            held = row != 0
            seg_out = ~uni[t + 1:end] & held[None, :]
            seg = w[t + 1:end]
            seg[seg_out] = 0.0
    return w
