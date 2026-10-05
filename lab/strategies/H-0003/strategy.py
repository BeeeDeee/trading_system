"""H-0003: volatility-timed equity exposure (SPY when short-run SPY vol <= long-run vol, else IEF), weekly.

Row t is decided after the close of row t with rows 0..t only; the engine executes at the open of t+1.
"""
import numpy as np

PARAMS = {
    "short_window_days": 21,
    "long_window_days": 252,
}

_FRIDAY = 4  # weekday index with Monday = 0


def _weekday_and_week(dates):
    """Weekday (Mon=0..Sun=6) and Monday-based week number from the date of each row alone.

    Day 0 of the epoch was a Thursday, so (day + 3) % 7 is the weekday and (day + 3) // 7 the week.
    """
    days = dates.astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7, (days + 3) // 7


def decision_days(dates):
    """Point-in-time approximation of 'last trading day of each calendar week'.

    A row is a decision day if it is a Friday (weekday arithmetic, exchange holidays unknown). If a week had
    no Friday row (e.g. Good Friday), its last row is only recognisable as such on the next row; the first
    row of the following week is then the (one row late) decision day for the missed week.
    """
    T = len(dates)
    dec = np.zeros(T, dtype=bool)
    if T == 0:
        return dec
    wd, wk = _weekday_and_week(dates)
    dec |= wd == _FRIDAY
    if T > 1:
        new_week = wk[1:] != wk[:-1]
        prev_before_friday = wd[:-1] < _FRIDAY
        dec[1:] |= new_week & prev_before_friday
    return dec


def _trailing_std_of_valid(x, valid, n):
    """Sample std (ddof=1) of the last n valid observations of x ending at each row (NaN until n exist)."""
    T = len(x)
    out = np.full(T, np.nan)
    xs = x[valid]
    m = len(xs)
    if n < 2 or m < n:
        return out
    win = np.lib.stride_tricks.sliding_window_view(xs, n)
    s = win.std(axis=1, ddof=1)               # s[j] = std of xs[j .. j+n-1]
    k = np.cumsum(valid)                      # number of valid observations up to and including row t
    j = k - n
    ok = j >= 0
    out[ok] = s[j[ok]]
    return out


def spy_log_returns(data):
    """Daily close-to-close log total return of SPY and a mask of rows that carry a valid return.

    A row carries a return only if SPY has a bar on it (listed, finite close) and on some earlier row
    (the first SPY bar has no previous close). Rows without a bar are skipped by the windows.
    """
    i = data.col("SPY")
    close = np.asarray(data.close, dtype=float)[:, i]
    bar = np.isfinite(close)
    if data.listed is not None:
        bar &= np.asarray(data.listed, dtype=bool)[:, i]
    r = (1.0 + np.asarray(data.ret_co, dtype=float)[:, i]) * (1.0 + np.asarray(data.ret_oc, dtype=float)[:, i]) - 1.0
    had_prev_bar = np.concatenate(([False], np.cumsum(bar)[:-1] > 0))
    valid = bar & had_prev_bar & np.isfinite(r) & (r > -1.0)
    logr = np.zeros(len(r))
    logr[valid] = np.log1p(r[valid])
    return logr, valid


def vol_state(data, params):
    """(calm, ready): calm[t] = sigma_short <= sigma_long at t; ready[t] = both windows full and IEF priced."""
    n_short = int(params["short_window_days"])
    n_long = int(params["long_window_days"])
    logr, valid = spy_log_returns(data)
    s_short = _trailing_std_of_valid(logr, valid, n_short)
    s_long = _trailing_std_of_valid(logr, valid, n_long)
    calm = s_short <= s_long
    j = data.col("IEF")
    ief_priced = np.isfinite(np.asarray(data.close, dtype=float)[:, j])
    if data.listed is not None:
        ief_priced &= np.asarray(data.listed, dtype=bool)[:, j]
    ready = np.isfinite(s_short) & np.isfinite(s_long) & ief_priced
    return calm, ready


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")
    w = np.full((T, N), np.nan)
    if T == 0:
        return w

    calm, ready = vol_state(data, params)
    dec = decision_days(data.dates)
    # Start: the first decision date with IEF priced and both windows full; every later decision date decides.
    started = np.maximum.accumulate(dec & ready)
    decide = dec & started

    tradable = np.asarray(data.tradable, dtype=bool)
    both_tradable = tradable[:, i_spy] & tradable[:, i_ief]

    # Re-emit the current target on the rows after a decision while the execution open was not tradable for
    # both instruments, so that a switch blocked by a missing bar is executed at the next open where both trade.
    emit = decide.copy()
    state = np.zeros(T, dtype=bool)   # True = calm (SPY), False = stressed (IEF); valid where emit
    cur = False
    pending = False
    for t in range(T):
        if decide[t]:
            cur = bool(calm[t])
            pending = True
        elif pending and not both_tradable[t]:
            emit[t] = True
        else:
            pending = False
        state[t] = cur

    rows = np.flatnonzero(emit)
    w[rows, :] = 0.0
    w[rows, i_spy] = np.where(state[rows], 1.0, 0.0)
    w[rows, i_ief] = np.where(state[rows], 0.0, 1.0)
    return w   # NaN rows: no decision, keep positions
