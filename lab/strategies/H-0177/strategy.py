"""H-0177 cross-asset time-series momentum on a 60/40 core.

Monthly, after the close of the last session of each month:
  B = cumulative total return of IEF over the last lookback_days sessions minus that of SHY
  E = the same for SPY minus SHY
  sB = +1 if B > 0 else -1 ; sE = +1 if E > 0 else -1
  w_SPY = 0.60 + 0.20 sB ; w_IEF = 0.40 - 0.20 sE
  if w_SPY + w_IEF > 1: scale both to sum 1, w_SHY = 0, else w_SHY = 1 - w_SPY - w_IEF
Before a full lookback exists for SPY, IEF and SHY: 0.60 SPY / 0.40 IEF, rebalanced on the same dates.
SHV, BIL, VGSH, SCHO are peer-only (diagnostic): weight 0, never read here.
"""
import numpy as np

PARAMS = {"lookback_days": 252}

BASE_SPY = 0.60
BASE_IEF = 0.40
TILT = 0.20          # fixed by design (card), not a parameter


def decision_rows(dates):
    """Point-in-time month-end flag from the date of each row alone.

    Row t is a decision row if the next weekday after its date lies in another month (weekday
    approximation of 'last XNYS session of the month'; exchange holidays are ignored). If the last
    weekday of a month is an exchange holiday (e.g. Memorial Day on May 31, Good Friday on the last
    weekday of March) no row of that month is flagged; then the first session of the next month is
    the decision row (one session late), so no month is skipped.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="backward")
    last = nxt.astype("datetime64[M]") != month
    dec = last.copy()
    if T > 1:
        missed = (month[1:] != month[:-1]) & ~last[:-1]
        dec[1:] = dec[1:] | missed
    return dec


def _session_log_returns(data, j):
    """Daily total log return close-to-close of column j; a missing return counts as 0."""
    co = np.nan_to_num(np.asarray(data.ret_co[:, j], dtype=np.float64), nan=0.0)
    oc = np.nan_to_num(np.asarray(data.ret_oc[:, j], dtype=np.float64), nan=0.0)
    return np.log1p(co) + np.log1p(oc)


def trailing_cum_return(data, j, lookback):
    """Cumulative total return over the last `lookback` sessions (rows t-L+1..t), NaN if t < L."""
    lr = _session_log_returns(data, j)
    T = lr.shape[0]
    c = np.concatenate([[0.0], np.cumsum(lr)])          # c[k] = sum of lr[0..k-1] (prefix, point-in-time)
    out = np.full(T, np.nan)
    if T > lookback:
        t = np.arange(lookback, T)
        out[lookback:] = np.expm1(c[t + 1] - c[t + 1 - lookback])
    return out


def full_lookback(data, cols, lookback):
    """True on row t if t >= L and every column is listed on rows t-L and t (full price window)."""
    T = data.dates.shape[0]
    ok = np.zeros(T, dtype=bool)
    if T > lookback:
        listed = np.asarray(data.listed, dtype=bool)
        good = np.ones(T - lookback, dtype=bool)
        for j in cols:
            good &= listed[:T - lookback, j] & listed[lookback:, j]
        ok[lookback:] = good
    return ok


def signals(data, lookback):
    """Return (sB, sE, full) arrays over rows; sB, sE in {+1, -1} (meaningful where full)."""
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")
    i_shy = data.col("SHY")
    r_spy = trailing_cum_return(data, i_spy, lookback)
    r_ief = trailing_cum_return(data, i_ief, lookback)
    r_shy = trailing_cum_return(data, i_shy, lookback)
    full = full_lookback(data, (i_spy, i_ief, i_shy), lookback)
    with np.errstate(invalid="ignore"):
        b = r_ief - r_shy
        e = r_spy - r_shy
        sb = np.where(b > 0, 1.0, -1.0)
        se = np.where(e > 0, 1.0, -1.0)
    return sb, se, full


def book_weights(sb, se):
    """Vectorized SPY / IEF / SHY weights for sign arrays."""
    w_spy = BASE_SPY + TILT * sb
    w_ief = BASE_IEF - TILT * se
    s = w_spy + w_ief
    over = s > 1.0
    w_spy = np.where(over, w_spy / s, w_spy)
    w_ief = np.where(over, w_ief / s, w_ief)
    w_shy = np.where(over, 0.0, 1.0 - w_spy - w_ief)
    w_shy = np.maximum(w_shy, 0.0)
    return w_spy, w_ief, w_shy


def target_weights(data, params):
    lookback = int(params["lookback_days"])
    if lookback < 1:
        raise ValueError("lookback_days must be >= 1")
    T = data.dates.shape[0]
    N = len(data.instruments)
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")
    i_shy = data.col("SHY")

    sb, se, full = signals(data, lookback)
    w_spy, w_ief, w_shy = book_weights(sb, se)
    # benchmark 60/40 before the first full lookback
    w_spy = np.where(full, w_spy, BASE_SPY)
    w_ief = np.where(full, w_ief, BASE_IEF)
    w_shy = np.where(full, w_shy, 0.0)

    dec = decision_rows(data.dates)
    w = np.full((T, N), np.nan)
    w[dec, :] = 0.0                      # peer-only instruments (and anything else) at 0
    w[dec, i_spy] = w_spy[dec]
    w[dec, i_ief] = w_ief[dec]
    w[dec, i_shy] = w_shy[dec]
    return w
