"""H-0008: earnings-announcement premium via volume seasonality (weekly LIQ-500 long/short).

Rules of the card and where they are implemented:
  Step 1  AV(i,d) = dv(i,d) / median(dv(i, d-63 .. d-1)), >= 40 valid days        -> abnormal_volume
          AVn(i,d) = AV(i,d) / median_{j in LIQ-500(d)} AV(j,d)                     -> normalized_av
  Step 2  event day: AVn >= av_threshold and AVn = max AVn over rows d-21 .. d+21   -> event_days
          (only used for d with d + 21 <= t)
  Step 3  predicted announcer: event in [D1-364-tol, D5-364+tol] and in
          [D1-91-10, D5-91+10] (calendar dates, D1..D5 = Mon..Fri of the holding week) -> target_weights
  Step 4  eligible: LIQ-500 member and tradable at the decision row, AVn defined on
          >= 80 % of the trading rows of the last 380 calendar days                -> target_weights
  Step 5  long w = L / N_long with L = min(0.5, 0.05 N_long); short the other eligible
          stocks, total short gross L; flat if N_long = 0                         -> target_weights

Decision calendar (point-in-time approximation): a row is a decision row if its date is a Friday
(weekday arithmetic, holidays unknown). If a week ended before Friday (e.g. Good Friday), no Friday row
exists; then the first trading row of the following week is a catch-up decision for that (current) week.
"""
import numpy as np

PARAMS = {"av_threshold": 2.0, "tol_days": 3}

BASE_WIN = 63        # baseline window of the daily abnormal volume (trading rows d-63 .. d-1)
MIN_VALID = 40       # minimum valid days in the baseline window
EVENT_HALF = 21      # event day = maximum over rows d-21 .. d+21
ANN_LAG = 364        # 52 weeks, calendar days
Q_LAG = 91           # 13 weeks, calendar days
Q_TOL = 10           # fixed tolerance of the quarterly confirmation, calendar days
ELIG_DAYS = 380      # eligibility look-back, calendar days
ELIG_NUM, ELIG_DEN = 4, 5   # 80 % of trading rows with defined AVn
MAX_W = 0.05         # max weight per long name
MAX_LEG = 0.5        # max gross of the long leg


def _day_ints(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def _weekday(days):
    """Monday = 0 .. Sunday = 6 (day 0 of the epoch was a Thursday)."""
    return (days + 3) % 7


def abnormal_volume(dv):
    """AV(i,d) = dv(i,d) / median of the valid dv in rows d-63 .. d-1; NaN if < 40 valid days.

    Valid = finite and > 0 (a missing or zero dollar volume is not a trading day of the stock)."""
    dv = np.asarray(dv, dtype=float)
    T, N = dv.shape
    valid = np.isfinite(dv) & (dv > 0)
    med = np.full((T, N), np.nan)
    if T == 0 or N == 0:
        return med
    x = np.where(valid, dv, np.inf)
    padded = np.concatenate([np.full((BASE_WIN, N), np.inf), x], axis=0)       # (T+63, N)
    windows = np.lib.stride_tricks.sliding_window_view(padded, BASE_WIN, axis=0)  # window d = rows d-63..d-1
    cv = np.zeros((T + 1, N), dtype=np.int64)
    np.cumsum(valid, axis=0, out=cv[1:])
    rows = np.arange(T)
    n_valid = cv[rows] - cv[np.maximum(rows - BASE_WIN, 0)]                     # valid count in d-63..d-1
    chunk = max(1, 4_000_000 // (N * BASE_WIN))
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        nn = n_valid[c0:c1]
        ok = nn >= MIN_VALID
        cols = np.nonzero(ok.any(axis=0))[0]
        if cols.size == 0:
            continue
        s = np.sort(windows[c0:c1][:, cols, :], axis=-1)    # inf (invalid) sorts last
        nc = nn[:, cols]
        k_lo = np.clip((nc - 1) // 2, 0, BASE_WIN - 1)[..., None]
        k_hi = np.clip(nc // 2, 0, BASE_WIN - 1)[..., None]
        m = 0.5 * (np.take_along_axis(s, k_lo, axis=-1)[..., 0] + np.take_along_axis(s, k_hi, axis=-1)[..., 0])
        block = med[c0:c1]
        block[:, cols] = np.where(ok[:, cols], m, np.nan)
    good = valid & np.isfinite(med) & (med > 0)
    av = np.full((T, N), np.nan)
    av[good] = dv[good] / med[good]
    return av


def normalized_av(av, universe):
    """AVn(i,d) = AV(i,d) / median over LIQ-500 members j on day d (with defined AV) of AV(j,d)."""
    T, N = av.shape
    mask = np.asarray(universe, dtype=bool) & np.isfinite(av)
    n = mask.sum(axis=1)
    s = np.sort(np.where(mask, av, np.inf), axis=1)
    k_lo = np.clip((n - 1) // 2, 0, max(N - 1, 0))[:, None]
    k_hi = np.clip(n // 2, 0, max(N - 1, 0))[:, None]
    cm = np.full(T, np.nan)
    if N > 0:
        m = 0.5 * (np.take_along_axis(s, k_lo, axis=1)[:, 0] + np.take_along_axis(s, k_hi, axis=1)[:, 0])
        cm = np.where(n > 0, m, np.nan)
    del s
    avn = np.full((T, N), np.nan)
    good = np.isfinite(av) & np.isfinite(cm)[:, None] & (cm > 0)[:, None]
    avn[good] = (av / np.where(np.isfinite(cm) & (cm > 0), cm, 1.0)[:, None])[good]
    return avn


def centered_max(avn):
    """Max of AVn over rows d-21 .. d+21 (NaN ignored, -inf if none)."""
    T, N = avn.shape
    h = EVENT_HALF
    y = np.where(np.isfinite(avn), avn, -np.inf)
    pad = np.full((h, N), -np.inf)
    padded = np.concatenate([pad, y, pad], axis=0)                                # (T+2h, N)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * h + 1, axis=0)  # window d = rows d-h..d+h
    out = np.full((T, N), -np.inf)
    if T == 0 or N == 0:
        return out
    chunk = max(1, 4_000_000 // (N * (2 * h + 1)))
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        out[c0:c1] = windows[c0:c1].max(axis=-1)
    return out


def event_days(avn, cmax, av_threshold):
    """Event day: AVn >= threshold and AVn is the max over rows d-21 .. d+21. Rows whose forward
    window is not complete in the data (d + 21 > last row) are never events."""
    T = avn.shape[0]
    with np.errstate(invalid="ignore"):
        ev = np.isfinite(avn) & (avn >= av_threshold) & (avn >= cmax)
    ev[max(T - EVENT_HALF, 0):] = False
    return ev


def _row_range(days, lo_day, hi_day, t):
    """Row slice [lo, hi) of rows with lo_day <= date <= hi_day, restricted to rows <= t - 21."""
    lo = np.searchsorted(days, lo_day, side="left")
    hi = np.searchsorted(days, hi_day, side="right")
    hi = np.minimum(hi, t - EVENT_HALF + 1)
    hi = np.maximum(hi, lo)
    return lo, hi


def decision_rows(days):
    """Fridays, plus a catch-up decision on the first row of a week whose previous week had no Friday row.
    Returns the decision rows and D1 (Monday of the holding week, day integer)."""
    T = days.shape[0]
    wd = _weekday(days)
    is_fri = wd == 4
    week = (days + 3) // 7          # Monday-based week number
    first_of_week = np.zeros(T, dtype=bool)
    prev_not_fri = np.zeros(T, dtype=bool)
    if T > 1:
        first_of_week[1:] = week[1:] != week[:-1]
        prev_not_fri[1:] = wd[:-1] != 4
    catch_up = first_of_week & prev_not_fri & ~is_fri
    dec = np.nonzero(is_fri | catch_up)[0]
    monday = days - wd
    d1 = np.where(is_fri, monday + 7, monday)[dec]
    return dec, d1


def target_weights(data, params):
    av_threshold = float(params["av_threshold"])
    tol_days = int(params["tol_days"])

    if data.universe is None:
        raise ValueError("H-0008 needs the point-in-time LIQ-500 universe (data.universe is None)")
    dates = data.dates
    days = _day_ints(dates)
    T = days.shape[0]
    N = len(data.instruments)
    dv = np.asarray(data.dollar_volume, dtype=float)
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    if dv.shape != (T, N) or uni.shape != (T, N) or trad.shape != (T, N):
        raise ValueError("input shapes do not match (T, N)")

    w = np.full((T, N), np.nan)
    if T == 0 or N == 0:
        return w

    # Steps 1-2
    av = abnormal_volume(dv)
    avn = normalized_av(av, uni)
    del av
    ev = event_days(avn, centered_max(avn), av_threshold)
    ce = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(ev, axis=0, out=ce[1:])
    del ev
    cd = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(np.isfinite(avn), axis=0, out=cd[1:])
    del avn

    dec, d1 = decision_rows(days)
    if dec.size == 0:
        return w
    d5 = d1 + 4

    # Step 3: annual lag and quarterly confirmation (calendar-date windows mapped to rows <= t-21)
    lo_a, hi_a = _row_range(days, d1 - ANN_LAG - tol_days, d5 - ANN_LAG + tol_days, dec)
    lo_q, hi_q = _row_range(days, d1 - Q_LAG - Q_TOL, d5 - Q_LAG + Q_TOL, dec)
    annual = (ce[hi_a] - ce[lo_a]) > 0
    quarterly = (ce[hi_q] - ce[lo_q]) > 0

    # Step 4: eligibility at the decision close
    lo_e = np.searchsorted(days, days[dec] - ELIG_DAYS + 1, side="left")   # dates in (t-380, t]
    n_rows = (dec + 1 - lo_e)[:, None]
    n_def = cd[dec + 1] - cd[lo_e]
    eligible = uni[dec] & trad[dec] & (ELIG_DEN * n_def >= ELIG_NUM * n_rows)

    # Step 5: weights
    long_ = eligible & annual & quarterly
    short = eligible & ~long_
    n_long = long_.sum(axis=1)
    n_short = short.sum(axis=1)
    leg = np.minimum(MAX_LEG, MAX_W * n_long)
    active = (n_long > 0) & (n_short > 0)
    leg = np.where(active, leg, 0.0)
    w_long = leg / np.maximum(n_long, 1)
    w_short = leg / np.maximum(n_short, 1)
    rows = np.where(long_, w_long[:, None], 0.0) - np.where(short, w_short[:, None], 0.0)
    w[dec] = rows
    return w
