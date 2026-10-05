"""H-0007: intraday-component reversal confirmed by intraday-component momentum.

Weekly long/short in the PIT LIQ-500. At the last trading day t of each calendar week:
  l_d       = ln(1 + ret_oc_d), 0 on days without a valid ret_oc
  IR_fast   = sum l_d, d = t-fast_days+1 .. t
  IR_slow   = sum l_d, d = t-slow_days+1 .. t-21
  eligible  = PIT member at t, tradable at t, >= 90 % valid days in both windows,
              listed for >= slow_days rows
  split eligible at the IR_slow median into W (top half) / L (bottom half);
  long  = round(20 % of |W|) names of W with the lowest IR_fast   (+0.5 / n_long each)
  short = round(20 % of |L|) names of L with the highest IR_fast  (-0.5 / n_short each)
  fewer than 10 names per leg -> flat that week. Ties broken by instrument id.
"""
import numpy as np

PARAMS = {"fast_days": 5, "slow_days": 252}

SKIP_DAYS = 21          # most recent trading days excluded from the slow window
MIN_VALID_SHARE = 0.9   # share of days with a valid ret_oc required in each window
LEG_SHARE = 0.2         # quintile of each half
MIN_LEG = 10            # minimum names per leg, otherwise flat
LEG_GROSS = 0.5         # gross per leg (dollar neutral, total gross 1.0)
FIXED_SCALE = 1e12      # fixed-point resolution of the log-return sums (exact integer window sums)


# --------------------------------------------------------------------------- calendar

def _civil_from_days(days):
    """Proleptic Gregorian (year, month, day) from days since the Unix epoch (integer arithmetic)."""
    z = days + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    y = y + (m <= 2)
    return y, m, d


def _easter(y):
    """Month and day of (Gregorian) Easter Sunday, anonymous Gregorian algorithm."""
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return month, day


def _is_scheduled_friday_holiday(days):
    """True where the day (assumed a Friday) is a regular, pre-announced NYSE full-day holiday."""
    y, m, d = _civil_from_days(days)
    new_year = (m == 1) & (d == 1)                       # Jan 1 on a Friday (no Dec 31 observance)
    july4 = (m == 7) & ((d == 4) | (d == 3))             # Jul 4 on Fri, or observed Fri Jul 3
    xmas = (m == 12) & ((d == 25) | (d == 24))           # Dec 25 on Fri, or observed Fri Dec 24
    juneteenth = (y >= 2022) & (m == 6) & ((d == 19) | (d == 18))
    sy, sm, sd = _civil_from_days(days + 2)              # the Sunday after
    em, ed = _easter(sy)
    good_friday = (sm == em) & (sd == ed)
    return new_year | july4 | xmas | juneteenth | good_friday


def decision_days(dates):
    """Last trading day of each calendar week, computed from the date of row t alone.

    A row is a decision day if it is a Friday, or a Thursday whose Friday is a scheduled NYSE
    holiday (Good Friday, New Year, Independence Day, Christmas, Juneteenth from 2022).
    Unscheduled closures cannot be known in advance; such weeks have no decision.
    """
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7                              # Monday = 0 ... Sunday = 6
    fri = weekday == 4
    thu = weekday == 3
    return fri | (thu & _is_scheduled_friday_holiday(days + 1))


# --------------------------------------------------------------------------- signal

def _window_sum(padded_cum, end, length):
    """Sum over rows end-length+1 .. end (inclusive) from a cumsum padded with a leading zero row.

    Rows where the window starts before row 0 are NaN.
    """
    T = padded_cum.shape[0] - 1
    out = np.full((T,) + padded_cum.shape[1:], np.nan)
    t = np.arange(T)
    e = t + end                     # window end row (end <= 0 is an offset from t)
    s = e - length + 1              # window start row
    ok = s >= 0
    out[ok] = padded_cum[e[ok] + 1] - padded_cum[s[ok]]
    return out


def signals(data, params):
    fast = int(params["fast_days"])
    slow = int(params["slow_days"])
    if fast < 1 or slow - SKIP_DAYS < 1:
        raise ValueError("invalid window parameters")

    ret_oc = np.asarray(data.ret_oc, dtype=float)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    T, N = ret_oc.shape

    valid = listed & tradable & np.isfinite(ret_oc) & np.isfinite(close) & (ret_oc > -1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        logr = np.where(valid, np.log1p(np.where(valid, ret_oc, 0.0)), 0.0)

    # Window sums from an exact integer (fixed-point, 1e-12) cumulative sum: float cumsum differences
    # carry rounding noise that would break genuine ties arbitrarily instead of by instrument id.
    zero = np.zeros((1, N), dtype=np.int64)
    fixed = np.rint(np.clip(logr, -20.0, 20.0) * FIXED_SCALE).astype(np.int64)
    cum_l = np.concatenate([zero, np.cumsum(fixed, axis=0)], axis=0)
    cum_v = np.concatenate([zero, np.cumsum(valid.astype(np.int64), axis=0)], axis=0)

    ir_fast = _window_sum(cum_l, 0, fast) / FIXED_SCALE
    ir_slow = _window_sum(cum_l, -SKIP_DAYS, slow - SKIP_DAYS) / FIXED_SCALE
    n_fast = _window_sum(cum_v, 0, fast)
    n_slow = _window_sum(cum_v, -SKIP_DAYS, slow - SKIP_DAYS)

    # rows since first listing (inclusive); "listed for less than slow_days" -> not eligible
    ever_listed = np.maximum.accumulate(listed.astype(np.int64), axis=0)
    age = np.cumsum(ever_listed, axis=0)

    member = np.ones((T, N), dtype=bool) if data.universe is None else np.asarray(data.universe, dtype=bool)

    with np.errstate(invalid="ignore"):
        eligible = (
            member
            & tradable
            & listed
            & (age >= slow)
            & (n_fast >= MIN_VALID_SHARE * fast - 1e-9)
            & (n_slow >= MIN_VALID_SHARE * (slow - SKIP_DAYS) - 1e-9)
            & np.isfinite(ir_fast)
            & np.isfinite(ir_slow)
        )
    return ir_fast, ir_slow, eligible


def _id_rank(instruments):
    """Rank of each instrument id in sorted (lexicographic) order, for deterministic tie-breaks."""
    names = list(instruments)
    order = sorted(range(len(names)), key=lambda j: names[j])
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(names))
    return rank


def form_book(ir_fast_row, ir_slow_row, elig_row, id_rank):
    """Weights for one decision row."""
    w = np.zeros(elig_row.shape[0])
    idx = np.flatnonzero(elig_row)
    n = idx.size
    half = n // 2
    k = int(np.floor(LEG_SHARE * half + 0.5))
    if k < MIN_LEG:
        return w
    # median split on IR_slow (ties by id); with an odd count the median name is in neither half
    o = np.lexsort((id_rank[idx], ir_slow_row[idx]))
    losers_slow = idx[o[:half]]
    winners_slow = idx[o[n - half:]]
    # long: lowest IR_fast within W
    ol = np.lexsort((id_rank[winners_slow], ir_fast_row[winners_slow]))
    longs = winners_slow[ol[:k]]
    # short: highest IR_fast within L
    os_ = np.lexsort((id_rank[losers_slow], -ir_fast_row[losers_slow]))
    shorts = losers_slow[os_[:k]]
    w[longs] = LEG_GROSS / k
    w[shorts] = -LEG_GROSS / k
    return w


def target_weights(data, params):
    ir_fast, ir_slow, eligible = signals(data, params)
    T, N = ir_fast.shape
    dec = decision_days(data.dates)
    id_rank = _id_rank(data.instruments)
    out = np.full((T, N), np.nan)
    for t in np.flatnonzero(dec):
        out[t] = form_book(ir_fast[t], ir_slow[t], eligible[t], id_rank)
    return out
