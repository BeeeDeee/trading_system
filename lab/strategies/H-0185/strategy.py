"""H-0185 v2: breakeven-inflation reversion, TIPS (TIP/SCHP) vs IEF, weekly 52-week cohorts.

TLT, SHY and IEI are peer-only instruments (event-study peers): they always get weight 0.
"""
import numpy as np

PARAMS = {"hold_weeks": 52, "lookback_days": 756}

BE_KEY = "fred_macro.T5YIE"
ENTRY_Z = 1.0          # fixed by the card, not a parameter
MIN_VALID = 0.8        # share of valid values the lookback window needs


def _ffill(x):
    """Forward-fill NaNs with the last valid value (point-in-time)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(n, np.nan)
    ok = idx >= 0
    out[ok] = x[idx[ok]]
    return out


def decision_days(dates):
    """Weekly decision rows: the last trading session of each week, point-in-time.

    A row is a decision if it is a Friday. If a week ended without a Friday session
    (Friday holiday), that is only visible on the first row of the next week, which then
    takes that week's decision (one session late). Uses only the dates of rows <= t.
    """
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    wd = (d + 3) % 7            # Monday = 0 ... Friday = 4 (1970-01-01 was a Thursday)
    week = (d + 3) // 7         # Monday-based week number
    dec = wd == 4
    if d.shape[0] > 1:
        new_week = week[1:] != week[:-1]
        prev_not_friday = wd[:-1] != 4
        dec[1:] |= new_week & prev_not_friday
    return dec


def breakeven_z(be_row, lookback):
    """z of the breakeven over the `lookback` rows ending at row t.

    `be_row[t]` is the aligned series (clock next_morning): the last T5YIE value known after
    the close of t, i.e. the value dated t-1 or earlier. The window is the `lookback` rows
    ending at row t (= sessions ending t-1). Gaps are forward-filled; z is NaN if fewer than
    80 % of the window's rows hold a valid (raw) value or the window is not yet complete.
    """
    raw = np.asarray(be_row, dtype=float)
    x = _ffill(raw)
    n = x.shape[0]
    L = int(lookback)
    z = np.full(n, np.nan)
    if n < L or L < 2:
        return z
    valid_raw = np.isfinite(raw).astype(float)
    have = np.isfinite(x)
    xv = np.where(have, x, 0.0)
    c_raw = np.concatenate([[0.0], np.cumsum(valid_raw)])
    c_n = np.concatenate([[0.0], np.cumsum(have.astype(float))])
    c_s = np.concatenate([[0.0], np.cumsum(xv)])
    c_q = np.concatenate([[0.0], np.cumsum(xv * xv)])
    end = np.arange(L, n + 1)           # window rows [end-L, end)
    start = end - L
    nraw = c_raw[end] - c_raw[start]
    cnt = c_n[end] - c_n[start]
    s = c_s[end] - c_s[start]
    q = c_q[end] - c_q[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
        var = (q - cnt * mean * mean) / (cnt - 1.0)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        zz = (x[end - 1] - mean) / sd
    ok = (nraw >= MIN_VALID * L) & (cnt >= 2) & (sd > 1e-12) & np.isfinite(x[end - 1])
    z[end - 1] = np.where(ok, zz, np.nan)
    return z


def cohort_signs(z):
    """+1 (long TIPS / short IEF) if z <= -1, -1 if z >= +1, 0 otherwise or if z is missing."""
    s = np.zeros(z.shape[0])
    s[np.isfinite(z) & (z <= -ENTRY_Z)] = 1.0
    s[np.isfinite(z) & (z >= ENTRY_Z)] = -1.0
    return s


def _rolling_sum_last(v, h):
    c = np.concatenate([[0.0], np.cumsum(v)])
    k = np.arange(v.shape[0])
    lo = np.maximum(k - h + 1, 0)
    return c[k + 1] - c[lo]


def book_weights(signs, a_tip, a_schp, a_ief, hold):
    """Book over the decision sequence.

    signs: (K,) cohort signs at each decision; a_*: (K,) availability (has a price) at each
    decision. Returns (K, 3) weights for TIP, SCHP, IEF. Each cohort fixes its TIPS split at
    formation (equal TIP/SCHP if both priced, else the one priced; flat for life if neither).
    At later decisions a leg on an unavailable TIPS ETF moves to the other one; if neither is
    available the cohort is flat (both legs). Live cohorts: the `hold` most recent.
    """
    signs = np.asarray(signs, dtype=float)
    a_tip = np.asarray(a_tip, dtype=bool)
    a_schp = np.asarray(a_schp, dtype=bool)
    a_ief = np.asarray(a_ief, dtype=bool)
    H = int(hold)
    both0 = a_tip & a_schp
    tip0 = a_tip & ~a_schp
    schp0 = ~a_tip & a_schp
    n_both = _rolling_sum_last(signs * both0, H)
    n_tip = _rolling_sum_last(signs * tip0, H)
    n_schp = _rolling_sum_last(signs * schp0, H)

    leg = 0.5 / H
    w_tip = np.zeros(signs.shape[0])
    w_schp = np.zeros(signs.shape[0])
    active = np.zeros(signs.shape[0])     # signed cohort count whose TIPS leg exists now

    # cohorts formed with both: split equally; one missing -> all to the other
    both_now = a_tip & a_schp
    w_tip += np.where(both_now, 0.5 * n_both, np.where(a_tip, n_both, 0.0))
    w_schp += np.where(both_now, 0.5 * n_both, np.where(a_schp, n_both, 0.0))
    active += np.where(a_tip | a_schp, n_both, 0.0)
    # cohorts formed TIP-only
    w_tip += np.where(a_tip, n_tip, 0.0)
    w_schp += np.where(~a_tip & a_schp, n_tip, 0.0)
    active += np.where(a_tip | a_schp, n_tip, 0.0)
    # cohorts formed SCHP-only
    w_schp += np.where(a_schp, n_schp, 0.0)
    w_tip += np.where(~a_schp & a_tip, n_schp, 0.0)
    active += np.where(a_tip | a_schp, n_schp, 0.0)

    w_ief = np.where(a_ief, -active, 0.0)
    return np.stack([w_tip * leg, w_schp * leg, w_ief * leg], axis=1)


def _available(data, j):
    close = np.asarray(data.close[:, j], dtype=float)
    ok = np.isfinite(close) & np.asarray(data.listed[:, j], dtype=bool)
    if data.universe is not None:
        ok &= np.asarray(data.universe[:, j], dtype=bool)
    return ok


def target_weights(data, params):
    hold = int(params["hold_weeks"])
    lookback = int(params["lookback_days"])
    T = len(data.dates)
    N = len(data.instruments)

    j_tip = data.col("TIP")
    j_schp = data.col("SCHP")
    j_ief = data.col("IEF")
    # peer-only instruments: must exist in the view, never traded
    for name in ("TLT", "SHY", "IEI"):
        data.col(name)

    be = np.asarray(data.series[BE_KEY], dtype=float)
    z = breakeven_z(be, lookback)
    dec = decision_days(data.dates)
    rows = np.flatnonzero(dec)

    out = np.full((T, N), np.nan)
    if rows.size == 0:
        return out
    a_tip = _available(data, j_tip)[rows]
    a_schp = _available(data, j_schp)[rows]
    a_ief = _available(data, j_ief)[rows]
    signs = cohort_signs(z[rows])
    # a cohort formed with no TIPS ETF priced is flat for life
    signs = np.where(a_tip | a_schp, signs, 0.0)
    w = book_weights(signs, a_tip, a_schp, a_ief, hold)

    out[rows, :] = 0.0
    out[rows, j_tip] = w[:, 0]
    out[rows, j_schp] = w[:, 1]
    out[rows, j_ief] = w[:, 2]
    return out
