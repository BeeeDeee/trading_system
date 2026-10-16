"""H-0042 Macro-announcement beta premium.

Always-invested 60/40-like core (SPY 0.60 - tilt/5, IEF 0.40 + tilt/5). Over the two sessions {A-1, A} of
every scheduled CPI / Employment Situation / FOMC-statement day A whose window does not touch the month turn,
`tilt` of IEF is swapped into the three eligible ETFs (other than IEF) with the highest trailing SPY beta,
tilt/3 each, chosen at the entry decision (close of A-2) and fixed for the whole (possibly joined) window.
Targets are re-applied only when the tilt state changes and on the first decision of each month.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"beta_lookback": 252, "tilt": 0.4}

UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "SHY", "IEF", "TLT", "LQD", "GLD")
CAL = "macro_release_calendar_1d."
TYPE_KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
MIN_ELIGIBLE = 9
N_TOP = 3


def _calendar_key(data, key):
    """Sessions to the next release (integer), -1 where unknown (NaN = no announcement known)."""
    x = np.asarray(data.series[CAL + key], dtype=float).reshape(-1)
    if x.shape[0] != len(data.dates):
        raise ValueError("calendar series %s has %d rows, data has %d" % (key, x.shape[0], len(data.dates)))
    out = np.full(x.shape, -1.0)
    fin = np.isfinite(x)
    out[fin] = np.round(x[fin])
    return out


def in_month_turn(x):
    """Weekday approximation of the month-turn window (last session of a month .. third session of the next).

    x: datetime64[D] array of (weekday) dates. A date is the last session of its month if the next weekday
    falls in another month; it is among the first three sessions if fewer than three weekdays of its month
    precede it. Exchange holidays are ignored (np.busday_* weekday arithmetic).
    """
    x = np.asarray(x, dtype="datetime64[D]")
    month = x.astype("datetime64[M]")
    nxt = np.busday_offset(x, 1, roll="forward")
    last = nxt.astype("datetime64[M]") != month
    first3 = np.busday_count(month.astype("datetime64[D]"), x) < 3
    return last | first3


def announcement_windows(data):
    """Per decision row d (after the close of d):

    a1[d]: session d+1 is an announcement day A;  a2[d]: session d+2 is an announcement day A;
    v2[d]: the A at d+2 is valid (neither d+1 nor d+2 in the month-turn window);
    tilt_session[d]: session d+1 is a tilt session (A-1 or A of a valid A).
    The validity of an A is fixed when it is first seen two sessions ahead (close of A-2) and re-used at A-1.
    """
    dates = np.asarray(data.dates, dtype="datetime64[D]")
    nrel = _calendar_key(data, "next_release_sessions")
    types = [_calendar_key(data, k) for k in TYPE_KEYS]
    a1 = nrel == 1
    a2 = nrel == 2
    for t in types:
        a1 = a1 | (t == 1)
        a2 = a2 | (t == 2)
    d1 = np.busday_offset(dates, 1, roll="forward")
    d2 = np.busday_offset(dates, 2, roll="forward")
    mt0 = in_month_turn(dates)
    mt1 = in_month_turn(d1)
    mt2 = in_month_turn(d2)
    v1 = ~(mt0 | mt1)          # A = d+1: A-1 = d (real date), A = d+1 (weekday estimate)
    v2 = ~(mt1 | mt2)          # A = d+2: A-1 = d+1, A = d+2 (weekday estimates)
    valid1 = v1.copy()
    if len(dates) > 1:
        valid1[1:] = np.where(a2[:-1], v2[:-1], v1[1:])
    tilt_session = (a1 & valid1) | (a2 & v2)
    return a1, a2, v2, tilt_session


def _rolling_sum(a, n):
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= n:
        c = np.cumsum(a, axis=0, dtype=float)
        lag = np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-n]], axis=0)
        out[n - 1:] = c[n - 1:] - lag
    return out


def betas(data, lookback):
    """(T, 10) OLS beta of each universe ETF's close-to-close total return on SPY's over the last `lookback`
    sessions up to and including row t, and (T, 10) eligibility (a full window of returns)."""
    cols = [data.col(n) for n in UNIVERSE]
    rco = np.asarray(data.ret_co, dtype=float)[:, cols]
    roc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    ok = np.isfinite(r) & listed
    ok[1:] &= listed[:-1]          # a return needs the previous close
    ok[0] = False
    r = np.where(ok, r, 0.0)
    s = UNIVERSE.index("SPY")
    x = r[:, [s]]
    n = float(lookback)
    cnt = _rolling_sum(ok.astype(float), lookback)
    sx = _rolling_sum(x, lookback)
    sy = _rolling_sum(r, lookback)
    sxx = _rolling_sum(x * x, lookback)
    sxy = _rolling_sum(x * r, lookback)
    var = sxx - sx * sx / n
    cov = sxy - sx * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    elig = np.nan_to_num(cnt, nan=0.0) >= n - 0.5
    elig &= elig[:, [s]]
    beta[:, s] = np.where(elig[:, s], 1.0, np.nan)
    elig &= np.isfinite(beta)
    return beta, elig


def target_weights(data, params):
    lookback = int(params["beta_lookback"])
    tilt = float(params["tilt"])
    dates = np.asarray(data.dates, dtype="datetime64[D]")
    T = len(dates)
    N = len(data.instruments)
    cols = np.array([data.col(n) for n in UNIVERSE])
    spy = UNIVERSE.index("SPY")
    ief = UNIVERSE.index("IEF")

    beta, elig = betas(data, lookback)
    n_elig = elig.sum(axis=1)
    _, _, _, tilt_session = announcement_windows(data)

    # runs of consecutive tilt sessions (joined windows) share the entry decision of their first row
    prev = np.zeros(T, dtype=bool)
    prev[1:] = tilt_session[:-1]
    start = tilt_session & ~prev
    run_start = np.maximum.accumulate(np.where(start, np.arange(T), 0))
    tilt_on = tilt_session & (n_elig[run_start] >= MIN_ELIGIBLE)

    # top three by beta at the entry decision, IEF excluded, ineligible excluded
    score = np.where(elig, beta, -np.inf)
    score[:, ief] = -np.inf
    order = np.argsort(-score, axis=1, kind="stable")[:, :N_TOP]
    top = order[run_start]

    W = np.zeros((T, len(UNIVERSE)))
    W[:, spy] = round(0.60 - tilt / 5.0, 12)
    W[:, ief] = round(0.40 + tilt / 5.0, 12)
    rows = np.nonzero(tilt_on)[0]
    W[rows, ief] = round(0.40 + tilt / 5.0 - tilt, 12)
    for k in range(N_TOP):
        W[rows, top[rows, k]] += tilt / N_TOP

    # decisions: tilt state changes and the first decision of each calendar month
    prev_on = np.zeros(T, dtype=bool)
    prev_on[1:] = tilt_on[:-1]
    month_first = np.asarray(period_starts(dates, "M"), dtype=bool)
    dec = (tilt_on != prev_on) | month_first
    if T > 0:
        dec[0] = True

    out = np.full((T, N), np.nan)
    rows_dec = np.nonzero(dec)[0]
    full = np.zeros((len(rows_dec), N))
    full[:, cols] = W[rows_dec]
    # an ETF without a bar on the decision day keeps its weight
    close = np.asarray(data.close, dtype=float)[rows_dec][:, cols]
    sub = full[:, cols]
    sub[~np.isfinite(close)] = np.nan
    full[:, cols] = sub
    out[rows_dec] = full
    return out
