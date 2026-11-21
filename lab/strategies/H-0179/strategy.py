"""H-0179 FX delayed overshooting: six G10 currency trusts, monthly dollar-neutral cohorts.

Decision after the close of the 3rd session of each calendar month (rows of the XNYS data calendar).
dy_c = latest carried-forward OECD 10y yield of c minus the value sampled at the decision date
lookback_months calendar months earlier (= lookback_months monthly observations earlier under the
m+2 stamp rule). Long the 2 largest dy, short the 2 smallest, +-0.25/hold_months per name, each cohort
held hold_months*21 sessions; at most hold_months cohorts alive (the oldest is closed when the
hold_months-th later cohort forms, so total gross <= 1).
"""
import numpy as np

PARAMS = {"hold_months": 2, "lookback_months": 3}

# trust ticker -> OECD 10y yield key (DE stands for the euro area), alphabetical by ticker
PAIRS = (("FXA", "AU"), ("FXB", "GB"), ("FXC", "CA"), ("FXE", "DE"), ("FXF", "CH"), ("FXY", "JP"))
SERIES_PREFIX = "fred_oecd_10y_yields_1d."
DECISION_SESSION = 3        # third session of the calendar month
SESSIONS_PER_MONTH = 21
MIN_RANKED = 5
N_LEG = 2
LEG_WEIGHT = 0.25           # per name per cohort, divided by hold_months


def month_ids(dates):
    return np.asarray(dates).astype("datetime64[M]").astype(np.int64)


def decision_rows(dates):
    """Bool (T,): row t is the third session of its calendar month (rows 0..t only).

    The first month of the data counts only if row 0 is the first weekday of its month
    (weekday arithmetic, exchange holidays ignored), otherwise its session count is unknown."""
    dates = np.asarray(dates).astype("datetime64[D]")
    T = len(dates)
    out = np.zeros(T, dtype=bool)
    if T == 0:
        return out
    mon = month_ids(dates)
    idx = np.arange(T)
    new_month = np.ones(T, dtype=bool)
    new_month[1:] = mon[1:] != mon[:-1]
    start = np.maximum.accumulate(np.where(new_month, idx, 0))
    pos = idx - start + 1
    month_start0 = dates[:1].astype("datetime64[M]").astype("datetime64[D]")
    first_ok = bool(np.busday_count(month_start0, dates[:1])[0] == 0)
    known = (mon != mon[0]) | first_ok
    out[:] = (pos == DECISION_SESSION) & known
    return out


def ranked_signal(data, params):
    """Returns (drows, dy, elig, cols, names): decision rows, (D, 6) yield changes, (D, 6) eligibility."""
    L = int(params["lookback_months"])
    dates = data.dates
    names = [tk for tk, _ in PAIRS]
    cols = np.array([data.col(tk) for tk in names], dtype=int)
    Y = np.column_stack([np.asarray(data.series[SERIES_PREFIX + key], dtype=float) for _, key in PAIRS])
    drows = np.flatnonzero(decision_rows(dates))
    D = len(drows)
    y_now = Y[drows]
    dmon = month_ids(dates)[drows]
    y_lag = np.full_like(y_now, np.nan)
    if D > 0:
        target = dmon - L
        pos = np.searchsorted(dmon, target)
        posc = np.minimum(pos, D - 1)
        ok = (pos < D) & (dmon[posc] == target)
        y_lag[ok] = y_now[posc[ok]]
    dy = y_now - y_lag
    trad = np.asarray(data.tradable, dtype=bool)[drows][:, cols]
    listed = np.asarray(data.listed, dtype=bool)[drows][:, cols]
    elig = np.isfinite(dy) & trad & listed
    return drows, dy, elig, cols, names


def legs(dy_row, elig_row, names):
    """Long / short leg (indices into PAIRS) of one decision, or None with fewer than MIN_RANKED eligible."""
    e = [j for j in range(len(names)) if elig_row[j]]
    if len(e) < MIN_RANKED:
        return None
    longs = sorted(e, key=lambda j: (-dy_row[j], -abs(dy_row[j]), names[j]))[:N_LEG]
    rest = [j for j in e if j not in longs]   # legs disjoint even if all dy tie
    shorts = sorted(rest, key=lambda j: (dy_row[j], -abs(dy_row[j]), names[j]))[:N_LEG]
    return longs, shorts


def target_weights(data, params):
    h = int(params["hold_months"])
    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)
    drows, dy, elig, cols, names = ranked_signal(data, params)
    cohorts = []
    for k in range(len(drows)):
        lg = legs(dy[k], elig[k], names)
        if lg is not None:
            cohorts.append((int(drows[k]), lg[0], lg[1]))
    per_name = LEG_WEIGHT / h
    life = h * SESSIONS_PER_MONTH
    for i, (t, longs, shorts) in enumerate(cohorts):
        end = t + life                       # zero target from row t+life: closed at the open after its last session
        if i + h < len(cohorts):
            end = min(end, cohorts[i + h][0])  # cap: at most hold_months live cohorts
        end = min(end, T)
        for j in longs:
            w[t:end, cols[j]] += per_name
        for j in shorts:
            w[t:end, cols[j]] -= per_name
    return w
