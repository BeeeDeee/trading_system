"""H-0179 mechanism-test events: at each monthly decision date with >= 5 ranked currencies, mark only the
single largest home-yield rise (+1) and the single largest fall (-1); same eligibility and tie-breaks as the
strategy (code copied from strategy.py, diagnostic.py is a separate file)."""
import numpy as np

PAIRS = (("FXA", "AU"), ("FXB", "GB"), ("FXC", "CA"), ("FXE", "DE"), ("FXF", "CH"), ("FXY", "JP"))
SERIES_PREFIX = "fred_oecd_10y_yields_1d."
DECISION_SESSION = 3
MIN_RANKED = 5
N_LEG = 2


def month_ids(dates):
    return np.asarray(dates).astype("datetime64[M]").astype(np.int64)


def decision_rows(dates):
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


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    drows, dy, elig, cols, names = ranked_signal(data, params)
    uni = data.universe
    for k in range(len(drows)):
        t = int(drows[k])
        e = [j for j in range(len(names)) if elig[k, j]]
        if len(e) < MIN_RANKED:
            continue
        longs = sorted(e, key=lambda j: (-dy[k, j], -abs(dy[k, j]), names[j]))[:N_LEG]
        rest = [j for j in e if j not in longs]
        shorts = sorted(rest, key=lambda j: (dy[k, j], -abs(dy[k, j]), names[j]))[:N_LEG]
        top = longs[0]      # first name of the strategy's long leg
        bot = shorts[0]     # first name of the strategy's short leg
        for j, s in ((top, 1.0), (bot, -1.0)):
            c = cols[j]
            if uni is not None and not bool(uni[t, c]):
                continue
            mask[t, c] = True
            side[t, c] = s
    return mask, side
