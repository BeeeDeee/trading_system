"""H-0042 mechanism events: after the close of A-2 for every valid scheduled announcement day A (window {A-1, A}
not touching the month turn), the three eligible ETFs with the highest trailing SPY beta get side +1 and the
three with the lowest get side -1; the rest are unmarked peers. Same calendar and beta logic as strategy.py."""
import numpy as np

UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "SHY", "IEF", "TLT", "LQD", "GLD")
CAL = "macro_release_calendar_1d."
TYPE_KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
MIN_ELIGIBLE = 9
N_SIDE = 3


def _calendar_key(data, key):
    x = np.asarray(data.series[CAL + key], dtype=float).reshape(-1)
    if x.shape[0] != len(data.dates):
        raise ValueError("calendar series %s has %d rows, data has %d" % (key, x.shape[0], len(data.dates)))
    out = np.full(x.shape, -1.0)
    fin = np.isfinite(x)
    out[fin] = np.round(x[fin])
    return out


def _in_month_turn(x):
    x = np.asarray(x, dtype="datetime64[D]")
    month = x.astype("datetime64[M]")
    nxt = np.busday_offset(x, 1, roll="forward")
    last = nxt.astype("datetime64[M]") != month
    first3 = np.busday_count(month.astype("datetime64[D]"), x) < 3
    return last | first3


def _rolling_sum(a, n):
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= n:
        c = np.cumsum(a, axis=0, dtype=float)
        lag = np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-n]], axis=0)
        out[n - 1:] = c[n - 1:] - lag
    return out


def _betas(data, lookback):
    cols = [data.col(n) for n in UNIVERSE]
    rco = np.asarray(data.ret_co, dtype=float)[:, cols]
    roc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    ok = np.isfinite(r) & listed
    ok[1:] &= listed[:-1]
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


def events(data, params):
    lookback = int(params["beta_lookback"])
    dates = np.asarray(data.dates, dtype="datetime64[D]")
    T = len(dates)
    N = len(data.instruments)
    cols = np.array([data.col(n) for n in UNIVERSE])

    nrel = _calendar_key(data, "next_release_sessions")
    a2 = nrel == 2
    for k in TYPE_KEYS:
        a2 = a2 | (_calendar_key(data, k) == 2)
    d1 = np.busday_offset(dates, 1, roll="forward")
    d2 = np.busday_offset(dates, 2, roll="forward")
    valid = ~(_in_month_turn(d1) | _in_month_turn(d2))

    beta, elig = _betas(data, lookback)
    ev = a2 & valid & (elig.sum(axis=1) >= MIN_ELIGIBLE)

    hi = np.argsort(-np.where(elig, beta, -np.inf), axis=1, kind="stable")[:, :N_SIDE]
    lo = np.argsort(np.where(elig, beta, np.inf), axis=1, kind="stable")[:, :N_SIDE]
    rows = np.nonzero(ev)[0]
    m = np.zeros((T, len(UNIVERSE)), dtype=bool)
    sd = np.zeros((T, len(UNIVERSE)))
    for k in range(N_SIDE):
        m[rows, hi[rows, k]] = True
        sd[rows, hi[rows, k]] = 1.0
        m[rows, lo[rows, k]] = True
        sd[rows, lo[rows, k]] = -1.0

    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    m &= tradable
    if data.universe is not None:
        m &= np.asarray(data.universe, dtype=bool)[:, cols]

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    mask[:, cols] = m
    side[:, cols] = np.where(m, sd, 0.0)
    return mask, side
