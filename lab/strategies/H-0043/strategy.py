"""H-0043  Credit-ETF gap to its bond index.

Long-only 60/40 SPY/IEF core; the IEF sleeve is tilted into HYG / JNK / LQD for HOLD_DAYS sessions after the
ETF's cumulative return over the last gap_days sessions falls well below what same-day changes of its ICE BofA
OAS and the 2y / 10y Treasury yields imply (out-of-sample rolling OLS residuals, z-scored).

Indexing convention ("known row"): FRED has the next_morning clock, so the series row k holds the observation
dated session k-1.  Every session-d quantity (ETF return of row d, FRED change dated d) is therefore stored on
row k = d + 1, the first row after whose close it is known.  The decision of row t uses rows <= t only, i.e.
sessions <= t-1, exactly as the card says.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"gap_days": 5, "z_entry": 2.0}

HOLD_DAYS = 5          # card: hold_days (5 sessions; named in the text, not a grid parameter)
REG_WINDOW = 252       # sessions in the rolling regression and in the residual standard deviation
MIN_VALID = 200        # minimum valid sessions in a 252-session window, else no signal
HISTORY_MIN = 252      # sessions of history an ETF needs before it can have events
CORE_SPY = 0.60
CORE_IEF = 0.40
SLEEVE = 0.40          # each active slot gets SLEEVE / max(2, n_active), taken out of IEF

CREDIT = ("HYG", "JNK", "LQD")
OAS_OF = {"HYG": "BAMLH0A0HYM2", "JNK": "BAMLH0A0HYM2", "LQD": "BAMLC0A0CM"}
FRED_KEYS = ("BAMLH0A0HYM2", "BAMLC0A0CM", "DGS2", "DGS10")
SERIES_NAME = {
    "BAMLH0A0HYM2": "fred_macro.BAMLH0A0HYM2",
    "BAMLC0A0CM": "fred_macro.BAMLC0A0CM",
    "DGS2": "fred_macro.DGS2",
    "DGS10": "fred_macro.DGS10",
}


# ----------------------------------------------------------------------------------------------- inputs
def load_fred(data):
    """(T, 4) float64 matrix of the FRED series in FRED_KEYS order, as delivered (row k = obs dated k-1).

    A missing series raises KeyError (K-0010).  Coverage cannot be checked point-in-time (a truncated prefix
    legitimately ends before a late series starts); leading NaNs simply mean no signal."""
    T = len(data.dates)
    cols = []
    for key in FRED_KEYS:
        x = np.asarray(data.series[SERIES_NAME[key]], dtype=np.float64).reshape(-1)
        if x.shape[0] != T:
            raise ValueError("FRED series has the wrong length")
        cols.append(x)
    return np.stack(cols, axis=1)


def fred_changes(F):
    """Changes between consecutive valid FRED observations, on known rows.

    A row is a valid FRED day when all four series are finite and it is a new observation (not every series
    identical to the previous row: a forward-filled FRED holiday).  dx[k] = F[k] - F[last valid row before k];
    returns (dx (T, 4), valid (T,) bool) where valid also requires an earlier valid row."""
    T = F.shape[0]
    finite = np.all(np.isfinite(F), axis=1)
    same = np.zeros(T, dtype=bool)
    if T > 1:
        same[1:] = np.all(F[1:] == F[:-1], axis=1)
    fv = finite & ~same
    rows = np.arange(T)
    last_valid = np.maximum.accumulate(np.where(fv, rows, -1))     # last valid row <= k
    prev = np.full(T, -1)
    if T > 1:
        prev[1:] = last_valid[:-1]                                # last valid row < k
    valid = fv & (prev >= 0)
    dx = np.full(F.shape, np.nan)
    idx = np.flatnonzero(valid)
    dx[idx] = F[idx] - F[prev[idx]]
    return dx, valid


def rolling_ols_residuals(y, Z, s, window=REG_WINDOW, min_valid=MIN_VALID):
    """Out-of-sample residual e[k] = y[k] - Z[k] @ beta_k, beta_k = OLS on the sample rows in k-window..k-1.

    y (T,), Z (T, p) design with intercept, s (T,) bool sample mask.  NaN where s is false or the window has
    fewer than min_valid sample rows."""
    T, p = Z.shape
    Zs = np.where(s[:, None], Z, 0.0)
    ys = np.where(s, y, 0.0)
    cxx = np.zeros((T + 1, p, p))
    cxx[1:] = np.cumsum(Zs[:, :, None] * Zs[:, None, :], axis=0)
    cxy = np.zeros((T + 1, p))
    cxy[1:] = np.cumsum(Zs * ys[:, None], axis=0)
    cn = np.zeros(T + 1, dtype=np.int64)
    cn[1:] = np.cumsum(s)
    k = np.arange(T)
    lo = np.maximum(k - window, 0)
    n = cn[k] - cn[lo]
    ok = s & (n >= min_valid)
    e = np.full(T, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size:
        xtx = cxx[idx] - cxx[lo[idx]]
        xty = cxy[idx] - cxy[lo[idx]]
        beta = np.einsum("tij,tj->ti", np.linalg.pinv(xtx), xty)
        e[idx] = y[idx] - np.sum(Z[idx] * beta, axis=1)
    return e


def gap_zscore(e, gap_days, window=REG_WINDOW, min_valid=MIN_VALID):
    """z[t] = (sum of the last gap_days residuals on rows <= t) / (sqrt(gap_days) * std of residuals on rows
    t-window+1..t), NaN with fewer than min_valid residuals in that window (sample std, ddof 1)."""
    T = e.shape[0]
    r = np.isfinite(e)
    ev = np.where(r, e, 0.0)
    c1 = np.zeros(T + 1)
    c1[1:] = np.cumsum(ev)
    c2 = np.zeros(T + 1)
    c2[1:] = np.cumsum(ev * ev)
    cn = np.zeros(T + 1, dtype=np.int64)
    cn[1:] = np.cumsum(r)
    hi = np.arange(T) + 1
    lo = np.maximum(hi - window, 0)
    n = cn[hi] - cn[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    nn = np.maximum(n, 2).astype(np.float64)
    var = (s2 - s1 * s1 / nn) / (nn - 1.0)
    ce = np.zeros(int(r.sum()) + 1)
    ce[1:] = np.cumsum(e[r])
    cnt = cn[hi]                                    # residuals on rows <= t
    G = ce[cnt] - ce[np.maximum(cnt - gap_days, 0)]
    good = (n >= min_valid) & (cnt >= gap_days) & (var > 0)
    z = np.full(T, np.nan)
    z[good] = G[good] / (np.sqrt(gap_days) * np.sqrt(var[good]))
    return z


def spaced_events(z, eligible, z_entry, hold=HOLD_DAYS):
    """Discount (z <= -z_entry) and premium (z >= z_entry) events on eligible rows; an event needs no event of
    either sign in the previous `hold` rows."""
    T = z.shape[0]
    disc = np.zeros(T, dtype=bool)
    prem = np.zeros(T, dtype=bool)
    with np.errstate(invalid="ignore"):
        cand = eligible & np.isfinite(z) & (np.abs(z) >= z_entry)
    last = -(10 ** 9)
    for t in np.flatnonzero(cand):
        if t - last > hold:
            if z[t] <= -z_entry:
                disc[t] = True
            else:
                prem[t] = True
            last = t
    return disc, prem


def credit_events(data, params):
    """{etf: (discount (T,) bool, premium (T,) bool)} after the close of each row."""
    gap_days = int(params["gap_days"])
    z_entry = float(params["z_entry"])
    T = len(data.dates)
    F = load_fred(data)
    dx, fred_ok = fred_changes(F)
    key_col = {k: i for i, k in enumerate(FRED_KEYS)}
    out = {}
    for name in CREDIT:
        j = data.col(name)
        listed = np.asarray(data.listed[:, j], dtype=bool)
        close = np.asarray(data.close[:, j], dtype=np.float64)
        bar = listed & np.isfinite(close)
        ret = np.asarray(data.ret_co[:, j], dtype=np.float64) + np.asarray(data.ret_oc[:, j], dtype=np.float64)
        # session d = row k-1 is known on row k: shift the ETF side forward by one row
        y = np.full(T, np.nan)
        etf_ok = np.zeros(T, dtype=bool)
        if T > 2:
            y[1:] = ret[:-1]
            etf_ok[2:] = bar[1:-1] & bar[:-2]
        Z = np.column_stack([
            np.ones(T),
            dx[:, key_col[OAS_OF[name]]],
            dx[:, key_col["DGS2"]],
            dx[:, key_col["DGS10"]],
        ])
        s = fred_ok & etf_ok & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        e = rolling_ols_residuals(y, Z, s)
        z = gap_zscore(e, gap_days)
        history = np.cumsum(bar)
        eligible = bar & np.asarray(data.tradable[:, j], dtype=bool) & (history >= HISTORY_MIN)
        if data.universe is not None:
            eligible = eligible & np.asarray(data.universe[:, j], dtype=bool)
        out[name] = spaced_events(z, eligible, z_entry)
    return out


# ------------------------------------------------------------------------------------------------- book
def month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool and ps.shape[0] == T:
        return ps
    mask = np.zeros(T, dtype=bool)
    mask[ps.astype(np.int64)] = True
    return mask


def slot_book(disc, hold=HOLD_DAYS):
    """disc (T, 3) discount events -> (active (T, 3) bool, credit weights (T, 3), IEF weight (T,)).

    A discount event on row t is held at the opens of t+1..t+hold, i.e. active on decision rows t..t+hold-1."""
    T, m = disc.shape
    active = np.zeros((T, m), dtype=bool)
    for c in range(m):
        for t in np.flatnonzero(disc[:, c]):
            active[t:t + hold, c] = True
    n = active.sum(axis=1)
    per = SLEEVE / np.maximum(2, n)
    credit = active * per[:, None]
    ief = CORE_IEF - credit.sum(axis=1)
    return active, credit, ief


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    ev = credit_events(data, params)
    disc = np.column_stack([ev[name][0] for name in CREDIT])
    active, credit, ief = slot_book(disc)

    w = np.zeros((T, N))
    w[:, data.col("SPY")] = CORE_SPY
    w[:, data.col("IEF")] = ief
    for c, name in enumerate(CREDIT):
        w[:, data.col(name)] = credit[:, c]
    data.col("TLT")      # peers of the event study: never traded, must exist
    data.col("SHY")

    decide = month_start_mask(data.dates).copy()
    if T:
        decide[0] = True
    if T > 1:
        decide[1:] |= np.any(active[1:] != active[:-1], axis=1)
    w[~decide] = np.nan
    return w
