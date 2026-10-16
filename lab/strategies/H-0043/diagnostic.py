"""H-0043 mechanism diagnostic: gap events of HYG / JNK / LQD, both signs.

Same event code as strategy.py (duplicated because the diagnostic may only import numpy and lab.framework):
side +1 for a discount (ETF fell more than its bonds imply), -1 for a premium. SPY, IEF, TLT, SHY never marked.
"""
import numpy as np

PARAMS = {"gap_days": 5, "z_entry": 2.0}

HOLD_DAYS = 5
REG_WINDOW = 252
MIN_VALID = 200
HISTORY_MIN = 252

CREDIT = ("HYG", "JNK", "LQD")
OAS_OF = {"HYG": "BAMLH0A0HYM2", "JNK": "BAMLH0A0HYM2", "LQD": "BAMLC0A0CM"}
FRED_KEYS = ("BAMLH0A0HYM2", "BAMLC0A0CM", "DGS2", "DGS10")
SERIES_NAME = {
    "BAMLH0A0HYM2": "fred_macro.BAMLH0A0HYM2",
    "BAMLC0A0CM": "fred_macro.BAMLC0A0CM",
    "DGS2": "fred_macro.DGS2",
    "DGS10": "fred_macro.DGS10",
}


def load_fred(data):
    T = len(data.dates)
    cols = []
    for key in FRED_KEYS:
        x = np.asarray(data.series[SERIES_NAME[key]], dtype=np.float64).reshape(-1)
        if x.shape[0] != T:
            raise ValueError("FRED series has the wrong length")
        cols.append(x)
    return np.stack(cols, axis=1)


def fred_changes(F):
    T = F.shape[0]
    finite = np.all(np.isfinite(F), axis=1)
    same = np.zeros(T, dtype=bool)
    if T > 1:
        same[1:] = np.all(F[1:] == F[:-1], axis=1)
    fv = finite & ~same
    rows = np.arange(T)
    last_valid = np.maximum.accumulate(np.where(fv, rows, -1))
    prev = np.full(T, -1)
    if T > 1:
        prev[1:] = last_valid[:-1]
    valid = fv & (prev >= 0)
    dx = np.full(F.shape, np.nan)
    idx = np.flatnonzero(valid)
    dx[idx] = F[idx] - F[prev[idx]]
    return dx, valid


def rolling_ols_residuals(y, Z, s, window=REG_WINDOW, min_valid=MIN_VALID):
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
    cnt = cn[hi]
    G = ce[cnt] - ce[np.maximum(cnt - gap_days, 0)]
    good = (n >= min_valid) & (cnt >= gap_days) & (var > 0)
    z = np.full(T, np.nan)
    z[good] = G[good] / (np.sqrt(gap_days) * np.sqrt(var[good]))
    return z


def spaced_events(z, eligible, z_entry, hold=HOLD_DAYS):
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


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    ev = credit_events(data, params)
    for name in CREDIT:
        j = data.col(name)
        disc, prem = ev[name]
        mask[:, j] = disc | prem
        side[:, j] = np.where(disc, 1.0, np.where(prem, -1.0, 0.0))
    return mask, side
