"""H-0169 mechanism test: bystanders of a shock day of the replicated CTA trend book.

mask[t, j] = True for every bystander j of a shock day t (same pipeline as strategy.py, copied);
side[t, j] = -s[j, t-1] (against its prevailing trend).
"""
import numpy as np

SECTORS = (
    ("SPY", "IWM", "EFA", "EEM", "EWJ"),            # equity
    ("IEF", "TLT"),                                 # bonds
    ("GLD", "SLV", "USO", "DBA", "DBB"),            # commodities
    ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF"),     # currencies
)
VOL_WIN = 63
VOL_MIN = 50
Z_WIN = 252
Z_MIN = 200
MIN_ELIG = 6
MIN_SECTORS = 3


def _shift(x, k, fill):
    out = np.full(x.shape, fill, dtype=x.dtype)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _rolling_count(valid, n):
    c = np.cumsum(valid.astype(np.int64), axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def _rolling_sum(x, n):
    c = np.cumsum(x, axis=0)
    out = c.copy()
    if n < c.shape[0]:
        out[n:] = c[n:] - c[:-n]
    return out


def _rolling_nanstd(x, valid, n, min_count):
    x0 = np.where(valid, x, 0.0)
    cnt = _rolling_count(valid, n).astype(np.float64)
    s1 = _rolling_sum(x0, n)
    s2 = _rolling_sum(x0 * x0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1.0)
    var = np.maximum(var, 0.0)
    return np.where((cnt >= min_count) & (cnt >= 2), np.sqrt(var), np.nan)


def _bystanders(data, params):
    L = int(params["trend_lookback_days"])
    k = float(params["shock_k"])
    sector_cols = [[data.col(name) for name in sec] for sec in SECTORS]
    cols = [c for sec in sector_cols for c in sec]
    T = len(data.dates)

    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    close = np.asarray(data.close, dtype=np.float64)[:, cols]
    ret_co = np.asarray(data.ret_co, dtype=np.float64)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)[:, cols]

    bar = listed & tradable & np.isfinite(close)
    gr = (1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc)) - 1.0
    validret = bar & _shift(bar, 1, False) & np.isfinite(gr)
    r = np.where(validret, gr, np.nan)

    lg = np.log1p(np.where(gr > -1.0, gr, 0.0))
    lg = np.where(np.isfinite(lg), lg, 0.0)
    clog = np.cumsum(lg, axis=0)
    trend = np.full(clog.shape, np.nan)
    if L < T:
        trend[L:] = clog[L:] - clog[:-L]
    s = np.sign(trend)

    sig = _rolling_nanstd(gr, validret, VOL_WIN, 2)
    cnt_all = np.cumsum(validret.astype(np.int64), axis=0)
    cnt_vol = _rolling_count(validret, VOL_WIN)

    s_prev = _shift(s, 1, np.nan)
    sig_prev = _shift(sig, 1, np.nan)
    cnt_all_prev = _shift(cnt_all, 1, 0)
    cnt_vol_prev = _shift(cnt_vol, 1, 0)

    elig = (validret
            & (cnt_all_prev >= L + 1)
            & (cnt_vol_prev >= VOL_MIN)
            & np.isfinite(s_prev) & (s_prev != 0)
            & np.isfinite(sig_prev) & (sig_prev > 0))
    if data.universe is not None:
        elig &= np.asarray(data.universe, dtype=bool)[:, cols]

    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(elig, s_prev * r / sig_prev, 0.0)

    sec_sum = np.zeros(T)
    n_sec = np.zeros(T, dtype=np.int64)
    n_elig = np.zeros(T, dtype=np.int64)
    pos = 0
    for sec in sector_cols:
        sl = slice(pos, pos + len(sec))
        pos += len(sec)
        ne = elig[:, sl].sum(axis=1)
        tot = u[:, sl].sum(axis=1)
        has = ne > 0
        sec_sum += np.where(has, tot / np.maximum(ne, 1), 0.0)
        n_sec += has
        n_elig += ne
    ok = (n_elig >= MIN_ELIG) & (n_sec >= MIN_SECTORS)
    P = np.where(ok, sec_sum / np.maximum(n_sec, 1), np.nan)

    pv = np.isfinite(P)
    sdP = _shift(_rolling_nanstd(np.where(pv, P, 0.0), pv, Z_WIN, Z_MIN), 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(pv & np.isfinite(sdP) & (sdP > 0), P / sdP, np.nan)
    shock = np.isfinite(z) & (z <= -k)

    with np.errstate(invalid="ignore"):
        not_lost = s_prev * r >= 0
    byst = elig & shock[:, None] & not_lost
    return cols, byst, s_prev


def events(data, params):
    cols, byst, s_prev = _bystanders(data, params)
    T, N = len(data.dates), len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    mask[:, cols] = byst            # bystanders have a valid (tradable) bar on t and are universe members
    side[:, cols] = np.where(byst, -s_prev, 0.0)
    return mask, side
