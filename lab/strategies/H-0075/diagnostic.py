"""H-0075 mechanism-test events: the same event and the same side logic as strategy.py (self-contained copy).

Event row d = the first session at whose close dy(t) of the shock day t is known (next_morning: d = t + 1).
Marked: the two rankable sector ETFs with the lowest beta_i * dy(t) (side +1, the long leg) and the two with
the highest (side -1, the short leg); only rankable ETFs (in the universe, tradable on d) are marked.
"""
import numpy as np

UNIVERSE = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR")
YIELD_SERIES = "fred_macro.DGS10"

VOL_WINDOW = 63
EQ_STRESS_SIGMA = 2.0
N_PER_SIDE = 2
MIN_RANKABLE = 6
BETA_COVERAGE = 0.8


def _prev_window_std(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if T <= n:
        return out
    fin = np.isfinite(x)
    xv = np.where(fin, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([pad, np.cumsum(xv, axis=0)])
    c2 = np.concatenate([pad, np.cumsum(xv * xv, axis=0)])
    cn = np.concatenate([pad, np.cumsum(fin.astype(np.float64), axis=0)])
    s1 = c1[n:T] - c1[0:T - n]
    s2 = c2[n:T] - c2[0:T - n]
    k = cn[n:T] - cn[0:T - n]
    var = np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0)
    sd = np.sqrt(var)
    sd = np.where(k > n - 0.5, sd, np.nan)
    out[n:T] = sd
    return out


def _yield_changes(yld):
    yld = np.asarray(yld, dtype=np.float64)
    T = yld.shape[0]
    dyk = np.full(T, np.nan)
    sig_y = np.full(T, np.nan)
    idx = np.flatnonzero(np.isfinite(yld))
    if idx.size >= 2:
        dvals = np.diff(yld[idx])
        dyk[idx[1:]] = dvals
        sig_y[idx[1:]] = _prev_window_std(dvals, VOL_WINDOW)
    return dyk, sig_y


def _session_returns(ret_co, ret_oc, close, listed, tradable):
    ret_co = np.asarray(ret_co, dtype=np.float64)
    ret_oc = np.asarray(ret_oc, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    valid = (np.asarray(listed, dtype=bool) & np.asarray(tradable, dtype=bool)
             & np.isfinite(close) & np.isfinite(prev_close) & np.isfinite(r))
    return np.where(valid, r, np.nan), valid


def _marks(ret_co, ret_oc, close, listed, tradable, member, yld, beta_days, k_sigma):
    beta_days = int(beta_days)
    k_sigma = float(k_sigma)
    tradable = np.asarray(tradable, dtype=bool)
    listed = np.asarray(listed, dtype=bool)
    member = np.asarray(member, dtype=bool)
    T, K = tradable.shape

    r, valid_r = _session_returns(ret_co, ret_oc, close, listed, tradable)
    cnt = valid_r.sum(axis=1)
    ew = np.full(T, np.nan)
    has = cnt > 0
    ew[has] = np.where(valid_r[has], r[has], 0.0).sum(axis=1) / cnt[has]
    rel = r - ew[:, None]
    sig_eq = _prev_window_std(ew, VOL_WINDOW)

    dyk, sig_y = _yield_changes(yld)

    x = np.full(T, np.nan)
    x[:-1] = dyk[1:]
    m = np.isfinite(x)[:, None] & valid_r
    xm = np.where(m, x[:, None], 0.0)
    ym = np.where(m, rel, 0.0)
    pad = np.zeros((1, K))
    Cn = np.concatenate([pad, np.cumsum(m.astype(np.float64), axis=0)])
    Cx = np.concatenate([pad, np.cumsum(xm, axis=0)])
    Cy = np.concatenate([pad, np.cumsum(ym, axis=0)])
    Cxy = np.concatenate([pad, np.cumsum(xm * ym, axis=0)])
    Cxx = np.concatenate([pad, np.cumsum(xm * xm, axis=0)])
    d = np.arange(T)
    hi = np.clip(d - 1, 0, T)
    lo = np.clip(d - 1 - beta_days, 0, T)
    n = Cn[hi] - Cn[lo]
    sx = Cx[hi] - Cx[lo]
    sy = Cy[hi] - Cy[lo]
    sxy = Cxy[hi] - Cxy[lo]
    sxx = Cxx[hi] - Cxx[lo]
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)

    started = np.maximum.accumulate(valid_r.astype(np.int64), axis=0)
    n_since = np.cumsum(started, axis=0)
    hist = np.zeros((T, K), dtype=bool)
    if T > 2:
        hist[2:] = n_since[:-2] >= beta_days

    valid_prev = np.zeros((T, K), dtype=bool)
    valid_prev[1:] = valid_r[:-1]
    rankable = (hist & (n >= BETA_COVERAGE * beta_days - 1e-9) & np.isfinite(beta)
                & valid_prev & tradable & listed & member)
    n_rank = rankable.sum(axis=1)

    ew_t = np.full(T, np.nan)
    ew_t[1:] = ew[:-1]
    sq_t = np.full(T, np.nan)
    sq_t[1:] = sig_eq[:-1]
    with np.errstate(invalid="ignore"):
        shock = np.isfinite(dyk) & np.isfinite(sig_y) & (sig_y > 0) & (np.abs(dyk) >= k_sigma * sig_y)
        calm = np.isfinite(ew_t) & np.isfinite(sq_t) & (np.abs(ew_t) < EQ_STRESS_SIGMA * sq_t)
    event = shock & calm & (n_rank >= MIN_RANKABLE)
    event[0] = False

    mask = np.zeros((T, K), dtype=bool)
    side = np.zeros((T, K))
    for row in np.flatnonzero(event):
        cols = np.flatnonzero(rankable[row])
        p = beta[row, cols] * dyk[row]
        order = cols[np.argsort(p, kind="stable")]
        mask[row, order[:N_PER_SIDE]] = True
        side[row, order[:N_PER_SIDE]] = 1.0
        mask[row, order[-N_PER_SIDE:]] = True
        side[row, order[-N_PER_SIDE:]] = -1.0
    return mask, side


def events(data, params):
    cols = [data.col(name) for name in UNIVERSE]
    yld = np.asarray(data.series[YIELD_SERIES], dtype=np.float64)
    T = len(data.dates)
    if yld.shape[0] != T:
        raise ValueError("DGS10 series not aligned to the calendar")
    if data.universe is None:
        member = np.ones((T, len(cols)), dtype=bool)
    else:
        member = np.asarray(data.universe, dtype=bool)[:, cols]
    m, s = _marks(np.asarray(data.ret_co)[:, cols], np.asarray(data.ret_oc)[:, cols],
                  np.asarray(data.close)[:, cols], np.asarray(data.listed)[:, cols],
                  np.asarray(data.tradable)[:, cols], member, yld,
                  params["beta_days"], params["k_sigma"])
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.ones((T, N))
    mask[:, cols] = m
    side[:, cols] = np.where(m, s, 1.0)
    return mask, side
