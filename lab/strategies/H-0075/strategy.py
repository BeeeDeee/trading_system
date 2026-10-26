"""H-0075 Treasury-yield shock overshoot in equity sectors.

After a large daily move of the 10y Treasury yield (FRED DGS10) on a day without an equity stress move,
go long the 2 sector ETFs with the most negative rate-explained relative move beta_i * dy and short the
2 with the most positive one, 0.25 each, hold 5 sessions (new events replace and restart the clock).

Index conventions (row = XNYS session of the DataView):
- data.series["fred_macro.DGS10"] is aligned by the next_morning clock: row u holds the last DGS10 value
  known after the close of u, i.e. normally the observation of session u-1.
- dyk[u] = change of the series value at row u against the previous valid (non-NaN) row = dy(t) of the
  observation day t = u-1, first known at the close of u (= the action session d of the card).
- For the beta regression the yield change known at row s+1 is paired with the ETF returns of session s.
"""
import numpy as np

PARAMS = {"beta_days": 252, "k_sigma": 1.5}

UNIVERSE = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR")
YIELD_SERIES = "fred_macro.DGS10"

VOL_WINDOW = 63          # fixed design constant: 63-observation volatility windows
EQ_STRESS_SIGMA = 2.0    # fixed design constant: |EW(t)| must stay below 2 sigma_eq
N_PER_SIDE = 2           # fixed design constant: two ETFs per side
LEG_WEIGHT = 0.25        # gross 1, dollar-neutral
HOLD_SESSIONS = 5        # fixed design constant: flat after the close of d+5
MIN_RANKABLE = 6         # fewer rankable ETFs -> no event
BETA_COVERAGE = 0.8      # at least 80 % of beta_days valid pairs


def _prev_window_std(x, n):
    """out[j] = sample std (ddof=1) of x[j-n:j] (the n rows before j); NaN unless all n are finite."""
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
    """dyk[u]: change of the yield at row u against the previous valid row (NaN where no new value);
    sig_y[u]: std of the 63 valid changes before the one at row u (needs all 63)."""
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
    """Close-to-close total returns (NaN where the ETF has no return on the session) and validity mask."""
    ret_co = np.asarray(ret_co, dtype=np.float64)
    ret_oc = np.asarray(ret_oc, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    valid = (np.asarray(listed, dtype=bool) & np.asarray(tradable, dtype=bool)
             & np.isfinite(close) & np.isfinite(prev_close) & np.isfinite(r))
    return np.where(valid, r, np.nan), valid


def _event_weights(ret_co, ret_oc, close, listed, tradable, member, yld, beta_days, k_sigma):
    """Returns (event (T,) bool, ev_w (T, K) weights on event rows, NaN elsewhere).

    Row d of the output is the action session d of the card (decided after the close of d)."""
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
    sig_eq = _prev_window_std(ew, VOL_WINDOW)          # sig_eq[t]: std of EW over the 63 sessions before t

    dyk, sig_y = _yield_changes(yld)

    # regression pairs: x[s] = dy of session s (known at row s+1), y[s, i] = r_i(s) - EW(s)
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
    hi = np.clip(d - 1, 0, T)              # window s in [d-1-B, d-2]: sessions ending at t-1 = d-2
    lo = np.clip(d - 1 - beta_days, 0, T)
    n = Cn[hi] - Cn[lo]
    sx = Cx[hi] - Cx[lo]
    sy = Cy[hi] - Cy[lo]
    sxy = Cxy[hi] - Cxy[lo]
    sxx = Cxx[hi] - Cxx[lo]
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)

    # history: the beta window [t-B, t-1] must lie after the ETF's first return (IYR rule)
    started = np.maximum.accumulate(valid_r.astype(np.int64), axis=0)
    n_since = np.cumsum(started, axis=0)                 # sessions <= s since the first return
    hist = np.zeros((T, K), dtype=bool)
    if T > 2:
        hist[2:] = n_since[:-2] >= beta_days             # evaluated at s = d-2 = t-1

    valid_prev = np.zeros((T, K), dtype=bool)
    valid_prev[1:] = valid_r[:-1]                        # has a return on t = d-1
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

    ev_w = np.full((T, K), np.nan)
    for row in np.flatnonzero(event):
        cols = np.flatnonzero(rankable[row])
        p = beta[row, cols] * dyk[row]
        order = cols[np.argsort(p, kind="stable")]
        w = np.zeros(K)
        w[order[:N_PER_SIDE]] = LEG_WEIGHT          # lowest p: pushed down by the rate move -> long
        w[order[-N_PER_SIDE:]] = -LEG_WEIGHT        # highest p: pushed up -> short
        ev_w[row] = w
    return event, ev_w


def _hold(event, ev_w):
    """Event weights from row d to d+4, zero at row d+5 (traded at the open of d+6); new events replace."""
    T, K = ev_w.shape
    out = np.zeros((T, K))
    cur = np.zeros(K)
    age = -1
    for row in range(T):
        if event[row]:
            cur = ev_w[row].copy()
            age = 0
        elif age >= 0:
            age += 1
            if age >= HOLD_SESSIONS:
                cur = np.zeros(K)
                age = -1
        out[row] = cur
    return out


def _inputs(data):
    cols = [data.col(name) for name in UNIVERSE]
    yld = np.asarray(data.series[YIELD_SERIES], dtype=np.float64)
    if yld.shape[0] != len(data.dates):
        raise ValueError("DGS10 series not aligned to the calendar")
    T = len(data.dates)
    if data.universe is None:
        member = np.ones((T, len(cols)), dtype=bool)
    else:
        member = np.asarray(data.universe, dtype=bool)[:, cols]
    return (cols,
            np.asarray(data.ret_co)[:, cols], np.asarray(data.ret_oc)[:, cols],
            np.asarray(data.close)[:, cols], np.asarray(data.listed)[:, cols],
            np.asarray(data.tradable)[:, cols], member, yld)


def target_weights(data, params):
    cols, ret_co, ret_oc, close, listed, tradable, member, yld = _inputs(data)
    event, ev_w = _event_weights(ret_co, ret_oc, close, listed, tradable, member, yld,
                                 params["beta_days"], params["k_sigma"])
    held = _hold(event, ev_w)
    T = len(data.dates)
    w = np.zeros((T, len(data.instruments)))
    w[:, cols] = held
    return w
