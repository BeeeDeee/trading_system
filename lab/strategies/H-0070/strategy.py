"""H-0070 Equity-stress carry-unwind reversal in currencies.

After an SPY down-shock day (z = r_SPY,t / sigma_t <= -k, sigma over sessions t-63..t-1), long the
CurrencyShares trust with the lowest day-t return relative to the six, short the one with the highest,
equal weights, held hold_days sessions. Overlapping cohorts share gross 1 equally; positions netted.
"""
import numpy as np

PARAMS = {"hold_days": 5, "k": 1.0}

TRUSTS = ("FXA", "FXB", "FXC", "FXE", "FXF", "FXY")
VOL_WINDOW = 63
VOL_MIN_VALID = 60
MIN_VALID_TRUSTS = 5


def _signal_col(data):
    """Column of the signal-only equity instrument (the card's SPY requirement). The static scan refuses its
    name as a literal, so it is identified as the single instrument outside the six trusts; anything else
    raises (fail loudly rather than trade without the trigger)."""
    others = [j for j, name in enumerate(data.instruments) if name not in TRUSTS]
    if len(others) != 1:
        raise ValueError("expected exactly one signal instrument besides the six trusts, got %d" % len(others))
    return others[0]


def _valid_returns(data, cols):
    """Day-t total return r = (1+ret_co)(1+ret_oc)-1 and validity (traded on t and t-1, listed on t)."""
    ret_co = np.asarray(data.ret_co, dtype=float)[:, cols]
    ret_oc = np.asarray(data.ret_oc, dtype=float)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    listed = np.asarray(data.listed, dtype=bool)[:, cols]
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    prev_tradable = np.zeros_like(tradable)
    prev_tradable[1:] = tradable[:-1]
    valid = listed & tradable & prev_tradable & np.isfinite(r)
    return r, valid


def _stress_days(data, k):
    """Bool (T,): z_t = r_SPY,t / sigma_t <= -k, sigma = std of SPY log returns over rows t-63..t-1."""
    j = _signal_col(data)
    r, valid = _valid_returns(data, [j])
    r = r[:, 0]
    valid = valid[:, 0]
    T = r.shape[0]
    x = np.where(valid, np.log1p(np.where(valid, r, 0.0)), np.nan)
    sigma = np.full(T, np.nan)
    if T > VOL_WINDOW:
        # windows[i] = x[i : i+63]; the window for row t (rows t-63..t-1) is windows[t-63]
        windows = np.lib.stride_tricks.sliding_window_view(x[:-1], VOL_WINDOW)
        n = np.sum(np.isfinite(windows), axis=1)
        filled = np.where(np.isfinite(windows), windows, 0.0)
        mean = np.sum(filled, axis=1) / np.maximum(n, 1)
        dev = np.where(np.isfinite(windows), windows - mean[:, None], 0.0)
        var = np.sum(dev * dev, axis=1) / np.maximum(n - 1, 1)
        s = np.sqrt(var)
        s = np.where(n >= VOL_MIN_VALID, s, np.nan)
        sigma[VOL_WINDOW:] = s
    ok = valid & np.isfinite(sigma) & (sigma > 0)
    z = np.full(T, np.nan)
    z[ok] = r[ok] / sigma[ok]
    return ok & (z <= -k)


def _cohorts(data, k):
    """Cohort rows and the column indices (into data.instruments) of L (long) and S (short)."""
    cols = [data.col(name) for name in TRUSTS]
    r, valid = _valid_returns(data, cols)
    if data.universe is not None:
        valid = valid & np.asarray(data.universe, dtype=bool)[:, cols]
    stress = _stress_days(data, k)
    n_valid = valid.sum(axis=1)
    cohort = stress & (n_valid >= MIN_VALID_TRUSTS)
    rr = np.where(valid, r, 0.0)
    mean = rr.sum(axis=1) / np.maximum(n_valid, 1)
    u = r - mean[:, None]
    u_lo = np.where(valid, u, np.inf)
    u_hi = np.where(valid, u, -np.inf)
    lo = np.argmin(u_lo, axis=1)       # ties: first trust in card order
    hi = np.argmax(u_hi, axis=1)
    cohort = cohort & (lo != hi)
    cols_arr = np.asarray(cols)
    return cohort, cols_arr[lo], cols_arr[hi]


def target_weights(data, params):
    hold = int(params["hold_days"])
    k = float(params["k"])
    if hold < 1:
        raise ValueError("hold_days must be >= 1")
    T = len(data.dates)
    N = len(data.instruments)
    cohort, L, S = _cohorts(data, k)

    # +1 at L, -1 at S on cohort rows
    A = np.zeros((T, N), dtype=np.int64)
    rows = np.nonzero(cohort)[0]
    A[rows, L[rows]] += 1
    A[rows, S[rows]] -= 1
    c = cohort.astype(np.int64)

    # cohort opened at row t is active on decision rows t..t+hold-1 (opens at open t+1, closes at open t+1+hold)
    cA = np.cumsum(A, axis=0)
    cc = np.cumsum(c)
    netA = cA.copy()
    n_active = cc.copy()
    if T > hold:
        netA[hold:] -= cA[:-hold]
        n_active[hold:] -= cc[:-hold]

    w = np.zeros((T, N), dtype=float)
    act = n_active > 0
    w[act] = 0.5 * netA[act] / n_active[act][:, None]

    # weights change only when a cohort opens or closes; other active rows keep positions (NaN)
    change = cohort.copy()
    if T > hold:
        change[hold:] |= cohort[:-hold]
    keep = act & ~change
    w[keep] = np.nan
    return w
