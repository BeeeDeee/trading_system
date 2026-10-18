"""H-0055 Variance-risk-premium equity tilt on a 60/40 SPY/IEF book, weekly.

VRP_d = VIX_d^2 / 12 - RV_d (percent squared, monthly).  At the weekly decision session t the
signal uses v = VRP_{t-1}; with the next_morning clock, row t of the VIXCLS series holds VIX_{t-1},
so the strategy works with the row-aligned series W[t] = vix_row[t]^2 / 12 - RV[t-1] = VRP_{t-1}.
"""
import numpy as np

PARAMS = {"rank_days": 756, "tilt": 0.3}

VIX_KEY = "fred_macro.VIXCLS"
RV_WINDOW = 21          # sessions in the realized-variance window (fixed by the card)
RV_MIN_PRESENT = 15     # VRP missing below this many present SPY returns
VIX_MAX_AGE = 5         # sessions a VIX value may be carried over a FRED holiday
MIN_VALID = 252         # minimum valid VRP values for a percentile
BASE_SPY = 0.6          # benchmark equity weight


def spy_realized_variance(data):
    """RV_d: sum over the 21 sessions ending at d of (100 x SPY daily log return)^2, rescaled to
    21 / n_present when bars are missing, NaN if n_present < 15."""
    j = data.col("SPY")
    rco = np.asarray(data.ret_co[:, j], dtype=float)
    roc = np.asarray(data.ret_oc[:, j], dtype=float)
    close = np.asarray(data.close[:, j], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log1p(rco) + np.log1p(roc)
    present = np.isfinite(close) & np.isfinite(r)
    sq = np.where(present, (100.0 * np.where(present, r, 0.0)) ** 2, 0.0)
    T = sq.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    cn = np.concatenate([[0], np.cumsum(present.astype(np.int64))])
    idx = np.arange(T)
    lo = np.maximum(idx - (RV_WINDOW - 1), 0)
    s = cs[idx + 1] - cs[lo]
    n = cn[idx + 1] - cn[lo]
    rv = np.full(T, np.nan)
    ok = n >= RV_MIN_PRESENT
    rv[ok] = s[ok] * RV_WINDOW / n[ok]
    return rv


def usable_vix(data):
    """Row t: the latest VIX known after the close of t (VIX of session t-1 under next_morning),
    carried at most VIX_MAX_AGE sessions after the last fresh observation, else NaN."""
    x = np.asarray(data.series[VIX_KEY], dtype=float).reshape(-1)
    T = x.shape[0]
    idx = np.arange(T)
    valid = np.isfinite(x)
    last_valid = np.maximum.accumulate(np.where(valid, idx, -1))
    ff = np.where(last_valid >= 0, x[np.maximum(last_valid, 0)], np.nan)
    prev_ff = np.concatenate([[np.nan], ff[:-1]])
    # a fresh observation: a value on this row that is not a carried copy of the previous one
    fresh = valid & ~(prev_ff == x)
    last_fresh = np.maximum.accumulate(np.where(fresh, idx, -1))
    age = idx - last_fresh
    return np.where((last_fresh >= 0) & (age <= VIX_MAX_AGE), ff, np.nan)


def lagged_vrp(data):
    """W[t] = VRP_{t-1} = VIX_{t-1}^2 / 12 - RV_{t-1}; NaN where either input is missing."""
    vix = usable_vix(data)
    rv = spy_realized_variance(data)
    w = np.full(vix.shape[0], np.nan)
    w[1:] = vix[1:] ** 2 / 12.0 - rv[:-1]
    return w


def decision_schedule(dates):
    """Weekly decision rows and the row whose signal each one uses.

    A row is the last session of its calendar week when the next weekday (np.busday_offset,
    no exchange holidays) lies in the next Monday-based week, i.e. it is a Friday.  If a week has
    no such row (its Friday is an exchange holiday), its last row is only known to have been the
    last session at the first row of the following week; the decision computed at that last row
    is then placed on the first row of the next week (one session later)."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    di = d.astype(np.int64)
    wk = (di + 3) // 7                       # 1970-01-01 is a Thursday -> Monday-based weeks
    nxt = np.busday_offset(d, 1, roll="forward")
    flagged = ((nxt.astype(np.int64) + 3) // 7) != wk
    new_week = np.concatenate([[True], wk[1:] != wk[:-1]])
    seg = np.cumsum(new_week) - 1
    nseg = int(seg[-1]) + 1
    flag_count = np.bincount(seg, weights=flagged.astype(float), minlength=nseg)
    starts = np.nonzero(new_week)[0]
    fb_rows = starts[1:]
    fb_rows = fb_rows[flag_count[seg[fb_rows] - 1] == 0]
    dec = np.concatenate([np.nonzero(flagged)[0], fb_rows])
    src = np.concatenate([np.nonzero(flagged)[0], fb_rows - 1])
    order = np.argsort(dec, kind="stable")
    return dec[order].astype(np.int64), src[order].astype(np.int64)


def percentile_ranks(w, rows, rank_days):
    """p at each row t: (#valid W in rows t-rank_days+1..t that are <= W[t]) / #valid, W[t] included;
    NaN if W[t] is missing or fewer than MIN_VALID valid values."""
    p = np.full(len(rows), np.nan)
    for k, t in enumerate(rows):
        v = w[t]
        if not np.isfinite(v):
            continue
        win = w[max(0, t - rank_days + 1): t + 1]
        win = win[np.isfinite(win)]
        if win.size < MIN_VALID:
            continue
        p[k] = np.count_nonzero(win <= v) / win.size
    return p


def weekly_signal(data, rank_days):
    """Decision rows and the percentile p used at each (NaN = no signal, hold)."""
    w = lagged_vrp(data)
    dec, src = decision_schedule(data.dates)
    return dec, percentile_ranks(w, src, rank_days)


def target_weights(data, params):
    rank_days = int(params["rank_days"])
    tilt = float(params["tilt"])
    T = len(data.dates)
    N = len(data.instruments)
    spy = data.col("SPY")
    ief = data.col("IEF")

    dec, p = weekly_signal(data, rank_days)

    spy_w = np.full(T, np.nan)
    if T > 0:
        spy_w[0] = BASE_SPY                  # warm-up: 60/40 from the start
    ok = np.isfinite(p)
    spy_w[dec[ok]] = BASE_SPY + tilt * (2.0 * p[ok] - 1.0)

    # A missing SPY or IEF bar on the trade day defers the trade: re-issue the target on that row
    # (for the next open) until a session with both bars, unless a new decision comes first.
    trad = np.asarray(data.tradable[:, spy], dtype=bool) & np.asarray(data.tradable[:, ief], dtype=bool)
    emitted = np.isfinite(spy_w)
    for r in np.nonzero(emitted)[0]:
        u = r + 1
        while u < T and not emitted[u] and not trad[u]:
            spy_w[u] = spy_w[r]
            u += 1

    out = np.full((T, N), np.nan)
    rows = np.isfinite(spy_w)
    out[rows, :] = 0.0
    out[rows, spy] = spy_w[rows]
    out[rows, ief] = 1.0 - spy_w[rows]
    return out
