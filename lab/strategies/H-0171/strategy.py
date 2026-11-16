"""H-0171: VIX jump -> funding sort within BTC-beta halves on the top-30 Binance perps, hold 5 days.

Rules of the card and where they live:
  (1) shock date      -> _shock_rows: new VIX observation on row t with log(VIX_obs / VIX_prev_obs) >= vix_shock
  (2) crowding        -> _trailing_mean_all_valid(funding, funding_days): needs funding_days valid values
  (3) beta halves     -> _rolling_beta (60 rows, >= 45 valid pairs, BTCUSDT.P = 1) and _select (median split)
  (4) book            -> _select: short 3 highest / long 3 lowest crowd per half, ties by higher 30d quote volume,
                         weight 1/12, no book if a half has < 6 eligible coins
  (5) holding         -> target_weights: hold 5 rows (entry open t+1, exit open t+6), new shock replaces the book
                         and restarts the clock, a coin leaving the universe is closed and not replaced.
"""
import numpy as np

PARAMS = {"funding_days": 7, "vix_shock": 0.1}

BTC = "BTCUSDT.P"
VIX_KEY = "fred_macro.VIXCLS"
TOP_N = 30            # card universe: crypto_top_n, n = 30
QV_DAYS = 30          # 30-day quote volume (tie-break, fallback universe)
BETA_WINDOW = 60
BETA_MIN_PAIRS = 45
N_PER_LEG_HALF = 3
MIN_HALF = 6
HOLD_DAYS = 5


def _shift(x, k):
    """x shifted down by k rows (row t holds x[t-k]); NaN where unavailable."""
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _trailing_mean_all_valid(x, n):
    """Mean of x over rows t-n+1..t; NaN unless all n values are finite (and t >= n-1)."""
    x = np.asarray(x, dtype=float)
    acc = np.zeros(x.shape)
    ok = np.ones(x.shape, dtype=bool)
    for k in range(int(n)):
        s = _shift(x, k)
        fin = np.isfinite(s)
        ok &= fin
        acc += np.where(fin, s, 0.0)
    return np.where(ok, acc / float(n), np.nan)


def _trailing_nansum(x, n):
    """Sum of the finite values of x over rows t-n+1..t (0 if none)."""
    x = np.asarray(x, dtype=float)
    acc = np.zeros(x.shape)
    for k in range(int(n)):
        s = _shift(x, k)
        acc += np.where(np.isfinite(s), s, 0.0)
    return acc


def _ffill_1d(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    idx = np.where(np.isfinite(v), np.arange(v.shape[0]), -1)
    idx = np.maximum.accumulate(idx) if v.shape[0] else idx
    return np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)


def _shock_rows(vix, vix_shock):
    """Bool (T,): row t carries a new VIX observation whose log change vs the previous observation >= vix_shock.

    The series is aligned by its clock (row t = last value known after the close of t). A new observation is a
    row whose (forward-filled) value differs from the row before; the previous observation is then the value of
    row t-1. Equal consecutive observations have a log change of 0 and could never fire anyway.
    """
    v = _ffill_1d(vix)
    prev = np.full(v.shape, np.nan)
    prev[1:] = v[:-1]
    ok = np.isfinite(v) & np.isfinite(prev) & (v > 0) & (prev > 0)
    lc = np.full(v.shape, np.nan)
    lc[ok] = np.log(v[ok] / prev[ok])
    new = ok & (v != prev)
    return new & (lc >= float(vix_shock))


def _cc_returns(data):
    """Close-to-close total return per row; NaN unless the instrument has a price at t and t-1."""
    r = (1.0 + np.asarray(data.ret_co, float)) * (1.0 + np.asarray(data.ret_oc, float)) - 1.0
    close = np.asarray(data.close, float)
    has = np.isfinite(close) & np.asarray(data.listed, bool)
    prev = np.zeros(has.shape, dtype=bool)
    prev[1:] = has[:-1]
    return np.where(has & prev & np.isfinite(r), r, np.nan)


def _rolling_beta(r, j_btc, window=BETA_WINDOW, min_pairs=BETA_MIN_PAIRS):
    """OLS slope of each column's return on column j_btc's over the last `window` rows, >= min_pairs pairs."""
    x = np.broadcast_to(r[:, j_btc : j_btc + 1], r.shape)
    valid = np.isfinite(r) & np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, r, 0.0)
    T = r.shape[0]
    pad = np.zeros((1, r.shape[1]))

    def wsum(a):
        cs = np.concatenate([pad, np.cumsum(a, axis=0)])
        hi = np.arange(1, T + 1)
        lo = np.maximum(hi - window, 0)
        return cs[hi] - cs[lo]

    n = wsum(valid.astype(float))
    sx, sy, sxy, sxx = wsum(xv), wsum(yv), wsum(xv * yv), wsum(xv * xv)
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (n * sxy - sx * sy) / den
    good = (n >= min_pairs) & (den > 1e-18)
    beta = np.where(good, beta, np.nan)
    beta[:, j_btc] = np.where(n[:, j_btc] >= min_pairs, 1.0, np.nan)   # BTC itself: beta 1 by definition
    return beta


def _universe(data, qv30):
    """Point-in-time top-30 membership; data.universe when the gate provides it, else top 30 by 30d quote volume."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    listed = np.isfinite(np.asarray(data.close, float)) & np.asarray(data.listed, bool)
    key = np.where(listed & (qv30 > 0), -qv30, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    return (rank < TOP_N) & np.isfinite(key)


def _select(crowd_t, beta_t, qv_t, elig_t):
    """Book of one shock date: (shorts, longs) column arrays, or None if a beta half has < MIN_HALF coins."""
    cols = np.flatnonzero(elig_t)
    if cols.size < 2 * MIN_HALF:
        return None
    order = cols[np.lexsort((cols, beta_t[cols]))]        # ascending beta, ties by column
    h = cols.size // 2                                    # odd count: the median coin is in neither half
    halves = (order[:h], order[cols.size - h :])
    if h < MIN_HALF:
        return None
    shorts, longs = [], []
    for half in halves:
        q = np.where(np.isfinite(qv_t[half]), qv_t[half], -np.inf)
        c = crowd_t[half]
        hi = half[np.lexsort((half, -q, -c))]             # highest crowd first, ties: higher quote volume
        lo = half[np.lexsort((half, -q, c))]              # lowest crowd first, ties: higher quote volume
        s = hi[:N_PER_LEG_HALF]
        lo = lo[~np.isin(lo, s)]                          # legs disjoint when many crowd values tie
        shorts.extend(s.tolist())
        longs.extend(lo[:N_PER_LEG_HALF].tolist())
    return np.array(shorts, dtype=int), np.array(longs, dtype=int)


def books(data, params):
    """Point-in-time list of (row, shorts, longs) for every shock date on which a book can be opened,
    plus the membership matrix."""
    fd = int(params["funding_days"])
    vix = data.series[VIX_KEY]
    funding = np.asarray(data.extras["funding"], dtype=float)
    j_btc = data.col(BTC)
    qv30 = _trailing_nansum(np.asarray(data.dollar_volume, float), QV_DAYS)
    member = _universe(data, qv30)
    crowd = _trailing_mean_all_valid(funding, fd)
    beta = _rolling_beta(_cc_returns(data), j_btc)
    shock = _shock_rows(vix, params["vix_shock"])
    out = []
    for t in np.flatnonzero(shock):
        elig = member[t] & np.isfinite(crowd[t]) & np.isfinite(beta[t])
        sel = _select(crowd[t], beta[t], qv30[t], elig)
        if sel is not None:
            out.append((int(t), sel[0], sel[1]))
    return out, member


def target_weights(data, params):
    T, N = len(data.dates), len(data.instruments)
    book_list, member = books(data, params)
    by_row = {t: (s, l) for t, s, l in book_list}
    w = np.zeros((T, N))
    held = []
    start = -1
    wgt = 1.0 / (4 * N_PER_LEG_HALF)                      # 12 positions, 1/12 each
    for t in range(T):
        if t in by_row:
            s, l = by_row[t]
            w[t, s] = -wgt
            w[t, l] = wgt
            held = s.tolist() + l.tolist()
            start = t
        elif held and t - start < HOLD_DAYS:
            w[t, :] = np.nan                               # keep the book (no rebalancing)
            keep = []
            for j in held:
                if member[t, j]:
                    keep.append(j)
                else:
                    w[t, j] = 0.0                          # left the top 30: close, do not replace
            held = keep
        else:
            held = []                                      # t - start == 5: close at the open of start + 6
    return w
