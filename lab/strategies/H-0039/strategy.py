"""H-0039 Perp leverage-crowding unwind.

Top-20 Binance perps (point-in-time universe), dollar-neutral. The universe is data.universe when the gate
supplies it; otherwise (binance_perp_1d delivers none) it is built point-in-time from the data: top 20 by
quote volume (dollar_volume) summed over rows t-29..t, among names listed with a price at t.
After the close of day t:
  (1) eligible = in universe at t, funding non-missing on rows t-fd+1..t, close non-missing on rows t-6..t
  (2) F = sum of daily funding over rows t-fd+1..t
  (3) R = compounded total return over rows t-6..t (close t-7 -> close t, from ret_co and ret_oc);
      split eligible names at the median of R (odd extra name goes to the high half)
  (4) within each half: k = max(1, floor(0.25 * half + 0.5)); k highest F -> short, k lowest F -> long;
      ties broken by liquidity (trailing 30-day mean quote volume, more liquid first)
  (5) longs +0.5 / N_long, shorts -0.5 / N_short
  (6) rebalance on rows whose index since the first row with >= 8 eligible names is a multiple of
      hold_days; NaN (keep) in between, except a held name that leaves the universe or loses its price is
      set to 0 on that row (no renormalisation). Fewer than 8 eligible on a rebalance row -> flat.
"""
import numpy as np

PARAMS = {"funding_days": 3, "hold_days": 5}

RET_DAYS = 7          # past-week return window (card: fixed)
MIN_ELIGIBLE = 8      # card: fixed
LEG_FRACTION = 0.25   # card: fixed structural choice
LIQ_DAYS = 30         # tie-break liquidity window (same window as the universe's quote-volume ranking)
UNIVERSE_N = 20       # card: universe crypto_top_n, n = 20 (used only if the gate supplies no universe mask)


def _lag(x, k, fill):
    """x shifted down by k rows (row t holds x[t-k]); first k rows = fill."""
    if k == 0:
        return x.copy()
    out = np.full_like(x, fill)
    out[k:] = x[:-k]
    return out


def top_n_universe(close, dv, listed, n=UNIVERSE_N, days=LIQ_DAYS):
    """Point-in-time crypto_top_n: at row t, the n instruments with the largest quote volume summed over
    rows t-days+1..t (missing days count 0), among those listed with a price at t. Ties: column order."""
    T, N = close.shape
    qv = np.zeros((T, N), dtype=np.float64)
    for k in range(days):
        dl = _lag(dv, k, np.nan)
        qv += np.where(np.isnan(dl), 0.0, dl)
    cand = listed & ~np.isnan(close) & (qv > 0.0)
    score = np.where(cand, qv, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty((T, N), dtype=np.int64)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return cand & (rank < n)


def _universe(data, close, dv):
    """The card's universe: the gate's PIT mask when supplied, else the same rule built from the data."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    return top_n_universe(close, dv, listed)


def _panel(data, funding_days):
    """Point-in-time panels: eligibility, funding score F, 7-day return R, liquidity."""
    fund = np.asarray(data.extras["funding"], dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    uni = _universe(data, close, dv)
    T, N = close.shape

    # (1)+(2) funding present on each of the last funding_days rows, and its sum (direct sum, so that
    # equal funding histories give bit-identical F and ties stay ties)
    fok = np.ones((T, N), dtype=bool)
    F = np.zeros((T, N), dtype=np.float64)
    for k in range(funding_days):
        fl = _lag(fund, k, np.nan)
        fok &= ~np.isnan(fl)
        F += np.where(np.isnan(fl), 0.0, fl)

    # price on each of the last 7 rows
    pok = np.ones((T, N), dtype=bool)
    for k in range(RET_DAYS):
        pok &= ~np.isnan(_lag(close, k, np.nan))

    # (3) compounded total return over rows t-6..t = close t-7 -> close t
    gross = (1.0 + np.nan_to_num(ret_co)) * (1.0 + np.nan_to_num(ret_oc))
    logg = np.log(gross)
    R = np.zeros((T, N), dtype=np.float64)
    for k in range(RET_DAYS):
        R += _lag(logg, k, 0.0)
    R = np.expm1(R)

    # tie-break liquidity: trailing 30-day mean quote volume (NaN days skipped)
    s = np.zeros((T, N), dtype=np.float64)
    c = np.zeros((T, N), dtype=np.float64)
    for k in range(LIQ_DAYS):
        dl = _lag(dv, k, np.nan)
        ok = ~np.isnan(dl)
        s += np.where(ok, dl, 0.0)
        c += ok
    liq = np.where(c > 0, s / np.maximum(c, 1.0), -np.inf)

    elig = uni & fok & pok
    return elig, F, R, liq, uni, close


def _legs(idx, R, F, liq):
    """Steps (3)-(4) on one row: idx = eligible column indices; returns (longs, shorts)."""
    order = np.lexsort((idx, R[idx]))          # by R ascending, column index as last resort
    srt = idx[order]
    n_low = len(srt) // 2                      # odd extra name -> high half
    longs, shorts = [], []
    for half in (srt[:n_low], srt[n_low:]):
        m = len(half)
        if m == 0:
            continue
        k = max(1, int(np.floor(LEG_FRACTION * m + 0.5)))
        f = F[half]
        q = liq[half]
        # lexsort: last key is primary. Short: highest F first, then most liquid first.
        sh = half[np.lexsort((-q, -f))][:k]
        # Long: lowest F first, then most liquid first, among names not already short (with ties in F
        # across the half the two orderings could pick the same name; 2k <= half size, so k remain).
        lo_order = half[np.lexsort((-q, f))]
        lo = lo_order[~np.isin(lo_order, sh)][:k]
        shorts.extend(sh.tolist())
        longs.extend(lo.tolist())
    return np.array(longs, dtype=int), np.array(shorts, dtype=int)


def target_weights(data, params):
    fd = int(params["funding_days"])
    hold = int(params["hold_days"])
    elig, F, R, liq, uni, close = _panel(data, fd)
    T, N = elig.shape
    out = np.full((T, N), np.nan)

    n_elig = elig.sum(axis=1)
    first = np.flatnonzero(n_elig >= MIN_ELIGIBLE)
    if len(first) == 0:
        return out
    anchor = int(first[0])

    cur = np.zeros(N)
    for t in range(anchor, T):
        if (t - anchor) % hold == 0:
            w = np.zeros(N)
            if n_elig[t] >= MIN_ELIGIBLE:
                idx = np.flatnonzero(elig[t])
                longs, shorts = _legs(idx, R[t], F[t], liq[t])
                w[longs] = 0.5 / len(longs)
                w[shorts] = -0.5 / len(shorts)
            out[t] = w
            cur = w
        else:
            bad = (cur != 0.0) & (~uni[t] | np.isnan(close[t]))
            if bad.any():
                out[t, bad] = 0.0
                cur = cur.copy()
                cur[bad] = 0.0
    return out
