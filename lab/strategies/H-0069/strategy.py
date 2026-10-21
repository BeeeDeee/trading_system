"""H-0069 Retail opening-auction attention premium (LIQ-500, market-neutral).

After the close of day t:
  r[j,t]   = (1 + ret_co) * (1 + ret_oc) - 1             (close t-1 -> close t)
  sigma    = sample std of r over sessions t-L .. t-1 (t excluded), needs >= 2/3 L valid returns
  m[t]     = cross-sectional median of r[.,t] over members with a valid return on t
  z        = (r - m) / sigma
  event    = member & tradable & valid r & enough history & |z| >= k_sigma   (both directions)
  short    = at most 20 events with the largest |z|, -0.025 each
  hedge    = the 100 eligible non-event members with the best (lowest) liq_rank, equal weights,
             total long = total short (dollar neutral)
  no event -> flat.  Every row is a decision row (daily, one-session hold).
"""
import numpy as np

PARAMS = {"k_sigma": 3.0, "lookback_days": 60}

MAX_SHORTS = 20
SHORT_WEIGHT = 0.025
N_HEDGE = 100
MIN_HISTORY_FRACTION = 2.0 / 3.0


def _row_median(x, valid):
    """Median per row over the elements where valid is True; NaN where no valid element."""
    T, N = x.shape
    vals = np.where(valid, x, np.inf)
    s = np.sort(vals, axis=1)
    n = valid.sum(axis=1)
    lo = np.clip((n - 1) // 2, 0, max(N - 1, 0))
    hi = np.clip(n // 2, 0, max(N - 1, 0))
    rows = np.arange(T)
    med = np.full(T, np.nan)
    has = n > 0
    if N > 0:
        a = s[rows, lo]
        b = s[rows, hi]
        med[has] = 0.5 * (a[has] + b[has])
    return med


def _rank_positions(key):
    """Position of each element in the stable ascending order of its row (0 = first)."""
    T, N = key.shape
    order = np.argsort(key, axis=1, kind="stable")
    pos = np.empty((T, N), dtype=np.int64)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(N), (T, N)), axis=1)
    return pos


def compute_signal(data, params):
    """Returns z (T,N), eligible (T,N) bool, event (T,N) bool."""
    k_sigma = float(params["k_sigma"])
    lookback = int(params["lookback_days"])
    if data.universe is None:
        raise ValueError("H-0069 needs the point-in-time LIQ-500 universe (data.universe is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    T, N = ret_co.shape

    # (0) daily close-to-close total return and its validity
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    has_px = np.isfinite(close) & listed
    prev_px = np.zeros((T, N), dtype=bool)
    if T > 1:
        prev_px[1:] = has_px[:-1]
    valid = has_px & prev_px & tradable & np.isfinite(r)

    # (1) trailing sigma over t-L .. t-1 (prefix sums; rows before the data start count as missing)
    rv = np.where(valid, r, 0.0)
    P1 = np.zeros((T + 1, N))
    P2 = np.zeros((T + 1, N))
    PC = np.zeros((T + 1, N))
    np.cumsum(rv, axis=0, out=P1[1:])
    np.cumsum(rv * rv, axis=0, out=P2[1:])
    np.cumsum(valid.astype(np.float64), axis=0, out=PC[1:])
    t_idx = np.arange(T)
    start = np.maximum(t_idx - lookback, 0)
    s1 = P1[t_idx] - P1[start]
    s2 = P2[t_idx] - P2[start]
    cnt = np.rint(PC[t_idx] - PC[start])
    enough = cnt >= MIN_HISTORY_FRACTION * lookback - 1e-9
    enough &= cnt >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.where(cnt > 0, cnt, 1.0)) / np.where(cnt > 1, cnt - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    sigma = np.sqrt(var)

    # (2) cross-sectional median over members with a valid return on t
    m = _row_median(r, valid & universe)

    # (3) standardized stock-specific move
    base = universe & tradable & valid & enough & (sigma > 0) & np.isfinite(m)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(base, (r - m[:, None]) / np.where(sigma > 0, sigma, 1.0), np.nan)
    eligible = base & np.isfinite(z)

    # (4) events: both directions
    event = eligible & (np.abs(np.where(eligible, z, 0.0)) >= k_sigma)
    return z, eligible, event


def target_weights(data, params):
    z, eligible, event = compute_signal(data, params)
    liq_rank = np.asarray(data.extras["liq_rank"], dtype=np.float64)
    T, N = z.shape

    # (5) short leg: at most 20 events by |z| descending, -0.025 each
    key_s = np.where(event, -np.abs(np.where(event, z, 0.0)), np.inf)
    kept = event & (_rank_positions(key_s) < MAX_SHORTS)
    n_s = kept.sum(axis=1)

    # (6) hedge: the 100 best liq_rank among eligible non-event tradable members
    hedge_ok = eligible & ~event & np.isfinite(liq_rank)
    key_h = np.where(hedge_ok, liq_rank, np.inf)
    hedge = hedge_ok & (_rank_positions(key_h) < N_HEDGE)
    n_h = hedge.sum(axis=1)

    w = np.zeros((T, N), dtype=np.float64)
    w[kept] = -SHORT_WEIGHT
    short_total = SHORT_WEIGHT * n_s
    per_long = np.where(n_h > 0, short_total / np.maximum(n_h, 1), 0.0)
    w += hedge * per_long[:, None]
    # rows without events are flat (all zeros); every row is a decision row
    return w
