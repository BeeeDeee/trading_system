"""H-0187: daily stop-cascade breakout on Binance perps (top-30), BTC-hedged event slots.

Rules (card signal.description):
 (1) R_t = log(max/min of the N closes t-N .. t-1), N = range_days (all N closes valid).
 (2) quiet: R_t <= 25th percentile (numpy linear) of R_{t-90} .. R_{t-1} (all 90 valid),
     i.e. N + 90 valid closes before t.
 (3) up if close_t > max(range), down if close_t < min(range).
 (4) quote_volume_t >= median(quote_volume t-20 .. t-1) (all 20 valid, day t valid).
 (5) coin close-to-close return on t minus BTCUSDT.P's has the breakout's sign; BTC is never an event.
 (6) coin in the universe on day t (and tradable on row t).
 Book: slot +-0.15 from decision row t (fill at open t+1) for hold_days rows (closed at the
 open of t+1+hold_days), opposite BTCUSDT.P hedge of the same notional, hedges netted; a held
 coin ignores events until its slot closes; leaving the universe closes the slot at that row;
 gross > 1 is scaled to 1.
"""
import numpy as np

from lab.framework.api import only_on  # noqa: F401  (every row is a decision row)

PARAMS = {"hold_days": 3, "range_days": 20}

BTC = "BTCUSDT.P"
SLOT_WEIGHT = 0.15
QUIET_LOOKBACK = 90
QUIET_PCTL = 0.25
VOL_LOOKBACK = 20
UNIVERSE_N = 30
UNIVERSE_VOL_DAYS = 30


def _swv(x, window, axis=0):
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)


def fallback_universe(close, dollar_volume, n=UNIVERSE_N, days=UNIVERSE_VOL_DAYS):
    """Point-in-time top-n by trailing `days`-day quote volume (rows t-days+1 .. t), only used
    when the gate passes no crypto_top_n universe. Eligible: valid close on t, volume sum > 0."""
    T, N = close.shape
    v = np.where(np.isfinite(dollar_volume), dollar_volume, 0.0).astype(np.float64)
    cs = np.cumsum(v, axis=0)
    prev = np.zeros_like(cs)
    if T > days:
        prev[days:] = cs[:-days]
    vol = cs - prev
    ok = np.isfinite(close) & (vol > 0)
    key = np.where(ok, vol, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return ok & (rank < n)


def breakout_events(data, params, universe):
    """(T, N) side array: +1 up-event, -1 down-event, 0 none. Rows <= t only."""
    rd = int(params["range_days"])
    close = np.asarray(data.close, dtype=np.float64)
    qv = np.asarray(data.dollar_volume, dtype=np.float64)
    T, N = close.shape
    btc = data.col(BTC)
    side = np.zeros((T, N), dtype=np.float64)
    if T < rd + 1:
        return side

    # (1) range over the rd closes before t (NaN propagates: all must be valid)
    hi = np.full((T, N), np.nan)
    lo = np.full((T, N), np.nan)
    win = _swv(close, rd, axis=0)                 # (T-rd+1, N, rd), window ending at s = i+rd-1
    with np.errstate(invalid="ignore"):
        mx = win.max(axis=-1)
        mn = win.min(axis=-1)
    hi[rd:] = mx[:-1]                             # window ending at t-1
    lo[rd:] = mn[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.log(hi / lo)
    R[~np.isfinite(R)] = np.nan

    # (2) quiet: R_t <= 25th pct of R_{t-90..t-1}; (4) volume >= median of t-20..t-1
    quiet = np.zeros((T, N), dtype=bool)
    volok = np.zeros((T, N), dtype=bool)
    cols = np.flatnonzero(universe.any(axis=0))
    k = QUIET_PCTL * (QUIET_LOOKBACK - 1)
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, QUIET_LOOKBACK - 1)
    frac = k - k0
    m0 = VOL_LOOKBACK // 2 - 1
    m1 = VOL_LOOKBACK // 2
    for j in cols:
        if T > QUIET_LOOKBACK:
            Rj = R[:, j]
            w = _swv(Rj, QUIET_LOOKBACK)          # row i covers R[i .. i+89]; for t use i = t-90
            bad = np.isnan(w).any(axis=1)
            p = np.partition(np.where(np.isnan(w), np.inf, w), [k0, k1], axis=1)
            with np.errstate(invalid="ignore"):
                q = p[:, k0] + frac * (p[:, k1] - p[:, k0])
            q[bad] = np.nan
            qt = np.full(T, np.nan)
            qt[QUIET_LOOKBACK:] = q[:-1]
            with np.errstate(invalid="ignore"):
                quiet[:, j] = np.isfinite(Rj) & np.isfinite(qt) & (Rj <= qt)
        if T > VOL_LOOKBACK:
            vj = qv[:, j]
            w = _swv(vj, VOL_LOOKBACK)
            bad = np.isnan(w).any(axis=1)
            p = np.partition(np.where(np.isnan(w), np.inf, w), [m0, m1], axis=1)
            med = 0.5 * (p[:, m0] + p[:, m1])
            med[bad] = np.nan
            mt = np.full(T, np.nan)
            mt[VOL_LOOKBACK:] = med[:-1]
            with np.errstate(invalid="ignore"):
                volok[:, j] = np.isfinite(vj) & np.isfinite(mt) & (vj >= mt)

    # (5) coin-specific: day-t close-to-close return minus BTC's
    ret = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = close[1:] / close[:-1] - 1.0
    rel = ret - ret[:, [btc]]

    valid = np.isfinite(close) & np.isfinite(qv) & np.isfinite(rel)
    base = valid & quiet & volok & universe & np.asarray(data.tradable, dtype=bool)
    base[:, btc] = False
    with np.errstate(invalid="ignore"):
        up = base & (close > hi) & (rel > 0)
        dn = base & (close < lo) & (rel < 0)
    side[up] = 1.0
    side[dn] = -1.0
    return side


def book(side, universe, btc, hold_days, slot_weight=SLOT_WEIGHT):
    """Event slots with a netted BTC hedge and gross <= 1."""
    T, N = side.shape
    w = np.zeros((T, N), dtype=np.float64)
    end = np.full(N, -1, dtype=np.int64)          # last decision row of the open slot
    sgn = np.zeros(N, dtype=np.float64)
    for t in range(T):
        active = (end >= t) & universe[t]         # early exit when the coin left the universe
        end[~active] = -1
        new = (side[t] != 0) & ~active
        sgn[new] = side[t, new]
        end[new] = t + hold_days - 1
        active = active | new
        row = np.where(active, slot_weight * sgn, 0.0)
        row[btc] = -row.sum()
        if not universe[t, btc]:
            row[btc] = 0.0                        # never on real data (BTC is always top-30)
        g = np.abs(row).sum()
        if g > 1.0:
            row /= g
        w[t] = row
    return w


def get_universe(data):
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    return fallback_universe(np.asarray(data.close, dtype=np.float64),
                             np.asarray(data.dollar_volume, dtype=np.float64))


def target_weights(data, params):
    hold = int(params["hold_days"])
    uni = get_universe(data)
    btc = data.col(BTC)
    side = breakout_events(data, params, uni)
    return book(side, uni, btc, hold)
