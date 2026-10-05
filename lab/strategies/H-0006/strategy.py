"""H-0006 Crypto liquidation-flush rebound.

Buy liquid Binance spot pairs after an extreme daily drop (r <= -k_sigma * sigma30), only while the BTCUSDT
weekly regime is ON (Sunday close > SMA of the last regime_weeks Sunday closes), hold hold_days days,
weight 1 / max(4, n_open) per open position, recomputed daily.

Row t is decided after the close of day t; the engine executes it at the open of day t+1.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"hold_days": 3, "k_sigma": 2.0, "regime_weeks": 20}

UNIVERSE_N = 50        # card: top 50 pairs (fixed convention)
VOL_WINDOW = 30        # card: mean daily quote_volume over the 30 days ending at t
MIN_HISTORY = 60       # card: at least 60 days of price history at t
SIGMA_WINDOW = 30      # card: std of daily log returns over the 30 days ending at t-1
MIN_WEIGHT_DEN = 4     # card: weight 1 / max(4, n_open)
REGIME_PAIR = "BTCUSDT"


def _lag(x, k, fill):
    """x shifted down by k rows (row t holds x[t-k]); first k rows = fill."""
    out = np.empty_like(x)
    T = x.shape[0]
    k = min(k, T)
    out[:k] = fill
    out[k:] = x[: T - k]
    return out


def _weekday(dates):
    """Monday = 0 ... Sunday = 6 (day 0 of the epoch was a Thursday)."""
    d = dates.astype("datetime64[D]").astype(np.int64)
    return (d + 3) % 7


def _ffill(x):
    """Forward-fill NaNs of a 1-D array (past values only)."""
    valid = ~np.isnan(x)
    idx = np.where(valid, np.arange(x.shape[0]), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(x.shape, np.nan)
    ok = idx >= 0
    out[ok] = x[idx[ok]]
    return out


def regime_series(btc_close, dates, regime_weeks):
    """(T,) bool: weekly BTC regime in force on each row.

    Weekly close = BTC close of the Sunday row (forward-filled if the Sunday bar is missing). At each Sunday,
    ON iff weekly close > mean of the last regime_weeks weekly closes (incl. the current one); OFF until
    regime_weeks weekly closes exist. Row t uses the regime set at the last Sunday on or before t.
    """
    T = dates.shape[0]
    n = int(regime_weeks)
    c = _ffill(np.asarray(btc_close, dtype=float))
    sundays = np.flatnonzero((_weekday(dates) == 6) & ~np.isnan(c))
    out = np.zeros(T, dtype=bool)
    if sundays.size == 0:
        return out
    wc = c[sundays]
    K = wc.shape[0]
    on = np.zeros(K, dtype=bool)
    if K >= n:
        s = np.zeros(K - n + 1)
        for j in range(n):
            s += wc[j: K - n + 1 + j]
        sma = s / n
        on[n - 1:] = wc[n - 1:] > sma
    pos = np.searchsorted(sundays, np.arange(T), side="right") - 1
    has = pos >= 0
    out[has] = on[pos[has]]
    return out


def card_universe(close, volume):
    """(T, N) bool: top UNIVERSE_N pairs by mean quote volume over the last VOL_WINDOW days, among pairs
    with at least MIN_HISTORY observed closes up to t."""
    T, N = close.shape
    hist = np.cumsum(~np.isnan(close), axis=0)
    v = np.nan_to_num(volume, nan=0.0)
    s = np.zeros((T, N))
    for k in range(VOL_WINDOW):
        s += _lag(v, k, 0.0)
    mean_vol = s / VOL_WINDOW
    qual = hist >= MIN_HISTORY
    score = np.where(qual, mean_vol, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    return qual & (rank < UNIVERSE_N)


def triggers(close, k_sigma):
    """(T, N) bool: r_t <= -k_sigma * sigma_{t-1}, sigma over the SIGMA_WINDOW returns ending at t-1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close / _lag(close, 1, np.nan))           # NaN if close t or t-1 missing
    valid = ~np.isnan(r)
    r0 = np.where(valid, r, 0.0)
    cnt = np.zeros(r.shape)
    s = np.zeros(r.shape)
    for k in range(1, SIGMA_WINDOW + 1):                      # rows t-30 .. t-1
        cnt += _lag(valid.astype(float), k, 0.0)
        s += _lag(r0, k, 0.0)
    full = cnt >= SIGMA_WINDOW
    mean = np.where(full, s / SIGMA_WINDOW, 0.0)
    ss = np.zeros(r.shape)
    for k in range(1, SIGMA_WINDOW + 1):
        ss += (_lag(r0, k, 0.0) - mean) ** 2
    sigma = np.sqrt(ss / (SIGMA_WINDOW - 1))                  # sample std (ddof = 1)
    sigma = np.where(full, sigma, np.nan)
    with np.errstate(invalid="ignore"):
        trig = valid & full & (sigma > 0) & (r <= -float(k_sigma) * sigma)
    return trig


def target_weights(data, params):
    hold = int(params["hold_days"])
    k_sigma = float(params["k_sigma"])
    regime_weeks = int(params["regime_weeks"])

    close = np.asarray(data.close, dtype=float)
    volume = np.asarray(data.dollar_volume, dtype=float)      # binance_spot_1d quote_volume
    T, N = close.shape

    if REGIME_PAIR in data.instruments:
        regime_close = close[:, data.col(REGIME_PAIR)]
    else:
        # Synthetic G0 market only (it has no BTCUSDT): equal-weight total-return index as a stand-in.
        tri = total_return_index(data)
        with np.errstate(invalid="ignore"):
            regime_close = np.nanmean(np.where(np.isfinite(tri), tri, np.nan), axis=1)
    regime = regime_series(regime_close, data.dates, regime_weeks)

    member = np.asarray(data.universe, dtype=bool) if data.universe is not None else np.ones((T, N), bool)
    uni = card_universe(close, volume) & member
    trig = triggers(close, k_sigma) & uni & regime[:, None]

    # A trigger at s keeps the pair open on decision rows s .. s+hold-1 (entry open s+1, exit open s+1+hold);
    # a new trigger while held resets the exit, i.e. open = any trigger in the last `hold` rows.
    is_open = np.zeros((T, N), dtype=bool)
    for k in range(hold):
        is_open |= _lag(trig, k, False)
    is_open &= member                                          # contract: hold only universe members

    n_open = is_open.sum(axis=1)
    w_each = 1.0 / np.maximum(MIN_WEIGHT_DEN, n_open)
    return np.where(is_open, w_each[:, None], 0.0)
