"""H-0005: stablecoin net issuance as crypto inflow gauge.

Weekly decision after the close of every Sunday (UTC day t), executed at the Monday open.
Regime: G_t = ln(S_{t-1} / S_{t-1-L}) of the aggregate USD stablecoin supply; risk-on if G_t > 0.
Risk-on portfolio: top 10 pairs by 30-day median quote volume (>= 60 bars), inverse-volatility
weights (60-day std of daily close-to-close log returns), capped at 0.25 with proportional
redistribution. Risk-off or missing/stale supply data: cash.
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"lookback_days": 28}

SUPPLY_KEYS = ("stablecoin_supply_1d.total_circulating_usd", "stablecoin_supply_1d.total")
TOP_N = 10
VOLUME_WINDOW_DAYS = 30
MIN_BARS = 60
VOL_WINDOW_DAYS = 60
MAX_WEIGHT = 0.25
MAX_STALENESS_DAYS = 7
SUNDAY = 6  # Monday = 0


def weekday(dates):
    """Monday = 0 ... Sunday = 6, from the date of the row alone (day 0 of the epoch was a Thursday)."""
    days = dates.astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7


def supply_series(data):
    series = data.series if data.series is not None else {}
    for key in SUPPLY_KEYS:
        if key in series:
            s = np.asarray(series[key], dtype=float)
            if s.ndim == 2:
                s = s[:, 0]
            return s
    return None


def regime_signal(dates, supply, lookback_days):
    """Return (G, valid) per row.

    G[t] = ln(S(d_t - 1) / S(d_t - 1 - L)) where S(d) is the last supply value observed at or before
    calendar day d (rows 0..t only). valid[t] is False when the history is shorter than L + 1 days
    or the value used for d_t - 1 is older than 7 days.
    """
    T = len(dates)
    day = dates.astype("datetime64[D]").astype(np.int64)
    G = np.full(T, np.nan)
    valid = np.zeros(T, dtype=bool)
    if supply is None or T == 0:
        return G, valid
    obs = np.isfinite(supply) & (supply > 0)
    idx = np.where(obs, np.arange(T), -1)
    last_obs = np.maximum.accumulate(idx)  # last row <= t with an observation
    first_obs_rows = np.flatnonzero(obs)
    if first_obs_rows.size == 0:
        return G, valid
    first_day = day[first_obs_rows[0]]

    # row of the last date <= d - k, using only dates up to the row itself (dates are sorted)
    def row_at_or_before(target_day):
        r = np.searchsorted(day, target_day, side="right") - 1
        return r

    r1 = row_at_or_before(day - 1)
    r0 = row_at_or_before(day - 1 - lookback_days)
    ok = (r1 >= 0) & (r0 >= 0)
    r1c = np.clip(r1, 0, T - 1)
    r0c = np.clip(r0, 0, T - 1)
    o1 = np.where(ok, last_obs[r1c], -1)
    o0 = np.where(ok, last_obs[r0c], -1)
    ok &= (o1 >= 0) & (o0 >= 0)
    # at least L + 1 days of supply history up to d_t - 1
    ok &= (day - 1 - lookback_days) >= first_day
    # value used for d_t - 1 not older than 7 days
    o1c = np.clip(o1, 0, T - 1)
    ok &= ((day - 1) - day[o1c]) <= MAX_STALENESS_DAYS
    o0c = np.clip(o0, 0, T - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(supply[o1c] / supply[o0c])
    ok &= np.isfinite(g)
    G[ok] = g[ok]
    valid = ok
    return G, valid


def cap_weights(w, cap):
    """Normalise to 1, then cap at `cap`, redistributing the excess proportionally to uncapped names."""
    w = np.asarray(w, dtype=float).copy()
    total = w.sum()
    if total <= 0:
        return np.zeros_like(w)
    w = w / total
    capped = np.zeros(w.shape, dtype=bool)
    for _ in range(len(w) + 1):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        w[capped] = cap
        free = ~capped & (w > 0)
        remaining = 1.0 - cap * capped.sum()
        free_sum = w[free].sum()
        if free_sum <= 0 or remaining <= 0:
            break  # fewer than 1/cap names: everything at the cap, rest stays in cash
        w[free] = w[free] / free_sum * remaining
    return w


def target_weights(data, params):
    L = int(params["lookback_days"])
    dates = data.dates
    T = len(dates)
    N = len(data.instruments)
    out = np.full((T, N), np.nan)
    if T == 0:
        return out

    decision = weekday(dates) == SUNDAY
    G, valid = regime_signal(dates, supply_series(data), L)
    risk_on = valid & (G > 0)

    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close) & (close > 0)
    bars = np.cumsum(has_bar, axis=0)
    qv = np.asarray(data.dollar_volume, dtype=float)
    qv = np.where(has_bar, qv, np.nan)

    tri = np.asarray(total_return_index(data), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.full((T, N), np.nan)
        logret[1:] = np.log(tri[1:] / tri[:-1])
    logret[1:] = np.where(has_bar[1:] & has_bar[:-1], logret[1:], np.nan)
    logret[~np.isfinite(logret)] = np.nan

    listed = np.asarray(data.listed, dtype=bool) if data.listed is not None else np.ones((T, N), bool)
    universe = np.asarray(data.universe, dtype=bool) if data.universe is not None else None

    for t in np.flatnonzero(decision):
        w = np.zeros(N)
        if risk_on[t]:
            eligible = (bars[t] >= MIN_BARS) & has_bar[t] & listed[t]
            if universe is not None:
                eligible &= universe[t]
            lo = max(0, t - VOLUME_WINDOW_DAYS + 1)
            with np.errstate(all="ignore"):
                win = qv[lo:t + 1]
                cnt = np.isfinite(win).sum(axis=0)
                med = np.full(N, np.nan)
                has = cnt > 0
                if has.any():
                    med[has] = np.nanmedian(win[:, has], axis=0)
            eligible &= np.isfinite(med)
            cand = np.flatnonzero(eligible)
            if cand.size > 0:
                # sort by median volume descending, ties broken by column order (deterministic)
                order = cand[np.lexsort((cand, -med[cand]))]
                top = order[:TOP_N]
                lo_v = max(1, t - VOL_WINDOW_DAYS + 1)
                rw = logret[lo_v:t + 1][:, top]
                n_ok = np.isfinite(rw).sum(axis=0)
                with np.errstate(all="ignore"):
                    sig = np.full(top.size, np.nan)
                    good = n_ok >= 2
                    if good.any():
                        sig[good] = np.nanstd(rw[:, good], axis=0, ddof=1)
                inv = np.where(np.isfinite(sig) & (sig > 0), 1.0 / sig, 0.0)
                w[top] = cap_weights(inv, MAX_WEIGHT)
        out[t] = w
    return out
