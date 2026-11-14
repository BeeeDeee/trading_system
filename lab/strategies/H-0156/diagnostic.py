"""H-0156 mechanism test: on each monthly decision date with s_t != 0, mark every eligible EM country ETF
with side s_t. SPY, DIA, QQQ, IWM, MDY are never marked (peers). Same signal code as strategy.py (copied)."""
import numpy as np
from lab.framework.api import period_starts

EM_ETFS = ("EWZ", "EWW", "EWY", "EWT", "EWM", "EZA", "FXI", "ECH", "TUR", "THD",
           "EIDO", "EPOL", "EPHE", "EPU", "INDA")
PEERS = ("SPY", "DIA", "QQQ", "IWM", "MDY")

FX_FOREIGN_PER_USD = ("fred_macro.DEXJPUS", "fred_macro.DEXCAUS", "fred_macro.DEXSZUS",
                      "fred_macro.DEXSDUS")
FX_USD_PER_FOREIGN = ("fred_macro.DEXUSEU", "fred_macro.DEXUSUK", "fred_macro.DEXUSAL")

SESSIONS_PER_MONTH = 21
MIN_CURRENCIES = 5
MIN_PRIOR_PRICED = 21


def _ffill(x):
    x = np.asarray(x, dtype=float)
    idx = np.where(np.isfinite(x), np.arange(x.shape[0]), -1)
    idx = np.maximum.accumulate(idx)
    out = x[np.maximum(idx, 0)]
    out[idx < 0] = np.nan
    return out


def _log_dollar(v, invert):
    v = _ffill(v)
    x = np.full(v.shape, np.nan)
    pos = np.isfinite(v) & (v > 0)
    x[pos] = np.log(v[pos])
    return -x if invert else x


def _dollar_change(data, lookback_months):
    T = len(data.dates)
    lag = int(lookback_months) * SESSIONS_PER_MONTH
    rows = []
    for key in FX_FOREIGN_PER_USD:
        rows.append(_log_dollar(data.series[key], False))
    for key in FX_USD_PER_FOREIGN:
        rows.append(_log_dollar(data.series[key], True))
    diffs = np.full((len(rows), T), np.nan)
    for k, x in enumerate(rows):
        if 0 < lag < T:
            diffs[k, lag:] = x[lag:] - x[:T - lag]
    ok = np.isfinite(diffs)
    n = ok.sum(axis=0)
    total = np.where(ok, diffs, 0.0).sum(axis=0)
    D = np.full(T, np.nan)
    good = n >= MIN_CURRENCIES
    D[good] = total[good] / n[good]
    return D


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    em_cols = np.array([data.col(name) for name in EM_ETFS], dtype=int)
    for name in PEERS:
        data.col(name)                        # fail loudly if a peer is missing

    D = _dollar_change(data, params["lookback_months"])
    s = np.zeros(T)
    s[D < 0] = 1.0
    s[D > 0] = -1.0

    dec = np.asarray(period_starts(data.dates, "M"), dtype=bool)
    has = np.isfinite(np.asarray(data.close, dtype=float)[:, em_cols])
    prior = np.cumsum(has, axis=0) - has
    elig = has & (prior >= MIN_PRIOR_PRICED)
    elig &= np.asarray(data.tradable, dtype=bool)[:, em_cols]
    if data.universe is not None:
        elig &= np.asarray(data.universe, dtype=bool)[:, em_cols]

    active = dec & (s != 0)
    mask = np.zeros((T, N), dtype=bool)
    mask[:, em_cols] = elig & active[:, None]
    side = np.zeros((T, N))
    side[:, em_cols] = np.where(mask[:, em_cols], s[:, None], 0.0)
    return mask, side
