"""H-0185 v2 mechanism test: weekly breakeven extremes mark TIP and SCHP.

Event: at each weekly decision date (last session of the week, same rule as the strategy)
with |z| >= 1 of T5YIE (through the prior session) vs its trailing lookback_days mean / sd,
each TIPS ETF with a price (TIP, SCHP) is marked: side +1 if z <= -1, -1 if z >= +1.
IEF, TLT, SHY and IEI are never marked (peers).
"""
import numpy as np

BE_KEY = "fred_macro.T5YIE"
ENTRY_Z = 1.0
MIN_VALID = 0.8


def _ffill(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    idx = np.where(np.isfinite(x), np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(n, np.nan)
    ok = idx >= 0
    out[ok] = x[idx[ok]]
    return out


def _decision_days(dates):
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    wd = (d + 3) % 7
    week = (d + 3) // 7
    dec = wd == 4
    if d.shape[0] > 1:
        dec[1:] |= (week[1:] != week[:-1]) & (wd[:-1] != 4)
    return dec


def _breakeven_z(be_row, lookback):
    raw = np.asarray(be_row, dtype=float)
    x = _ffill(raw)
    n = x.shape[0]
    L = int(lookback)
    z = np.full(n, np.nan)
    if n < L or L < 2:
        return z
    valid_raw = np.isfinite(raw).astype(float)
    have = np.isfinite(x)
    xv = np.where(have, x, 0.0)
    c_raw = np.concatenate([[0.0], np.cumsum(valid_raw)])
    c_n = np.concatenate([[0.0], np.cumsum(have.astype(float))])
    c_s = np.concatenate([[0.0], np.cumsum(xv)])
    c_q = np.concatenate([[0.0], np.cumsum(xv * xv)])
    end = np.arange(L, n + 1)
    start = end - L
    nraw = c_raw[end] - c_raw[start]
    cnt = c_n[end] - c_n[start]
    s = c_s[end] - c_s[start]
    q = c_q[end] - c_q[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
        var = np.maximum((q - cnt * mean * mean) / (cnt - 1.0), 0.0)
        sd = np.sqrt(var)
        zz = (x[end - 1] - mean) / sd
    ok = (nraw >= MIN_VALID * L) & (cnt >= 2) & (sd > 1e-12) & np.isfinite(x[end - 1])
    z[end - 1] = np.where(ok, zz, np.nan)
    return z


def events(data, params):
    lookback = int(params["lookback_days"])
    T = len(data.dates)
    N = len(data.instruments)
    j_tip = data.col("TIP")
    j_schp = data.col("SCHP")
    for name in ("IEF", "TLT", "SHY", "IEI"):
        data.col(name)

    be = np.asarray(data.series[BE_KEY], dtype=float)
    z = _breakeven_z(be, lookback)
    dec = _decision_days(data.dates)
    fin = np.isfinite(z)
    cheap = dec & fin & (z <= -ENTRY_Z)
    rich = dec & fin & (z >= ENTRY_Z)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    uni = data.universe
    for j in (j_tip, j_schp):
        ok = (np.isfinite(np.asarray(data.close[:, j], dtype=float))
              & np.asarray(data.listed[:, j], dtype=bool)
              & np.asarray(data.tradable[:, j], dtype=bool))
        if uni is not None:
            ok &= np.asarray(uni[:, j], dtype=bool)
        mask[:, j] = (cheap | rich) & ok
        side[:, j] = np.where(cheap, 1.0, np.where(rich, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side
