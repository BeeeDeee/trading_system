"""H-0177 mechanism events.

At each monthly decision row (same rule as the strategy) with (a) a full lookback_days history for
SPY, IEF and SHY and (b) at least three of the five near-cash peers SHY, SHV, BIL, VGSH, SCHO with a
price on that row: SPY is marked with side sB and IEF with side -sE. Peers are never marked.
"""
import numpy as np

PEERS = ("SHY", "SHV", "BIL", "VGSH", "SCHO")
MIN_PEERS = 3


def _decision_rows(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="backward")
    last = nxt.astype("datetime64[M]") != month
    dec = last.copy()
    if T > 1:
        missed = (month[1:] != month[:-1]) & ~last[:-1]
        dec[1:] = dec[1:] | missed
    return dec


def _trailing_cum_return(data, j, lookback):
    co = np.nan_to_num(np.asarray(data.ret_co[:, j], dtype=np.float64), nan=0.0)
    oc = np.nan_to_num(np.asarray(data.ret_oc[:, j], dtype=np.float64), nan=0.0)
    lr = np.log1p(co) + np.log1p(oc)
    T = lr.shape[0]
    c = np.concatenate([[0.0], np.cumsum(lr)])
    out = np.full(T, np.nan)
    if T > lookback:
        t = np.arange(lookback, T)
        out[lookback:] = np.expm1(c[t + 1] - c[t + 1 - lookback])
    return out


def _full_lookback(data, cols, lookback):
    T = data.dates.shape[0]
    ok = np.zeros(T, dtype=bool)
    if T > lookback:
        listed = np.asarray(data.listed, dtype=bool)
        good = np.ones(T - lookback, dtype=bool)
        for j in cols:
            good &= listed[:T - lookback, j] & listed[lookback:, j]
        ok[lookback:] = good
    return ok


def events(data, params):
    lookback = int(params["lookback_days"])
    T = data.dates.shape[0]
    N = len(data.instruments)
    i_spy = data.col("SPY")
    i_ief = data.col("IEF")
    i_shy = data.col("SHY")
    peer_cols = [data.col(p) for p in PEERS]

    r_spy = _trailing_cum_return(data, i_spy, lookback)
    r_ief = _trailing_cum_return(data, i_ief, lookback)
    r_shy = _trailing_cum_return(data, i_shy, lookback)
    full = _full_lookback(data, (i_spy, i_ief, i_shy), lookback)
    with np.errstate(invalid="ignore"):
        sb = np.where(r_ief - r_shy > 0, 1.0, -1.0)
        se = np.where(r_spy - r_shy > 0, 1.0, -1.0)

    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    priced = np.zeros(T, dtype=np.int64)
    for j in peer_cols:
        priced += (np.isfinite(close[:, j]) & listed[:, j]).astype(np.int64)

    date_ok = _decision_rows(data.dates) & full & (priced >= MIN_PEERS)

    tradable = np.asarray(data.tradable, dtype=bool)
    uni = data.universe
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for j, s in ((i_spy, sb), (i_ief, -se)):
        ok = date_ok & tradable[:, j] & listed[:, j]
        if uni is not None:
            ok &= np.asarray(uni[:, j], dtype=bool)
        mask[:, j] = ok
        side[:, j] = np.where(ok, s, 0.0)
    return mask, side
