"""H-0047 mechanism events: on each release session t, the top_k ETFs by z = r_t / sigma_63 are
marked +1 and the bottom_k are marked -1 (same selection as the strategy's cohorts). Dates with
fewer than 2 * top_k + 3 eligible ETFs are not events. Point-in-time: rows <= t only."""
import numpy as np

WINDOW = 63
MIN_PEERS = 3
SERIES_KEY = "macro_release_calendar_1d.next_release_sessions"


def _returns(data):
    ret_co = np.asarray(data.ret_co, dtype=float)
    ret_oc = np.asarray(data.ret_oc, dtype=float)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=float)
    has = listed & np.isfinite(close)
    valid = np.zeros_like(has)
    valid[1:] = has[1:] & has[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def events(data, params):
    top_k = int(params["top_k"])
    T = len(data.dates)
    N = len(data.instruments)
    s = np.asarray(data.series[SERIES_KEY], dtype=float).reshape(-1)
    if s.shape[0] != T:
        raise ValueError("release calendar series does not match the data calendar")
    rel = np.zeros(T, dtype=bool)
    with np.errstate(invalid="ignore"):
        rel[1:] = np.rint(s[:-1]) == 1.0   # identical to == 1 on the integer series
    r, valid = _returns(data)
    tradable = np.asarray(data.tradable, dtype=bool)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    for t in np.flatnonzero(rel):
        t = int(t)
        if t < WINDOW:
            continue
        elig = valid[t] & valid[t - WINDOW:t].all(axis=0)
        idx = np.flatnonzero(elig)
        if idx.size == 0:
            continue
        sig = np.std(r[t - WINDOW:t, idx], axis=0, ddof=1)
        ok = np.isfinite(sig) & (sig > 0)
        idx = idx[ok]
        sig = sig[ok]
        if idx.size < 2 * top_k + MIN_PEERS:
            continue
        z = r[t, idx] / sig
        order = np.lexsort((idx, -z))
        longs = idx[order[:top_k]]
        shorts = idx[order[idx.size - top_k:]]
        mask[t, longs] = True
        side[t, longs] = 1.0
        mask[t, shorts] = True
        side[t, shorts] = -1.0
    mask &= tradable
    if uni is not None:
        mask &= uni
    side = np.where(mask, side, 0.0)
    return mask, side
