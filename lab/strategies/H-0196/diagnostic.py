"""H-0196 mechanism event: a LIQ-500 member after the close of t with an idiosyncratic close-to-close drop
z <= -z_min (residual std over t-63..t-1, >= 50 valid), no 8-K earnings release and no non-earnings 8-K
on t or t-1, 3..window_close sessions after its latest Item 2.02 reaction session (within 100 sessions),
and a positive trailing-twelve-month buyback yield -ncfcommon/marketcap in the top third of members.
Same code as strategy.event_table (copied: separate file). Side +1.
"""
import numpy as np

RESID_WIN = 63
RESID_MIN = 50
WINDOW_OPEN = 3
RELEASE_LOOKBACK = 100
TOP_FRACTION = 1.0 / 3.0


def _close_to_close(data):
    r = np.asarray(data.ret_co, dtype=np.float64) + 1.0
    r *= np.asarray(data.ret_oc, dtype=np.float64) + 1.0
    r -= 1.0
    has_px = np.isfinite(np.asarray(data.close)) & np.asarray(data.listed, dtype=bool)
    valid = has_px.copy()
    valid[0] = False
    valid[1:] &= has_px[:-1]
    valid &= np.isfinite(r)
    r[~valid] = np.nan
    return r


def _residual_z(r, uni):
    T, N = r.shape
    rv = np.isfinite(r)
    mem = uni & rv
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    mean = np.full(T, np.nan)
    ok = cnt > 0
    mean[ok] = tot[ok] / cnt[ok]
    del rv, mem, tot
    e = r - mean[:, None]
    ev = np.isfinite(e)
    lo = np.maximum(np.arange(T) - RESID_WIN, 0)
    cs = np.zeros((T + 1, N))
    np.cumsum(np.where(ev, e, 0.0), axis=0, out=cs[1:])
    S = cs[:T] - cs[lo]
    cs[1:] = np.where(ev, e * e, 0.0)
    np.cumsum(cs[1:], axis=0, out=cs[1:])
    Q = cs[:T] - cs[lo]
    del cs
    cc = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(ev, axis=0, out=cc[1:])
    n = (cc[:T] - cc[lo]).astype(np.float64)
    del cc
    with np.errstate(invalid="ignore", divide="ignore"):
        S *= S
        S /= n
        Q -= S
        del S
        Q /= n - 1.0
    np.maximum(Q, 0.0, out=Q)
    Q[n < RESID_MIN] = np.nan
    del n
    np.sqrt(Q, out=Q)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(Q) & (Q > 0) & ev
        e /= Q
    del Q
    e[~ok] = np.nan
    return e


def _no_news(ea, nk):
    quiet = (ea == 0) & (nk == 0)
    prev = np.zeros_like(quiet)
    prev[1:] = quiet[:-1]
    return quiet & prev


def _open_window(ea, window_close):
    T, N = ea.shape
    rows = np.arange(T, dtype=np.int32)[:, None]
    last = np.where(ea == 1, rows, np.int32(-1))
    np.maximum.accumulate(last, axis=0, out=last)
    ok = last >= 0
    last -= rows
    last *= -1
    ok &= (last >= WINDOW_OPEN) & (last <= window_close) & (last <= RELEASE_LOOKBACK)
    return ok


def _repurchaser(ncf, mcap, uni):
    with np.errstate(invalid="ignore", divide="ignore"):
        b = -np.asarray(ncf, dtype=np.float64)
        b /= mcap
        bv = np.isfinite(b) & np.isfinite(mcap) & (mcap > 0) & uni
    b[~bv] = np.nan
    bm = b
    n = bv.sum(axis=1)
    asc = np.sort(bm, axis=1)
    m = np.ceil(n * TOP_FRACTION).astype(np.int64)
    pos = np.clip(n - m, 0, bm.shape[1] - 1)
    thr = np.take_along_axis(asc, pos[:, None], axis=1)[:, 0]
    thr = np.where(n > 0, thr, np.inf)
    with np.errstate(invalid="ignore"):
        return bv & (bm > 0) & (bm >= thr[:, None])


def events(data, params):
    if data.universe is None:
        raise ValueError("card needs the liq_n point-in-time universe")
    uni = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    ea = np.asarray(data.extras["ea_reaction_day"])
    nk = np.asarray(data.extras["nonearn_8k_reaction_day"])
    ncf = np.asarray(data.extras["sf1_art_ncfcommon"])
    mcap = np.asarray(data.extras["sf1_art_marketcap"])
    z_min = float(params["z_min"])
    window_close = int(params["window_close"])

    r = _close_to_close(data)
    z = _residual_z(r, uni)
    del r
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(z) & (z <= -z_min)
    del z
    mask &= uni
    mask &= tradable
    mask &= _no_news(ea, nk)
    mask &= _open_window(ea, window_close)
    mask &= _repurchaser(ncf, mcap, uni)
    return mask, None
