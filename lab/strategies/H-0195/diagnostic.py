"""H-0195 mechanism_test event: a LIQ-500 stock (member, traded on t) with a market-adjusted overnight gap of
-gap_pct % or worse, ret_oc >= 0, dollar volume >= 4 x its median over the prior 63 sessions (>= 50 valid
values), and no 8-K earnings release on t-2, t-1 or t (flag not NaN on t). Side +1. Same code as conditions
(1)-(4) of strategy.py (copied: diagnostic.py is a separate file). The strategy's no-re-entry rule is a book
rule and is not part of the event text, so it is not applied here.
"""
import numpy as np

VOL_MULT = 4.0
VOL_WIN = 63
VOL_MIN_VALID = 50


def _cross_median(x, valid):
    T = x.shape[0]
    xm = np.where(valid, x, np.nan)
    med = np.full(T, np.nan)
    ok = valid.any(axis=1)
    if ok.any():
        med[ok] = np.nanmedian(xm[ok], axis=1)
    return med


def _shift_rows(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def events(data, params):
    gap = float(params["gap_pct"]) / 100.0
    uni = data.universe
    if uni is None:
        raise ValueError("H-0195 needs the point-in-time LIQ-500 universe (data.universe is None)")
    member = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = co.shape

    valid_co = member & tradable & np.isfinite(co)
    med = _cross_median(co, valid_co)
    g = co - med[:, None]
    base = valid_co & np.isfinite(oc) & np.isfinite(dv) & np.isfinite(g)
    cond = base & (g <= -gap) & (oc >= 0.0)

    ea1 = _shift_rows(ea, 1)
    ea2 = _shift_rows(ea, 2)
    with np.errstate(invalid="ignore"):
        ea_ok = np.isfinite(ea) & (ea != 1.0) & ~(ea1 == 1.0) & ~(ea2 == 1.0)
    cond &= ea_ok

    mask = np.zeros((T, N), dtype=bool)
    ti, ji = np.nonzero(cond)
    if ti.size > 0:
        offs = np.arange(-VOL_WIN, 0)
        rows = ti[:, None] + offs[None, :]
        inb = rows >= 0
        win = np.full(rows.shape, np.nan)
        win[inb] = dv[rows[inb], np.broadcast_to(ji[:, None], rows.shape)[inb]]
        nvalid = np.isfinite(win).sum(axis=1)
        enough = nvalid >= VOL_MIN_VALID
        vmed = np.full(ti.size, np.nan)
        if enough.any():
            vmed[enough] = np.nanmedian(win[enough], axis=1)
        with np.errstate(invalid="ignore"):
            vol_ok = enough & (dv[ti, ji] >= VOL_MULT * vmed)
        mask[ti[vol_ok], ji[vol_ok]] = True
    return mask, None
