"""H-0165 mechanism events: |z_t| >= z_threshold marks SPY, QQQ, IWM, DIA (+1 fear, -1 complacency).

z_t is computed exactly as in strategy.py (code copied, since this file is separate). Bond ETFs
IEF, TLT, SHY, AGG are never marked and serve as peers.
"""
import numpy as np

SMOOTH = 5
SMOOTH_MIN = 4
FIT_WINDOW = 252
FIT_MIN = 200
STALE_RUN = 3
EQUITY = ("SPY", "QQQ", "IWM", "DIA")


def fresh_values(pc):
    pc = np.asarray(pc, dtype=float)
    lp = np.full(pc.shape, np.nan)
    ok = np.isfinite(pc) & (pc > 0)
    lp[ok] = np.log(pc[ok])
    stale = np.zeros(pc.shape, dtype=bool)
    if pc.shape[0] >= STALE_RUN:
        same = np.ones(pc.shape[0] - (STALE_RUN - 1), dtype=bool)
        for k in range(1, STALE_RUN):
            a = pc[STALE_RUN - 1:]
            b = pc[STALE_RUN - 1 - k:pc.shape[0] - k]
            same &= np.isfinite(a) & (a == b)
        stale[STALE_RUN - 1:] = same
    lp[stale] = np.nan
    return lp


def trailing_valid_mean(x, n, min_valid):
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    out = np.full(T, np.nan)
    if T == 0:
        return out
    ok = np.isfinite(x)
    s = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    c = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    cnt = c[hi] - c[lo]
    full = (hi - n) >= 0
    good = full & (cnt >= min_valid)
    out[good] = (s[hi] - s[lo])[good] / cnt[good]
    return out


def residual_z(pc_aligned, vix_aligned, spy_logret_session):
    pc_aligned = np.asarray(pc_aligned, dtype=float)
    T = pc_aligned.shape[0]
    z = np.full(T, np.nan)
    if T < 2:
        return z
    p = trailing_valid_mean(fresh_values(pc_aligned), SMOOTH, SMOOTH_MIN)
    vix = np.asarray(vix_aligned, dtype=float)
    v = np.full(T, np.nan)
    okv = np.isfinite(vix) & (vix > 0)
    v[okv] = np.log(vix[okv])
    lr = np.asarray(spy_logret_session, dtype=float)
    lr_shift = np.full(T, np.nan)
    lr_shift[1:] = lr[:-1]
    x = trailing_valid_mean(lr_shift, SMOOTH, SMOOTH) * SMOOTH
    valid = np.isfinite(p) & np.isfinite(x) & np.isfinite(v)
    cnt = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    for t in range(T):
        if not valid[t]:
            continue
        lo = max(t - FIT_WINDOW + 1, 0)
        if t - FIT_WINDOW + 1 < 0:
            continue
        if cnt[t + 1] - cnt[lo] < FIT_MIN:
            continue
        sl = slice(lo, t + 1)
        m = valid[sl]
        xs = x[sl][m]
        vs = v[sl][m]
        ps = p[sl][m]
        A = np.column_stack([np.ones(xs.shape[0]), xs, vs])
        coef, _, rank, _ = np.linalg.lstsq(A, ps, rcond=None)
        if rank < 3:
            continue
        e = ps - A @ coef
        sd = float(np.std(e, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        z[t] = e[-1] / sd
    return z


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    js = data.col("SPY")
    rco = np.asarray(data.ret_co[:, js], dtype=float)
    roc = np.asarray(data.ret_oc[:, js], dtype=float)
    lr = np.log1p(rco) + np.log1p(roc)
    pc = np.asarray(data.series["cboe_putcall_1d.equity_pc"], dtype=float)
    vix = np.asarray(data.series["fred_macro.VIXCLS"], dtype=float)
    z = residual_z(pc, vix, lr)
    k = float(params["z_threshold"])
    fear = np.isfinite(z) & (z >= k)
    comp = np.isfinite(z) & (z <= -k)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    eligible = np.asarray(data.tradable, dtype=bool)
    if data.universe is not None:
        eligible = eligible & np.asarray(data.universe, dtype=bool)
    for name in EQUITY:
        j = data.col(name)
        mask[:, j] = (fear | comp) & eligible[:, j]
        side[:, j] = np.where(fear, 1.0, np.where(comp, -1.0, 0.0))
    side = np.where(mask, side, 0.0)
    return mask, side
