"""H-0117: buffered beta-neutral book on a ridge liquidity-provision score, LIQ-300 high-volatility half, weekly.

Decision after the close of the first session of each week (period_starts "W"), executed at the next open.
"""
import numpy as np
from lab.framework.api import only_on, period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"alpha": 10.0, "entry_frac": 0.15, "exit_frac": 0.4}

HORIZON = 5            # card ml.horizon_days (label and purge)
RETRAIN_EVERY = 63     # card ml.retrain_every_days
VOL_WIN = 63           # 63-day realized volatility, 63-day mean dollar volume
SHORT_WIN = 5          # 5-day sums / 5-day mean dollar volume / prior-week window length
BETA_WIN = 252
BETA_MIN_OBS = 126
MIN_LEG = 5
GROSS_LO = 0.3
GROSS_HI = 0.7
TOL = 1e-9


def _rsum(x, n):
    """Trailing sum over rows t-n+1..t; NaN unless all n values are finite (point-in-time via prefix cumsums)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if T < n:
        return out
    valid = np.isfinite(x)
    c = np.cumsum(np.where(valid, x, 0.0), axis=0)
    k = np.cumsum(valid, axis=0, dtype=np.int32)
    del valid
    s = c[n - 1:].copy()
    cnt = k[n - 1:].copy()
    if T > n:
        s[1:] -= c[:-n]
        cnt[1:] -= k[:-n]
    out[n - 1:] = np.where(cnt == n, s, np.nan)
    return out


def _inputs(data):
    if data.universe is None:
        raise ValueError("H-0117 needs the point-in-time LIQ-300 universe (data.universe is None)")
    uni = np.asarray(data.universe, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    has_px = listed & np.isfinite(np.asarray(data.close, dtype=float))
    rco = np.where(has_px, np.asarray(data.ret_co, dtype=float), np.nan)
    roc = np.where(has_px, np.asarray(data.ret_oc, dtype=float), np.nan)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    dv = np.asarray(data.dollar_volume, dtype=float)
    dv = np.where(has_px & np.isfinite(dv) & (dv >= 0), dv, np.nan)
    return uni, rco, roc, r, dv


def _demean(f, mask):
    cnt = mask.sum(axis=1)
    s = np.where(mask, f, 0.0).sum(axis=1)
    mu = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return np.where(mask, f - mu[:, None], np.nan)


def _features(uni, rco, roc, r, dv):
    """Raw features (before z-scoring) and the eligible mask (members, high-vol half, complete vector)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = _rsum(r, VOL_WIN)
        var = _rsum(r * r, VOL_WIN)
        var -= s1 * s1 / VOL_WIN
        del s1
        var /= (VOL_WIN - 1)
        vol = np.where(np.isfinite(var), np.sqrt(np.maximum(var, 0.0)), np.nan)   # 63-day realized vol
        del var
        f5 = np.where(vol > 0, np.log(vol), np.nan)
    # high-volatility half: vol >= median of the day's members with a valid volatility
    vm = uni & np.isfinite(vol)
    med = np.full(vol.shape[0], np.nan)
    rows = vm.any(axis=1)
    if rows.any():
        med[rows] = np.nanmedian(np.where(vm[rows], vol[rows], np.nan), axis=1)
    with np.errstate(invalid="ignore"):
        hv = vm & (vol >= med[:, None])
    del vol, vm
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = _rsum(roc, SHORT_WIN)                       # 5-day intraday sum
        f2 = _rsum(rco, SHORT_WIN)                       # 5-day overnight sum
        l5 = _rsum(np.log1p(r), SHORT_WIN)
        f3 = np.full(l5.shape, np.nan)                   # total return over sessions t-9..t-5
        if l5.shape[0] > SHORT_WIN:
            f3[SHORT_WIN:] = np.expm1(l5[:-SHORT_WIN])
        del l5
        m63 = _rsum(dv, VOL_WIN) / VOL_WIN
        f4 = _rsum(dv, SHORT_WIN) / SHORT_WIN
        f4 = np.where((f4 > 0) & (m63 > 0), np.log(f4 / m63), np.nan)
        del m63
    feats = [f1, f2, f3, f4, f5]
    complete = np.ones(hv.shape, dtype=bool)
    for f in feats:
        complete &= np.isfinite(f)
    elig = hv & complete
    del feats, hv, complete
    f1 = _demean(f1, elig)
    f2 = _demean(f2, elig)
    f3 = _demean(f3, elig)
    f4 = np.where(elig, f4, np.nan)
    f5 = np.where(elig, f5, np.nan)
    return [f1, f2, f3, f4, f5], elig


def _score(data, params, uni, rco, roc, r, dv):
    feats, elig = _features(uni, rco, roc, r, dv)
    zs = []
    for i in range(len(feats)):
        zs.append(np.where(elig, cs_zscore(feats[i], elig), np.nan).astype(np.float32))
        feats[i] = None
    X = stack(*zs)
    del zs, feats
    y = cs_rank(forward_return(data, HORIZON), elig)
    score = walk_forward(X, y, horizon=HORIZON, model="ridge", params={"alpha": float(params["alpha"])},
                         retrain_every=RETRAIN_EVERY, universe=elig)
    score = np.where(elig, np.asarray(score, dtype=float), np.nan)
    return score, elig


def _market(r, uni):
    mm = uni & np.isfinite(r)
    cnt = mm.sum(axis=1)
    s = np.where(mm, r, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _betas(r, mkt, d, cols):
    """Slope of each column's daily total return on the member market return over the last 252 rows to d."""
    lo = max(0, d - BETA_WIN + 1)
    y = r[lo:d + 1][:, cols]
    x = np.broadcast_to(mkt[lo:d + 1][:, None], y.shape)
    v = np.isfinite(y) & np.isfinite(x)
    n = v.sum(axis=0)
    nn = np.maximum(n, 1)
    with np.errstate(invalid="ignore"):
        mx = np.where(v, x, 0.0).sum(axis=0) / nn
        my = np.where(v, y, 0.0).sum(axis=0) / nn
        dx = np.where(v, x - mx, 0.0)
        dy = np.where(v, y - my, 0.0)
        sxx = (dx * dx).sum(axis=0)
        sxy = (dx * dy).sum(axis=0)
        ok = (n >= BETA_MIN_OBS) & (sxx > 0)
        return np.where(ok, sxy / np.where(ok, sxx, 1.0), 1.0)


def _leg_gross(b_long, b_short):
    s = b_long + b_short
    if not (s > 0):
        gl, gs = 0.5, 0.5          # degenerate leg betas (sum <= 0): equal split
    else:
        gl, gs = b_short / s, b_long / s
    gl = min(max(gl, GROSS_LO), GROSS_HI)
    gs = min(max(gs, GROSS_LO), GROSS_HI)
    tot = gl + gs
    return gl / tot, gs / tot


def _ranks(s, ok):
    """Ordinal rank 0..n-1 (ascending score) among ok names; -1 elsewhere. Returns (pos, n)."""
    idx = np.flatnonzero(ok)
    pos = np.full(s.shape[0], -1.0)
    n = idx.size
    if n == 0:
        return pos, 0
    order = np.argsort(s[idx], kind="stable")
    rk = np.empty(n)
    rk[order] = np.arange(n, dtype=float)
    pos[idx] = rk
    return pos, n


def _legs(pos, n, ok, held_l, held_s, entry, exit_):
    """Buffered legs. pos = rank 0..n-1 (percentile = pos / (n - 1))."""
    nm1 = max(n - 1, 0)
    top_entry = ok & (pos >= (1.0 - entry) * nm1 - TOL)
    top_keep = ok & (pos >= (1.0 - exit_) * nm1 - TOL)
    bot_entry = ok & (pos <= entry * nm1 + TOL)
    bot_keep = ok & (pos <= exit_ * nm1 + TOL)
    want_l = (held_l & top_keep) | top_entry
    want_s = (held_s & bot_keep) | bot_entry
    both = want_l & want_s
    long_ = want_l & ~(both & ~held_s)     # in both: the newly entered side wins
    short_ = want_s & ~(both & ~held_l)
    return long_, short_


def _decision_days(dates):
    ps = period_starts(dates, "W")
    a = np.asarray(ps)
    if a.dtype == bool:
        return ps, np.flatnonzero(a)
    return ps, np.sort(a.astype(int))


def _book(score, elig, r, uni, days, entry, exit_):
    T, N = score.shape
    W = np.zeros((T, N))
    ev = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))
    mkt = _market(r, uni)
    held_l = np.zeros(N, dtype=bool)
    held_s = np.zeros(N, dtype=bool)
    for d in days:
        ok = elig[d] & np.isfinite(score[d])
        pos, n = _ranks(score[d], ok)
        long_, short_ = _legs(pos, n, ok, held_l, held_s, entry, exit_)
        nl = int(long_.sum())
        ns = int(short_.sum())
        if nl < MIN_LEG or ns < MIN_LEG:
            held_l = np.zeros(N, dtype=bool)
            held_s = np.zeros(N, dtype=bool)
            continue
        il = np.flatnonzero(long_)
        is_ = np.flatnonzero(short_)
        bl = float(np.mean(_betas(r, mkt, d, il)))
        bs = float(np.mean(_betas(r, mkt, d, is_)))
        gl, gs = _leg_gross(bl, bs)
        W[d, il] = gl * (1.0 - TOL) / nl          # (1 - 1e-9): keep float gross <= 1
        W[d, is_] = -gs * (1.0 - TOL) / ns
        new_l = long_ & ~held_l
        new_s = short_ & ~held_s
        ev[d] = new_l | new_s
        side[d] = np.where(new_l, 1.0, np.where(new_s, -1.0, 0.0))
        held_l = long_
        held_s = short_
    return W, ev, side


def target_weights(data, params):
    entry = float(params["entry_frac"])
    exit_ = float(params["exit_frac"])
    uni, rco, roc, r, dv = _inputs(data)
    score, elig = _score(data, params, uni, rco, roc, r, dv)
    del rco, roc, dv
    ps, days = _decision_days(data.dates)
    W, _, _ = _book(score, elig, r, uni, days, entry, exit_)
    return only_on(ps, W)
