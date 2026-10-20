"""H-0066 mechanism test: peer events of large 8-K release reactions (same rules as strategy.py).

The event logic is a verbatim copy of strategy.peer_events (the diagnostic may not import the strategy).
"""
import numpy as np

CORR_WINDOW = 252
CORR_MIN_OBS = 200
Z_WINDOW = 63
Z_MIN_OBS = 40
NO_REPORT_WINDOW = 40


def _valid_returns(ret_co, ret_oc, close, listed, tradable):
    co = np.asarray(ret_co, dtype=np.float32)
    oc = np.asarray(ret_oc, dtype=np.float32)
    r = (np.float32(1.0) + co) * (np.float32(1.0) + oc) - np.float32(1.0)
    fin_close = np.isfinite(close)
    prev_close = np.zeros_like(fin_close)
    prev_close[1:] = fin_close[:-1]
    valid = (np.asarray(listed, dtype=bool) & np.asarray(tradable, dtype=bool) & fin_close & prev_close
             & np.isfinite(r))
    return r, valid


def _market_adjusted(r, valid, uni):
    in_mean = valid & uni
    cnt = in_mean.sum(axis=1)
    tot = np.where(in_mean, r, np.float32(0.0)).sum(axis=1, dtype=np.float64)
    mm = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    mm[ok] = tot[ok] / cnt[ok]
    return np.where(valid, r - mm[:, None].astype(np.float32), np.float32(np.nan)).astype(np.float32)


def _peer_table(window, k, min_obs=CORR_MIN_OBS):
    W = np.asarray(window, dtype=np.float64)
    M = W.shape[1]
    out = np.full((M, k), -1, dtype=np.int64)
    if M < 2 or W.shape[0] == 0:
        return out
    V = np.isfinite(W)
    X = np.where(V, W, 0.0)
    Vf = V.astype(np.float64)
    n = Vf.T @ Vf
    sx = X.T @ Vf
    sxx = (X * X).T @ Vf
    sxy = X.T @ X
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        var_i = sxx - sx * sx / n
        var_j = var_i.T
        corr = cov / np.sqrt(var_i * var_j)
    qual = (n >= min_obs) & (var_i > 1e-14) & (var_j > 1e-14) & np.isfinite(corr)
    np.fill_diagonal(qual, False)
    score = np.where(qual, corr, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    taken = np.take_along_axis(qual, order, axis=1)
    kk = order.shape[1]
    out[:, :kk] = np.where(taken, order, -1)
    return out


def _announcer_z(a, cand_t, cand_j, window=Z_WINDOW, min_obs=Z_MIN_OBS):
    cand_t = np.asarray(cand_t, dtype=np.int64)
    cand_j = np.asarray(cand_j, dtype=np.int64)
    z = np.full(cand_t.shape[0], np.nan)
    if cand_t.shape[0] == 0:
        return z
    rows = cand_t[:, None] + np.arange(-window, 0)[None, :]
    inside = rows >= 0
    vals = a[np.clip(rows, 0, None), cand_j[:, None]].astype(np.float64)
    vals = np.where(inside, vals, np.nan)
    ok = np.isfinite(vals)
    cnt = ok.sum(axis=1)
    x = np.where(ok, vals, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = x.sum(axis=1) / cnt
        ss = (np.where(ok, vals - mean[:, None], 0.0) ** 2).sum(axis=1)
        sd = np.sqrt(ss / (cnt - 1))
        zz = a[cand_t, cand_j].astype(np.float64) / sd
    good = (cnt >= min_obs) & (sd > 0) & np.isfinite(zz)
    z[good] = zz[good]
    return z


def _month_starts(dates):
    m = np.asarray(dates).astype("datetime64[M]")
    s = np.zeros(m.shape[0], dtype=bool)
    s[1:] = m[1:] != m[:-1]
    return s


def _peer_events(data, params):
    k = int(params["peers_k"])
    thr = float(params["z_threshold"])

    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float32)
    uni_raw = data.universe
    if uni_raw is None:
        raise ValueError("H-0066 needs the point-in-time liq_n universe")
    uni = np.asarray(uni_raw, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    T, N = uni.shape

    r, valid = _valid_returns(data.ret_co, data.ret_oc, data.close, listed, tradable)
    a = _market_adjusted(r, valid, uni)
    del r

    rel = ea == 1
    cand_t, cand_j = np.nonzero(rel & uni & valid)
    z = _announcer_z(a, cand_t, cand_j)
    is_ann = np.isfinite(z) & (np.abs(z) >= thr)
    ann_t = cand_t[is_ann]
    ann_j = cand_j[is_ann]
    ann_s = np.sign(z[is_ann])
    announcer = np.zeros((T, N), dtype=bool)
    announcer[ann_t, ann_j] = True

    starts = _month_starts(data.dates)
    block = np.cumsum(starts)
    S = np.zeros((T, N), dtype=np.float64)
    start_rows = np.nonzero(starts)[0]
    ann_block = block[ann_t]
    for b, m in enumerate(start_rows, start=1):
        sel = ann_block == b
        if not sel.any():
            continue
        lo = max(0, m - CORR_WINDOW)
        members = np.nonzero(uni[m - 1])[0]
        if members.shape[0] < 2:
            continue
        tab = _peer_table(a[lo:m, members], k)
        peers_of = np.full((N, tab.shape[1]), -1, dtype=np.int64)
        peers_of[members] = np.where(tab >= 0, members[np.clip(tab, 0, None)], -1)
        pt = ann_t[sel]
        pj = ann_j[sel]
        ps = ann_s[sel]
        pp = peers_of[pj]
        ok = pp >= 0
        rows = np.broadcast_to(pt[:, None], pp.shape)[ok]
        vals = np.broadcast_to(ps[:, None], pp.shape)[ok]
        np.add.at(S, (rows, pp[ok]), vals)

    idx = np.where(rel, np.arange(T, dtype=np.int32)[:, None], np.int32(-10 ** 8))
    last_rel = np.maximum.accumulate(idx, axis=0)
    del idx
    not_reported = (np.arange(T, dtype=np.int32)[:, None] - last_rel) >= NO_REPORT_WINDOW
    del last_rel

    eligible = uni & valid & ~announcer & not_reported & np.isfinite(ea)
    return np.where(eligible & (S != 0), np.sign(S), 0.0)


def events(data, params):
    side = _peer_events(data, params)
    mask = side != 0
    return mask, np.where(mask, side, 0.0)
