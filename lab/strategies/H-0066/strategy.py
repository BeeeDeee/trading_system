"""H-0066 Earnings-news transfer to not-yet-reporting peers (LIQ-500, long/short).

Decision after the close of row t with rows 0..t only; the engine fills at the open of t+1.
"""
import numpy as np

PARAMS = {"hold_days": 5, "peers_k": 10, "z_threshold": 3.0}

CORR_WINDOW = 252        # sessions of market-adjusted returns for the peer correlation
CORR_MIN_OBS = 200       # common valid observations a pair needs
Z_WINDOW = 63            # sessions ending t-1 for the announcer's sd
Z_MIN_OBS = 40           # valid observations the sd needs
NO_REPORT_WINDOW = 40    # sessions ending t without an own release session
SLOT_WEIGHT = 0.01       # weight per open position
MAX_SLOTS = 100          # above this, every position gets 1 / N_open


# ----------------------------------------------------------------------------------------------
# building blocks (pure numpy, tested separately)
# ----------------------------------------------------------------------------------------------

def valid_returns(ret_co, ret_oc, close, listed, tradable):
    """Close-to-close total return r and the mask of rows where it is a real return.

    A return is valid on t when the stock is listed and traded on t and has a close on t and t-1.
    """
    co = np.asarray(ret_co, dtype=np.float32)
    oc = np.asarray(ret_oc, dtype=np.float32)
    r = (np.float32(1.0) + co) * (np.float32(1.0) + oc) - np.float32(1.0)
    fin_close = np.isfinite(close)
    prev_close = np.zeros_like(fin_close)
    prev_close[1:] = fin_close[:-1]
    valid = (np.asarray(listed, dtype=bool) & np.asarray(tradable, dtype=bool) & fin_close & prev_close
             & np.isfinite(r))
    return r, valid


def market_adjusted(r, valid, uni):
    """a = r minus the equal-weighted mean r of the universe members with a valid return that row."""
    in_mean = valid & uni
    cnt = in_mean.sum(axis=1)
    tot = np.where(in_mean, r, np.float32(0.0)).sum(axis=1, dtype=np.float64)
    mm = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    mm[ok] = tot[ok] / cnt[ok]
    a = np.where(valid, r - mm[:, None].astype(np.float32), np.float32(np.nan)).astype(np.float32)
    return a


def peer_table(window, k, min_obs=CORR_MIN_OBS):
    """Top-k Pearson-correlated peers per column of `window` (rows x M, NaN = no observation).

    Pairwise-complete correlation; a pair qualifies with at least `min_obs` common observations and
    non-zero variance on them. Returns an (M, k) int array of column positions, -1 where fewer than k
    columns qualify. Ties are broken by the lower column position (stable sort).
    """
    W = np.asarray(window, dtype=np.float64)
    M = W.shape[1]
    out = np.full((M, k), -1, dtype=np.int64)
    if M < 2 or W.shape[0] == 0:
        return out
    V = np.isfinite(W)
    X = np.where(V, W, 0.0)
    Vf = V.astype(np.float64)
    n = Vf.T @ Vf                      # common observations
    sx = X.T @ Vf                      # sx[i, j] = sum of x_i where both i and j are valid
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


def announcer_z(a, cand_t, cand_j, window=Z_WINDOW, min_obs=Z_MIN_OBS):
    """z = a[t, j] / sd(a[t-window .. t-1, j]) (ddof 1, >= min_obs valid values), NaN otherwise."""
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


def month_starts(dates):
    """True on the first row of a calendar month (point-in-time: uses the dates of rows t-1 and t)."""
    m = np.asarray(dates).astype("datetime64[M]")
    s = np.zeros(m.shape[0], dtype=bool)
    s[1:] = m[1:] != m[:-1]
    return s


def build_book(event_side, ea, uni, tradable, listed, hold_days):
    """Event book: open on the event row, hold `hold_days` decision rows, close early on own release,
    universe exit or no trading. Weight side * 0.01, or side / N_open above 100 open positions."""
    T, N = event_side.shape
    W = np.zeros((T, N), dtype=np.float64)
    side = np.zeros(N, dtype=np.float64)
    entry = np.full(N, -1, dtype=np.int64)
    for t in range(T):
        held = side != 0
        if held.any():
            stop = held & ((t - entry >= hold_days) | (ea[t] == 1) | ~uni[t] | ~tradable[t] | ~listed[t])
            side[stop] = 0.0
        ev = event_side[t] != 0
        if ev.any():
            side[ev] = event_side[t, ev]
            entry[ev] = t
        n_open = np.count_nonzero(side)
        if n_open:
            w = SLOT_WEIGHT if n_open <= MAX_SLOTS else 1.0 / n_open
            W[t] = side * w
    return W


# ----------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------

def peer_events(data, params):
    """(T, N) float: +1 / -1 where a peer event fires after the close of row t, 0 elsewhere."""
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

    r, valid = valid_returns(data.ret_co, data.ret_oc, data.close, listed, tradable)
    a = market_adjusted(r, valid, uni)
    del r

    # announcers: own release session, member, valid return, |z| >= threshold
    rel = ea == 1
    cand_t, cand_j = np.nonzero(rel & uni & valid)
    z = announcer_z(a, cand_t, cand_j)
    is_ann = np.isfinite(z) & (np.abs(z) >= thr)
    ann_t = cand_t[is_ann]
    ann_j = cand_j[is_ann]
    ann_s = np.sign(z[is_ann])
    announcer = np.zeros((T, N), dtype=bool)
    announcer[ann_t, ann_j] = True

    # monthly peer maps: computed on the first row of each month from the CORR_WINDOW rows ending
    # at the previous row (= the last session of the previous month), used for the whole month
    starts = month_starts(data.dates)
    block = np.cumsum(starts)              # 0 = before the first month boundary: no map
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
        tab = peer_table(a[lo:m, members], k)
        peers_of = np.full((N, tab.shape[1]), -1, dtype=np.int64)
        peers_of[members] = np.where(tab >= 0, members[np.clip(tab, 0, None)], -1)
        pt = ann_t[sel]
        pj = ann_j[sel]
        ps = ann_s[sel]
        pp = peers_of[pj]                                   # (n_ann, k)
        ok = pp >= 0
        rows = np.broadcast_to(pt[:, None], pp.shape)[ok]
        vals = np.broadcast_to(ps[:, None], pp.shape)[ok]
        np.add.at(S, (rows, pp[ok]), vals)

    # not reported yet: no own release session in the NO_REPORT_WINDOW rows ending t
    idx = np.where(rel, np.arange(T, dtype=np.int32)[:, None], np.int32(-10 ** 8))
    last_rel = np.maximum.accumulate(idx, axis=0)
    del idx
    not_reported = (np.arange(T, dtype=np.int32)[:, None] - last_rel) >= NO_REPORT_WINDOW
    del last_rel

    eligible = uni & valid & ~announcer & not_reported & np.isfinite(ea)
    side = np.where(eligible & (S != 0), np.sign(S), 0.0)
    return side


def target_weights(data, params):
    hold = int(params["hold_days"])
    side = peer_events(data, params)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float32)
    uni = np.asarray(data.universe, dtype=bool)
    return build_book(side, ea, uni, np.asarray(data.tradable, dtype=bool),
                      np.asarray(data.listed, dtype=bool), hold)
