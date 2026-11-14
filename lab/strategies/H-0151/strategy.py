"""H-0151: healthy cheap stocks vs value traps, learned (LIQ-500, long only, monthly buffered top fraction).

Features (per stock and day, rows <= t only), each winsorized 1/99 % and z-scored across the day's scored members:
  BM   = (1 / sf1_arq_pb) / total return since the pb value became usable (pb is frozen at the filing date)
  GPA  = sf1_art_gp / sf1_arq_assets
  ACC  = (sf1_art_netinc - sf1_art_ncfo) / sf1_arq_assets
  DROA = netinc/assets at t minus the same ratio 252 rows earlier
  VOL  = std of daily total returns over the last 252 rows, >= 200 valid
Label: framework forward_return over 60 rows, ranked across members. Model: gbm via walk_forward, expanding,
refit every 126 rows. Book: monthly (last trading day of the month), buy in the top `top`, keep while in the top
2 x `top`, equal weight, gross 1, long only; flat with fewer than 20 scored members.
"""
import numpy as np
from lab.framework.api import total_return_index
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.2}

HORIZON = 60            # card ml.horizon_days
RETRAIN_EVERY = 126     # card ml.retrain_every_days
VOL_WINDOW = 252
VOL_MIN_VALID = 200
DROA_LAG = 252
MIN_SCORED = 20
WINSOR_LO = 1.0
WINSOR_HI = 99.0
GBM_FIXED = {"learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 1000, "subsample": 0.5,
             "colsample": 1.0}


# ----------------------------------------------------------------------------------------------- calendar
def decision_rows(dates):
    """Row t is a decision row if it is the last weekday of its month (next weekday by np.busday_offset is in
    another month; exchange holidays ignored). If a month's last weekday is a holiday, no row of that month is
    flagged; then the first row of the next month is the decision row (one day late, point-in-time)."""
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    month = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    last = nxt.astype("datetime64[M]") != month
    new_month = np.zeros(T, dtype=bool)
    new_month[1:] = month[1:] != month[:-1]
    prev_last = np.zeros(T, dtype=bool)
    prev_last[1:] = last[:-1]
    return last | (new_month & ~prev_last)


# ----------------------------------------------------------------------------------------------- features
def _lag_rows(x, k):
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def book_to_market(pb, idx):
    """(1/pb) moved with the total return since the first row the current pb value was usable.
    The anchor is the close of the row before that first usable row (the filing-day close), so the
    growth factor includes the return of the first usable day. pb <= 0 or missing -> NaN."""
    pb = np.asarray(pb, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.float64)
    T, N = pb.shape
    prev = np.full_like(pb, np.nan)
    prev[1:] = pb[:-1]
    changed = ~(pb == prev)            # NaN on either side counts as a change
    changed[0] = True
    rows = np.arange(T)[:, None]
    anchor = np.maximum.accumulate(np.where(changed, rows, 0), axis=0)
    base_row = np.maximum(anchor - 1, 0)
    base = np.take_along_axis(idx, base_row, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = idx / base
        inv_pb = np.where(pb > 0, 1.0 / pb, np.nan)
        bm = inv_pb / grow
    bm[~np.isfinite(bm) | ~(base > 0)] = np.nan
    return bm


def _ratio(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def realized_vol(r, valid, window=VOL_WINDOW, min_valid=VOL_MIN_VALID):
    """Sample std of daily total returns over the last `window` rows using only valid rows; NaN below min_valid."""
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(r)
    x = np.where(v, r, 0.0)
    T, N = x.shape
    c1 = np.zeros((T + 1, N))
    c2 = np.zeros((T + 1, N))
    cn = np.zeros((T + 1, N))
    c1[1:] = np.cumsum(x, axis=0)
    c2[1:] = np.cumsum(x * x, axis=0)
    cn[1:] = np.cumsum(v, axis=0)
    lo = np.maximum(np.arange(1, T + 1) - window, 0)
    hi = np.arange(1, T + 1)
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    n = cn[hi] - cn[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    vol = np.sqrt(var)
    vol[n < min_valid] = np.nan
    return vol


def winsorize_rows(x, lo=WINSOR_LO, hi=WINSOR_HI):
    """Clip each row at its own lo/hi percentile (linear interpolation) over its finite values."""
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x)
    n = fin.sum(axis=1)
    srt = np.sort(np.where(fin, x, np.inf), axis=1)
    out = x.copy()
    rows = np.nonzero(n > 0)[0]
    if rows.size == 0:
        return out
    nn = n[rows].astype(np.float64)

    def q(p):
        pos = (nn - 1.0) * p / 100.0
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n[rows] - 1)
        frac = pos - i0
        a = srt[rows, i0]
        b = srt[rows, i1]
        return a + (b - a) * frac

    qlo = q(lo)[:, None]
    qhi = q(hi)[:, None]
    out[rows] = np.clip(x[rows], qlo, qhi)
    out[~fin] = np.nan
    return out


def raw_features(data):
    ex = data.extras
    pb = ex["sf1_arq_pb"]
    assets = ex["sf1_arq_assets"]
    gp = ex["sf1_art_gp"]
    netinc = np.asarray(ex["sf1_art_netinc"], dtype=np.float64)
    ncfo = np.asarray(ex["sf1_art_ncfo"], dtype=np.float64)
    idx = total_return_index(data)
    bm = book_to_market(pb, idx)
    gpa = _ratio(gp, assets)
    acc = _ratio(netinc - ncfo, assets)
    roa = _ratio(netinc, assets)
    droa = roa - _lag_rows(roa, DROA_LAG)
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    valid = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    vol = realized_vol(r, valid)
    return [bm, gpa, acc, droa, vol]


def model_inputs(data):
    """Winsorized, cross-sectionally z-scored features on the complete-case member set, shape (T, N, 5)."""
    uni = data.universe
    feats = raw_features(data)
    complete = np.asarray(uni, dtype=bool).copy()
    for f in feats:
        complete &= np.isfinite(f)
    zs = [cs_zscore(winsorize_rows(np.where(complete, f, np.nan)), uni) for f in feats]
    return stack(*zs)


def model_scores(data, params):
    uni = data.universe
    X = model_inputs(data)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    score = np.asarray(score, dtype=np.float64)
    score[~np.asarray(uni, dtype=bool)] = np.nan
    return score


# ----------------------------------------------------------------------------------------------- book
def rank_sets(s, ok, top):
    """Return (in_top, in_keep, n_scored) for one decision row; ranks by descending score, ties by column."""
    N = s.shape[0]
    n = int(ok.sum())
    in_top = np.zeros(N, dtype=bool)
    in_keep = np.zeros(N, dtype=bool)
    if n < MIN_SCORED:
        return in_top, in_keep, n
    cols = np.nonzero(ok)[0]
    order = cols[np.lexsort((cols, -s[cols]))]
    n_top = int(np.floor(top * n + 1e-9))
    n_keep = int(np.floor(2.0 * top * n + 1e-9))
    in_top[order[:n_top]] = True
    in_keep[order[:n_keep]] = True
    return in_top, in_keep, n


def buffered_book(score, member, dec, top):
    """Monthly buffered equal-weight long book. Non-decision rows are NaN (keep) except that a held stock
    leaving the universe is set to 0 on the first row it is outside."""
    score = np.asarray(score, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    T, N = score.shape
    w = np.full((T, N), np.nan)
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        if dec[t]:
            ok = np.isfinite(score[t]) & member[t]
            in_top, in_keep, n = rank_sets(score[t], ok, top)
            new = in_top | (held & in_keep)
            row = np.zeros(N)
            k = int(new.sum())
            if k > 0:
                row[new] = 1.0 / k
            w[t] = row
            held = new
        elif held.any():
            out = held & ~member[t]
            if out.any():
                w[t, out] = 0.0
                held = held & ~out
    return w


def fresh_entries(score, member, tradable, dec, top):
    """Mechanism-test event (identical copy in diagnostic.py, kept here so the tests can reach it): at a
    decision row, in the top `top` of scored members and not in the top 2 x `top` at the previous decision."""
    score = np.asarray(score, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    T, N = score.shape
    mask = np.zeros((T, N), dtype=bool)
    prev_keep = np.zeros(N, dtype=bool)
    for t in np.nonzero(dec)[0]:
        ok = np.isfinite(score[t]) & member[t]
        in_top, in_keep, n = rank_sets(score[t], ok, top)
        mask[t] = in_top & ~prev_keep & member[t] & tradable[t]
        prev_keep = in_keep
    return mask


def target_weights(data, params):
    top = float(params["top"])
    score = model_scores(data, params)
    dec = decision_rows(data.dates)
    return buffered_book(score, data.universe, dec, top)
