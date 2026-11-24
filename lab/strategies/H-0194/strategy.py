"""H-0194: learned ranking of new insider purchase filings in LIQ-500, long the top fraction for 5 sessions,
beta-scaled hedge with the 50 most liquid members.

Pipeline (all point-in-time, row t decided after the close of t, executed at the open of t+1):
  1. events: member & tradable at t, ins_buy_n_91d rises from t-1 to t, and no such candidate in t-5..t-1
  2. seven features on event rows only (NaN elsewhere, NaN when any input is missing)
  3. walk-forward gbm (framework) on event rows, label = cross-sectional rank of the 5-day forward return
  4. entry: score >= (1 - top) quantile of the scored events of rows t-252..t-1 (at least 50 of them)
  5. book: 0.08 long per entry, 0.08 x beta short spread over the 50 most liquid members not held long,
     exit at the open of t+6, early close when the stock leaves the universe or is not tradable,
     pro-rata scaling when the slot gross exceeds 1.
"""
import numpy as np
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

PARAMS = {"n_estimators": 100, "top": 0.5}

HORIZON = 5            # label horizon and hold (exit at the open of t+6)
RETRAIN_EVERY = 63     # card ml.retrain_every_days
DEDUP = 5              # no candidate in t-5..t-1
Q_WINDOW = 252         # trailing window of prior scored events for the entry quantile
MIN_PRIOR = 50         # minimum prior scored events
SLOT_W = 0.08          # long weight per entry
N_HEDGE = 50           # liquid hedge basket size
DV_WINDOW = 63         # median dollar volume window
BETA_WINDOW = 252
BETA_MIN = 126
BETA_LO, BETA_HI = 0.5, 1.5
GBM_FIXED = {"num_leaves": 8, "min_child_samples": 50, "learning_rate": 0.05}


# ----------------------------------------------------------------------------------------------- inputs
def daily_returns(data):
    """(T, N) daily total return close-to-close, NaN where the instrument is not listed."""
    r = (1.0 + np.asarray(data.ret_co, dtype=np.float64)) * (1.0 + np.asarray(data.ret_oc, dtype=np.float64)) - 1.0
    listed = np.asarray(data.listed, dtype=bool)
    return np.where(listed & np.isfinite(r), r, np.nan)


def _window_sum(x, n):
    """Trailing sum over rows t-n+1..t of x (NaN -> 0) and the count of finite values."""
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    cc = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(v, axis=0, out=cs[1:])
    np.cumsum(valid, axis=0, out=cc[1:])
    lo = np.maximum(np.arange(1, T + 1) - n, 0)
    s = cs[1:] - cs[lo]
    c = cc[1:] - cc[lo]
    return s, c


def trailing_compound_return(r, n):
    """n-session total return up to the close of t; NaN unless all n daily returns are known."""
    lr = np.log1p(np.where(np.isfinite(r), np.maximum(r, -0.999999), np.nan))
    s, c = _window_sum(lr, n)
    out = np.expm1(s)
    out[c < n] = np.nan
    return out


def trailing_vol(r, n):
    """n-session realized volatility (population std) of daily returns; NaN unless all n are known."""
    s, c = _window_sum(r, n)
    s2, _ = _window_sum(r * r, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / c
        var = s2 / c - mean * mean
    out = np.sqrt(np.maximum(var, 0.0))
    out[c < n] = np.nan
    return out


def row_median(x, mask):
    """Median over the columns where mask is true and x is finite, per row (NaN if none)."""
    xm = np.where(mask & np.isfinite(x), x, np.nan)
    out = np.full(x.shape[0], np.nan)
    rows = np.isfinite(xm).any(axis=1)
    if rows.any():
        out[rows] = np.nanmedian(xm[rows], axis=1)
    return out


# ----------------------------------------------------------------------------------------------- events
def detect_events(buy_n91, uni, tradable, dedup=DEDUP):
    """Candidate: member & tradable at t and ins_buy_n_91d[t] > ins_buy_n_91d[t-1] (NaN never counts).
    Event: candidate with no candidate of the same stock in rows t-dedup..t-1 (a burst counts once)."""
    b = np.asarray(buy_n91, dtype=np.float64)
    T = b.shape[0]
    rise = np.zeros(b.shape, dtype=bool)
    if T > 1:
        with np.errstate(invalid="ignore"):
            rise[1:] = b[1:] > b[:-1]
    cand = rise & np.asarray(uni, dtype=bool) & np.asarray(tradable, dtype=bool)
    cs = np.zeros((T + 1,) + b.shape[1:], dtype=np.int32)
    np.cumsum(cand, axis=0, out=cs[1:])
    t = np.arange(T)
    lo = np.maximum(t - dedup, 0)
    prior = cs[t] - cs[lo]          # candidates in rows lo..t-1
    return cand & (prior == 0)


def build_features(data, event, uni):
    """Seven features on event rows, NaN elsewhere and where any input is missing."""
    ex = data.extras
    # (6) 21-session total return minus the members' median, (7) 63-session realized vol; z-scored over members
    r = daily_returns(data)
    r21 = trailing_compound_return(r, 21)
    rel21 = r21 - row_median(r21, uni)[:, None]
    del r21
    z6 = np.asarray(cs_zscore(rel21, uni), dtype=np.float32)
    del rel21
    vol63 = trailing_vol(r, 63)
    del r
    z7 = np.asarray(cs_zscore(vol63, uni), dtype=np.float32)
    del vol63

    # (1)-(5) insider features, fed raw (float32, as delivered)
    bv91 = np.asarray(ex["ins_buy_value_91d"], dtype=np.float32)
    bn91 = np.asarray(ex["ins_buy_n_91d"], dtype=np.float32)
    bn365 = np.asarray(ex["ins_buy_n_365d"], dtype=np.float32)
    sv182 = np.asarray(ex["ins_sell_value_182d"], dtype=np.float32)
    mcap = np.asarray(ex["sf1_arq_marketcap"], dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcap = np.where(np.isfinite(mcap) & (mcap > 0), mcap, np.float32(np.nan))
        f1 = np.log1p(np.where(bv91 >= 0, bv91, np.float32(np.nan)))
        f2 = bv91 / mcap
        f4 = bn365 - bn91
        f5 = sv182 / mcap
    del mcap
    feats = [f1, f2, bn91, f4, f5, z6, z7]
    complete = event.copy()
    for f in feats:
        complete &= np.isfinite(f)
    out = []
    for i in range(len(feats)):
        out.append(np.where(complete, feats[i], np.float32(np.nan)).astype(np.float32, copy=False))
        feats[i] = None
    return out, complete


def model_scores(data, params, uni):
    """Event mask, out-of-sample gbm scores on complete event rows (NaN elsewhere)."""
    if uni is None:
        raise ValueError("H-0194 needs the point-in-time LIQ-500 universe")
    ex = data.extras
    tradable = np.asarray(data.tradable, dtype=bool)
    event = detect_events(ex["ins_buy_n_91d"], uni, tradable)
    feats, complete = build_features(data, event, uni)
    X = stack(*feats)
    del feats
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = dict(GBM_FIXED)
    gbm["n_estimators"] = int(params["n_estimators"])
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    del X, y
    score = np.asarray(score, dtype=np.float64)
    score = np.where(complete & np.isfinite(score), score, np.nan)
    return event, score


# ----------------------------------------------------------------------------------------------- entries
def select_entries(score, top, window=Q_WINDOW, min_prior=MIN_PRIOR):
    """Entry at (t, j): finite score >= the (1 - top) quantile of all finite scores in rows t-window..t-1,
    which must hold at least min_prior scores."""
    T = score.shape[0]
    entry = np.zeros(score.shape, dtype=bool)
    rows, cols = np.nonzero(np.isfinite(score))        # row-major: sorted by row
    if rows.size == 0:
        return entry
    vals = score[rows, cols]
    q = 1.0 - float(top)
    for t in np.unique(rows):
        lo = np.searchsorted(rows, t - window, side="left")
        hi = np.searchsorted(rows, t, side="left")
        if hi - lo < min_prior:
            continue
        thr = np.quantile(vals[lo:hi], q)
        a = np.searchsorted(rows, t, side="left")
        b = np.searchsorted(rows, t, side="right")
        sel = vals[a:b] >= thr
        entry[t, cols[a:b][sel]] = True
    return entry


# ----------------------------------------------------------------------------------------------- book
def median_dollar_volume(dvol, t, cols, window=DV_WINDOW):
    lo = max(0, t - window + 1)
    blk = np.asarray(dvol[lo:t + 1][:, cols], dtype=np.float64)
    out = np.full(len(cols), np.nan)
    ok = np.isfinite(blk).any(axis=0)
    if ok.any():
        out[ok] = np.nanmedian(blk[:, ok], axis=0)
    return out


def hedge_basket(dvol, uni_row, t, exclude, n=N_HEDGE):
    """The n members at t with the highest 63-day median dollar volume, excluding the given columns."""
    members = np.nonzero(uni_row)[0]
    if exclude:
        members = members[~np.isin(members, np.fromiter(exclude, dtype=np.int64))]
    if members.size == 0:
        return members
    med = median_dollar_volume(dvol, t, members)
    ok = np.isfinite(med)
    members, med = members[ok], med[ok]
    order = np.lexsort((members, -med))                 # highest first, ties by column
    return members[order[:n]]


def slot_beta(r, t, j, basket, window=BETA_WINDOW, min_obs=BETA_MIN):
    """252-session OLS beta of j's daily return on the basket's equal-weight return, clipped; 1.0 if < 126."""
    if basket.size == 0:
        return 1.0
    lo = max(0, t - window + 1)
    rj = r[lo:t + 1, j]
    rb_blk = r[lo:t + 1][:, basket]
    cnt = np.isfinite(rb_blk).sum(axis=1)
    rb = np.where(cnt > 0, np.nansum(rb_blk, axis=1) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(rj) & np.isfinite(rb)
    if ok.sum() < min_obs:
        return 1.0
    x = rb[ok] - rb[ok].mean()
    yv = rj[ok] - rj[ok].mean()
    var = float(np.dot(x, x))
    if var <= 0:
        return 1.0
    return float(np.clip(np.dot(x, yv) / var, BETA_LO, BETA_HI))


def build_book(entry, uni, tradable, r, dvol, hold=HORIZON, slot_w=SLOT_W, n_hedge=N_HEDGE):
    """Slot book. A slot entered at decision row t is held on rows t..t+hold-1 and closed on row t+hold
    (exit at the open of t+hold+1 = t+6). Closed early from the first row where the long is not a member
    or not tradable; a hedge name that leaves the universe is dropped from its slot."""
    T, N = entry.shape
    w = np.zeros((T, N), dtype=np.float64)
    slots = []           # dicts: j, t0, hcols (array), hw (array), alive (bool array)
    for t in range(T):
        keep = []
        for s in slots:
            if t - s["t0"] >= hold:
                continue
            if not (uni[t, s["j"]] and tradable[t, s["j"]]):
                continue
            s["alive"] &= uni[t, s["hcols"]]
            keep.append(s)
        slots = keep
        new = np.nonzero(entry[t])[0]
        if new.size:
            held = {s["j"] for s in slots}
            new = [int(j) for j in new if int(j) not in held and uni[t, j] and tradable[t, j]]
            exclude = held | set(new)
            if new:
                basket = hedge_basket(dvol, uni[t], t, exclude, n_hedge)
                for j in new:
                    beta = slot_beta(r, t, j, basket)
                    hw = np.full(basket.size, slot_w * beta / max(basket.size, 1))
                    slots.append({"j": j, "t0": t, "hcols": basket, "hw": hw,
                                  "alive": np.ones(basket.size, dtype=bool)})
        if not slots:
            continue
        gross = 0.0
        row = np.zeros(N, dtype=np.float64)
        for s in slots:
            row[s["j"]] += slot_w
            hw = np.where(s["alive"], s["hw"], 0.0)
            np.subtract.at(row, s["hcols"], hw)
            gross += slot_w + float(hw.sum())
        if gross > 1.0:
            row /= gross
        w[t] = row
    w[~uni] = 0.0
    return w


def target_weights(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0194 needs the point-in-time LIQ-500 universe")
    uni = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    event, score = model_scores(data, params, uni)
    entry = select_entries(score, params["top"])
    r = daily_returns(data)
    dvol = np.asarray(data.dollar_volume, dtype=np.float64)
    return build_book(entry, uni, tradable, r, dvol)
