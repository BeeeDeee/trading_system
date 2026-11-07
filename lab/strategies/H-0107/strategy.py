"""H-0107: wrapper-specific ETF flow-day reversal, learned (LightGBM), S&P 500 core plus 5-day event slots.

Row t is decided after the close of t from rows 0..t only; the engine executes at the open of t+1.
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, walk_forward

PARAMS = {"flow_sigma": 2.0, "n_estimators": 100}

MARKET = "SPY"
WRAPPERS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SMH", "IBB", "IYR", "IGV")

BETA_WIN = 126        # (a)/(b) trailing window t-125..t
MIN_OBS_LONG = 100    # (a) at least 100 valid returns (also used for the (b) correlation window)
BASE_WIN = 63         # (c) baselines over t-63..t-1
MIN_OBS_BASE = 50     # minimum valid values in a 63-session baseline window (implementation choice)
MIN_CORR = 0.30       # (b)
VOL_MULT = 2.0        # (c)
HORIZON = 5           # (e)
RETRAIN = 126         # (e)
GBM_FIXED = {"num_leaves": 8, "learning_rate": 0.05, "min_child_samples": 200}
SLOT_W = 0.03         # (g)
SLOT_DAYS = 5         # (g)
SLOT_CAP = 0.50       # (g)
CORE_WIN = 63         # (g) liquidity proxy window t-62..t
VAR_TOL = 1e-12       # numerical floor for a per-observation variance (constant series have no beta/correlation)


# ----------------------------------------------------------------------------------------------- helpers
def _csum(x):
    """Prefix sums with a leading zero row (float64)."""
    c = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    return c


def _wsum(c, w, lag=0):
    """Rolling sum over rows t-lag-w+1 .. t-lag from prefix sums c (works for any number of rows)."""
    T = c.shape[0] - 1
    hi = np.clip(np.arange(T) + 1 - lag, 0, T)
    lo = np.clip(hi - w, 0, T)
    return c[hi] - c[lo]


def _rsum(x, w, lag=0):
    return _wsum(_csum(x), w, lag)


def _returns(data, cols):
    """Close-to-close total return (1+ret_co)(1+ret_oc)-1; NaN unless a price exists at t and t-1."""
    rc = np.asarray(data.ret_co[:, cols], dtype=np.float64)
    ro = np.asarray(data.ret_oc[:, cols], dtype=np.float64)
    r = (1.0 + rc) * (1.0 + ro) - 1.0
    px = np.isfinite(np.asarray(data.close[:, cols], dtype=np.float64))
    v = px.copy()
    v[0] = False
    v[1:] &= px[:-1]
    v &= np.isfinite(r)
    return np.where(v, r, np.nan)


def _beta_resid(r, rm, w=BETA_WIN, min_obs=MIN_OBS_LONG):
    """Residual e_t = r_t - beta_t * rm_t with beta_t the OLS slope over t-w+1..t (>= min_obs pairs)."""
    M = np.isfinite(r) & np.isfinite(rm)[:, None]
    x = np.where(M, rm[:, None], 0.0)
    y = np.where(M, r, 0.0)
    n = _rsum(M.astype(np.float64), w)
    sx = _rsum(x, w)
    sy = _rsum(y, w)
    sxx = _rsum(x * x, w)
    sxy = _rsum(x * y, w)
    del x, y, M
    with np.errstate(divide="ignore", invalid="ignore"):
        varx = sxx - sx * sx / n
        cov = sxy - sx * sy / n
        beta = np.where((n >= min_obs) & (varx > VAR_TOL * n), cov / varx, np.nan)
    return r - beta * rm[:, None]


def _prior_std(e, w=BASE_WIN, min_obs=MIN_OBS_BASE):
    """Sample std (ddof 1) of e over rows t-w..t-1 with at least min_obs valid values, else NaN."""
    M = np.isfinite(e)
    z = np.where(M, e, 0.0)
    n = _rsum(M.astype(np.float64), w, lag=1)
    s = _rsum(z, w, lag=1)
    ss = _rsum(z * z, w, lag=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (ss - s * s / n) / (n - 1.0)
        out = np.sqrt(np.maximum(var, 0.0))
    return np.where(n >= min_obs, out, np.nan)


def _median_axis0(win, min_obs):
    """Median along axis 0 ignoring NaN; NaN where fewer than min_obs valid values."""
    cnt = np.isfinite(win).sum(axis=0)
    s = np.sort(win, axis=0)          # NaN sorts last
    lo = np.clip((cnt - 1) // 2, 0, None)
    hi = np.clip(cnt // 2, 0, max(win.shape[0] - 1, 0))
    lo = np.minimum(lo, max(win.shape[0] - 1, 0))
    a = np.take_along_axis(s, lo[None], axis=0)[0]
    b = np.take_along_axis(s, hi[None], axis=0)[0]
    med = 0.5 * (a + b)
    return np.where(cnt >= max(min_obs, 1), med, np.nan)


def _prior_median(x, w=BASE_WIN, min_obs=MIN_OBS_BASE, chunk=256):
    """Median of x over rows t-w..t-1 (NaN ignored, at least min_obs valid values), any number of rows."""
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    pad = np.full((w,) + x.shape[1:], np.nan, dtype=np.float32)
    xp = np.concatenate([pad, x], axis=0)          # xp[t + w] = x[t]; window of t = xp[t .. t+w-1]
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        win = np.stack([xp[a + i:b + i] for i in range(w)], axis=0)
        out[a:b] = _median_axis0(win, min_obs)
    return out


def _positive(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x) & (x > 0), x, np.nan)


# ----------------------------------------------------------------------------------------------- signal
def wrapper_flow_days(ek, dvk, sigma):
    """(c) Down-/up-flow days of each wrapper: dv_t >= 2 x median(t-63..t-1) and e_t beyond sigma x std(t-63..t-1)."""
    medk = _prior_median(dvk).astype(np.float64)
    sdk = _prior_std(ek)
    with np.errstate(invalid="ignore", divide="ignore"):
        volok = np.isfinite(dvk) & np.isfinite(medk) & (medk > 0) & (dvk >= VOL_MULT * medk)
        ok = volok & np.isfinite(ek) & np.isfinite(sdk) & (sdk > 0)
        down = ok & (ek <= -sigma * sdk)
        up = ok & (ek >= sigma * sdk)
        intensity = np.where(np.isfinite(medk) & (medk > 0), np.log(dvk / medk), np.nan)
    return down, up, intensity


def assign_wrappers(ej, ek, w=BETA_WIN, min_obs=MIN_OBS_LONG):
    """(b) For each stock and row: wrapper with the highest trailing residual correlation, its slope b_jk."""
    T, n = ej.shape
    best_c = np.full((T, n), -np.inf)
    best_k = np.zeros((T, n), dtype=np.int64)
    best_b = np.full((T, n), np.nan)
    fj = np.isfinite(ej)
    for i in range(ek.shape[1]):
        e2 = ek[:, i]
        M = fj & np.isfinite(e2)[:, None]
        a = np.where(M, ej, 0.0)
        b = np.where(M, e2[:, None], 0.0)
        cnt = _rsum(M.astype(np.float64), w)
        sa = _rsum(a, w)
        sb = _rsum(b, w)
        saa = _rsum(a * a, w)
        sbb = _rsum(b * b, w)
        sab = _rsum(a * b, w)
        del a, b, M
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sab - sa * sb / cnt
            va = saa - sa * sa / cnt
            vb = sbb - sb * sb / cnt
            ok = (cnt >= min_obs) & (va > VAR_TOL * cnt) & (vb > VAR_TOL * cnt)
            corr = np.where(ok, cov / np.sqrt(va * vb), np.nan)
            slope = np.where(ok, cov / vb, np.nan)
            better = ok & np.isfinite(corr) & (corr > best_c)
        best_c = np.where(better, corr, best_c)
        best_k = np.where(better, i, best_k)
        best_b = np.where(better, slope, best_b)
    assigned = np.isfinite(best_c) & (best_c >= MIN_CORR)
    best_c = np.where(assigned, best_c, np.nan)
    best_b = np.where(assigned, best_b, np.nan)
    return assigned, best_k, best_c, best_b


def top_half_events(cand, score):
    """(f) Per date: candidates with a score; >= 2 of them -> the top ceil(n/2) by score are events."""
    T, N = cand.shape
    ev = np.zeros((T, N), dtype=bool)
    c = cand & np.isfinite(score)
    counts = c.sum(axis=1)
    for t in np.nonzero(counts >= 2)[0]:
        idx = np.nonzero(c[t])[0]
        k = (len(idx) + 1) // 2
        order = np.argsort(-score[t, idx], kind="stable")
        ev[t, idx[order[:k]]] = True
    return ev


def flow_pipeline(data, params):
    """Rules (a)-(f). Returns (events, flow_rows, down_candidates, score), all (T, N)."""
    sigma = float(params["flow_sigma"])
    n_est = int(params["n_estimators"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0107 needs the point-in-time S&P 500 universe (data.universe)")
    uni = np.asarray(uni, dtype=bool)
    T, N = uni.shape
    im = data.col(MARKET)
    ik = [data.col(k) for k in WRAPPERS]
    ea_all = data.extras["ea_reaction_day"]
    ne_all = data.extras["nonearn_8k_reaction_day"]
    etf = set([im] + ik)
    # columns that are a member on some row of this view; each column's features use only its own rows <= t
    # and the wrappers, so leaving out never-member columns changes nothing on rows <= t (no layout effect:
    # everything is scattered back into the full (T, N) layout before the model)
    cols = np.array([j for j in range(N) if j not in etf and uni[:, j].any()], dtype=np.int64)

    flow_full = np.zeros((T, N), dtype=bool)
    cand_full = np.zeros((T, N), dtype=bool)
    X = None

    rm = _returns(data, [im])[:, 0]
    rk = _returns(data, ik)
    ek = _beta_resid(rk, rm)                                     # (a) wrapper residuals
    dvk = _positive(data.dollar_volume[:, ik])
    down_k, up_k, inten_k = wrapper_flow_days(ek, dvk, sigma)    # (c)

    if len(cols) > 0:
        mem = uni[:, cols]
        rj = _returns(data, cols)
        ej = _beta_resid(rj, rm)                                 # (a) stock residuals
        del rj
        assigned, kk, tie, bjk = assign_wrappers(ej, ek)         # (b)

        ek_j = np.take_along_axis(ek, kk, axis=1)
        down_j = np.take_along_axis(down_k, kk, axis=1) & assigned
        up_j = np.take_along_axis(up_k, kk, axis=1) & assigned
        inten_j = np.take_along_axis(inten_k, kk, axis=1)
        dvk_j = np.take_along_axis(dvk, kk, axis=1)

        dvj = _positive(data.dollar_volume[:, cols])
        medj = _prior_median(dvj).astype(np.float64)
        medj = np.where(np.isfinite(medj) & (medj > 0), medj, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            push = bjk * ek_j                                    # feature 1
            idio = ej - push                                     # feature 2
            rel = np.log(dvk_j / medj)                           # feature 4
            own = np.log(dvj / medj)                             # feature 6
        risk = _prior_std(ej)                                    # feature 7
        del ek_j, dvk_j, dvj, medj
        raw = [push, idio, inten_j, rel, tie, own, risk]
        na = ~assigned
        Z = []
        for f in raw:
            f = np.where(na, np.nan, f).astype(np.float32)
            Z.append(np.asarray(cs_zscore(f, mem), dtype=np.float32))
        del raw, push, idio, inten_j, rel, tie, own, risk
        complete = np.ones((T, len(cols)), dtype=bool)
        for z in Z:
            complete &= np.isfinite(z)

        ea = np.asarray(ea_all[:, cols])
        ne = np.asarray(ne_all[:, cols])
        news = (ea == 1) | (ne == 1)                             # NaN flags count as no filing
        flow = mem & assigned & (down_j | up_j) & ~news & complete   # (d) flow rows (both directions)
        cand = flow & down_j                                     # (d) down-push candidates
        flow_full[:, cols] = flow
        cand_full[:, cols] = cand
        X = np.full((T, N, 7), np.nan, dtype=np.float32)   # allocated late to keep the peak memory low
        for i, z in enumerate(Z):
            X[:, cols, i] = np.where(flow, z, np.nan)
        del Z, ej, assigned, kk, bjk, mem
    if X is None:
        X = np.full((T, N, 7), np.nan, dtype=np.float32)

    # (e) model: label = cross-sectional rank among all members of the open(t+1) -> open(t+6) return
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm_params = dict(GBM_FIXED)
    gbm_params["n_estimators"] = n_est
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm_params,
                         retrain_every=RETRAIN, universe=flow_full)
    score = np.asarray(score, dtype=np.float64)
    events = top_half_events(cand_full, score)                   # (f)
    return events, flow_full, cand_full, score


# ----------------------------------------------------------------------------------------------- book
def core_base(dv, mem, t, win=CORE_WIN):
    """(g) 63-session (t-62..t) median dollar volume of the current members; window shortened at the view start."""
    lo = max(0, t - win + 1)
    block = np.asarray(dv[lo:t + 1], dtype=np.float64)
    block = np.where(np.isfinite(block) & (block > 0), block, np.nan)
    med = _median_axis0(block, 1)
    return np.where(mem & np.isfinite(med), med, 0.0)


def build_book(events, uni, dv, core_days):
    """(g) w = (1 - T) c + slots; slots 3 % for 5 sessions, renewals extend, new entries scaled to the 50 % cap."""
    T, N = uni.shape
    W = np.zeros((T, N), dtype=np.float64)
    base = np.zeros(N)
    slot_w = np.zeros(N)
    slot_end = np.full(N, -1, dtype=np.int64)
    for t in range(T):
        mem = uni[t]
        if core_days[t]:
            base = core_base(dv, mem, t)
        c = np.where(mem, base, 0.0)                 # leavers dropped, rest renormalized
        s = c.sum()
        if s > 0:
            c = c / s
        active = (slot_end >= t) & mem               # expired slots and slots of index leavers are closed
        slot_w = np.where(active, slot_w, 0.0)
        slot_end = np.where(active, slot_end, -1)
        e = events[t] & mem
        renew = e & active
        slot_end[renew] = t + SLOT_DAYS - 1          # renewed, not doubled
        new = e & ~active
        nn = int(new.sum())
        if nn > 0:
            room = max(0.0, SLOT_CAP - float(slot_w.sum()))
            per = min(SLOT_W, room / nn)
            if per > 0:
                slot_w[new] = per
                slot_end[new] = t + SLOT_DAYS - 1
        tot = float(slot_w.sum())
        W[t] = (1.0 - tot) * c + slot_w
    return W


def target_weights(data, params):
    events = flow_pipeline(data, params)[0]
    uni = np.asarray(data.universe, dtype=bool)
    core_days = np.asarray(period_starts(data.dates, "M"), dtype=bool).copy()
    if len(core_days) > 0:
        core_days[0] = True                          # the first row is the first decision date of its month
    return build_book(events, uni, data.dollar_volume, core_days)
