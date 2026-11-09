"""H-0106 mechanism_test: flow-day entries.

Event at decision date t for stock i: i newly enters the long leg (top `top` fraction of the model scores, not in
it at t-1; side +1) or the short leg (bottom `top` fraction, not in it at t-1; side -1), and its assigned sector
SPDR had a flow session at t: dollar volume >= 2 x its median over t-63..t-1 and |excess return over SPY| on t
>= 2 x the standard deviation of that excess return over t-63..t-1.
The feature / walk_forward pipeline is a copy of strategy.py (same code, same parameters).
"""
import numpy as np
from lab.framework.api import period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

MARKET_ETF = "SPY"
SECTOR_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
CORR_WINDOW = 252
MIN_PAIRS = 189
FLOW_WINDOW = 63
SUM_WINDOW = 5
HORIZON = 5
REFIT = 126
MIN_NAMES = 100
GBM_FIXED = {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 500}
FLOW_DV_MULT = 2.0       # ETF dollar volume >= 2 x its 63-session median
FLOW_RET_SIGMA = 2.0     # |ETF excess return| >= 2 x its 63-session standard deviation


# ---------------------------------------------------------------- copy of the strategy pipeline
def _cc_return(data):
    r = (1.0 + np.asarray(data.ret_co, np.float64)) * (1.0 + np.asarray(data.ret_oc, np.float64)) - 1.0
    valid = np.asarray(data.listed, bool) & np.isfinite(np.asarray(data.close, np.float64))
    return np.where(valid, r, np.nan)


def _trailing_sum(x, n):
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, np.nan)
    T = x.shape[0]
    if T >= n:
        acc = x[n - 1:].copy()
        for j in range(1, n):
            acc = acc + x[n - 1 - j:T - j]
        out[n - 1:] = acc
    return out


def _lagged_median(v, need, window=FLOW_WINDOW, chunk=128):
    v = np.asarray(v, np.float64)
    need = np.asarray(need, bool)
    T, M = v.shape
    out = np.full((T, M), np.nan)
    for start in range(window, T, chunk):
        stop = min(T, start + chunk)
        cols = np.nonzero(need[start:stop].any(axis=0))[0]
        if cols.size == 0:
            continue
        sub = v[start - window:stop][:, cols]
        rows = stop - start
        win = np.stack([sub[j:j + rows] for j in range(window)], axis=-1)
        n = np.isfinite(win).sum(axis=-1)
        s = np.sort(win, axis=-1)
        lo = np.clip((n - 1) // 2, 0, window - 1)[..., None]
        hi = np.clip(n // 2, 0, window - 1)[..., None]
        med = 0.5 * (np.take_along_axis(s, lo, axis=-1)[..., 0] + np.take_along_axis(s, hi, axis=-1)[..., 0])
        med[n == 0] = np.nan
        out[start:stop, cols] = med
    return np.where(need, out, np.nan)


def _lagged_std(v, window=FLOW_WINDOW):
    """Sample standard deviation (ddof 1) of the finite values of v over rows t-window..t-1; NaN before
    `window` rows exist or with fewer than 2 finite values."""
    v = np.asarray(v, np.float64)
    T, M = v.shape
    out = np.full((T, M), np.nan)
    if T > window:
        rows = T - window
        win = np.stack([v[j:j + rows] for j in range(window)], axis=-1)    # rows t-window+j, t = window..T-1
        ok = np.isfinite(win)
        n = ok.sum(axis=-1)
        w0 = np.where(ok, win, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = w0.sum(axis=-1) / n
            dev = np.where(ok, win - mean[..., None], 0.0)
            var = (dev * dev).sum(axis=-1) / (n - 1)
        out[window:] = np.where(n >= 2, np.sqrt(var), np.nan)
    return out


def _sector_assignment(e, x, starts, window=CORR_WINDOW, min_pairs=MIN_PAIRS):
    e = np.asarray(e, np.float64)
    x = np.asarray(x, np.float64)
    starts = np.asarray(starts, bool)
    T, N = e.shape
    idx = np.nonzero(starts)[0]
    km = np.full((len(idx) + 1, N), -1, np.int64)
    bm = np.full((len(idx) + 1, N), np.nan)
    rm = np.full((len(idx) + 1, N), np.nan)
    for m, t0 in enumerate(idx):
        lo = max(0, t0 - window + 1)
        E = e[lo:t0 + 1]
        X = x[lo:t0 + 1]
        ve = np.isfinite(E).astype(np.float64)
        E0 = np.where(ve > 0, E, 0.0)
        vx = np.isfinite(X).astype(np.float64)
        X0 = np.where(vx > 0, X, 0.0)
        n = vx.T @ ve
        sx = X0.T @ ve
        sxx = (X0 * X0).T @ ve
        se = vx.T @ E0
        see = vx.T @ (E0 * E0)
        sxe = X0.T @ E0
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = sx / n
            me = se / n
            cov = sxe / n - mx * me
            varx = sxx / n - mx * mx
            vare = see / n - me * me
            rho = cov / np.sqrt(varx * vare)
            beta = cov / varx
        ok = (n >= min_pairs) & (varx > 0) & (vare > 0) & np.isfinite(rho) & np.isfinite(beta)
        score = np.where(ok, rho, -np.inf)
        k = np.argmax(score, axis=0)
        cols = np.arange(N)
        assigned = ok.any(axis=0)
        km[m] = np.where(assigned, k, -1)
        bm[m] = np.where(assigned, beta[k, cols], np.nan)
        rm[m] = np.where(assigned, rho[k, cols], np.nan)
    month = np.cumsum(starts) - 1
    month = np.where(month < 0, len(idx), month)
    return km[month], bm[month], rm[month]


def _gather(a, kstar):
    a = np.asarray(a, np.float64)
    g = np.take_along_axis(a, np.clip(kstar, 0, None), axis=1)
    return np.where(kstar >= 0, g, np.nan)


def _news_flag(ea, ne, n=SUM_WINDOW):
    ea = np.asarray(ea, np.float64)
    ne = np.asarray(ne, np.float64)
    hit = ((ea == 1.0) | (ne == 1.0)).astype(np.float64)
    known = (np.isfinite(ea) & np.isfinite(ne)).astype(np.float64)
    any_hit = _trailing_sum(hit, n)
    all_known = _trailing_sum(known, n)
    return np.where(any_hit > 0, 1.0, np.where(all_known == n, 0.0, np.nan))


def _log_ratio(v, med):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(v / med)
    return np.where(np.isfinite(out), out, np.nan)


def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0106 needs the point-in-time S&P 500 universe (data.universe is None)")
    return np.asarray(uni, bool)


def _inputs(data):
    uni = _universe(data)
    r = _cc_return(data)
    r_spy = r[:, data.col(MARKET_ETF)]
    etf_cols = [data.col(k) for k in SECTOR_ETFS]
    x = r[:, etf_cols] - r_spy[:, None]
    e = r - r_spy[:, None]
    del r
    dv = np.asarray(data.dollar_volume, np.float64)
    dv = np.where(dv > 0, dv, np.nan)

    starts = period_starts(data.dates, "M")
    kstar, beta, rho = _sector_assignment(e, x, starts)

    x5 = _trailing_sum(x, SUM_WINDOW)
    xk = _gather(x, kstar)
    xk5 = _gather(x5, kstar)

    dv_etf = dv[:, etf_cols]
    med_etf = _lagged_median(dv_etf, np.ones(dv_etf.shape, bool))
    flow_etf = _log_ratio(dv_etf, med_etf)

    push_1d = beta * xk
    push_5d = beta * xk5
    etf_flow = _gather(flow_etf, kstar)
    idio_5d = _trailing_sum(e, SUM_WINDOW) - beta * xk5
    del e
    own_flow = _log_ratio(dv, _lagged_median(dv, uni))
    own_news = _news_flag(data.extras["ea_reaction_day"], data.extras["nonearn_8k_reaction_day"])
    loading = rho
    feats = {
        "push_1d": push_1d, "push_5d": push_5d, "etf_flow": etf_flow, "idio_5d": idio_5d,
        "own_flow": own_flow, "own_news": own_news, "loading": loading,
    }
    feats = {k: np.where(uni, v, np.nan).astype(np.float32) for k, v in feats.items()}
    return {"uni": uni, "kstar": kstar, "x": x, "dv_etf": dv_etf, "med_etf": med_etf, "feats": feats}


def _feature_matrix(inp):
    uni = inp["uni"]
    f = inp["feats"]
    z = lambda name: cs_zscore(f[name], uni)
    return stack(z("push_1d"), z("push_5d"), z("etf_flow"), z("idio_5d"), z("own_flow"),
                 f["own_news"], z("loading"))


def _model_params(params):
    p = dict(GBM_FIXED)
    p["n_estimators"] = int(params["n_estimators"])
    return p


def _scores(data, params, inp):
    X = _feature_matrix(inp)
    uni = inp["uni"]
    y = cs_rank(forward_return(data, HORIZON), uni)
    return walk_forward(X, y, horizon=HORIZON, model="gbm", params=_model_params(params),
                        retrain_every=REFIT, universe=uni)


# ---------------------------------------------------------------- the event
def _flow_sessions(x, dv_etf, med_etf):
    """(T, 9) bool: ETF k had a flow session at t (volume and excess-return conditions, both on t-63..t-1)."""
    sd = _lagged_std(x)
    with np.errstate(invalid="ignore"):
        vol_ok = dv_etf >= FLOW_DV_MULT * med_etf
        ret_ok = np.abs(x) >= FLOW_RET_SIGMA * sd
    return np.isfinite(dv_etf) & np.isfinite(med_etf) & np.isfinite(x) & np.isfinite(sd) & vol_ok & ret_ok


def _new_entries(sw):
    """New long-leg and short-leg members at t (not in the same leg at t-1)."""
    sw = np.nan_to_num(np.asarray(sw, np.float64))
    lng = sw > 0
    sht = sw < 0
    prev_l = np.zeros_like(lng)
    prev_s = np.zeros_like(sht)
    prev_l[1:] = lng[:-1]
    prev_s[1:] = sht[:-1]
    return lng & ~prev_l, sht & ~prev_s


def events(data, params):
    inp = _inputs(data)
    uni = inp["uni"]
    score = _scores(data, params, inp)
    top = float(params["top"])
    sw = score_weights(score, uni, top=top, bottom=top, min_names=MIN_NAMES)
    new_l, new_s = _new_entries(sw)
    flow = _flow_sessions(inp["x"], inp["dv_etf"], inp["med_etf"]).astype(np.float64)
    flow_i = _gather(flow, inp["kstar"]) == 1.0
    tradable = np.asarray(data.tradable, bool)
    mask = (new_l | new_s) & flow_i & uni & tradable
    side = np.where(new_l, 1.0, -1.0)
    return mask, side
