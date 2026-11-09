"""H-0114 mechanism event: on an active weekly decision date (|S10| >= 1, >= 50 scored members) a scored
LIQ-500 member enters the top fraction `top` of the gbm score (side +1) or the bottom fraction `top` (side -1).

Same feature, walk_forward and book code as strategy.py (copied: diagnostic.py is a separate file).
See strategy.py for the indexing convention of the oil inputs (information row k = session d + 1).
"""
import numpy as np
from lab.framework.api import only_on, period_starts
from lab.framework.ml import cs_rank, cs_zscore, forward_return, score_weights, stack, walk_forward

OIL_KEY = "fred_macro.DCOILWTICO"
BETA_WINDOW = 252
MIN_PAIRS = 200
SHOCK_WINDOW = 10
MIN_SHOCK_OBS = 8
CLIP = 0.20
RESID_WINDOW = 5
VOL_WINDOW = 63
S10_THRESHOLD = 1.0
MIN_SCORED = 50
HORIZON = 5
RETRAIN_EVERY = 126
CHUNK = 256


def roll_sum(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    cs = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=cs[1:])
    lo = np.maximum(np.arange(T) + 1 - n, 0)
    return cs[1:] - cs[lo]


def shift_down(x, k=1):
    out = np.empty_like(x)
    if x.dtype == bool:
        out[:k] = False
    else:
        out[:k] = np.nan
    out[k:] = x[:-k] if x.shape[0] > k else x[:0]
    return out


def daily_returns(ret_co, ret_oc, close):
    ret_co = np.asarray(ret_co, dtype=np.float64)
    ret_oc = np.asarray(ret_oc, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    has = np.isfinite(close)
    valid = has & shift_down(has, 1) & np.isfinite(ret_co) & np.isfinite(ret_oc)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    return np.where(valid, r, np.nan)


def market_return(ret_co, ret_oc, close, uni):
    T, N = uni.shape
    s = np.zeros(T)
    c = np.zeros(T)
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        r = daily_returns(ret_co[:, a:b], ret_oc[:, a:b], close[:, a:b])
        ok = np.isfinite(r) & uni[:, a:b]
        s += np.where(ok, r, 0.0).sum(axis=1)
        c += ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def oil_changes(S):
    S = np.asarray(S, dtype=np.float64)
    T = S.shape[0]
    pos = np.isfinite(S) & (S > 0)
    idx = np.where(pos, np.arange(T), -1)
    last = np.maximum.accumulate(idx) if T else idx
    prev = np.full(T, -1)
    prev[1:] = last[:-1]
    ok = pos & (prev >= 0)
    O = np.full(T, np.nan)
    if ok.any():
        O[ok] = np.log(S[ok] / S[prev[ok]])
    return np.clip(O, -CLIP, CLIP)


def oil_shock(O, M1):
    T = O.shape[0]
    v = np.isfinite(O) & np.isfinite(M1)
    o0 = np.where(v, O, 0.0)
    m0 = np.where(v, M1, 0.0)
    n = roll_sum(v.astype(np.float64), BETA_WINDOW)
    so = roll_sum(o0, BETA_WINDOW)
    sm = roll_sum(m0, BETA_WINDOW)
    soo = roll_sum(o0 * o0, BETA_WINDOW)
    smm = roll_sum(m0 * m0, BETA_WINDOW)
    som = roll_sum(o0 * m0, BETA_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1.0)
        cmm = smm - sm * sm / nn
        coo = soo - so * so / nn
        com = som - so * sm / nn
        okc = (n >= MIN_PAIRS) & (cmm > 0)
        c = np.where(okc, com / np.where(cmm > 0, cmm, 1.0), np.nan)
        var_u = (coo - 2.0 * c * com + c * c * cmm) / np.maximum(n - 1.0, 1.0)
        s_u = np.where(okc & (var_u > 0), np.sqrt(np.maximum(var_u, 0.0)), np.nan)
        cnt10 = roll_sum(v.astype(np.float64), SHOCK_WINDOW)
        so10 = roll_sum(o0, SHOCK_WINDOW)
        sm10 = roll_sum(m0, SHOCK_WINDOW)
        U10 = np.where(cnt10 >= MIN_SHOCK_OBS, so10 - c * sm10, np.nan)
        cntl = np.full(T, 0.0)
        sol = np.zeros(T)
        sml = np.zeros(T)
        if T > SHOCK_WINDOW:
            cntl[SHOCK_WINDOW:] = cnt10[:-SHOCK_WINDOW]
            sol[SHOCK_WINDOW:] = so10[:-SHOCK_WINDOW]
            sml[SHOCK_WINDOW:] = sm10[:-SHOCK_WINDOW]
        U10lag = np.where(cntl >= MIN_SHOCK_OBS, sol - c * sml, np.nan)
        S10 = U10 / (s_u * np.sqrt(float(SHOCK_WINDOW)))
    return c, s_u, U10, U10lag, S10


def stock_regression(R1, M1, O):
    P = np.isfinite(R1) & np.isfinite(M1)[:, None] & np.isfinite(O)[:, None]
    Pf = P.astype(np.float64)
    r0 = np.where(P, R1, 0.0)
    m0 = np.where(P, np.where(np.isfinite(M1), M1, 0.0)[:, None], 0.0)
    o0 = np.where(P, np.where(np.isfinite(O), O, 0.0)[:, None], 0.0)
    w = BETA_WINDOW
    n = roll_sum(Pf, w)
    sr, sm, so = roll_sum(r0, w), roll_sum(m0, w), roll_sum(o0, w)
    smm, soo, smo = roll_sum(m0 * m0, w), roll_sum(o0 * o0, w), roll_sum(m0 * o0, w)
    srm, sro, srr = roll_sum(r0 * m0, w), roll_sum(r0 * o0, w), roll_sum(r0 * r0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1.0)
        cmm = smm - sm * sm / nn
        coo = soo - so * so / nn
        cmo = smo - sm * so / nn
        crm = srm - sr * sm / nn
        cro = sro - sr * so / nn
        crr = srr - sr * sr / nn
        D = cmm * coo - cmo * cmo
        Ds = np.where(D > 0, D, 1.0)
        b_m = (crm * coo - cro * cmo) / Ds
        b_o = (cro * cmm - crm * cmo) / Ds
        rss = np.maximum(crr - b_m * crm - b_o * cro, 0.0)
        se2 = rss / np.maximum(n - 3.0, 1.0) * cmm / Ds
        ok = (n >= MIN_PAIRS) & (D > 0) & (se2 > 0)
        tst = b_o / np.sqrt(np.where(se2 > 0, se2, 1.0))
    nan = np.nan
    return np.where(ok, b_m, nan), np.where(ok, b_o, nan), np.where(ok, tst, nan)


def raw_features(ret_co, ret_oc, close, uni, S):
    T, N = uni.shape
    m = market_return(ret_co, ret_oc, close, uni)
    M1 = shift_down(m, 1)
    O = oil_changes(S)
    _, _, U10, U10lag, S10 = oil_shock(O, M1)
    names = ("b_o", "tstat", "P10", "P10lag", "resid5", "vol63")
    out = {k: np.full((T, N), np.nan, dtype=np.float32) for k in names}
    vm = np.isfinite(m)
    m_0 = np.where(vm, m, 0.0)
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        r = daily_returns(ret_co[:, a:b], ret_oc[:, a:b], close[:, a:b])
        b_m, b_o, tst = stock_regression(shift_down(r, 1), M1, O)
        out["b_o"][:, a:b] = b_o
        out["tstat"][:, a:b] = tst
        out["P10"][:, a:b] = b_o * U10[:, None]
        out["P10lag"][:, a:b] = b_o * U10lag[:, None]
        v = np.isfinite(r)
        r0 = np.where(v, r, 0.0)
        v5 = v & vm[:, None]
        n5 = roll_sum(v5.astype(np.float64), RESID_WINDOW)
        sr5 = roll_sum(np.where(v5, r0, 0.0), RESID_WINDOW)
        sm5 = roll_sum(np.where(v5, m_0[:, None], 0.0), RESID_WINDOW)
        with np.errstate(invalid="ignore"):
            out["resid5"][:, a:b] = np.where(n5 >= RESID_WINDOW, sr5 - b_m * sm5, np.nan)
        n63 = roll_sum(v.astype(np.float64), VOL_WINDOW)
        s1 = roll_sum(r0, VOL_WINDOW)
        s2 = roll_sum(r0 * r0, VOL_WINDOW)
        var = (s2 - s1 * s1 / VOL_WINDOW) / (VOL_WINDOW - 1.0)
        out["vol63"][:, a:b] = np.where(n63 >= VOL_WINDOW, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return out, S10


def build_features(data, uni):
    S = data.series[OIL_KEY]
    raw, S10 = raw_features(data.ret_co, data.ret_oc, data.close, uni, S)
    feats = []
    for k in ("b_o", "tstat", "P10", "P10lag", "resid5", "vol63"):
        z = np.asarray(cs_zscore(raw.pop(k), uni), dtype=np.float32)
        feats.append(z)
    s10f = np.where(uni, S10[:, None], np.nan).astype(np.float32)
    feats.append(s10f)
    X = stack(*feats)
    del feats
    return X, S10


def model_scores(data, params, uni):
    X, S10 = build_features(data, uni)
    y = cs_rank(forward_return(data, HORIZON), uni)
    gbm = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 1000,
        "subsample": 0.8,
        "colsample": 1.0,
    }
    score = walk_forward(X, y, horizon=HORIZON, model="gbm", params=gbm,
                         retrain_every=RETRAIN_EVERY, universe=uni)
    return np.asarray(score, dtype=np.float64), S10


def decision_book(score, S10, uni, weekly, top, min_scored=MIN_SCORED):
    weekly = np.asarray(weekly, dtype=bool)
    book = np.asarray(score_weights(score, uni, top=top, bottom=top, min_names=min_scored), dtype=np.float64)
    book = np.where(np.isfinite(book), book, 0.0)
    nscored = (np.isfinite(score) & uni).sum(axis=1)
    with np.errstate(invalid="ignore"):
        active = np.isfinite(S10) & (np.abs(S10) >= S10_THRESHOLD) & (nscored >= min_scored)
    w = np.where(active[:, None], book, 0.0)
    w = np.where(uni, w, 0.0)
    return only_on(weekly, w)


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0114 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = np.asarray(uni, dtype=bool)
    top = float(params["top"])
    score, S10 = model_scores(data, params, uni)
    weekly = period_starts(data.dates, "W")
    w = decision_book(score, S10, uni, weekly, top)
    w = np.where(np.isfinite(w), w, 0.0)
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = (w != 0.0) & uni & tradable
    side = np.where(w > 0, 1.0, np.where(w < 0, -1.0, 0.0))
    return mask, side
