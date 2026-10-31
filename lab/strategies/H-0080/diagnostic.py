"""H-0080 mechanism event: credit-type ETF closes with 5-session residual z <= -k_sigma
(trailing 252-session OLS on SPY and IEF ending 5 sessions before t), no event in the previous
hold_days sessions. side = +1 (the candidate should outperform the unmarked peers).
Same computation as strategy.py (duplicated: diagnostic files may import only the allowed modules).
"""
import numpy as np

CANDIDATES = ("LQD", "HYG", "JNK", "EMB", "MUB", "BKLN")
ANCHOR_EQ = "SPY"
ANCHOR_RATE = "IEF"
FIT_LEN = 252
FIT_GAP = 5
FIT_MIN_OBS = 200
SIG_LEN = 5
MIN_HISTORY = 257


def _rolling_sum(x, n):
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    c[1:] = np.cumsum(x, axis=0)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if T >= n:
        out[n - 1:] = c[n:] - c[:T - n + 1]
    return out


def _lag(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] > k:
        out[k:] = x[:x.shape[0] - k]
    return out


def _residual_z(data):
    rco = np.nan_to_num(np.asarray(data.ret_co, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    roc = np.nan_to_num(np.asarray(data.ret_oc, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    present = np.asarray(data.listed, dtype=bool) & np.isfinite(np.asarray(data.close, dtype=np.float64))
    iS = data.col(ANCHOR_EQ)
    iI = data.col(ANCHOR_RATE)
    T = r.shape[0]
    z = np.full((T, len(CANDIDATES)), np.nan, dtype=np.float64)
    for k, name in enumerate(CANDIDATES):
        j = data.col(name)
        valid = present[:, j] & present[:, iS] & present[:, iI]
        v = valid.astype(np.float64)
        y = np.where(valid, r[:, j], 0.0)
        x1 = np.where(valid, r[:, iS], 0.0)
        x2 = np.where(valid, r[:, iI], 0.0)
        stack = np.stack([v, x1, x2, y, x1 * x1, x2 * x2, x1 * x2, x1 * y, x2 * y, y * y], axis=1)
        F = _lag(_rolling_sum(stack, FIT_LEN), FIT_GAP)
        n, s1, s2, sy, s11, s22, s12, s1y, s2y, syy = (F[:, i] for i in range(10))
        fit_ok = np.isfinite(n) & (n >= FIT_MIN_OBS)
        nn = np.where(fit_ok, n, 1.0)
        m1, m2, my = s1 / nn, s2 / nn, sy / nn
        c11 = s11 / nn - m1 * m1
        c22 = s22 / nn - m2 * m2
        c12 = s12 / nn - m1 * m2
        c1y = s1y / nn - m1 * my
        c2y = s2y / nn - m2 * my
        cyy = syy / nn - my * my
        det = c11 * c22 - c12 * c12
        det_ok = fit_ok & np.isfinite(det) & (det > 1e-10 * np.abs(c11 * c22))
        dd = np.where(det_ok, det, 1.0)
        b1 = (c22 * c1y - c12 * c2y) / dd
        b2 = (c11 * c2y - c12 * c1y) / dd
        a = my - b1 * m1 - b2 * m2
        ssr = np.maximum(nn * (cyy - b1 * c1y - b2 * c2y), 0.0)
        sigma = np.sqrt(ssr / np.maximum(nn - 1.0, 1.0))
        W = _rolling_sum(np.stack([v, y, x1, x2], axis=1), SIG_LEN)
        full = np.isfinite(W[:, 0]) & (W[:, 0] >= SIG_LEN - 0.5)
        S = W[:, 1] - SIG_LEN * a - b1 * W[:, 2] - b2 * W[:, 3]
        hist_ok = np.cumsum(present[:, j]) >= MIN_HISTORY
        ok = det_ok & full & hist_ok & np.isfinite(sigma) & (sigma > 0) & np.isfinite(S)
        zz = np.full(T, np.nan)
        zz[ok] = S[ok] / (sigma[ok] * np.sqrt(SIG_LEN))
        z[:, k] = zz
    return z


def events(data, params):
    hold_days = int(params["hold_days"])
    k_sigma = float(params["k_sigma"])
    z = _residual_z(data)
    hit = np.where(np.isfinite(z), z, np.inf) <= -k_sigma
    T, N = len(data.dates), len(data.instruments)
    tradable = np.asarray(data.tradable, dtype=bool)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    for k, name in enumerate(CANDIDATES):
        j = data.col(name)
        last = -10 ** 9
        for t in np.flatnonzero(hit[:, k]):
            if t - last > hold_days:
                last = t      # an event counts for the spacing rule whether or not it is marked
                if tradable[t, j] and (uni is None or uni[t, j]):
                    mask[t, j] = True
    side = np.ones((T, N), dtype=np.float64)
    return mask, side
