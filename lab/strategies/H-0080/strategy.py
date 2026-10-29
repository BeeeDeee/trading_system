"""H-0080: credit-ETF liquidity overshoot vs SPY/IEF anchors.

60/40 SPY/IEF book; up to 2 tilt slots of 0.20 each taken from IEF and given to a credit-type ETF
for hold_days sessions after its 5-session residual (vs a trailing OLS on SPY and IEF) drops to
z <= -k_sigma.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 5, "k_sigma": 2.0}

CANDIDATES = ("LQD", "HYG", "JNK", "EMB", "MUB", "BKLN")
ANCHOR_EQ = "SPY"
ANCHOR_RATE = "IEF"

FIT_LEN = 252          # sessions t-256 .. t-5
FIT_GAP = 5            # fit window ends 5 sessions before t
FIT_MIN_OBS = 200      # at least 200 of the 252 sessions with all three returns
SIG_LEN = 5            # signal window t-4 .. t
MIN_HISTORY = 257      # sessions of price history a candidate needs
W_SPY = 0.60
W_IEF = 0.40
SLOT_WEIGHT = 0.20
N_SLOTS = 2


def _rolling_sum(x, n):
    """Trailing sum over rows u-n+1 .. u (inclusive); NaN where the window is incomplete."""
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


def _present(data):
    listed = np.asarray(data.listed, dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    return listed & np.isfinite(close)


def _session_returns(data):
    rco = np.nan_to_num(np.asarray(data.ret_co, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    roc = np.nan_to_num(np.asarray(data.ret_oc, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return (1.0 + rco) * (1.0 + roc) - 1.0


def residual_z(data):
    """z_j,t for each candidate (T, len(CANDIDATES)); NaN where j has no signal at t."""
    r = _session_returns(data)
    present = _present(data)
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
        F = _lag(_rolling_sum(stack, FIT_LEN), FIT_GAP)       # fit on t-256 .. t-5
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
        ssr = nn * (cyy - b1 * c1y - b2 * c2y)          # residual sum of squares (residual mean is 0)
        ssr = np.maximum(ssr, 0.0)
        sigma = np.sqrt(ssr / np.maximum(nn - 1.0, 1.0))  # sample std of the fitted residuals

        W = _rolling_sum(np.stack([v, y, x1, x2], axis=1), SIG_LEN)   # signal window t-4 .. t
        full = np.isfinite(W[:, 0]) & (W[:, 0] >= SIG_LEN - 0.5)
        S = W[:, 1] - SIG_LEN * a - b1 * W[:, 2] - b2 * W[:, 3]

        hist_ok = np.cumsum(present[:, j]) >= MIN_HISTORY
        ok = det_ok & full & hist_ok & np.isfinite(sigma) & (sigma > 0) & np.isfinite(S)
        zz = np.full(T, np.nan)
        zz[ok] = S[ok] / (sigma[ok] * np.sqrt(SIG_LEN))
        z[:, k] = zz
    return z


def events_from_z(z, k_sigma, hold_days):
    """Event: z <= -k_sigma and no event of the same candidate in the previous hold_days sessions."""
    hit = np.where(np.isfinite(z), z, np.inf) <= -float(k_sigma)
    ev = np.zeros(z.shape, dtype=bool)
    hd = int(hold_days)
    for k in range(z.shape[1]):
        last = -10 ** 9
        for t in np.flatnonzero(hit[:, k]):
            if t - last > hd:
                ev[t, k] = True
                last = t
    return ev


def _month_starts(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return ps.copy()
    m = np.zeros(len(dates), dtype=bool)
    m[ps.astype(np.int64)] = True
    return m


def target_weights(data, params):
    hold_days = int(params["hold_days"])
    k_sigma = float(params["k_sigma"])
    T = len(data.dates)
    N = len(data.instruments)
    iS = data.col(ANCHOR_EQ)
    iI = data.col(ANCHOR_RATE)
    cols = [data.col(c) for c in CANDIDATES]

    z = residual_z(data)
    ev = events_from_z(z, k_sigma, hold_days)
    present = _present(data)
    delisting = np.asarray(data.delisting, dtype=bool) if data.delisting is not None else np.zeros((T, N), dtype=bool)
    uni = None if data.universe is None else np.asarray(data.universe, dtype=bool)

    month_start = _month_starts(data.dates)
    W = np.full((T, N), np.nan, dtype=np.float64)
    slots = []          # list of (candidate index k, decision row of entry)
    prev = None
    for t in range(T):
        # release slots whose hold ends (held from open t0+1 to open t0+1+hold_days)
        # or whose instrument lost its price / left the universe
        keep = []
        for (k, t0) in slots:
            j = cols[k]
            if t - t0 >= hold_days:
                continue
            if (not present[t, j]) or delisting[t, j] or (uni is not None and not uni[t, j]):
                continue
            keep.append((k, t0))
        slots = keep
        free = N_SLOTS - len(slots)
        if free > 0:
            new = np.flatnonzero(ev[t])
            if new.size:
                held = {k for (k, _) in slots}
                order = new[np.argsort(z[t, new], kind="stable")]
                for k in order:
                    if free <= 0:
                        break
                    if k in held:
                        continue
                    j = cols[k]
                    if uni is not None and not uni[t, j]:
                        continue
                    slots.append((int(k), t))
                    held.add(int(k))
                    free -= 1
        state = tuple(sorted(k for (k, _) in slots))
        if t == 0 or month_start[t] or state != prev:
            row = np.zeros(N, dtype=np.float64)
            row[iS] = W_SPY
            row[iI] = W_IEF - SLOT_WEIGHT * len(slots)
            for k in state:
                row[cols[k]] = SLOT_WEIGHT
            W[t] = row
        prev = state
    return W
