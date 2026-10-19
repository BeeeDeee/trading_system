"""H-0060 mechanism events: on each monthly decision date, long-leg ETFs (side +1) and short-leg ETFs
(side -1), selected exactly as in strategy.py (same code, duplicated so the file stands alone)."""
import numpy as np
from lab.framework.api import period_starts

UNIVERSE = (
    "SPY", "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM",
    "EWN", "EWO", "EWP", "EWQ", "EWS", "EWU", "EWW", "EWY", "EWT", "EWZ",
)

SESSIONS_PER_MONTH = 21
SKIP = 252
VOL_WINDOW = 252


def _decision_rows(dates):
    m = np.asarray(period_starts(dates, "M"))
    if m.dtype == bool:
        return np.flatnonzero(m)
    return np.asarray(m, dtype=np.int64)


def events(data, params):
    F = SESSIONS_PER_MONTH * int(params["formation_months"])
    L = int(params["leg_size"])
    T = len(data.dates)
    N = len(data.instruments)
    cols = np.array([data.col(name) for name in UNIVERSE], dtype=np.int64)
    n = len(cols)
    rco = np.asarray(data.ret_co, dtype=np.float64)[:, cols]
    roc = np.asarray(data.ret_oc, dtype=np.float64)[:, cols]
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    r = np.where(np.isfinite(r), r, 0.0)
    close = np.asarray(data.close, dtype=np.float64)[:, cols]
    has_price = np.isfinite(close)
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)[:, cols]
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]
    order = np.arange(n)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in _decision_rows(data.dates):
        t = int(t)
        s0 = t - (F + SKIP)
        s1 = t - SKIP
        if s0 < 0:
            continue
        elig = has_price[s0] & has_price[t]
        if universe is not None:
            elig &= universe[t]
        score = np.prod(1.0 + r[s0 + 1:s1 + 1], axis=0) - 1.0
        sigma = np.std(r[t - VOL_WINDOW + 1:t + 1], axis=0, ddof=1)
        elig &= np.isfinite(score) & np.isfinite(sigma) & (sigma > 0)
        idx = np.flatnonzero(elig)
        if idx.size < 2 * L + 3:
            continue
        sc = score[idx]
        longs = idx[np.lexsort((order[idx], sc))][:L]
        shorts = idx[np.lexsort((order[idx], -sc))][:L]
        for pos, sgn in ((longs, 1.0), (shorts, -1.0)):
            c = cols[pos]
            ok = tradable[t, pos]
            if universe is not None:
                ok = ok & universe[t, pos]
            mask[t, c[ok]] = True
            side[t, c[ok]] = sgn
    return mask, side
