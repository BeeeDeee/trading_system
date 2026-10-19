"""H-0060 Cross-country long-term reversal.

21 country equity ETFs (SPY + 20 iShares MSCI single-country ETFs). On the first XNYS trading day of each
month (after its close) rank eligible ETFs by the cumulative total return from the close of session
t-(F+252) to the close of session t-252 (F = 21 x formation_months sessions). Long the leg_size lowest,
short the leg_size highest, inverse-volatility weights (252-session std of daily total returns), gross 1.
Between decision dates the weights drift (NaN rows = keep positions).
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"formation_months": 48, "leg_size": 5}

# The card's universe in the card's order (ties are broken by this order).
UNIVERSE = (
    "SPY", "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM",
    "EWN", "EWO", "EWP", "EWQ", "EWS", "EWU", "EWW", "EWY", "EWT", "EWZ",
)

SESSIONS_PER_MONTH = 21
SKIP = 252        # the most recent 252 sessions are skipped in the formation score
VOL_WINDOW = 252  # sessions for sigma_i


def _decision_rows(dates):
    """Row indices of the first trading day of each calendar month (point-in-time)."""
    m = np.asarray(period_starts(dates, "M"))
    if m.dtype == bool:
        return np.flatnonzero(m)
    return np.asarray(m, dtype=np.int64)


def _daily_total_return(data, cols):
    """Close-to-close total return per row: (1+ret_co)(1+ret_oc)-1; a day without a bar counts as 0."""
    rco = np.asarray(data.ret_co, dtype=np.float64)[:, cols]
    roc = np.asarray(data.ret_oc, dtype=np.float64)[:, cols]
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def select(data, params):
    """Per decision row: (row, eligible-count, long idx, short idx, sigma) over UNIVERSE positions.

    Returns the list of decision rows, the column indices of UNIVERSE in data, and a dict
    row -> (long_positions, short_positions, sigma_vector) for rows where a book is formed
    (rows that are flat are absent).
    """
    F = SESSIONS_PER_MONTH * int(params["formation_months"])
    L = int(params["leg_size"])
    cols = np.array([data.col(name) for name in UNIVERSE], dtype=np.int64)
    n = len(cols)
    r = _daily_total_return(data, cols)
    close = np.asarray(data.close, dtype=np.float64)[:, cols]
    has_price = np.isfinite(close)
    universe = None if data.universe is None else np.asarray(data.universe, dtype=bool)[:, cols]
    order = np.arange(n)

    rows = _decision_rows(data.dates)
    books = {}
    for t in rows:
        t = int(t)
        s0 = t - (F + SKIP)   # formation start (close)
        s1 = t - SKIP         # formation end (close)
        if s0 < 0:
            continue
        elig = has_price[s0] & has_price[t]
        if universe is not None:
            elig &= universe[t]
        # cumulative total return from close of s0 to close of s1 = product over rows s0+1..s1
        score = np.prod(1.0 + r[s0 + 1:s1 + 1], axis=0) - 1.0
        sigma = np.std(r[t - VOL_WINDOW + 1:t + 1], axis=0, ddof=1)
        # an ETF whose weight cannot be formed (zero/undefined sigma) cannot be held: treat as ineligible
        elig &= np.isfinite(score) & np.isfinite(sigma) & (sigma > 0)
        idx = np.flatnonzero(elig)
        if idx.size < 2 * L + 3:
            books[t] = None
            continue
        sc = score[idx]
        asc = idx[np.lexsort((order[idx], sc))]     # lowest score first, ties by universe order
        desc = idx[np.lexsort((order[idx], -sc))]   # highest score first, ties by universe order
        books[t] = (asc[:L], desc[:L], sigma)
    return rows, cols, books


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    rows, cols, books = select(data, params)
    for t in rows:
        t = int(t)
        w[t, :] = 0.0
        book = books.get(t)
        if book is None:
            continue
        longs, shorts, sigma = book
        raw = np.zeros(len(cols))
        raw[longs] = 1.0 / sigma[longs]
        raw[shorts] = -1.0 / sigma[shorts]
        raw /= np.sum(np.abs(raw))
        w[t, cols] = raw
    return w
