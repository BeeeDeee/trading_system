"""H-0058 mechanism events: the short leg S at each Sunday decision (side -1).

The leg logic is a verbatim copy of strategy.py (kept self-contained so the diagnostic does not depend on
importing the strategy module). Events are the members of S on Sundays where S has at least 3 members
(the card: dates with fewer than 3 young coins are skipped), and only where the perp is tradable.
"""
import numpy as np
from lab.framework.api import total_return_index

SEASONED_DAYS = 730
MIN_RETURNS = 30
BETA_WINDOW = 60
BETA_MIN_OBS = 30
MIN_LEG = 3
SHRINK = 0.5
PERP_SUFFIX = ".P"


def _day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def sunday_rows(dates):
    d = _day_numbers(dates)
    return (d + 3) % 7 == 6


def listing_info(data):
    T = len(data.dates)
    names = tuple(data.instruments)
    ac = np.asarray(data.asset_class)
    is_perp = ac == "crypto_perp"
    close = np.asarray(data.close, dtype=float)
    has_bar = np.isfinite(close) | np.asarray(data.listed, dtype=bool)
    first = np.where(has_bar.any(axis=0), has_bar.argmax(axis=0), T).astype(np.int64)

    spot_cols = {}
    for j, nm in enumerate(names):
        if ac[j] == "crypto_spot":
            spot_cols[nm] = j
    dotp = [j for j, nm in enumerate(names) if is_perp[j] and nm.endswith(PERP_SUFFIX)]
    if dotp and not spot_cols:
        raise ValueError("binance_spot_1d not delivered: no crypto_spot instruments for the perp listing dates")

    listing_row = np.full(len(names), T, dtype=np.int64)
    censored = np.zeros(len(names), dtype=bool)
    for j in np.flatnonzero(is_perp):
        r = first[j]
        nm = names[j]
        if nm.endswith(PERP_SUFFIX):
            sj = spot_cols.get(nm[: -len(PERP_SUFFIX)])
            if sj is not None:
                rs = first[sj]
                if rs < T and rs == 0:
                    censored[j] = True
                r = min(r, rs)
        listing_row[j] = r
    return listing_row, censored, is_perp, has_bar


def daily_log_returns(data, has_bar):
    tri = np.asarray(total_return_index(data), dtype=float)
    T, N = tri.shape
    lr = np.full((T, N), np.nan)
    valid = np.zeros((T, N), dtype=bool)
    if T > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(tri[1:] / tri[:-1])
        ok = has_bar[1:] & has_bar[:-1] & np.isfinite(r)
        lr[1:][ok] = r[ok]
        valid[1:] = ok
    return lr, valid


def market_return(lr, valid, members):
    use = valid & members
    cnt = use.sum(axis=1)
    tot = np.where(use, lr, 0.0).sum(axis=1)
    m = np.full(lr.shape[0], np.nan)
    has = cnt > 0
    m[has] = tot[has] / cnt[has]
    return m


def betas_at(t, lr, valid, mkt):
    lo = max(0, t - BETA_WINDOW + 1)
    x = mkt[lo:t + 1][:, None]
    y = lr[lo:t + 1]
    mask = valid[lo:t + 1] & np.isfinite(x)
    n = mask.sum(axis=0).astype(float)
    xm = np.where(mask, x, 0.0)
    ym = np.where(mask, y, 0.0)
    sx = xm.sum(axis=0)
    sy = ym.sum(axis=0)
    sxx = (xm * xm).sum(axis=0)
    sxy = (xm * ym).sum(axis=0)
    den = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(den > 1e-18, (n * sxy - sx * sy) / den, np.nan)
    return beta, n


class Book:
    def __init__(self, data, params):
        self.min_age = float(params["min_age_days"])
        self.young_max = float(params["young_max_days"])
        U = data.universe
        if U is None:
            raise ValueError("card needs the point-in-time crypto_top_n universe; data.universe is None")
        self.T = len(data.dates)
        self.N = len(data.instruments)
        self.days = _day_numbers(data.dates)
        self.names = tuple(data.instruments)
        self.listing_row, self.censored, self.is_perp, has_bar = listing_info(data)
        self.members = np.asarray(U, dtype=bool) & self.is_perp[None, :]
        self.lr, self.valid = daily_log_returns(data, has_bar)
        self.n_ret = np.cumsum(self.valid, axis=0)
        self.mkt = market_return(self.lr, self.valid, self.members)

    def legs(self, t):
        mem = self.members[t]
        listed = self.listing_row <= t
        age = np.full(self.N, np.nan)
        lrow = np.minimum(self.listing_row, self.T - 1)
        age[listed] = (self.days[t] - self.days[lrow])[listed]
        age[self.censored & listed] = np.inf
        beta, nobs = betas_at(t, self.lr, self.valid, self.mkt)
        elig = mem & listed & (self.n_ret[t] >= MIN_RETURNS) & (nobs >= BETA_MIN_OBS) & np.isfinite(beta)
        S = elig & (age >= self.min_age) & (age <= self.young_max)
        L = elig & (age > SEASONED_DAYS)
        return S, L, SHRINK * beta + (1.0 - SHRINK)


def events(data, params):
    book = Book(data, params)
    T, N = book.T, book.N
    tradable = np.asarray(data.tradable, dtype=bool)
    mask = np.zeros((T, N), dtype=bool)
    for t in np.flatnonzero(sunday_rows(data.dates)):
        S, _, _ = book.legs(int(t))
        if int(S.sum()) >= MIN_LEG:
            mask[t] = S & tradable[t]
    side = np.where(mask, -1.0, 0.0)
    return mask, side
