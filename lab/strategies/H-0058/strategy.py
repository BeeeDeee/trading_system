"""H-0058 Token vesting overhang on Binance perps (beta-neutral, weekly).

Short perps aged min_age_days..young_max_days since listing, long perps listed > 730 days,
both legs from the point-in-time top-30 universe, beta-neutral gross split, equal weight in a leg,
reset every Sunday close (filled Monday open).
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"min_age_days": 30, "young_max_days": 365}

SEASONED_DAYS = 730      # long leg: age > 730 days (fixed by the card, not a parameter)
MIN_RETURNS = 30         # at least 30 daily perp returns
BETA_WINDOW = 60         # trailing 60 days for the OLS beta
BETA_MIN_OBS = 30        # at least 30 observations in the beta window
MIN_LEG = 3              # fewer than 3 members in S or L -> flat
SHRINK = 0.5             # shrunk beta = 0.5 * beta + 0.5
PERP_SUFFIX = ".P"


def _day_numbers(dates):
    return np.asarray(dates).astype("datetime64[D]").astype(np.int64)


def sunday_rows(dates):
    """Rows whose UTC day is a Sunday (day number 0 of the datetime64 epoch was a Thursday)."""
    d = _day_numbers(dates)
    return (d + 3) % 7 == 6


def listing_info(data):
    """Per instrument: listing row (first daily bar of spot pair or perp, the earlier) and censoring flag.

    listing_row[j] = T means no bar at all in the data. censored[j] = the spot pair's first bar is the
    first row of the data (age censored from below: the card says such pairs are simply old).
    A listing row is only used at decision rows t >= listing_row, so it is point-in-time.
    """
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
        # The listing age needs binance_spot_1d: fail loudly instead of silently using perp bars only.
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
    """Close-to-close log return of each instrument; valid only where there is a bar today and yesterday."""
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
    """Equal-weight mean daily log return of the universe members of each day (PIT membership of that day)."""
    use = valid & members
    cnt = use.sum(axis=1)
    tot = np.where(use, lr, 0.0).sum(axis=1)
    m = np.full(lr.shape[0], np.nan)
    has = cnt > 0
    m[has] = tot[has] / cnt[has]
    return m


def betas_at(t, lr, valid, mkt):
    """OLS slope of each instrument's daily log return on the market over rows t-59..t, and the obs count."""
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
    """Everything the legs need, computed once; legs(t) uses rows <= t only."""

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
        # A member without any spot or perp bar on or before t has no listing date: not eligible (card rule).
        # (A per-member exception is not possible: G0's noise rows contain universe members without bars.)
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

    def weights(self, t):
        S, L, b = self.legs(t)
        w = np.zeros(self.N)
        nS, nL = int(S.sum()), int(L.sum())
        if nS < MIN_LEG or nL < MIN_LEG:
            return w
        BL = float(b[L].mean())
        BS = float(b[S].mean())
        if not (BL > 0.0 and BS > 0.0):
            # undefined in the card: a non-positive mean shrunk beta would flip a leg; stay flat this week
            return w
        GL = BS / (BL + BS)
        GS = BL / (BL + BS)
        w[L] = GL / nL
        w[S] = -GS / nS
        return w


def target_weights(data, params):
    book = Book(data, params)
    T, N = book.T, book.N
    sun = sunday_rows(data.dates)
    W = np.full((T, N), np.nan)
    cur = np.zeros(N)
    for t in range(T):
        if sun[t]:
            cur = book.weights(t)
            W[t] = cur
        else:
            out = (cur != 0.0) & ~book.members[t]
            if out.any():
                # coin left the PIT universe: close it now (K-0051), its weight stays in cash until Sunday
                W[t, out] = 0.0
                cur[out] = 0.0
    return W
