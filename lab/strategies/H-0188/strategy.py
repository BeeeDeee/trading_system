"""H-0188: no-news residual reversal at the monthly horizon, S&P 500, beta-neutral legs.

Rules of the card and where they live:
  timing      month_end_rows(): last XNYS session of each calendar month (weekday arithmetic, see below)
  (1) eligibility   _book_on_row(): member, tradable, 252 valid return sessions (window t-251..t),
                    ea_reaction_day == 0 (not NaN) on every session t-20..t
  (2) peers         top peers_k correlations (t-251..t) among members with 252 sessions of history
  (3) residual      R_j = cum. return t-20..t minus the mean of the same for its peers
  (4) legs          floor(leg_quantile * n_eligible) lowest R long, highest R short, equal weight
  (5) sizing        OLS beta on the equal-weight mean member return, leg beta floored at 0.3,
                    w_L = b_S / (b_L + b_S), w_S = b_L / (b_L + b_S)
  (6) no re-weighting between decisions (NaN rows); index exits / delistings are left to the engine
"""
import numpy as np

PARAMS = {"leg_quantile": 0.2, "peers_k": 20}

WINDOW = 252        # sessions t-251 .. t for peers, beta and the history requirement
FORM = 21           # formation window t-20 .. t
BETA_FLOOR = 0.3


def month_end_rows(dates):
    """Boolean (T,) flag of the decision rows, from the date of each row alone.

    Row t is the month-end if the next weekday after its date falls in another month
    (np.busday_offset, exchange holidays ignored). If the last weekday(s) of a month were exchange
    holidays, no row of that month is flagged; then the first row of the next month is flagged
    instead (one session late), which is known at that row from its own date and the previous row's.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    T = d.shape[0]
    if T == 0:
        return np.zeros(0, dtype=bool)
    mon = d.astype("datetime64[M]")
    nxt = np.busday_offset(d, 1, roll="forward")
    flag = nxt.astype("datetime64[M]") != mon
    out = flag.copy()
    if T > 1:
        first_of_month = np.zeros(T, dtype=bool)
        first_of_month[1:] = mon[1:] != mon[:-1]
        prev_flag = np.zeros(T, dtype=bool)
        prev_flag[1:] = flag[:-1]
        out = out | (first_of_month & ~prev_flag)
    return out


def daily_returns(ret_co, ret_oc, close, listed):
    """Session total returns and their validity (price at s and s-1, listed at s and s-1)."""
    r = (1.0 + np.asarray(ret_co, dtype=float)) * (1.0 + np.asarray(ret_oc, dtype=float)) - 1.0
    has_px = np.asarray(listed, dtype=bool) & np.isfinite(np.asarray(close, dtype=float))
    valid = np.zeros_like(has_px)
    valid[1:] = has_px[1:] & has_px[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def market_return(r, valid, universe):
    """Equal-weight mean return of the members of each day (members with a valid return)."""
    m = universe & valid
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


def _book_on_row(t, r, valid_cnt, mkt, universe, tradable, ea, leg_quantile, peers_k):
    """Weights (N,) decided after the close of row t (rows <= t only)."""
    N = r.shape[1]
    w = np.zeros(N)
    if t < WINDOW - 1:
        return w
    cnt = valid_cnt[t] - (valid_cnt[t - WINDOW] if t >= WINDOW else 0)
    cand = universe[t] & (cnt == WINDOW)                     # members with 252 sessions of history
    cidx = np.flatnonzero(cand)
    if cidx.size < 2:
        return w
    X = r[t - WINDOW + 1:t + 1][:, cidx]                     # (252, C), all finite by construction
    mw = mkt[t - WINDOW + 1:t + 1]
    sd = X.std(axis=0)
    keep = np.isfinite(sd) & (sd > 0)
    cidx, X, sd = cidx[keep], X[:, keep], sd[keep]
    C = cidx.size
    if C < 2:
        return w
    # (2) peers: correlation over t-251 .. t
    Z = (X - X.mean(axis=0)) / sd
    corr = (Z.T @ Z) / WINDOW
    np.fill_diagonal(corr, -np.inf)
    # (3) formation return over t-20 .. t
    cum = np.prod(1.0 + X[-FORM:], axis=0) - 1.0
    # (1) eligibility
    ea_win = ea[t - FORM + 1:t + 1][:, cidx]
    no_news = np.all(ea_win == 0, axis=0)                    # NaN or 1 on any session -> not eligible
    elig = tradable[t, cidx] & no_news
    k = int(peers_k)
    if k < 1 or C - 1 < k:
        return w
    epos = np.flatnonzero(elig)
    if epos.size == 0:
        return w
    order = np.argsort(-corr[epos], axis=1, kind="stable")[:, :k]
    R = cum[epos] - cum[order].mean(axis=1)
    fin = np.isfinite(R)
    epos, R = epos[fin], R[fin]
    n = epos.size
    n_leg = int(np.floor(float(leg_quantile) * n + 1e-9))
    if n_leg < 1 or 2 * n_leg > n:
        return w
    # (4) legs
    rk = np.argsort(R, kind="stable")
    long_pos = epos[rk[:n_leg]]
    short_pos = epos[rk[n - n_leg:]]
    # (5) beta-neutral sizing
    if not np.all(np.isfinite(mw)):
        return w
    mc = mw - mw.mean()
    vm = float((mc * mc).sum())
    if not vm > 0:
        return w
    beta = ((X - X.mean(axis=0)) * mc[:, None]).sum(axis=0) / vm
    b_l = max(float(beta[long_pos].mean()), BETA_FLOOR)
    b_s = max(float(beta[short_pos].mean()), BETA_FLOOR)
    w_l = b_s / (b_l + b_s)
    w_s = b_l / (b_l + b_s)
    w[cidx[long_pos]] = w_l / n_leg
    w[cidx[short_pos]] = -w_s / n_leg
    return w


def decision_book(dates, ret_co, ret_oc, close, listed, tradable, universe, ea, leg_quantile, peers_k):
    """(T, N) weights on decision rows, NaN elsewhere; plus the decision flag."""
    universe = np.asarray(universe, dtype=bool)
    tradable = np.asarray(tradable, dtype=bool)
    ea = np.asarray(ea, dtype=float)
    r, valid = daily_returns(ret_co, ret_oc, close, listed)
    valid_cnt = np.cumsum(valid, axis=0)
    mkt = market_return(r, valid, universe)
    dec = month_end_rows(dates)
    T, N = r.shape
    W = np.full((T, N), np.nan)
    for t in np.flatnonzero(dec):
        W[t] = _book_on_row(int(t), r, valid_cnt, mkt, universe, tradable, ea, leg_quantile, peers_k)
    return W, dec


def target_weights(data, params):
    """Month-end decision rows carry the full book; every other row is NaN (hold, no re-weighting).
    (6) A held name that leaves the index or delists mid-month is exited by the engine (weight to
    cash, delisting return applied) and is not replaced until the next month-end."""
    uni = data.universe
    if uni is None:
        raise ValueError("H-0188 needs the point-in-time S&P 500 universe (data.universe is None)")
    ea = data.extras["ea_reaction_day"]
    W, dec = decision_book(data.dates, data.ret_co, data.ret_oc, data.close, data.listed, data.tradable,
                           uni, ea, params["leg_quantile"], params["peers_k"])
    return W
