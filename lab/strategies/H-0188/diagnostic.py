"""H-0188 mechanism events: at each month-end decision t, the eligible S&P 500 members assigned to the
long leg (bottom leg_quantile of the peer-adjusted 21-session return, side +1) or to the short leg
(top leg_quantile, side -1). Same pipeline as strategy.py (copied: diagnostic.py is a separate file).
Eligibility already requires membership and tradability at t."""
import numpy as np

WINDOW = 252
FORM = 21
BETA_FLOOR = 0.3


def month_end_rows(dates):
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
    r = (1.0 + np.asarray(ret_co, dtype=float)) * (1.0 + np.asarray(ret_oc, dtype=float)) - 1.0
    has_px = np.asarray(listed, dtype=bool) & np.isfinite(np.asarray(close, dtype=float))
    valid = np.zeros_like(has_px)
    valid[1:] = has_px[1:] & has_px[:-1]
    valid &= np.isfinite(r)
    return np.where(valid, r, np.nan), valid


def market_return(r, valid, universe):
    m = universe & valid
    cnt = m.sum(axis=1)
    s = np.where(m, r, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


def _book_on_row(t, r, valid_cnt, mkt, universe, tradable, ea, leg_quantile, peers_k):
    N = r.shape[1]
    w = np.zeros(N)
    if t < WINDOW - 1:
        return w
    cnt = valid_cnt[t] - (valid_cnt[t - WINDOW] if t >= WINDOW else 0)
    cand = universe[t] & (cnt == WINDOW)
    cidx = np.flatnonzero(cand)
    if cidx.size < 2:
        return w
    X = r[t - WINDOW + 1:t + 1][:, cidx]
    mw = mkt[t - WINDOW + 1:t + 1]
    sd = X.std(axis=0)
    keep = np.isfinite(sd) & (sd > 0)
    cidx, X, sd = cidx[keep], X[:, keep], sd[keep]
    C = cidx.size
    if C < 2:
        return w
    Z = (X - X.mean(axis=0)) / sd
    corr = (Z.T @ Z) / WINDOW
    np.fill_diagonal(corr, -np.inf)
    cum = np.prod(1.0 + X[-FORM:], axis=0) - 1.0
    ea_win = ea[t - FORM + 1:t + 1][:, cidx]
    no_news = np.all(ea_win == 0, axis=0)
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
    rk = np.argsort(R, kind="stable")
    long_pos = epos[rk[:n_leg]]
    short_pos = epos[rk[n - n_leg:]]
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


def events(data, params):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0188 needs the point-in-time S&P 500 universe (data.universe is None)")
    ea = data.extras["ea_reaction_day"]
    W, dec = decision_book(data.dates, data.ret_co, data.ret_oc, data.close, data.listed, data.tradable,
                           uni, ea, params["leg_quantile"], params["peers_k"])
    Wd = np.where(np.isfinite(W), W, 0.0)
    mask = dec[:, None] & (Wd != 0)
    side = np.where(Wd < 0, -1.0, 1.0)
    return mask, side
