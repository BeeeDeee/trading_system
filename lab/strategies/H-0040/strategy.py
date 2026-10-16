"""H-0040 BTC-led catch-up in lagging altcoins (15m perps).

After a 1-hour BTC jump of k_sigma * (2 * 7-day 15m sigma) or more, go long +0.1 in every
universe follower (ex BTC, ETH) whose own 1-hour log return is less than half of BTC's,
hold exactly hold_candles decision rows, ignore signals while held (incl. the exit row).
"""
import numpy as np

PARAMS = {"hold_candles": 16, "k_sigma": 3.0}

LEADER = "BTCUSDT"
EXCLUDED = "ETHUSDT"
VOL_WINDOW = 672      # 7 days of 15m candles (fixed by design)
RET_WINDOW = 4        # 1 hour (fixed by design)
LAG_RATIO = 0.5       # fixed by design
WEIGHT = 0.1          # fixed by design


def _shift(x, k):
    """x shifted down by k rows (row t holds x[t-k]); first k rows NaN / False."""
    out = np.empty_like(x)
    if x.dtype == bool:
        out[:k] = False
    else:
        out[:k] = np.nan
    out[k:] = x[: x.shape[0] - k]
    return out


def _all_real_trailing(real, n):
    """True at t when real[t-n+1 .. t] are all True (trailing window, point-in-time)."""
    r = real.astype(np.int64)
    cs = np.cumsum(r, axis=0)
    cnt = cs.copy()
    if cs.shape[0] > n:
        cnt[n:] = cs[n:] - cs[:-n]
    ok = cnt == n
    ok[: n - 1] = False
    return ok


def _rolling_std_ddof1(x, n):
    """Sample std (ddof=1) of x[t-n+1 .. t]; NaN if the window holds any NaN or is incomplete."""
    T = x.shape[0]
    out = np.full(T, np.nan)
    if T < n:
        return out
    finite = np.isfinite(x)
    xz = np.where(finite, x, 0.0)
    nf = np.concatenate([[0], np.cumsum(~finite)])
    s1 = np.concatenate([[0.0], np.cumsum(xz)])
    s2 = np.concatenate([[0.0], np.cumsum(xz * xz)])
    w1 = s1[n:] - s1[:-n]
    w2 = s2[n:] - s2[:-n]
    bad = (nf[n:] - nf[:-n]) > 0
    var = (w2 - w1 * w1 / n) / (n - 1)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd[bad] = np.nan
    out[n - 1:] = sd
    return out


def target_positions(data, params):
    hold = int(params["hold_candles"])
    k_sigma = float(params["k_sigma"])

    c = np.asarray(data.c, dtype=np.float64)
    real = np.asarray(data.real, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    if data.universe is None:
        raise ValueError("H-0040 needs the point-in-time crypto_top_n universe")
    universe = np.asarray(data.universe, dtype=bool)
    T, N = c.shape

    li = data.col(LEADER)
    ei = data.col(EXCLUDED)

    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(c)
        # Leader quantities
        lc = logc[:, li]
        lr1 = lc - _shift(lc, 1)                       # ln(c[s]/c[s-1])
        r_L = lc - _shift(lc, RET_WINDOW)              # ln(c[t]/c[t-4])
        sigma = _rolling_std_ddof1(lr1, VOL_WINDOW)    # s = t-671 .. t
        leader_ok = _all_real_trailing(real[:, li:li + 1], VOL_WINDOW)[:, 0]
        s_1h = 2.0 * sigma
        jump = leader_ok & np.isfinite(r_L) & np.isfinite(s_1h) & (r_L > 0) & (r_L >= k_sigma * s_1h)

        # Follower quantities
        r_F = logc - _shift(logc, RET_WINDOW)
        f_real = _all_real_trailing(real, RET_WINDOW + 1)  # candles t-4 .. t real
        lag = np.isfinite(r_F) & (r_F < LAG_RATIO * r_L[:, None])  # r_F / r_L < 0.5 with r_L > 0

    follower = np.ones(N, dtype=bool)
    follower[li] = False
    follower[ei] = False

    sig = jump[:, None] & lag & f_real & universe & listed & follower[None, :]

    w = np.zeros((T, N), dtype=np.float64)
    for j in np.flatnonzero(follower):
        rows = np.flatnonzero(sig[:, j])
        next_allowed = -1
        for t in rows:
            if t < next_allowed:
                continue
            w[t:t + hold, j] = WEIGHT
            next_allowed = t + hold + 1  # exit row t+hold also ignores signals

    # Never hold non-members: close at that row, hold clock keeps running (no resume).
    w[~universe] = 0.0

    # Gross normalisation only when more than 10 followers are held at once.
    gross = np.abs(w).sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / np.where(gross > 0, gross, 1.0), 1.0)
    w = w * scale[:, None]
    return w
