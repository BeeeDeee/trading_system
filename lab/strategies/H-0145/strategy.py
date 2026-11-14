"""H-0145 Slow leverage-crowding unwind.

Top-30 Binance perps (point-in-time), dollar-neutral weekly cohorts held hold_weeks weeks:
short the coins whose basis rose most above their own 120-day history (net of the 30-day
return, cross-sectional OLS residual), long the coins whose basis fell most, inside BTC-beta halves.
"""
import numpy as np

PARAMS = {"hold_weeks": 26, "lookback_days": 30}

BTC = "BTCUSDT.P"      # beta reference (inside the card's universe)
TOP_N = 30             # crypto_top_n n
VOL_WINDOW = 30        # trailing median of daily quote volume (fallback universe)
REF_DAYS = 120         # reference window of the basis
MIN_VALID = 0.8        # share of valid basis days in each window
R_DAYS = 30            # r30 = log(close_t / close_{t-30})
BETA_DAYS = 90         # beta window (daily log returns)
BETA_MIN = 60          # minimum valid paired returns for beta
MIN_ELIGIBLE = 10      # fewer eligible coins: no cohort that week
LEG_FRAC = 0.2         # k = max(1, round(0.2 x half size))
FORM_EVERY = 7         # one formation date every seventh day


# ----------------------------------------------------------------------------------------------
# universe

def _rolling_median(x, w):
    """Trailing median over the last min(w, rows so far) rows, NaNs ignored (NaN if none valid)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T == 0:
        return out
    chunk = 256
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        lo = max(0, start - w + 1)
        block = x[lo:stop]
        pad = (w - 1) - (start - lo)
        if pad > 0:
            block = np.concatenate([np.full((pad, N), np.nan), block], axis=0)
        win = np.lib.stride_tricks.sliding_window_view(block, w, axis=0)   # (stop-start, N, w)
        s = np.sort(win, axis=-1)                                            # NaN sorted last
        cnt = np.isfinite(win).sum(axis=-1)
        i_lo = np.maximum((cnt - 1) // 2, 0)
        i_hi = np.maximum(cnt // 2, 0)
        v_lo = np.take_along_axis(s, i_lo[..., None], axis=-1)[..., 0]
        v_hi = np.take_along_axis(s, i_hi[..., None], axis=-1)[..., 0]
        med = 0.5 * (v_lo + v_hi)
        out[start:stop] = np.where(cnt > 0, med, np.nan)
    return out


def _universe(data):
    """The gate's point-in-time universe; if none is passed, the PIT top 30 by trailing 30-day
    median daily quote volume among listed perps, from rows <= t only."""
    if data.universe is not None:
        return np.asarray(data.universe, dtype=bool)
    if "quote_volume" in data.extras:
        qv = np.asarray(data.extras["quote_volume"], dtype=np.float64)
    else:
        qv = np.asarray(data.dollar_volume, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    med = _rolling_median(qv, VOL_WINDOW)
    valid = listed & np.isfinite(med)
    key = np.where(valid, med, -np.inf)
    T, N = key.shape
    cols = np.broadcast_to(np.arange(N), (T, N))
    order = np.lexsort((cols, -key), axis=-1) if T > 0 else np.zeros((0, N), dtype=np.int64)
    rank = np.empty((T, N), dtype=np.int64)
    if T > 0:
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(N), (T, N)).copy(), axis=-1)
    return valid & (rank < TOP_N)


# ----------------------------------------------------------------------------------------------
# formation

def _cs_residual(x, y):
    """Residuals of the OLS regression of y on x with an intercept."""
    xm = x - x.mean()
    ym = y - y.mean()
    vx = float(np.sum(xm * xm))
    slope = float(np.sum(xm * ym)) / vx if vx > 0 else 0.0
    return ym - slope * xm


def _window_mean_count(x, a, b):
    """Sum-based mean and count of finite values of x[a:b] per column (rows before 0 count as missing)."""
    w = x[max(0, a):max(0, b)]
    ok = np.isfinite(w)
    n = ok.sum(axis=0)
    s = np.where(ok, w, 0.0).sum(axis=0)
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return w, ok, n, mean


def _formations(data, params, uni):
    """List of (row t, members (cols), weights, side) of every formed cohort. Rows <= t only."""
    L = int(params["lookback_days"])
    basis = np.asarray(data.extras["basis"], dtype=np.float64)
    close = np.asarray(data.close, dtype=np.float64)
    T, N = close.shape
    btc = data.col(BTC)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close > 0, close, np.nan))
    lr = np.full((T, N), np.nan)
    if T > 1:
        lr[1:] = logc[1:] - logc[:-1]
    if T == 0:
        return []
    dnum = (data.dates - data.dates[0]).astype(np.int64)
    form_rows = np.flatnonzero(dnum % FORM_EVERY == 0)
    out = []
    for t in form_rows:
        t = int(t)
        elig = uni[t] & np.isfinite(basis[t]) & np.isfinite(close[t]) & (close[t] > 0)
        if elig.sum() < MIN_ELIGIBLE:
            continue
        # (1) own-history basis score
        _, _, n_rec, b_rec = _window_mean_count(basis, t - L + 1, t + 1)
        ref_a, ref_b = t - L - REF_DAYS + 1, t - L + 1
        w_ref, ok_ref, n_ref, b_ref = _window_mean_count(basis, ref_a, ref_b)
        dev = np.where(ok_ref, w_ref - b_ref, 0.0)
        s_ref = np.sqrt(np.where(n_ref > 1, (dev * dev).sum(axis=0) / np.maximum(n_ref - 1, 1), np.nan))
        elig &= (n_rec >= MIN_VALID * L) & (n_ref >= MIN_VALID * REF_DAYS) & (s_ref > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (b_rec - b_ref) / s_ref
        elig &= np.isfinite(raw)
        # (2) 30-day log return
        if t < R_DAYS:
            continue
        r30 = logc[t] - logc[t - R_DAYS]
        elig &= np.isfinite(r30)
        # (3) beta on BTCUSDT.P over the last 90 daily log returns
        a = max(1, t - BETA_DAYS + 1)
        x = lr[a:t + 1, btc]
        y = lr[a:t + 1]
        pair = np.isfinite(x)[:, None] & np.isfinite(y)
        npair = pair.sum(axis=0)
        xx = np.where(pair, x[:, None], 0.0)
        yy = np.where(pair, y, 0.0)
        mx = xx.sum(axis=0) / np.maximum(npair, 1)
        my = yy.sum(axis=0) / np.maximum(npair, 1)
        dx = np.where(pair, xx - mx, 0.0)
        dy = np.where(pair, yy - my, 0.0)
        vx = (dx * dx).sum(axis=0)
        cxy = (dx * dy).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where((npair >= BETA_MIN) & (vx > 0), cxy / vx, np.nan)
        if npair[btc] >= BETA_MIN:
            beta[btc] = 1.0
        elig &= np.isfinite(beta)
        # (4) cohort
        idx = np.flatnonzero(elig)
        n = idx.size
        if n < MIN_ELIGIBLE:
            continue
        score = np.full(N, np.nan)
        score[idx] = _cs_residual(r30[idx], raw[idx])
        by_beta = idx[np.lexsort((idx, beta[idx]))]
        n_low = n // 2
        longs, shorts = [], []
        for half in (by_beta[:n_low], by_beta[n_low:]):
            k = max(1, int(np.floor(LEG_FRAC * half.size + 0.5)))
            ordered = half[np.lexsort((half, score[half]))]
            longs.extend(ordered[:k].tolist())
            shorts.extend(ordered[-k:].tolist())
        members = np.array(longs + shorts, dtype=np.int64)
        weights = np.concatenate([np.full(len(longs), 0.5 / len(longs)),
                                  np.full(len(shorts), -0.5 / len(shorts))])
        side = np.sign(weights)
        out.append((t, members, weights, side))
    return out


# ----------------------------------------------------------------------------------------------
# book

def target_weights(data, params):
    H = int(params["hold_weeks"])
    uni = _universe(data)
    T, N = uni.shape
    W = np.zeros((T, N))
    if T == 0:
        return W
    dnum = (data.dates - data.dates[0]).astype(np.int64)
    n_active = np.zeros(T)
    for t, members, weights, _side in _formations(data, params, uni):
        end = int(np.searchsorted(dnum, dnum[t] + FORM_EVERY * H, side="left"))
        alive = np.cumprod(uni[t:end][:, members], axis=0).astype(bool)   # leaves universe: 0 for good
        W[t:end, members] += weights[None, :] * alive
        n_active[t:end] += 1.0
    return np.where(n_active[:, None] > 0, W / np.maximum(n_active, 1.0)[:, None], 0.0)
