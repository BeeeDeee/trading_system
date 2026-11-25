"""H-0202: BTC-led catch-up sized by the beta-implied expected catch-up.

After a closed 15m candle t where BTC's last-hour log return is at least k_sigma of its own trailing
30-day standard deviation (window ending at t-4), trade the lagging altcoins of the top-10 (by 30-day
quote volume, BTC and ETH removed) of the point-in-time universe in BTC's direction, sized by
E_j = beta_j * r_BTC - r_j, and exit the whole book 16 candles after entry. One book at a time.
"""
import numpy as np

PARAMS = {"beta_days": 30, "k_sigma": 3.0}

LEADER = "BTCUSDT"
EXCLUDED_ALT = "ETHUSDT"
N_ALTS = 10                 # tradable set: the 10 highest members by 30-day quote volume
RET_CANDLES = 4             # one hour of 15m candles
SIGMA_CANDLES = 2880        # 30 days of 15m candles (sigma window)
VOLUME_CANDLES = 2880       # 30 days of 15m candles (quote-volume ranking)
HOLD_CANDLES = 16           # 4 hours, fixed by the card
FULL_SIZE_GAP = 0.02        # fixed scale of the card, not a parameter
SLOT_WEIGHT = 0.1           # one tenth of equity per altcoin at full size
MIN_REAL_SHARE = 0.8        # skip j if fewer than 80 % of the beta hours are real
# cost floor = modeled round trip = 2 * (taker fee + slippage tier), from the gate's cost model
FEE_BPS = 5.0
SLIPPAGE_TIERS_BPS = ((1.0e9, 1.5), (2.0e8, 3.0), (5.0e7, 6.0), (0.0, 15.0))
TIER_DAYS = 30              # slippage tier by the 30-day median daily quote volume


def _leader_z(c_btc):
    """r_BTC[t] = log(c[t] / c[t-4]); sigma[t] = std of the 2880 rolling 4-candle log returns
    ending at t-4; z = r_BTC / sigma. Causal prefix sums only."""
    T = c_btc.shape[0]
    with np.errstate(all="ignore"):
        lc = np.log(np.asarray(c_btc, dtype=np.float64))
    r4 = np.full(T, np.nan)
    if T > RET_CANDLES:
        r4[RET_CANDLES:] = lc[RET_CANDLES:] - lc[:-RET_CANDLES]
    ok = np.isfinite(r4)
    x = np.where(ok, r4, 0.0)
    s1 = np.concatenate(([0.0], np.cumsum(x)))
    s2 = np.concatenate(([0.0], np.cumsum(x * x)))
    sn = np.concatenate(([0], np.cumsum(ok.astype(np.int64))))
    t = np.arange(T)
    end = t - RET_CANDLES + 1           # window rows [t-4-2879, t-4] -> prefix indices [end-2880, end)
    start = end - SIGMA_CANDLES
    have = start >= 0
    sigma = np.full(T, np.nan)
    e = end[have]
    s = start[have]
    n = sn[e] - sn[s]
    m1 = s1[e] - s1[s]
    m2 = s2[e] - s2[s]
    w = float(SIGMA_CANDLES)
    var = (m2 - m1 * m1 / w) / (w - 1.0)
    sig = np.sqrt(np.maximum(var, 0.0))
    sig[n < SIGMA_CANDLES] = np.nan
    sigma[have] = sig
    z = np.full(T, np.nan)
    good = np.isfinite(sigma) & (sigma > 0) & np.isfinite(r4)
    z[good] = r4[good] / sigma[good]
    return z, r4


def _slippage_bps(median_daily_qv):
    out = np.full(median_daily_qv.shape, SLIPPAGE_TIERS_BPS[-1][1], dtype=np.float64)
    for threshold, bps in reversed(SLIPPAGE_TIERS_BPS):
        out[median_daily_qv >= threshold] = bps
    return out


def _event_book(t, c, real, uni, alt_ok, csq, day, jb, r_btc, n_hours):
    """Weights of the book opened after the close of candle t (rows <= t only)."""
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    if t < RET_CANDLES:
        return empty
    members = np.flatnonzero(uni[t] & alt_ok)
    if members.size == 0:
        return empty
    # tradable set: the 10 highest members by quote volume over the 2880 candles ending at t
    lo = max(t + 1 - VOLUME_CANDLES, 0)
    vol = csq[t + 1, members] - csq[lo, members]
    cols = members[np.argsort(-vol, kind="stable")[:N_ALTS]]

    with np.errstate(all="ignore"):
        # (4) gap candles t-3..t (t+1 is handled at row t+1, see _run)
        ok = real[t - 3:t + 1][:, cols].all(axis=0)
        # (3) r_j over the same four candles
        rj = np.log(c[t, cols].astype(np.float64) / c[t - RET_CANDLES, cols].astype(np.float64))
        ok &= np.isfinite(rj)
        # beta_j: OLS slope of non-overlapping hourly log returns ending at t-4, t-8, ..., over beta_days
        s = t - RET_CANDLES * np.arange(1, n_hours + 1)
        s = s[s >= RET_CANDLES]
        if s.size == 0:
            return empty
        x = np.log(c[s, jb].astype(np.float64) / c[s - RET_CANDLES, jb].astype(np.float64))
        y = np.log(c[np.ix_(s, cols)].astype(np.float64) / c[np.ix_(s - RET_CANDLES, cols)].astype(np.float64))
        vb = real[s, jb] & real[s - RET_CANDLES, jb] & np.isfinite(x)
        v = vb[:, None] & real[np.ix_(s, cols)] & real[np.ix_(s - RET_CANDLES, cols)] & np.isfinite(y)
        n = v.sum(axis=0)
        ok &= n >= MIN_REAL_SHARE * n_hours
        nn = np.maximum(n, 1)
        xx = np.where(v, x[:, None], 0.0)
        yy = np.where(v, y, 0.0)
        mx = xx.sum(axis=0) / nn
        my = yy.sum(axis=0) / nn
        dx = np.where(v, x[:, None] - mx, 0.0)
        dy = np.where(v, y - my, 0.0)
        sxx = (dx * dx).sum(axis=0)
        sxy = (dx * dy).sum(axis=0)
        beta = np.where(sxx > 0, sxy / np.where(sxx > 0, sxx, 1.0), np.nan)
        ok &= np.isfinite(beta) & (beta > 0)
        # expected catch-up and the lag side only
        e_gap = beta * r_btc - rj
        side = np.sign(r_btc)
        ok &= np.sign(e_gap) == side
        # cost floor: |E_j| >= 2 * (fee + j's slippage tier); tier from the median daily quote
        # volume of the last 30 completed UTC days
        days = np.arange(day[t] - TIER_DAYS, day[t])
        st = np.searchsorted(day, days, side="left")
        en = np.searchsorted(day, days, side="right")
        daily = csq[np.ix_(en, cols)] - csq[np.ix_(st, cols)]
        med = np.median(daily, axis=0)
        floor = 2.0 * (FEE_BPS + _slippage_bps(med)) / 1.0e4
        ok &= np.abs(e_gap) >= floor
        w = side * np.minimum(np.abs(e_gap) / FULL_SIZE_GAP, 1.0) * SLOT_WEIGHT
    ok &= np.isfinite(w)
    return cols[ok], w[ok], e_gap[ok], beta[ok]


def _run(data, params):
    k_sigma = float(params["k_sigma"])
    n_hours = int(round(float(params["beta_days"]) * 24))
    if data.universe is None:
        raise ValueError("H-0202 needs the point-in-time crypto_top_n universe")
    jb = data.col(LEADER)
    je = data.col(EXCLUDED_ALT)
    c = data.c
    T, N = c.shape
    real = np.asarray(data.real, dtype=bool)
    uni = np.asarray(data.universe, dtype=bool)
    csq = np.zeros((T + 1, N), dtype=np.float64)
    np.cumsum(np.nan_to_num(np.asarray(data.qv, dtype=np.float64), nan=0.0), axis=0, out=csq[1:])
    day = np.asarray(data.times).astype("datetime64[D]").astype(np.int64)

    z, r4b = _leader_z(c[:, jb])
    alt_ok = np.ones(N, dtype=bool)
    alt_ok[jb] = False
    alt_ok[je] = False

    pos = np.zeros((T, N), dtype=np.float64)
    trades = []
    cand = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= k_sigma))
    next_free = 0
    for t in cand:
        t = int(t)
        if t < next_free:          # a book is open: the candle is ignored (no stacking, no re-entry)
            continue
        cols, w, e_gap, beta = _event_book(t, c, real, uni, alt_ok, csq, day, jb, float(r4b[t]), n_hours)
        if cols.size == 0:         # nothing passes the rules: no book is opened
            continue
        end = min(t + HOLD_CANDLES, T)          # target rows t..t+15, zero at t+16 -> exit at open t+17
        block = np.empty((end - t, cols.size), dtype=np.float64)
        block[:] = w
        cancelled = np.zeros(cols.size, dtype=bool)
        if t + 1 < T:
            # gap candle at t+1: the entry order cannot fill there; cancel it at row t+1 (known then)
            cancelled = ~real[t + 1, cols]
            block[1:, cancelled] = 0.0
        # hold only universe members: close at the first row outside the universe, no re-entry
        out = np.logical_or.accumulate(~uni[t:end][:, cols], axis=0)
        block[out] = 0.0
        pos[t:end, cols] = block
        held = (block != 0).sum(axis=0)
        trades.append((t, cols, w, e_gap, beta, held, cancelled))
        next_free = t + HOLD_CANDLES + 1
    return pos, trades


def target_positions(data, params):
    pos, _ = _run(data, params)
    return pos


def trade_log(data, params):
    """Step (7) of the card, for the Librarian's falsification checks (not used by target_positions).
    Per trade: decision row, entry/exit fill rows, instrument, E_j, beta_j, side, weight, the gross
    open-to-open return of j from entry to exit and BTC's open-to-open log return over the same interval.
    Uses rows after the decision by construction (returns after entry); NaN where the exit is beyond the data."""
    _, trades = _run(data, params)
    o = data.o
    T = o.shape[0]
    jb = data.col(LEADER)
    rec = {k: [] for k in ("decision_row", "entry_row", "exit_row", "instrument", "E", "beta", "side",
                           "weight", "gross_return", "btc_log_return")}
    for t, cols, w, e_gap, beta, held, cancelled in trades:
        for i in range(cols.size):
            if cancelled[i]:
                continue
            j = int(cols[i])
            entry = t + 1
            exit_ = t + int(held[i]) + 1
            if entry < T and exit_ < T:
                g = float(o[exit_, j]) / float(o[entry, j]) - 1.0
                b = float(np.log(float(o[exit_, jb]) / float(o[entry, jb])))
            else:
                g = np.nan
                b = np.nan
            rec["decision_row"].append(t)
            rec["entry_row"].append(entry)
            rec["exit_row"].append(exit_)
            rec["instrument"].append(data.instruments[j])
            rec["E"].append(float(e_gap[i]))
            rec["beta"].append(float(beta[i]))
            rec["side"].append(float(np.sign(w[i])))
            rec["weight"].append(float(w[i]))
            rec["gross_return"].append(g)
            rec["btc_log_return"].append(b)
    return {k: np.asarray(v) for k, v in rec.items()}
