"""H-0024 Concurrent-filing earnings-surprise drift (LIQ-500, dollar-neutral, hold 5 sessions).

Rules of card.signal.description -> code:
 (1) filing detection          -> _detect_filings: sf1_art_netinc or sf1_art_marketcap differs from row t-1,
                                  with netinc and marketcap finite on both t and t-1.
 (2) surprise                  -> S = (netinc[t] - netinc[t-1]) / marketcap[t], skipped if marketcap <= 0.
 (3) concurrency               -> dollar_volume[t-1] or [t] >= vol_mult * median(dollar_volume[t-64 .. t-2]),
                                  at least 40 finite values in that 63-row window.
 (4) extremes                  -> q / 1-q quantiles of S over all detected filings of universe members in rows
                                  t-251 .. t (>= 200 filings, else no events); concurrent filings at or beyond
                                  them are long (+1) / short (-1) events.
 (5) book                      -> each event active on decision rows t .. t+4 (enter open t+1, exit open t+6);
                                  a new event replaces the old one; +0.10 / -0.10 per name, each side capped at
                                  0.5; net hedged equally over the other current universe members; a full row
                                  only when the signed set of active names changes, else NaN (keep).
 (6) missing data              -> an active name that leaves the universe / stops trading / delists is dropped
                                  (weight 0 at the next decision); held hedge names that leave the universe are
                                  set to 0 on the non-recomputed rows.
"""
import numpy as np

PARAMS = {"quantile": 0.2, "vol_mult": 2.0}

HOLD_SESSIONS = 5          # enter at the open of t+1, exit at the open of t+6
PER_NAME = 0.10            # target weight per active name
SIDE_CAP = 0.5             # max total per side
POOL_SESSIONS = 252        # rows t-251 .. t
MIN_POOL = 200             # minimum filings in the pool
VOL_FIRST = 64             # median window starts at t-64
VOL_LAST = 2               # ... and ends at t-2 (63 sessions)
MIN_VOL_OBS = 40           # minimum finite dollar-volume values in the window


def _as_bool(x):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    return np.nan_to_num(a.astype(np.float64), nan=0.0) != 0


def _eligible(data):
    """(T, N) bool: universe member, listed, tradable and not being delisted on row t."""
    if data.universe is None:
        raise ValueError("H-0024 needs the point-in-time LIQ-500 universe (data.universe is None)")
    uni = _as_bool(data.universe)
    ok = uni & _as_bool(data.listed) & _as_bool(data.tradable) & ~_as_bool(data.delisting)
    return uni, ok


def _detect_filings(data, uni):
    """Rows/columns/surprise of detected filings of universe members (sorted by row)."""
    ni = np.asarray(data.extras["sf1_art_netinc"])
    mc = np.asarray(data.extras["sf1_art_marketcap"])
    T, N = ni.shape
    new = np.zeros((T, N), dtype=bool)
    if T > 1:
        fin = np.isfinite(ni[1:]) & np.isfinite(ni[:-1]) & np.isfinite(mc[1:]) & np.isfinite(mc[:-1])
        changed = (ni[1:] != ni[:-1]) | (mc[1:] != mc[:-1])
        new[1:] = fin & changed
    with np.errstate(invalid="ignore"):
        new &= mc > 0                      # skip if marketcap <= 0 (NaN already excluded)
    new &= uni                             # pool: filings of universe members on row t
    tt, jj = np.nonzero(new)               # row-major -> sorted by row
    s = (ni[tt, jj].astype(np.float64) - ni[tt - 1, jj].astype(np.float64)) / mc[tt, jj].astype(np.float64)
    return tt, jj, s


def _concurrent(data, tt, jj, vol_mult):
    """Bool per filing: dollar volume on t-1 or t >= vol_mult x median over rows t-64 .. t-2."""
    dv = np.asarray(data.dollar_volume)
    out = np.zeros(tt.shape[0], dtype=bool)
    if tt.shape[0] == 0:
        return out
    offs = np.arange(-VOL_FIRST, -VOL_LAST + 1)        # -64 .. -2, 63 values
    rows = tt[:, None] + offs[None, :]
    inside = rows >= 0
    win = dv[np.clip(rows, 0, None), jj[:, None]].astype(np.float64)
    win[~inside] = np.nan
    nvalid = np.isfinite(win).sum(axis=1)
    enough = nvalid >= MIN_VOL_OBS
    if not enough.any():
        return out
    med = np.nanmedian(win[enough], axis=1)
    thr = vol_mult * med
    d_prev = dv[tt[enough] - 1, jj[enough]].astype(np.float64)
    d_cur = dv[tt[enough], jj[enough]].astype(np.float64)
    with np.errstate(invalid="ignore"):
        hit = (d_prev >= thr) | (d_cur >= thr)
    out[enough] = hit
    return out


def compute_events(data, params):
    """(T, N) int8 event side: +1 long event, -1 short event, 0 none (decided after the close of row t)."""
    q = float(params["quantile"])
    vol_mult = float(params["vol_mult"])
    uni, ok = _eligible(data)
    T, N = uni.shape
    side = np.zeros((T, N), dtype=np.int8)
    tt, jj, s = _detect_filings(data, uni)
    if tt.shape[0] == 0:
        return side
    cand = ok[tt, jj]
    cand_idx = np.nonzero(cand)[0]
    if cand_idx.shape[0] == 0:
        return side
    conc = _concurrent(data, tt[cand_idx], jj[cand_idx], vol_mult)
    cand_idx = cand_idx[conc]
    if cand_idx.shape[0] == 0:
        return side
    for d in np.unique(tt[cand_idx]):
        lo = np.searchsorted(tt, d - (POOL_SESSIONS - 1), side="left")
        hi = np.searchsorted(tt, d, side="right")
        if hi - lo < MIN_POOL:
            continue
        q_lo, q_hi = np.quantile(s[lo:hi], [q, 1.0 - q])
        today = cand_idx[tt[cand_idx] == d]
        sv = s[today]
        is_long = sv >= q_hi
        is_short = sv <= q_lo
        both = is_long & is_short            # only possible when q_lo == q_hi == S: ambiguous, no event
        side[d, jj[today[is_long & ~both]]] = 1
        side[d, jj[today[is_short & ~both]]] = -1
    return side


def target_weights(data, params):
    ev = compute_events(data, params)
    uni, ok = _eligible(data)
    T, N = uni.shape
    W = np.full((T, N), np.nan)
    end = np.full(N, -1, dtype=np.int64)      # last decision row on which the event is active
    side = np.zeros(N, dtype=np.int8)
    prev_sig = None
    last_full = np.zeros(N)
    ev_rows = np.nonzero(ev.any(axis=1))[0]
    ev_row_set = set(int(r) for r in ev_rows)
    for t in range(T):
        # (6) active names that left the universe / stopped trading / delist: exit at the next open
        active = end >= t
        drop = active & ~ok[t]
        if drop.any():
            end[drop] = -1
        # (5) new events: replace any older event of the same stock, clock restarts
        if t in ev_row_set:
            e = ev[t]
            js = np.nonzero(e)[0]
            end[js] = t + HOLD_SESSIONS - 1
            side[js] = e[js]
        sig = np.where(end >= t, side, 0).astype(np.int8)
        if prev_sig is None or np.any(sig != prev_sig):
            w = np.zeros(N)
            longs = sig > 0
            shorts = sig < 0
            n_l = int(longs.sum())
            n_s = int(shorts.sum())
            if n_l + n_s > 0:
                wl = min(PER_NAME, SIDE_CAP / n_l) if n_l else 0.0
                ws = min(PER_NAME, SIDE_CAP / n_s) if n_s else 0.0
                w[longs] = wl
                w[shorts] = -ws
                net = wl * n_l - ws * n_s
                hedge = uni[t] & (sig == 0)
                m = int(hedge.sum())
                if m > 0 and net != 0.0:
                    w[hedge] = -net / m
            W[t] = w
            last_full = w
            prev_sig = sig
        else:
            # no change of the active set: keep positions, but exit held names that left the universe
            gone = (last_full != 0) & ~uni[t]
            if gone.any():
                W[t, gone] = 0.0
    return W
