"""H-0157 Home-closure anchoring in Asian ADRs (LIQ-500).

Decision after the close of US session t, execution at the open of t+1.

Rules (card signal.description):
 (1) excess return = member total return minus the equal-weight mean total return of the LIQ-500
     members on that session.
 (2) home assignment, recomputed after the close of the first session of each month (period_starts "M")
     and held until the next one: Pearson correlation of the stock's excess returns with each country
     ETF's excess returns over the last 252 sessions (rows t-251..t), >= 200 pairwise-valid sessions, an
     ETF with fewer than 252 sessions since its first valid return is skipped; assigned to the exchange
     of the best ETF if corr >= 0.5 (fixed), else unassigned.
 (3) event at t for assigned stock j with home X: X.anchored(t) = 0, X.closed_next(t) = 1,
     X.anchored(t-1) = 1; m = excess(t); s = std of excess over rows t-63..t-1 (>= 50 valid);
     |m / s| >= z_min; side = -sign(m).
 (4) slot 0.20 x side on decision rows t .. t+hold_days-1 (flat from row t+hold_days, i.e. closed at
     the open of t+1+hold_days), closed early at the first later decision row where j is not a member
     or not tradable; slot gross capped at 0.5 by proportional scaling; net slot weight offset equally
     over the 50 members with the highest 63-session median dollar volume that are unassigned and have
     no open slot. Cash otherwise. The ETFs are signal only and never held.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 3, "z_min": 0.25}

ETF_MAP = (
    ("EWJ", "XTKS"),
    ("EWH", "XHKG"),
    ("FXI", "XHKG"),
    ("EWY", "XKRX"),
    ("EWT", "XTAI"),
    ("EWS", "XSES"),
    ("EWM", "XKLS"),
    ("EWA", "XASX"),
)
EXCHANGES = ("XTKS", "XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX")
HOLIDAY_DATASET = "exchange_holidays_1d"

SLOT_WEIGHT = 0.20
SLOT_GROSS_CAP = 0.5
N_HEDGE = 50
DV_WINDOW = 63
CORR_WINDOW = 252
CORR_MIN_OBS = 200
CORR_THRESHOLD = 0.5      # fixed by the card, not a parameter
VOL_WINDOW = 63
VOL_MIN_OBS = 50


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------
def _universe(data):
    uni = data.universe
    if uni is None:
        raise ValueError("H-0157 needs the LIQ-500 point-in-time universe (data.universe is None)")
    return np.asarray(uni, dtype=bool)


def _etf_columns(data):
    cols = np.array([data.col(name) for name, _ in ETF_MAP], dtype=np.int64)
    exch = np.array([EXCHANGES.index(x) for _, x in ETF_MAP], dtype=np.int64)
    return cols, exch


def _holiday_flags(data):
    T = len(data.dates)
    anchored = np.empty((T, len(EXCHANGES)), dtype=np.float64)
    closed_next = np.empty((T, len(EXCHANGES)), dtype=np.float64)
    for k, x in enumerate(EXCHANGES):
        a = np.asarray(data.series[HOLIDAY_DATASET + "." + x + ".anchored"], dtype=np.float64).reshape(-1)
        c = np.asarray(data.series[HOLIDAY_DATASET + "." + x + ".closed_next"], dtype=np.float64).reshape(-1)
        if a.shape[0] != T or c.shape[0] != T:
            raise ValueError("holiday flag series for " + x + " is not aligned to the view")
        anchored[:, k] = a
        closed_next[:, k] = c
    return anchored, closed_next


def _excess_returns(data, uni, etf_cols):
    """(T, N) float32 excess total return; NaN where the session return is not valid."""
    rco = np.asarray(data.ret_co, dtype=np.float64)
    roc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + rco) * (1.0 + roc) - 1.0
    close = np.asarray(data.close, dtype=np.float64)
    listed = np.asarray(data.listed, dtype=bool)
    ok = listed & np.isfinite(close) & np.isfinite(r)
    del close, rco, roc
    valid = ok.copy()
    valid[0] = False
    valid[1:] &= ok[:-1]          # a session return needs a price on the previous session too
    del ok
    mem = uni & valid
    mem[:, etf_cols] = False      # ETFs are never peers
    cnt = mem.sum(axis=1)
    tot = np.where(mem, r, 0.0).sum(axis=1)
    peer = np.full(r.shape[0], np.nan)
    has = cnt > 0
    peer[has] = tot[has] / cnt[has]
    ex = (r - peer[:, None]).astype(np.float32)
    ex[~valid] = np.nan
    ex[~has] = np.nan
    return ex


def _month_starts(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return np.flatnonzero(ps)
    return np.sort(ps.astype(np.int64))


# ----------------------------------------------------------------------------------------------
# (2) home assignment
# ----------------------------------------------------------------------------------------------
def _home_assignment(data, ex, uni, etf_cols, etf_exch):
    """(T, N) int8: index into EXCHANGES of the assigned home exchange, -1 = unassigned."""
    T, N = ex.shape
    home = np.full((T, N), -1, dtype=np.int8)
    etf_ex = ex[:, etf_cols].astype(np.float64)
    etf_valid = np.isfinite(etf_ex)
    anyv = etf_valid.any(axis=0)
    first_valid = np.where(anyv, np.argmax(etf_valid, axis=0), T)
    is_etf = np.zeros(N, dtype=bool)
    is_etf[etf_cols] = True

    starts = _month_starts(data.dates)
    for k, t in enumerate(starts):
        t = int(t)
        end = int(starts[k + 1]) if k + 1 < len(starts) else T
        if t < CORR_WINDOW - 1:
            continue
        lo = t - CORR_WINDOW + 1
        # ETF needs 252 sessions of history (since its first valid return) as of t
        etf_ok = first_valid <= lo
        if not etf_ok.any():
            continue
        members = np.flatnonzero(uni[t] & ~is_etf)
        if members.size == 0:
            continue
        X = ex[lo:t + 1][:, members].astype(np.float64)
        Y = etf_ex[lo:t + 1][:, etf_ok]
        Mx = np.isfinite(X)
        My = np.isfinite(Y)
        X0 = np.where(Mx, X, 0.0)
        Y0 = np.where(My, Y, 0.0)
        Mxf = Mx.astype(np.float64)
        Myf = My.astype(np.float64)
        n = Mxf.T @ Myf
        sx = X0.T @ Myf
        sy = Mxf.T @ Y0
        sxx = (X0 * X0).T @ Myf
        syy = Mxf.T @ (Y0 * Y0)
        sxy = X0.T @ Y0
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            corr = cov / np.sqrt(vx * vy)
        good = (n >= CORR_MIN_OBS) & (vx > 0) & (vy > 0) & np.isfinite(corr)
        corr = np.where(good, corr, -np.inf)
        best = np.argmax(corr, axis=1)
        best_c = corr[np.arange(members.size), best]
        assigned = best_c >= CORR_THRESHOLD
        exch_of_col = etf_exch[etf_ok]
        a = np.full(members.size, -1, dtype=np.int8)
        a[assigned] = exch_of_col[best[assigned]].astype(np.int8)
        home[t:end, members] = a[None, :]
    return home


# ----------------------------------------------------------------------------------------------
# (3) events
# ----------------------------------------------------------------------------------------------
def _closure_starts(anchored, closed_next):
    """(T, 7) bool: t is the first unanchored US session of a closure spanning >= 2 US sessions."""
    T = anchored.shape[0]
    e = np.zeros(anchored.shape, dtype=bool)
    if T > 1:
        e[1:] = (anchored[1:] == 0.0) & (closed_next[1:] == 1.0) & (anchored[:-1] == 1.0)
    return e


def _events(data, params, ex, home, uni, etf_cols):
    """mask (T, N) bool and side (T, N) float (+1 / -1, 0 where no event)."""
    z_min = float(params["z_min"])
    T, N = ex.shape
    anchored, closed_next = _holiday_flags(data)
    closure = _closure_starts(anchored, closed_next)
    tradable = np.asarray(data.tradable, dtype=bool)
    is_etf = np.zeros(N, dtype=bool)
    is_etf[etf_cols] = True
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    for t in np.flatnonzero(closure.any(axis=1)):
        h = home[t].astype(np.int64)
        assigned = h >= 0
        hit = np.zeros(N, dtype=bool)
        hit[assigned] = closure[t, h[assigned]]
        cand = np.flatnonzero(hit & uni[t] & tradable[t] & ~is_etf & np.isfinite(ex[t]))
        if cand.size == 0:
            continue
        lo = max(0, t - VOL_WINDOW)
        W = ex[lo:t][:, cand].astype(np.float64)
        V = np.isfinite(W)
        nv = V.sum(axis=0)
        W0 = np.where(V, W, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = W0.sum(axis=0) / nv
            dev = np.where(V, W - mu[None, :], 0.0)
            s = np.sqrt((dev * dev).sum(axis=0) / (nv - 1))
        m = ex[t, cand].astype(np.float64)
        ok = (nv >= VOL_MIN_OBS) & np.isfinite(s) & (s > 0) & (m != 0.0)
        z = np.zeros(cand.size)
        z[ok] = m[ok] / s[ok]
        fire = ok & (np.abs(z) >= z_min)
        mask[t, cand[fire]] = True
        side[t, cand[fire]] = -np.sign(m[fire])
    return mask, side


# ----------------------------------------------------------------------------------------------
# (4) book
# ----------------------------------------------------------------------------------------------
def target_weights(data, params):
    hold_days = int(params["hold_days"])
    uni = _universe(data)
    etf_cols, etf_exch = _etf_columns(data)
    T, N = uni.shape
    ex = _excess_returns(data, uni, etf_cols)
    home = _home_assignment(data, ex, uni, etf_cols, etf_exch)
    mask, side = _events(data, params, ex, home, uni, etf_cols)
    del ex

    tradable = np.asarray(data.tradable, dtype=bool)
    slots = np.zeros((T, N), dtype=np.float64)
    open_slot = np.zeros((T, N), dtype=bool)
    ev_t, ev_j = np.nonzero(mask)
    for t, j in zip(ev_t, ev_j):
        w = SLOT_WEIGHT * side[t, j]
        for u in range(t, min(T, t + hold_days)):
            if u > t and not (uni[u, j] and tradable[u, j]):
                break
            slots[u, j] += w
            open_slot[u, j] = True

    weights = np.zeros((T, N), dtype=np.float64)
    is_etf = np.zeros(N, dtype=bool)
    is_etf[etf_cols] = True
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    for u in np.flatnonzero(open_slot.any(axis=1)):
        row = slots[u].copy()
        gross = np.abs(row).sum()
        if gross > SLOT_GROSS_CAP:
            row *= SLOT_GROSS_CAP / gross
        net = row.sum()
        weights[u] = row
        if net == 0.0:
            continue
        elig = np.flatnonzero(uni[u] & ~is_etf & (home[u] < 0) & ~open_slot[u])
        if elig.size == 0:
            continue
        lo = max(0, u - DV_WINDOW + 1)
        D = dv[lo:u + 1][:, elig]
        has = np.isfinite(D).any(axis=0)
        elig = elig[has]
        if elig.size == 0:
            continue
        med = np.nanmedian(D[:, has], axis=0)
        order = np.lexsort((elig, -med))      # highest median first, ties by column index
        hedge = elig[order[:N_HEDGE]]
        weights[u, hedge] = -net / hedge.size
    return weights
