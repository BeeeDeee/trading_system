"""H-0034 Non-earnings 8-K news-reaction drift (LIQ-500, fixed event slots).

Decision after the close of row t from rows 0..t only; executed by the engine at the open of t+1.

Rules of the card:
 (1) r(i,t) = (1 + ret_co) * (1 + ret_oc) - 1; a(i,t) = r(i,t) - equal-weight mean of r over the LIQ-500
     members with a valid return on t.
 (2) sigma(i,t) = sample std (ddof=1) of the valid a(i,.) over rows t-60 .. t-1, needs >= 40 valid values.
 (3) z = a / sigma.
 (4) event: member at t, nonearn_8k_reaction_day == 1 at t, ea_reaction_day == 0 (not 1, not NaN) on
     rows t-2, t-1, t; |z| >= z_threshold; side = sign(z).
 (5) slot of side * 0.10 on decision rows t .. t+hold_days-1 (enter open t+1, exit open t+1+hold_days);
     a new event on a live stock replaces the slot (new side, clock reset); gross > 1 -> all live slots
     scaled pro rata to gross 1; a slot whose stock is not a member or not tradable on a row is closed
     (weight 0, permanently until a new event).
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_threshold": 2.0}

SLOT = 0.10          # fixed design constant of the card (not a parameter)
WINDOW = 60          # sessions t-60 .. t-1
MIN_OBS = 40         # minimum valid abnormal returns in the window
EA_LOOKBACK = 2      # no Item 2.02 reaction session in t-2 .. t


def compute_events(data, params):
    """Return (event (T, N) bool, side (T, N) int8) per the card's rules (1)-(4)."""
    thr = float(params["z_threshold"])
    if data.universe is None:
        raise ValueError("H-0034 needs the LIQ-500 point-in-time universe")
    univ = np.asarray(data.universe, dtype=bool)
    nonearn = np.asarray(data.extras["nonearn_8k_reaction_day"])
    ea = np.asarray(data.extras["ea_reaction_day"])

    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    valid = (np.asarray(data.listed, dtype=bool)
             & np.isfinite(np.asarray(data.close, dtype=np.float64))
             & np.isfinite(r))
    T, N = r.shape

    # (1) abnormal return against the equal-weight mean of valid members on the same row
    mv = univ & valid
    cnt = mv.sum(axis=1)
    tot = np.where(mv, r, 0.0).sum(axis=1)
    mkt = np.full(T, np.nan)
    has = cnt > 0
    mkt[has] = tot[has] / cnt[has]
    a = np.where(valid, r - mkt[:, None], np.nan)
    del r, tot
    a_ok = np.isfinite(a)

    # (4) flags: NaN never counts (NaN == 0 and NaN == 1 are False)
    ea_zero = ea == 0
    ea_ok = ea_zero.copy()
    ea_ok[:EA_LOOKBACK] = False
    for k in range(1, EA_LOOKBACK + 1):
        ea_ok[EA_LOOKBACK:] &= ea_zero[EA_LOOKBACK - k:T - k]
    cand = univ & (nonearn == 1) & ea_ok & a_ok

    event = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.int8)
    rows = np.flatnonzero(cand.any(axis=1))
    for t in rows:
        if t < MIN_OBS:
            continue
        idx = np.flatnonzero(cand[t])
        block = a[t - WINDOW if t >= WINDOW else 0:t][:, idx]     # rows t-60 .. t-1 only
        ok = np.isfinite(block)
        n = ok.sum(axis=0)
        b0 = np.where(ok, block, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = b0.sum(axis=0) / n
            dev = np.where(ok, block - mean, 0.0)
            var = (dev * dev).sum(axis=0) / (n - 1)
            sigma = np.sqrt(var)
            z = a[t, idx] / sigma
        good = (n >= MIN_OBS) & np.isfinite(z) & (sigma > 0) & (np.abs(z) >= thr)
        sel = idx[good]
        event[t, sel] = True
        side[t, sel] = np.sign(z[good]).astype(np.int8)
    return event, side


def target_weights(data, params):
    hold = int(params["hold_days"])
    event, side = compute_events(data, params)
    univ = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    T, N = event.shape

    w = np.zeros((T, N), dtype=np.float64)
    exit_row = np.full(N, -1, dtype=np.int64)   # first decision row on which the slot is 0
    sd = np.zeros(N, dtype=np.float64)
    for t in range(T):
        e = np.flatnonzero(event[t])
        if e.size:
            sd[e] = side[t, e]
            exit_row[e] = t + hold               # live on rows t .. t+hold-1
        # leaving the universe or not tradable while live: close the slot (weight 0)
        exit_row[~(univ[t] & trad[t])] = -1
        live = exit_row > t
        k = int(live.sum())
        if k == 0:
            continue
        row = np.where(live, SLOT * sd, 0.0)
        g = SLOT * k
        if g > 1.0:
            row = row / np.abs(row).sum() * (1.0 - 1e-12)   # pro rata to gross 1.0
        w[t] = row
    return w
