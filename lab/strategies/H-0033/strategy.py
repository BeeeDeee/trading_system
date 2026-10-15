"""H-0033: positive earnings-reaction tilt on a cap-weighted LIQ-500 core (long-only).

Rules of card.signal.description and where they live:
 (1) r = (1+ret_co)(1+ret_oc)-1; a = r - leave-one-out equal-weight mean of r over LIQ-500 members
     tradable on t; sigma = sample std of a over rows t-63..t-1 (>= 40 valid obs)   -> _reaction_z
 (2) event after close t: member & tradable & ea_reaction_day == 1 & a/sigma >= z_threshold
     (NaN flag = no event, negative reactions ignored)                              -> _events
 (3) core: cap = sf1_arq_marketcap at its last change (anchor) x total-return growth since the
     anchor; c = cap / sum(cap) over members with a value                           -> target_weights loop
 (4) slots: s = min(max_name_weight, (B_MAX - B_used)/n_new), fixed, closed after hold_days
     sessions or when the stock leaves the universe / stops trading                 -> target_weights loop
 (5) w = (1 - B_used_after_today) * c + slots; rows without a slot change or a change of the
     core membership are NaN (no decision: hold, positions drift with prices)       -> target_weights loop
"""
import numpy as np

PARAMS = {"hold_days": 5, "max_name_weight": 0.04, "z_threshold": 2.0}

B_MAX = 0.50            # fixed tilt budget of the card (not a grid parameter)
SIGMA_WINDOW = 63       # rows t-63 .. t-1
SIGMA_MIN_OBS = 40
_VAR_FLOOR = 1e-12      # numerical guard only: a zero-variance history gives no event
_ROOM_EPS = 1e-12       # numerical guard only: budget counted as full below this


def _inputs(data):
    """Fetch every input directly so a missing one raises (K-0010)."""
    if data.universe is None:
        raise ValueError("H-0033 needs the point-in-time LIQ-500 universe (data.universe is None)")
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    mc = np.asarray(data.extras["sf1_arq_marketcap"], dtype=np.float64)
    ret_co = np.asarray(data.ret_co, dtype=np.float64)
    ret_oc = np.asarray(data.ret_oc, dtype=np.float64)
    r = (1.0 + ret_co) * (1.0 + ret_oc) - 1.0
    trad = np.asarray(data.tradable, dtype=bool)
    elig = np.asarray(data.universe, dtype=bool) & trad
    if ea.shape != r.shape or mc.shape != r.shape:
        raise ValueError("extras have the wrong shape")
    return r, trad, elig, ea, mc


def _lagged_window_sum(x, window):
    """Sum of x over rows t-window .. t-1 (rows < t only), shape like x."""
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=np.float64)
    np.cumsum(x, axis=0, out=c[1:])
    lo = np.maximum(np.arange(T) - window, 0)
    return c[:T] - c[lo]


def _reaction_z(r, trad, elig):
    """a[t,j] / sigma[t,j]; NaN where a is not valid or sigma is not defined."""
    fin = np.isfinite(r)
    mem = elig & fin                        # members tradable on t with a return
    rm = np.where(mem, r, 0.0)
    S = rm.sum(axis=1, keepdims=True)
    K = mem.sum(axis=1, keepdims=True).astype(np.float64)
    k_ex = K - mem                          # members other than j
    valid = trad & fin & (k_ex >= 1)        # j had a session on t and has a comparison group
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(valid, r - (S - rm) / np.where(k_ex >= 1, k_ex, 1.0), 0.0)
    del rm
    n = _lagged_window_sum(valid.astype(np.float64), SIGMA_WINDOW)
    s1 = _lagged_window_sum(a, SIGMA_WINDOW)
    s2 = _lagged_window_sum(a * a, SIGMA_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1.0)
    del s1, s2
    ok = valid & (n >= SIGMA_MIN_OBS) & (var > _VAR_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ok, a / np.sqrt(np.where(ok, var, 1.0)), np.nan)
    return z


def _events(r, trad, elig, ea, z_threshold):
    z = _reaction_z(r, trad, elig)
    with np.errstate(invalid="ignore"):
        ev = elig & (ea == 1.0) & (z >= z_threshold)
    return ev


def target_weights(data, params):
    hold = int(params["hold_days"])
    mnw = float(params["max_name_weight"])
    zthr = float(params["z_threshold"])

    r, trad, elig, ea, mc = _inputs(data)
    T, N = r.shape
    ev = _events(r, trad, elig, ea, zthr)
    growth = np.where(np.isfinite(r), 1.0 + r, 1.0)

    W = np.full((T, N), np.nan, dtype=np.float64)
    cap = np.full(N, np.nan)
    prev_mc = np.full(N, np.nan)
    slot_w = np.zeros(N)
    slot_exit = np.zeros(N, dtype=np.int64)
    prev_core = np.zeros(N, dtype=bool)

    for t in range(T):
        # (3) cap estimate: drift with total return since the anchor, reset at a new filing value
        m = mc[t]
        mfin = np.isfinite(m)
        cap = cap * growth[t]
        changed = mfin & (~np.isfinite(prev_mc) | (m != prev_mc))
        cap = np.where(changed, m, cap)
        cap = np.where(mfin, cap, np.nan)
        prev_mc = m

        # (4) close slots that reached their exit or whose stock left the universe / stopped trading
        open_ = slot_w > 0
        closing = open_ & ((slot_exit <= t) | ~elig[t])
        slot_w[closing] = 0.0
        open_ = slot_w > 0
        b_used = float(slot_w.sum())

        # (4) open new slots for tonight's events (no second slot on a stock with an open slot)
        new = ev[t] & ~open_
        n_new = int(new.sum())
        opened = False
        if n_new > 0:
            room = B_MAX - b_used
            if room > _ROOM_EPS:
                s = min(mnw, room / n_new)
                slot_w[new] = s
                slot_exit[new] = t + hold
                opened = True
        b_after = float(slot_w.sum())

        # (3) core over members (tradable on t) with a market-cap value
        core = elig[t] & np.isfinite(cap) & (cap > 0)
        decide = (t == 0) or opened or bool(closing.any()) or bool((core != prev_core).any())
        prev_core = core
        if not decide:
            continue
        csum = float(cap[core].sum())
        w = np.zeros(N)
        if csum > 0:
            w[core] = cap[core] / csum
        # (5) pro-rata core plus fixed slots
        W[t] = (1.0 - b_after) * w + slot_w
    return W
