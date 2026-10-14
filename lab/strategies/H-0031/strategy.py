"""H-0031 Earnings-reaction drift with fixed risk-scaled event slots (LIQ-500).

Rules of card.signal.description and where they are implemented:

(1) r[t,j] = (1+ret_co)*(1+ret_oc) - 1; a[t,j] = r[t,j] - mean of r[t,.] over universe members
    with a price on day t                                                  -> _reaction_events
(2) sigma[t,j] = sample std (ddof=1) of a[s,j], s = t-63..t-1 (row t excluded), at least 50
    valid values (valid = listed, a price on day s, and a market mean on day s)  -> _reaction_events
(3) event: ea_reaction_day == 1 (NaN counts as 0), universe member at t, sigma valid,
    |z| >= z_threshold, z = a/sigma; side = sign(z)                        -> _reaction_events
(4) slot weight side * event_weight * clip(median_sigma_t / sigma, 0.5, 2.0), median over
    universe members with a valid sigma; held on decision rows t..t+hold_days-1 (filled at the
    open of t+1, closed at the open of t+1+hold_days); a new event replaces the slot; a slot is
    closed (target 0) on the first row where the stock is not a universe member or not
    tradable                                                               -> target_weights
(5) hedge -N_t (N_t = sum of signed slot weights) spread equally over a basket of the 100
    universe members with the best (lowest) liq_rank and no open slot, chosen on the first
    trading day of each month; names leaving the universe drop out; basket names with an open
    slot are skipped on that day; size reset every day                    -> target_weights
(6) if sum|slot w| + |hedge| > 1, all weights of the day are scaled down proportionally.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {
    "event_weight": 0.025,
    "hold_days": 5,
    "z_threshold": 2.0,
}

VOL_WINDOW = 63        # fixed design choice of the card
VOL_MIN_OBS = 50       # fixed design choice of the card
CLIP_LO = 0.5          # fixed design choice of the card
CLIP_HI = 2.0          # fixed design choice of the card
HEDGE_NAMES = 100      # fixed design choice of the card
MAX_GROSS = 1.0


def _reaction_events(ret_co, ret_oc, listed, close, universe, tradable, flag, z_threshold):
    """Point-in-time event detection, one forward pass over rows.

    Returns event (T, N) bool, z (T, N) float64 (NaN where not computable),
    sigma (T, N) float64 (NaN where invalid) and med_sigma (T,) float64.
    """
    T, N = ret_co.shape
    event = np.zeros((T, N), dtype=bool)
    z_out = np.full((T, N), np.nan)
    sig_out = np.full((T, N), np.nan)
    med_out = np.full(T, np.nan)

    buf_a = np.zeros((VOL_WINDOW, N))
    buf_v = np.zeros((VOL_WINDOW, N), dtype=np.int64)
    s1 = np.zeros(N)
    s2 = np.zeros(N)
    cnt = np.zeros(N, dtype=np.int64)

    for t in range(T):
        co = np.asarray(ret_co[t], dtype=np.float64)
        oc = np.asarray(ret_oc[t], dtype=np.float64)
        rt = (1.0 + co) * (1.0 + oc) - 1.0
        ok = np.asarray(listed[t], dtype=bool) & np.isfinite(np.asarray(close[t], dtype=np.float64)) & np.isfinite(rt)
        uni_t = np.asarray(universe[t], dtype=bool)
        mem = uni_t & ok
        if mem.any():
            a_ok = ok
            at = np.where(a_ok, rt - rt[mem].mean(), 0.0)
        else:
            a_ok = np.zeros(N, dtype=bool)
            at = np.zeros(N)

        # (2) sigma over rows t-63..t-1 (the running sums do not contain row t yet)
        enough = cnt >= VOL_MIN_OBS
        n = np.where(enough, cnt, 2).astype(np.float64)
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        var = np.where(enough, np.maximum(var, 0.0), np.nan)
        sigma = np.sqrt(var)
        sig_ok = enough & np.isfinite(sigma) & (sigma > 0.0)
        sigma = np.where(sig_ok, sigma, np.nan)
        sig_out[t] = sigma

        # (3) event
        zt = np.full(N, np.nan)
        zc = sig_ok & a_ok
        zt[zc] = at[zc] / sigma[zc]
        z_out[t] = zt
        f = np.asarray(flag[t], dtype=np.float64)
        flag_on = f == 1.0                      # NaN == 1 is False: NaN flags count as 0
        ev = flag_on & uni_t & np.asarray(tradable[t], dtype=bool) & zc & (np.abs(np.where(zc, zt, 0.0)) >= z_threshold)
        event[t] = ev

        med_mask = uni_t & sig_ok
        if med_mask.any():
            med_out[t] = float(np.median(sigma[med_mask]))

        # roll the window: drop row t-63, add row t
        k = t % VOL_WINDOW
        if t >= VOL_WINDOW:
            old = buf_a[k]
            s1 -= old
            s2 -= old * old
            cnt -= buf_v[k]
        buf_a[k] = at
        buf_v[k] = a_ok.astype(np.int64)
        s1 += at
        s2 += at * at
        cnt += buf_v[k]

    return event, z_out, sig_out, med_out


def _month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    T = len(dates)
    if ps.dtype == bool:
        return ps.reshape(T)
    mask = np.zeros(T, dtype=bool)
    mask[ps.astype(np.int64)] = True
    return mask


def _book(event, z, sigma, med, universe, tradable, liq_rank, month_start,
          event_weight, hold_days):
    T, N = event.shape
    w = np.zeros((T, N))
    start = np.full(N, -1, dtype=np.int64)
    wslot = np.zeros(N)
    basket = np.zeros(N, dtype=bool)
    formed = False

    for t in range(T):
        uni_t = np.asarray(universe[t], dtype=bool)
        trad_t = np.asarray(tradable[t], dtype=bool)

        # (4) expiry after hold_days rows, exit when out of the universe or not tradable
        open_ = (start >= 0) & ((t - start) < hold_days) & uni_t & trad_t
        start[~open_] = -1
        wslot[~open_] = 0.0

        # (4) new events open (or replace) a slot
        ev = event[t]
        if ev.any():
            ratio = np.clip(med[t] / sigma[t, ev], CLIP_LO, CLIP_HI)
            wslot[ev] = np.sign(z[t, ev]) * event_weight * ratio
            start[ev] = t

        has_slot = start >= 0

        # (5) hedge basket, refreshed on the first trading day of each month
        if month_start[t] or not formed:
            lr = np.asarray(liq_rank[t], dtype=np.float64)
            cand = np.flatnonzero(uni_t & ~has_slot & np.isfinite(lr))
            order = cand[np.argsort(lr[cand], kind="stable")][:HEDGE_NAMES]
            basket = np.zeros(N, dtype=bool)
            basket[order] = True
            formed = bool(basket.any())
        basket &= uni_t                        # leavers drop out until the next refresh
        active = basket & ~has_slot

        n_t = float(wslot.sum())
        row = wslot.copy()
        hedge_gross = 0.0
        n_active = int(active.sum())
        if n_active > 0 and n_t != 0.0:
            row[active] = -n_t / n_active
            hedge_gross = abs(n_t)

        # (6) gross cap
        gross = float(np.abs(wslot).sum()) + hedge_gross
        if gross > MAX_GROSS:
            row *= MAX_GROSS / gross
        w[t] = row
    return w


def target_weights(data, params):
    event_weight = float(params["event_weight"])
    hold_days = int(params["hold_days"])
    z_threshold = float(params["z_threshold"])

    universe = data.universe
    if universe is None:
        raise ValueError("H-0031 needs the point-in-time LIQ-500 universe")
    flag = data.extras["ea_reaction_day"]
    liq_rank = data.extras["liq_rank"]

    event, z, sigma, med = _reaction_events(
        data.ret_co, data.ret_oc, data.listed, data.close, universe, data.tradable, flag, z_threshold)
    month_start = _month_start_mask(data.dates)
    return _book(event, z, sigma, med, universe, data.tradable, liq_rank, month_start,
                 event_weight, hold_days)
