"""H-0079: pooled CEF catch-up to its asset-class ETF anchor.

Seven fund classes share one book of 2 slots. Each slot is long a CEF (0.25) and short its fixed
anchor ETF (0.25), entered at the open after a 5-session beta-adjusted residual of at least
z_entry standard deviations below zero, and held hold_days sessions.
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_entry": 2.0}

# Fixed CEF -> anchor map declared in the card (not chosen from data).
PAIRS = (
    ("NUV", "MUB"), ("NEA", "MUB"), ("NAD", "MUB"), ("NZF", "MUB"), ("NVG", "MUB"), ("PML", "MUB"),
    ("VKQ", "MUB"), ("VGM", "MUB"), ("EIM", "MUB"), ("MMU", "MUB"), ("MQY", "MUB"), ("MYI", "MUB"),
    ("HYT", "HYG"), ("HIX", "HYG"), ("PHK", "HYG"), ("EHI", "HYG"),
    ("EFT", "BKLN"), ("FRA", "BKLN"), ("JFR", "BKLN"), ("BGT", "BKLN"), ("VVR", "BKLN"),
    ("ETY", "SPY"), ("ETV", "SPY"), ("BXMX", "SPY"), ("GAB", "SPY"), ("ADX", "SPY"), ("TY", "SPY"),
    ("USA", "SPY"),
    ("RQI", "VNQ"), ("RNP", "VNQ"), ("RFI", "VNQ"),
    ("UTG", "XLU"), ("UTF", "XLU"), ("DNP", "XLU"),
    ("GF", "EWG"), ("MXF", "EWW"), ("IFN", "EPI"), ("JOF", "EWJ"),
)

BETA_WINDOW = 252      # sessions ending at t
RESID_WINDOW = 252     # sessions ending at t-5
MIN_OBS = 200          # minimum joint sessions / residual values
RET_DAYS = 5           # 5-session log return
BETA_LO, BETA_HI = 0.25, 2.0
SLOT_CAP = 2
LEG_WEIGHT = 0.25


def _rsum(a, n):
    """Trailing sum over the n rows ending at t (fewer at the start); point-in-time."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > n:
        out[n:] = c[n:] - c[:-n]
    return out


def _lag(a, k, fill):
    out = np.full(a.shape, fill, dtype=a.dtype)
    if a.shape[0] > k:
        out[k:] = a[:-k]
    return out


def pair_columns(data):
    cef = np.array([data.col(c) for c, _ in PAIRS], dtype=int)
    anc = np.array([data.col(a) for _, a in PAIRS], dtype=int)
    return cef, anc


def log_returns(data):
    """Daily total log returns r = log((1+ret_co)(1+ret_oc)) and a 'has a bar' mask."""
    gross = (1.0 + np.asarray(data.ret_co, dtype=float)) * (1.0 + np.asarray(data.ret_oc, dtype=float))
    bar = (np.asarray(data.tradable, dtype=bool) & np.asarray(data.listed, dtype=bool)
           & np.isfinite(gross) & (gross > 0))
    lr = np.where(bar, np.log(np.where(bar, gross, 1.0)), 0.0)
    return lr, bar


def zscores(y, x, by, bx):
    """z_i(t) for each pair column: y = CEF log returns, x = anchor log returns, by/bx bar masks.
    NaN where the card's history or bar rules are not met."""
    w = (by & bx).astype(float)
    xw = np.where(w > 0, x, 0.0)
    yw = np.where(w > 0, y, 0.0)
    # (1) beta over the 252 sessions ending at t, sessions where both have a bar, >= 200 of them
    n = _rsum(w, BETA_WINDOW)
    sx = _rsum(xw, BETA_WINDOW)
    sy = _rsum(yw, BETA_WINDOW)
    sxx = _rsum(xw * xw, BETA_WINDOW)
    sxy = _rsum(xw * yw, BETA_WINDOW)
    nn = np.maximum(n, 1.0)
    varx = sxx - sx * sx / nn
    cov = sxy - sx * sy / nn
    ok_b = (n >= MIN_OBS) & (varx > 0)
    beta = np.clip(cov / np.where(ok_b, varx, 1.0), BETA_LO, BETA_HI)
    beta = np.where(ok_b, beta, np.nan)
    # (2) 5-session log returns over s-4..s; valid only when both have a bar on all 5 sessions
    v5 = _rsum(w, RET_DAYS) >= RET_DAYS - 0.5
    a5 = np.where(v5, _rsum(yw, RET_DAYS), 0.0)
    b5 = np.where(v5, _rsum(xw, RET_DAYS), 0.0)
    vf = v5.astype(float)
    # (3) moments of the overlapping 5-session returns for s in the 252 sessions ending at t-5
    m = _lag(_rsum(vf, RESID_WINDOW), RET_DAYS, 0.0)
    sa = _lag(_rsum(a5, RESID_WINDOW), RET_DAYS, 0.0)
    sb = _lag(_rsum(b5, RESID_WINDOW), RET_DAYS, 0.0)
    saa = _lag(_rsum(a5 * a5, RESID_WINDOW), RET_DAYS, 0.0)
    sbb = _lag(_rsum(b5 * b5, RESID_WINDOW), RET_DAYS, 0.0)
    sab = _lag(_rsum(a5 * b5, RESID_WINDOW), RET_DAYS, 0.0)
    bb = np.where(ok_b, beta, 0.0)
    se = sa - bb * sb                                   # sum of e_i(s) with today's beta_i
    see = saa - 2.0 * bb * sab + bb * bb * sbb          # sum of e_i(s)^2
    mm = np.maximum(m, 2.0)
    var_e = (see - se * se / mm) / (mm - 1.0)           # sample variance (ddof = 1)
    ok_s = ok_b & (m >= MIN_OBS) & (var_e > 0)
    sd = np.sqrt(np.where(ok_s, var_e, 1.0))
    # (4) z = e_i(t) / sd_i
    e = a5 - bb * b5
    return np.where(v5 & ok_s, e / sd, np.nan)


def run_book(z, z_entry, hold, member, tradable, gone):
    """Event rule and 2-slot book over the pair columns.

    z, member, tradable, gone: (T, K) for the K CEFs (member = CEF and anchor may be traded,
    tradable = CEF can fill at the open of row t, gone = CEF not listed / delisting at row t).
    Returns events (T, K) bool and held (T, K) 0/1: slot k targeted after the close of row t."""
    T, K = z.shape
    hold = int(hold)
    cross = np.isfinite(z) & (np.where(np.isfinite(z), z, 0.0) <= -float(z_entry)) & member
    events = np.zeros((T, K), dtype=bool)
    held = np.zeros((T, K), dtype=float)
    last_ev = np.full(K, -(10 ** 9), dtype=np.int64)
    slots = []                                          # [k, t0] with t0 = event row
    for t in range(T):
        keep = []
        n_occ = 0
        has_slot = np.zeros(K, dtype=bool)
        for k, t0 in slots:
            if t == t0 + 1 and not tradable[t, k]:
                continue                                # no tradable bar at the entry open: skipped
            if t <= t0 + hold - 1:
                if gone[t, k]:
                    continue                            # delisted during the hold: paid out, slot closed
                keep.append((k, t0))
                n_occ += 1
                has_slot[k] = True
            elif t == t0 + hold:
                keep.append((k, t0))                    # exit ordered now, fills at the next open
                has_slot[k] = True
            else:
                if tradable[t, k] or gone[t, k]:
                    continue                            # delayed exit filled at this open
                keep.append((k, t0))                    # still stuck: the slot stays occupied
                n_occ += 1
                has_slot[k] = True
        slots = keep
        cand = np.flatnonzero(cross[t] & ((t - last_ev) > hold) & ~has_slot)
        if cand.size:
            events[t, cand] = True
            last_ev[cand] = t
            free = SLOT_CAP - n_occ
            if free > 0:
                order = cand[np.argsort(z[t, cand], kind="stable")]
                for k in order[:free]:
                    slots.append((int(k), t))
        for k, t0 in slots:
            if t0 <= t <= t0 + hold - 1:
                held[t, k] = 1.0
    return events, held


def _inputs(data):
    cef, anc = pair_columns(data)
    lr, bar = log_returns(data)
    z = zscores(lr[:, cef], lr[:, anc], bar[:, cef], bar[:, anc])
    tradable = np.asarray(data.tradable, dtype=bool)
    gone = ~np.asarray(data.listed, dtype=bool) | np.asarray(data.delisting, dtype=bool)
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
        member = uni[:, cef] & uni[:, anc]
    else:
        member = np.ones(z.shape, dtype=bool)
    member = member & bar[:, cef]
    return cef, anc, z, member, tradable[:, cef], gone[:, cef]


def target_weights(data, params):
    hold = int(params["hold_days"])
    z_entry = float(params["z_entry"])
    cef, anc, z, member, tr, gone = _inputs(data)
    _, held = run_book(z, z_entry, hold, member, tr, gone)
    T, N = len(data.dates), len(data.instruments)
    w = np.zeros((T, N), dtype=float)
    for k in range(len(cef)):
        w[:, cef[k]] += LEG_WEIGHT * held[:, k]
        w[:, anc[k]] -= LEG_WEIGHT * held[:, k]
    if data.universe is not None:
        w = np.where(np.asarray(data.universe, dtype=bool), w, 0.0)
    return w
