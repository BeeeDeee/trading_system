"""H-0079 mechanism-test events: a CEF's 5-session beta-adjusted residual vs its fixed anchor ETF is
at or below -z_entry standard deviations (first crossing: no event in the previous hold_days sessions
and no open slot), side +1. Same code as strategy.py (copied: the diagnostic may not import it)."""
import numpy as np

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

BETA_WINDOW = 252
RESID_WINDOW = 252
MIN_OBS = 200
RET_DAYS = 5
BETA_LO, BETA_HI = 0.25, 2.0
SLOT_CAP = 2


def _rsum(a, n):
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


def _zscores(y, x, by, bx):
    w = (by & bx).astype(float)
    xw = np.where(w > 0, x, 0.0)
    yw = np.where(w > 0, y, 0.0)
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
    v5 = _rsum(w, RET_DAYS) >= RET_DAYS - 0.5
    a5 = np.where(v5, _rsum(yw, RET_DAYS), 0.0)
    b5 = np.where(v5, _rsum(xw, RET_DAYS), 0.0)
    vf = v5.astype(float)
    m = _lag(_rsum(vf, RESID_WINDOW), RET_DAYS, 0.0)
    sa = _lag(_rsum(a5, RESID_WINDOW), RET_DAYS, 0.0)
    sb = _lag(_rsum(b5, RESID_WINDOW), RET_DAYS, 0.0)
    saa = _lag(_rsum(a5 * a5, RESID_WINDOW), RET_DAYS, 0.0)
    sbb = _lag(_rsum(b5 * b5, RESID_WINDOW), RET_DAYS, 0.0)
    sab = _lag(_rsum(a5 * b5, RESID_WINDOW), RET_DAYS, 0.0)
    bb = np.where(ok_b, beta, 0.0)
    se = sa - bb * sb
    see = saa - 2.0 * bb * sab + bb * bb * sbb
    mm = np.maximum(m, 2.0)
    var_e = (see - se * se / mm) / (mm - 1.0)
    ok_s = ok_b & (m >= MIN_OBS) & (var_e > 0)
    sd = np.sqrt(np.where(ok_s, var_e, 1.0))
    e = a5 - bb * b5
    return np.where(v5 & ok_s, e / sd, np.nan)


def _run_book(z, z_entry, hold, member, tradable, gone):
    T, K = z.shape
    hold = int(hold)
    cross = np.isfinite(z) & (np.where(np.isfinite(z), z, 0.0) <= -float(z_entry)) & member
    events = np.zeros((T, K), dtype=bool)
    last_ev = np.full(K, -(10 ** 9), dtype=np.int64)
    slots = []
    for t in range(T):
        keep = []
        n_occ = 0
        has_slot = np.zeros(K, dtype=bool)
        for k, t0 in slots:
            if t == t0 + 1 and not tradable[t, k]:
                continue
            if t <= t0 + hold - 1:
                if gone[t, k]:
                    continue
                keep.append((k, t0))
                n_occ += 1
                has_slot[k] = True
            elif t == t0 + hold:
                keep.append((k, t0))
                has_slot[k] = True
            else:
                if tradable[t, k] or gone[t, k]:
                    continue
                keep.append((k, t0))
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
    return events


def events(data, params):
    hold = int(params["hold_days"])
    z_entry = float(params["z_entry"])
    cef = np.array([data.col(c) for c, _ in PAIRS], dtype=int)
    anc = np.array([data.col(a) for _, a in PAIRS], dtype=int)
    gross = (1.0 + np.asarray(data.ret_co, dtype=float)) * (1.0 + np.asarray(data.ret_oc, dtype=float))
    tradable = np.asarray(data.tradable, dtype=bool)
    listed = np.asarray(data.listed, dtype=bool)
    bar = tradable & listed & np.isfinite(gross) & (gross > 0)
    lr = np.where(bar, np.log(np.where(bar, gross, 1.0)), 0.0)
    z = _zscores(lr[:, cef], lr[:, anc], bar[:, cef], bar[:, anc])
    gone = ~listed | np.asarray(data.delisting, dtype=bool)
    if data.universe is not None:
        uni = np.asarray(data.universe, dtype=bool)
        member = uni[:, cef] & uni[:, anc]
    else:
        member = np.ones(z.shape, dtype=bool)
    member = member & bar[:, cef]
    ev = _run_book(z, z_entry, hold, member, tradable[:, cef], gone[:, cef])
    T, N = len(data.dates), len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    mask[:, cef] = ev            # anchors are never marked; each CEF column appears once
    return mask, None
