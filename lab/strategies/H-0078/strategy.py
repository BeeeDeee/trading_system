"""H-0078: country closed-end fund catch-up to its country ETF.

Long a single-country CEF (+0.25) and short its fixed anchor ETF (-0.25) for hold_days sessions after the
fund's 5-session log return falls z_entry standard deviations below the anchor's (first session of the
crossing). At most 2 open events; most negative z first; skipped events are not queued.
"""
import numpy as np

PARAMS = {"hold_days": 5, "z_entry": 2.0}

# (closed-end fund, anchor ETF), fixed, never switched. Card v2: 22 pairs (CH->ECH and MF->EWM removed,
# not in Sharadar SFP; ECH and EWM stay in the universe but never trade).
PAIRS = (
    ("MXF", "EWW"), ("MXE", "EWW"), ("KF", "EWY"), ("TWN", "EWT"), ("IFN", "EPI"), ("IIF", "EPI"),
    ("CHN", "FXI"), ("TDF", "FXI"), ("GCH", "FXI"), ("JOF", "EWJ"), ("JEQ", "EWJ"), ("SWZ", "EWL"),
    ("GF", "EWG"), ("IF", "IDX"), ("TTF", "THD"), ("SGF", "EWS"),
    ("ISL", "EIS"), ("TKF", "TUR"), ("EMF", "EEM"), ("BZF", "EWZ"), ("SNF", "EWP"), ("IRL", "EIRL"),
)

FORM = 5            # formation window t-4..t
VOL_WIN = 126       # sigma window: 126 sessions ending t-5
VOL_LAG = 5
VOL_MIN = 100       # sessions with both legs having a bar inside the sigma window
DV_WIN = 21         # fund median dollar volume window
DV_FLOOR = 100000.0  # fixed feasibility floor (not a parameter)
SLOT_W = 0.25
MAX_SLOTS = 2


def _present_pairs(data):
    # strict: every card instrument must be in the data (data.col raises on a missing name); a fund or
    # anchor without prices simply has no bars and therefore no eligible days
    pairs = list(PAIRS)
    fcols = np.array([data.col(f) for f, _ in pairs], dtype=int)
    acols = np.array([data.col(a) for _, a in pairs], dtype=int)
    return pairs, fcols, acols


def _has_bar(data):
    close = np.asarray(data.close, dtype=float)
    bar = np.asarray(data.listed, dtype=bool) & np.asarray(data.tradable, dtype=bool) & np.isfinite(close)
    if data.universe is not None:
        bar = bar & np.asarray(data.universe, dtype=bool)
    return bar


def _lagged_window_sum(x, lag, length):
    """sum of x over rows t-lag-length+1 .. t-lag (truncated at row 0); uses rows <= t only."""
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=float)
    c[1:] = np.cumsum(x, axis=0)
    t = np.arange(T)
    hi = np.clip(t - lag + 1, 0, T)
    lo = np.clip(t - lag + 1 - length, 0, T)
    return c[hi] - c[lo]


def _shift(x, k, fill):
    """row t holds x[t-k] (k >= 0)."""
    out = np.full_like(x, fill)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def pair_signals(data, params):
    """z (T, P), event (T, P) bool and the pair columns. Row t uses rows 0..t only."""
    z_entry = float(params["z_entry"])
    pairs, fcols, acols = _present_pairs(data)

    ret_co = np.nan_to_num(np.asarray(data.ret_co, dtype=float), nan=0.0)
    ret_oc = np.nan_to_num(np.asarray(data.ret_oc, dtype=float), nan=0.0)
    r = np.log1p(ret_co) + np.log1p(ret_oc)          # close-to-close total log return
    bar = _has_bar(data)

    both = bar[:, fcols] & bar[:, acols]              # (T, P) both legs have a bar
    d = np.where(both, r[:, fcols] - r[:, acols], 0.0)  # hedge ratio 1
    bothf = both.astype(float)

    # formation: S = sum of d over t-4..t, all 5 sessions with both bars
    S = np.zeros_like(d)
    nform = np.zeros_like(d)
    for k in range(FORM):
        S += _shift(d, k, 0.0)
        nform += _shift(bothf, k, 0.0)
    form_ok = nform >= FORM - 0.5

    # sigma over the 126 sessions ending t-5, sample std (ddof=1) over sessions with both bars
    n = _lagged_window_sum(bothf, VOL_LAG, VOL_WIN)
    s1 = _lagged_window_sum(d, VOL_LAG, VOL_WIN)
    s2 = _lagged_window_sum(d * d, VOL_LAG, VOL_WIN)
    warm = n >= VOL_MIN - 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(warm, n, 2.0)
        var = (s2 - s1 * s1 / nn) / (nn - 1.0)
        sigma = np.sqrt(np.clip(var, 0.0, None))
        ok = form_ok & warm & (sigma > 0)
        z = np.where(ok, S / np.where(sigma > 0, sigma, 1.0) / np.sqrt(FORM), np.nan)

    # fund median dollar volume over the last 21 sessions; a session without a value counts as 0
    dv = np.asarray(data.dollar_volume, dtype=float)[:, fcols]
    dv = np.where(np.isfinite(dv), dv, 0.0)
    stack = np.stack([_shift(dv, k, np.nan) for k in range(DV_WIN)], axis=0)
    med_dv = np.median(stack, axis=0)                 # NaN for t < 20
    with np.errstate(invalid="ignore"):
        dv_ok = med_dv >= DV_FLOOR

    eligible = np.isfinite(z) & dv_ok
    z_prev = _shift(z, 1, np.nan)
    with np.errstate(invalid="ignore"):
        event = eligible & (z <= -z_entry) & (z_prev > -z_entry)
    return z, event, fcols, acols


def target_weights(data, params):
    hold = int(params["hold_days"])
    z, event, fcols, acols = pair_signals(data, params)
    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)

    open_ev = []                                      # (pair index, last held decision row)
    for t in range(T):
        open_ev = [e for e in open_ev if e[1] >= t]
        cand = np.nonzero(event[t])[0]
        if cand.size:
            held = set(p for p, _ in open_ev)         # one pair per fund: no new event on a held fund
            cand = [int(p) for p in cand if int(p) not in held]
            cand.sort(key=lambda p: (z[t, p], p))     # most negative z first
            free = MAX_SLOTS - len(open_ev)
            for p in cand[:max(free, 0)]:
                open_ev.append((p, t + hold - 1))     # rows t..t+hold-1, exit decided on row t+hold
        for p, _ in open_ev:
            w[t, fcols[p]] += SLOT_W
            w[t, acols[p]] -= SLOT_W                  # shared anchors add

    if data.universe is not None:
        w = np.where(np.asarray(data.universe, dtype=bool), w, 0.0)
    return w
