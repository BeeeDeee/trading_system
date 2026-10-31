"""H-0078 mechanism test events: every eligible first crossing of z <= -z_entry (no slot cap, no
held-fund rule: those belong to the book). Fund marked +1, its anchor ETF -1. Same signal code as
strategy.py (duplicated because the diagnostic may only import numpy and the lab framework)."""
import numpy as np

PAIRS = (
    ("MXF", "EWW"), ("MXE", "EWW"), ("KF", "EWY"), ("TWN", "EWT"), ("IFN", "EPI"), ("IIF", "EPI"),
    ("CHN", "FXI"), ("TDF", "FXI"), ("GCH", "FXI"), ("JOF", "EWJ"), ("JEQ", "EWJ"), ("SWZ", "EWL"),
    ("GF", "EWG"), ("IF", "IDX"), ("TTF", "THD"), ("SGF", "EWS"),
    ("ISL", "EIS"), ("TKF", "TUR"), ("EMF", "EEM"), ("BZF", "EWZ"), ("SNF", "EWP"), ("IRL", "EIRL"),
)

FORM = 5
VOL_WIN = 126
VOL_LAG = 5
VOL_MIN = 100
DV_WIN = 21
DV_FLOOR = 100000.0


def _present_pairs(data):
    pairs = list(PAIRS)                               # strict: data.col raises on a missing instrument
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
    T = x.shape[0]
    c = np.zeros((T + 1,) + x.shape[1:], dtype=float)
    c[1:] = np.cumsum(x, axis=0)
    t = np.arange(T)
    hi = np.clip(t - lag + 1, 0, T)
    lo = np.clip(t - lag + 1 - length, 0, T)
    return c[hi] - c[lo]


def _shift(x, k, fill):
    out = np.full_like(x, fill)
    if k == 0:
        out[...] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def _pair_signals(data, params):
    z_entry = float(params["z_entry"])
    pairs, fcols, acols = _present_pairs(data)

    ret_co = np.nan_to_num(np.asarray(data.ret_co, dtype=float), nan=0.0)
    ret_oc = np.nan_to_num(np.asarray(data.ret_oc, dtype=float), nan=0.0)
    r = np.log1p(ret_co) + np.log1p(ret_oc)
    bar = _has_bar(data)

    both = bar[:, fcols] & bar[:, acols]
    d = np.where(both, r[:, fcols] - r[:, acols], 0.0)
    bothf = both.astype(float)

    S = np.zeros_like(d)
    nform = np.zeros_like(d)
    for k in range(FORM):
        S += _shift(d, k, 0.0)
        nform += _shift(bothf, k, 0.0)
    form_ok = nform >= FORM - 0.5

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

    dv = np.asarray(data.dollar_volume, dtype=float)[:, fcols]
    dv = np.where(np.isfinite(dv), dv, 0.0)
    stack = np.stack([_shift(dv, k, np.nan) for k in range(DV_WIN)], axis=0)
    med_dv = np.median(stack, axis=0)
    with np.errstate(invalid="ignore"):
        dv_ok = med_dv >= DV_FLOOR

    eligible = np.isfinite(z) & dv_ok
    z_prev = _shift(z, 1, np.nan)
    with np.errstate(invalid="ignore"):
        event = eligible & (z <= -z_entry) & (z_prev > -z_entry)
    return z, event, fcols, acols


def events(data, params):
    z, event, fcols, acols = _pair_signals(data, params)
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=float)
    for p in range(event.shape[1]):
        rows = event[:, p]
        mask[rows, fcols[p]] = True
        side[rows, fcols[p]] = 1.0
        mask[rows, acols[p]] = True
        side[rows, acols[p]] = -1.0
    # eligibility already requires both legs to be listed, tradable and (if given) in the universe
    return mask, side
