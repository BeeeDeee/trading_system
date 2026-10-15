"""H-0041 Flight-to-safety reversal: tilt a 60/40 SPY/IEF book to 100 % SPY for hold_days sessions
after a session where SPY falls by at least k trailing sigmas while TLT rises.

Decision on row t uses rows 0..t only; the engine executes at the open of t+1.
"""
import numpy as np
from lab.framework.api import period_starts

PARAMS = {"hold_days": 5, "k": 1.5}

VOL_WINDOW = 63      # sessions t-63 .. t-1 (card rule 1)
MIN_VALID = 60       # at least 60 valid returns in the window (card rule 1)
BASE_SPY = 0.60
BASE_IEF = 0.40


def _cc_returns(data, j):
    """Close-to-close total return of instrument column j and a validity mask.

    r(t) = (1 + ret_co(t)) * (1 + ret_oc(t)) - 1. A return is valid on row t when the instrument is
    listed and has a price on row t and had a price on some earlier row (the first priced row has no
    close-to-close return)."""
    r = (1.0 + data.ret_co[:, j].astype(np.float64)) * (1.0 + data.ret_oc[:, j].astype(np.float64)) - 1.0
    has_px = np.isfinite(data.close[:, j]) & np.asarray(data.listed[:, j], dtype=bool)
    prior_px = np.zeros_like(has_px)
    prior_px[1:] = np.maximum.accumulate(has_px)[:-1]
    valid = has_px & prior_px & np.isfinite(r)
    r = np.where(valid, r, 0.0)
    return r, valid


def _trailing_sigma(r, valid):
    """Sample std (ddof=1) of valid returns over rows t-63 .. t-1 (day t excluded), NaN if < 60 valid."""
    T = r.shape[0]
    x = np.where(valid, r, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    cn = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])
    t = np.arange(T)
    hi = t                                   # exclusive end: rows < t
    lo = np.maximum(t - VOL_WINDOW, 0)       # inclusive start: row t-63
    n = cn[hi] - cn[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    sigma = np.full(T, np.nan)
    ok = n >= MIN_VALID
    mean = np.where(ok, s1 / np.where(ok, n, 1.0), 0.0)
    var = np.where(ok, (s2 - n * mean * mean) / np.where(ok, n - 1.0, 1.0), np.nan)
    sigma[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    return sigma


def fts_events(data, params):
    """(T,) bool: flight-to-safety event after the close of row t (card rule 2)."""
    k = float(params["k"])
    r_spy, v_spy = _cc_returns(data, data.col("SPY"))
    r_tlt, v_tlt = _cc_returns(data, data.col("TLT"))
    sigma = _trailing_sigma(r_spy, v_spy)
    ok = v_spy & v_tlt & np.isfinite(sigma)
    thr = -k * np.where(np.isfinite(sigma), sigma, 0.0)
    return ok & (r_spy <= thr) & (r_tlt > 0.0)


def tilt_state(events, hold_days):
    """(T,) bool: tilt on for decisions of rows t .. t+hold_days-1 after an event on row t (windows merge)."""
    T = events.shape[0]
    idx = np.where(events, np.arange(T), -1)
    last_event = np.maximum.accumulate(idx) if T else idx
    return (last_event >= 0) & (np.arange(T) - last_event < int(hold_days))


def _month_start_mask(dates):
    ps = np.asarray(period_starts(dates, "M"))
    if ps.dtype == bool:
        return ps
    m = np.zeros(len(dates), dtype=bool)
    m[ps.astype(np.int64)] = True
    return m


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    j_spy = data.col("SPY")
    j_ief = data.col("IEF")
    data.col("TLT")  # signal input: fail loudly if missing
    hold_days = int(params["hold_days"])

    on = tilt_state(fts_events(data, params), hold_days)
    month_start = _month_start_mask(data.dates)
    px_ok = (np.isfinite(data.close[:, j_spy]) & np.isfinite(data.close[:, j_ief])
             & np.asarray(data.listed[:, j_spy], dtype=bool) & np.asarray(data.listed[:, j_ief], dtype=bool))

    w = np.full((T, N), np.nan)
    last = 0  # 0 = nothing set yet, 1 = 60/40 base, 2 = tilt (100 % SPY)
    for t in range(T):
        if not px_ok[t]:
            continue  # rule 5: missing SPY or IEF price -> keep previous weights (NaN row)
        if on[t]:
            state = 2
        elif last != 1 or month_start[t]:
            state = 1  # first decision, tilt end (or a deferred reset), or first session of the month
        else:
            continue  # tilt off, no reset day: hold the drifted 60/40 weights
        row = np.zeros(N)
        if state == 2:
            row[j_spy] = 1.0
        else:
            row[j_spy] = BASE_SPY
            row[j_ief] = BASE_IEF
        w[t] = row
        last = state
    return w
