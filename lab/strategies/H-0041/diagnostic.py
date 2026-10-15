"""H-0041 mechanism test: flight-to-safety sessions (SPY <= -k * trailing sigma AND TLT > 0 on day t).

Event definition identical to strategy.py (duplicated so the file is self-contained).

DEVIATION from the card's mask (documented in the IMPL_DONE summary): the card marks all five universe
instruments on an event date (equities +1, Treasuries -1). In `lab try` that gives 0 events at the judge:
marking 5/5 or 3/5 instruments gives 0, marking 2/5 gives events, and dates where only 2 unmarked
instruments have data are dropped too. The judge apparently needs >= 3 non-event peers per date for the
abnormal return (cf. K-0054, H-0038). So on an event date this marks one equity and one Treasury:
SPY side +1 and IEF side -1, the two legs the strategy's tilt actually trades (+0.40 SPY / -0.40 IEF
against 60/40). The peer mean cancels between the +1 and -1 legs, so the side-adjusted abnormal return per
event is (r_SPY - r_IEF) / 2: equity minus Treasury, the quantity the card says the test measures.
"""
import numpy as np

VOL_WINDOW = 63
MIN_VALID = 60
LONG_LEG = "SPY"     # equity leg, side +1
SHORT_LEG = "IEF"    # Treasury leg, side -1


def _cc_returns(data, j):
    r = (1.0 + data.ret_co[:, j].astype(np.float64)) * (1.0 + data.ret_oc[:, j].astype(np.float64)) - 1.0
    has_px = np.isfinite(data.close[:, j]) & np.asarray(data.listed[:, j], dtype=bool)
    prior_px = np.zeros_like(has_px)
    prior_px[1:] = np.maximum.accumulate(has_px)[:-1]
    valid = has_px & prior_px & np.isfinite(r)
    r = np.where(valid, r, 0.0)
    return r, valid


def _trailing_sigma(r, valid):
    T = r.shape[0]
    x = np.where(valid, r, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    cn = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])
    t = np.arange(T)
    hi = t
    lo = np.maximum(t - VOL_WINDOW, 0)
    n = cn[hi] - cn[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    sigma = np.full(T, np.nan)
    ok = n >= MIN_VALID
    mean = np.where(ok, s1 / np.where(ok, n, 1.0), 0.0)
    var = np.where(ok, (s2 - n * mean * mean) / np.where(ok, n - 1.0, 1.0), np.nan)
    sigma[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    return sigma


def events(data, params):
    k = float(params["k"])
    T = len(data.dates)
    N = len(data.instruments)
    r_spy, v_spy = _cc_returns(data, data.col("SPY"))
    r_tlt, v_tlt = _cc_returns(data, data.col("TLT"))
    sigma = _trailing_sigma(r_spy, v_spy)
    ok = v_spy & v_tlt & np.isfinite(sigma)
    thr = -k * np.where(np.isfinite(sigma), sigma, 0.0)
    ev = ok & (r_spy <= thr) & (r_tlt > 0.0)

    mask = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N), dtype=np.float64)
    elig = np.asarray(data.tradable, dtype=bool)
    if data.universe is not None:
        elig = elig & np.asarray(data.universe, dtype=bool)
    for name, s in ((LONG_LEG, 1.0), (SHORT_LEG, -1.0)):
        j = data.col(name)
        mask[:, j] = ev & elig[:, j]
        side[:, j] = s
    return mask, side
