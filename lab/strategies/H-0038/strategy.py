"""H-0038 Treasury auction supply cycle.

After the close of session d (us_close clock), using the PIT auction-calendar series of
treasury_auctions_1d (5y-30y nominal coupon auctions, each visible from its announcement date):

  (1) 1 <= next_coupon5_sessions(d) <= window_days        -> TLT = -1 (pre-auction short)
  (2) else 0 <= last_coupon5_sessions(d) <= window_days-1 -> TLT = +1 (post-auction long)
  (3) else                                                -> TLT = 0  (cash)

SHY is always 0 (mechanism-test control only). NaN series values make the comparison false
(rule does not fire). If TLT has no bar on session d, the TLT element is NaN (keep position).
Weights are decided every session; the engine only trades when the target changes.
"""
import numpy as np

PARAMS = {"window_days": 3}

NEXT_KEY = "treasury_auctions_1d.next_coupon5_sessions"
LAST_KEY = "treasury_auctions_1d.last_coupon5_sessions"


def auction_states(data, params):
    """Return (short_state, long_state), two (T,) bool arrays (rule 1, and rule 2 without rule 1)."""
    w = int(params["window_days"])
    nxt = np.asarray(data.series[NEXT_KEY], dtype=float).reshape(-1)
    lst = np.asarray(data.series[LAST_KEY], dtype=float).reshape(-1)
    T = len(data.dates)
    if nxt.shape[0] != T or lst.shape[0] != T:
        raise ValueError("auction series not aligned with data.dates")
    with np.errstate(invalid="ignore"):
        short = (nxt >= 1) & (nxt <= w)                     # NaN -> False
        post = (lst >= 0) & (lst <= w - 1)                  # NaN -> False
    long_ = post & ~short                                   # rule (1) takes precedence
    return short, long_


def target_weights(data, params):
    tlt = data.col("TLT")
    shy = data.col("SHY")
    T = len(data.dates)
    N = len(data.instruments)

    short, long_ = auction_states(data, params)

    w = np.zeros((T, N), dtype=float)
    tlt_w = np.where(short, -1.0, np.where(long_, 1.0, 0.0))
    # no TLT bar on this session: keep the previous TLT weight
    no_bar = np.isnan(np.asarray(data.close, dtype=float)[:, tlt])
    tlt_w[no_bar] = np.nan
    w[:, tlt] = tlt_w
    w[:, shy] = 0.0
    return w
