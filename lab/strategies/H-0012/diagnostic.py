"""H-0012 mechanism test event: first open-market purchase filing usable on t after a quiet window.

ins_buy_n_<Q>d[t-1, j] == 0 and ins_buy_n_91d[t, j] > 0, j a LIQ-500 member tradable on t; side +1.
Q = params["quiet_window_days"] (182 at the primary value, as in the card's mechanism_test.event).
"""
import numpy as np

QUIET_FIELDS = {91: "ins_buy_n_91d", 182: "ins_buy_n_182d", 365: "ins_buy_n_365d"}


def events(data, params):
    Q = int(params["quiet_window_days"])
    if Q not in QUIET_FIELDS:
        raise ValueError("quiet_window_days must be one of 91, 182, 365")
    if data.universe is None:
        raise ValueError("card requires a LIQ-500 universe")
    member = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    quiet = np.asarray(data.extras[QUIET_FIELDS[Q]], dtype=np.float64)
    buy91 = np.asarray(data.extras["ins_buy_n_91d"], dtype=np.float64)
    T, N = buy91.shape
    quiet_prev = np.full((T, N), np.nan)
    if T > 1:
        quiet_prev[1:] = quiet[:-1]
    with np.errstate(invalid="ignore"):
        mask = member & tradable & (quiet_prev == 0.0) & (buy91 > 0.0)
    return mask, None
