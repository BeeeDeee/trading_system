"""H-0164 mechanism event: a Desk purchase in the ETF's maturity bucket is scheduled
for session d+1 (announced by d). Side +1. SPY is never marked. Dates with five or
more of the six Treasury ETFs marked are dropped."""
import numpy as np

DATASET = "nyfed_tsy_operations_1d"
BUCKETS = (
    ("b0_3", "SHY"),
    ("b3_7", "IEI"),
    ("b7_10", "IEF"),
    ("b10_20", "TLH"),
    ("b20_30", "TLT"),
    ("tips", "TIP"),
)
MAX_MARKED = 4


def events(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    close = np.asarray(data.close, dtype=np.float64)
    tradable = np.asarray(data.tradable, dtype=bool)
    uni = data.universe
    for key, etf in BUCKETS:
        j = data.col(etf)
        s = np.asarray(data.series[DATASET + ".buy_next." + key], dtype=np.float64)
        m = np.isfinite(s) & (s > 0.5) & np.isfinite(close[:, j]) & tradable[:, j]
        if uni is not None:
            m &= np.asarray(uni[:, j], dtype=bool)
        mask[:, j] = m
    n_marked = mask.sum(axis=1)
    mask[n_marked > MAX_MARKED, :] = False
    side = np.where(mask, 1.0, 0.0)
    return mask, side
