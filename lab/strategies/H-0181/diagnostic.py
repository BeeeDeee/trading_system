"""H-0181 mechanism events: S&P 500 membership flag changes of LIQ-500 members (same rule as the strategy)."""
import numpy as np

SKIP_ROWS = 5


def _bool(x):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    a = np.asarray(a, dtype=float)
    return np.nan_to_num(a, nan=0.0) > 0


def events(data, params):
    member = data.universe
    if member is None:
        raise ValueError("H-0181 needs the LIQ-500 point-in-time universe")
    member = _bool(member)
    tradable = _bool(data.tradable)
    sp = np.asarray(data.extras["sp500_member"], dtype=float)
    T, N = sp.shape
    prev = np.full((T, N), np.nan)
    prev[1:] = sp[:-1]
    ok = member & tradable
    ok[:SKIP_ROWS] = False
    dele = (prev == 1) & (sp == 0) & ok
    addi = (prev == 0) & (sp == 1) & ok
    mask = dele | addi
    side = np.where(dele, 1.0, np.where(addi, -1.0, 0.0))
    return mask, side
