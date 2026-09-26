"""Rebalancing schedules that only need the past.

"Last trading day of the week" would require knowing tomorrow's date (the look-ahead test catches
it on truncated data), so decisions are taken after the close of the first trading day of each
week or month instead.
"""

import numpy as np


def period_starts(dates: np.ndarray, unit: str) -> np.ndarray:
    """True on the first trading day of each week ('W', Monday-based) or month ('M')."""
    d = np.asarray(dates, dtype="datetime64[D]")
    if unit == "W":
        key = (d.astype(np.int64) + 3) // 7  # 1970-01-01 was a Thursday
    elif unit == "M":
        key = d.astype("datetime64[M]").astype(np.int64)
    else:
        raise ValueError(f"unknown unit {unit!r}")
    return np.concatenate(([True], key[1:] != key[:-1]))
