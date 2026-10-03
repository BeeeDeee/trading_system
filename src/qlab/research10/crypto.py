"""Crypto long/short trend sleeves of research 10 (prereg §5.2).

Point in time: the signal of day t uses closes <= t; a coin trades at the open of t+1 only when its own
signal changes (size = `size` x NAV at that open), NaN targets keep the other coin untouched.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.research9.strategy import sma

SIZE = 0.5
FUNDING_STRESS = -0.0003    # per day: the short pays ~11 % p. a. instead of the real funding


@dataclass(frozen=True)
class Config:
    kind: str    # LS | HS
    n: int       # SMA length

    @property
    def id(self) -> str:
        return f"{self.kind}_{self.n}"

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research10", "part": "C"}


def grid() -> list[Config]:
    return [Config(k, n) for k in ("LS", "HS") for n in (20, 50, 100, 200)]


def signal(close: np.ndarray, n: int) -> np.ndarray:
    """+1 close > SMA(n), -1 close <= SMA(n), 0 before n valid closes."""
    ref = sma(close, n)
    return np.where(np.isnan(ref), 0.0, np.where(close > ref, 1.0, -1.0))


def decisions(signals: dict[int, np.ndarray], N: int, first: int, short_only: bool = False,
              size: float = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Decision days (signal changes from `first` on) and targets; NaN = keep. LS: size x signal;
    short_only: -size below the SMA, else cash."""
    T = len(next(iter(signals.values())))
    tgt = {j: (np.minimum(s, 0.0) if short_only else s) * size for j, s in signals.items()}
    days, rows = [], []
    for t in range(first, T - 1):
        row = np.full(N, np.nan)
        for j, w in tgt.items():
            if t == first or w[t] != w[t - 1]:
                row[j] = w[t]
        if not np.isnan(row).all():
            days.append(t)
            rows.append(row)
    return np.array(days, dtype=int), np.array(rows).reshape(len(days), N)
