"""STR-TF cost model (research 5 pre-registration §6): per side, by period.

Slippage + half spread is multiplied by `open_spread_mult` for executions at the open; the MOC
variant executes at the close and uses multiplier 1.
"""

from dataclasses import dataclass

import numpy as np

from qlab.engine.costs import CostModel

# (first day, slippage + half spread bps, commission bps)
PERIODS = (("1900-01-01", 25.0, 5.0), ("2002-01-01", 10.0, 1.0), ("2008-01-01", 5.0, 0.5))
OPEN_SPREAD_MULT = 1.5


def period_cost(dates: np.ndarray, at_open: bool = True, multiplier: float = 1.0,
                open_spread_mult: float = OPEN_SPREAD_MULT) -> np.ndarray:
    """(T,) cost per side as a fraction of traded value."""
    d = np.asarray(dates, dtype="datetime64[D]")
    starts = np.array([np.datetime64(p[0]) for p in PERIODS])
    k = np.searchsorted(starts, d, side="right") - 1
    slip = np.array([p[1] for p in PERIODS])[k] * (open_spread_mult if at_open else 1.0)
    comm = np.array([p[2] for p in PERIODS])[k]
    return multiplier * (slip + comm) / 1e4


@dataclass(frozen=True)
class DateCost:
    """Cost that depends on the day only."""
    rates: np.ndarray  # (T,)

    def at(self, t: int, a: int) -> float:
        return float(self.rates[t])


@dataclass(frozen=True)
class RankCost:
    """Research 1 tier model by liquidity rank (sensitivity only)."""
    liq_rank: np.ndarray  # (T, N)
    dates: np.ndarray
    model: CostModel = CostModel()

    def at(self, t: int, a: int) -> float:
        return float(self.model.rate(np.array([[self.liq_rank[t, a]]], dtype=float),
                                     self.dates[t:t + 1])[0, 0])
