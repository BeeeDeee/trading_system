"""Transaction costs (spec §7.3) and interest on cash (spec §7.4)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Cost per unit of traded value: commission + max(floor, k * half-spread), all in bps.

    `multiplier` scales the whole cost for the mandatory x2 / x3 sensitivity runs. A missing spread
    estimate (NaN) is treated as the upper clip, i.e. as illiquid.
    """

    commission_bps: float = 0.0
    slip_floor_bps: float = 5.0
    spread_k: float = 1.0
    spread_min_bps: float = 2.0
    spread_max_bps: float = 200.0
    multiplier: float = 1.0

    def rate(self, spread_bps: np.ndarray) -> np.ndarray:
        spread = np.clip(np.nan_to_num(np.asarray(spread_bps, dtype=float), nan=self.spread_max_bps),
                         self.spread_min_bps, self.spread_max_bps)
        slippage = np.maximum(self.slip_floor_bps, self.spread_k * spread / 2.0)
        return self.multiplier * (self.commission_bps + slippage) / 1e4


def cash_returns(dates: np.ndarray, rate_dates: np.ndarray, rate_pct: np.ndarray) -> np.ndarray:
    """Daily return of cash for each trading day in `dates`.

    Day t earns the annual rate last published strictly before `dates[t]`, act/360 over the
    calendar days since the previous trading day. The first day earns nothing.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    rate_dates = np.asarray(rate_dates, dtype="datetime64[D]")
    rate_pct = np.asarray(rate_pct, dtype=float)
    ok = np.isfinite(rate_pct)
    rate_dates, rate_pct = rate_dates[ok], rate_pct[ok]
    idx = np.searchsorted(rate_dates, dates, side="left") - 1
    rate = np.where(idx >= 0, rate_pct[np.maximum(idx, 0)], 0.0) / 100.0
    days = np.diff(dates).astype(int)
    out = np.zeros(len(dates))
    out[1:] = rate[1:] * days / 360.0
    return out
