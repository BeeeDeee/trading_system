"""Transaction costs (spec §7.3) and interest on cash (spec §7.4)."""

from dataclasses import dataclass

import numpy as np


# Half-spread + slippage per side by liquidity rank (rank of the trailing median dollar volume
# among all eligible securities that day, 1 = most liquid). An OHLC spread estimator was tried
# and rejected: below ~30 bps it is dominated by volatility noise (see docs/DATA_FINDINGS.md).
DEFAULT_TIERS = ((200, 3.0), (500, 6.0), (1000, 12.0))
DECIMALIZATION = np.datetime64("2001-04-09")  # Nasdaq completed decimal quoting


@dataclass(frozen=True)
class CostModel:
    """Cost per unit of traded value (spec §7.3): commission + max(floor, tier half-spread).

    Tiers map liquidity rank to half-spread bps; unranked or rank beyond the last tier costs
    `default_bps`. Before decimalization the tier cost is multiplied by `pre_decimal_multiplier`.
    `multiplier` scales the whole cost for the mandatory x2 / x3 sensitivity runs.
    """

    commission_bps: float = 0.0
    floor_bps: float = 5.0
    tiers: tuple[tuple[int, float], ...] = DEFAULT_TIERS
    default_bps: float = 25.0
    pre_decimal_multiplier: float = 2.5
    multiplier: float = 1.0

    def rate(self, liquidity_rank: np.ndarray, dates: np.ndarray | None = None) -> np.ndarray:
        """Cost rates (fraction of traded value) for a (T, N) rank matrix; NaN = unranked."""
        rank = np.asarray(liquidity_rank, dtype=float)
        half = np.full(rank.shape, self.default_bps)
        for max_rank, bps in reversed(self.tiers):
            half = np.where(rank <= max_rank, bps, half)
        if dates is not None:
            early = np.asarray(dates, dtype="datetime64[D]") < DECIMALIZATION
            half = half * np.where(early, self.pre_decimal_multiplier, 1.0).reshape(
                (-1,) + (1,) * (half.ndim - 1))
        return self.multiplier * (self.commission_bps + np.maximum(self.floor_bps, half)) / 1e4


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
