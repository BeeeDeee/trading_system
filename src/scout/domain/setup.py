import math
from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware
from scout.domain.enums import Direction, Regime, SetupOutcome, VolBucket


@dataclass(frozen=True, slots=True)
class Setup:
    """A strategy's complete statement of an intended trade, in price space.
    Contains no money, no size, no probability, no score. A Setup is a
    geometric claim, nothing more.

    `target_price` may be None (ADR-019): no profit target; exit on time or rank.
    `stop_price` is always required.
    """

    symbol: str
    ts: datetime  # decision bar close; entry is at ts+1 open
    strategy_id: str  # registry key, e.g. "donchian_breakout_v1"
    direction: Direction
    regime: Regime  # regime at detection; recorded, not re-derived

    reference_price: float  # close at ts; entry is estimated from this
    stop_price: float  # protective stop, absolute
    target_price: float | None  # take profit, absolute; None = no target (ADR-019)
    max_hold_bars: int  # time stop

    trigger_note: str  # short, stable label for the exact rule that fired

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")
        if self.direction is Direction.LONG:
            if self.stop_price > self.reference_price:
                raise ValueError("stop must be on the losing side")
            if self.target_price is not None and self.target_price < self.reference_price:
                raise ValueError("target must be on the winning side")
        else:
            if self.stop_price < self.reference_price:
                raise ValueError("stop must be on the losing side")
            if self.target_price is not None and self.target_price > self.reference_price:
                raise ValueError("target must be on the winning side")
        if self.risk_per_unit <= 0 or not math.isfinite(self.risk_per_unit):
            raise ValueError("risk_per_unit must be > 0")
        if self.target_price is not None and (
            self.reward_risk_ratio <= 0 or not math.isfinite(self.reward_risk_ratio)
        ):
            raise ValueError("reward_risk_ratio must be > 0")

    @property
    def risk_per_unit(self) -> float:
        return abs(self.reference_price - self.stop_price)

    @property
    def reward_per_unit(self) -> float:
        if self.target_price is None:
            raise ValueError("reward_per_unit is undefined when target_price is None")
        return abs(self.target_price - self.reference_price)

    @property
    def reward_risk_ratio(self) -> float:
        return self.reward_per_unit / self.risk_per_unit


@dataclass(frozen=True, slots=True)
class ResolvedSetup:
    """A Setup carried forward through the triple-barrier labeling pass."""

    setup: Setup
    entry_ts: datetime
    entry_price: float
    resolution_ts: datetime | None  # None iff outcome is OPEN
    exit_price: float | None
    outcome: SetupOutcome
    bars_held: int
    realised_r_gross: float
    mae_r: float
    mfe_r: float
    vol_bucket: VolBucket  # at detection

    def __post_init__(self) -> None:
        require_aware(self.entry_ts, "entry_ts")
        require_aware(self.resolution_ts, "resolution_ts")
        open_outcome = self.outcome is SetupOutcome.OPEN
        if open_outcome != (self.resolution_ts is None):
            raise ValueError("resolution_ts must be None iff outcome is OPEN")
