"""Reference EMA crossover strategy (long / flat only)."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from app.domain.models.enums import Direction
from app.domain.models.market import FeatureSet, MarketSnapshot
from app.domain.models.state import PortfolioState
from app.domain.models.trading import TradingSignal


class EmaCrossStrategy:
    def __init__(
        self,
        strategy_id: str = "ema_cross",
        target_fraction: Decimal = Decimal("0.95"),
    ) -> None:
        self.strategy_id = strategy_id
        self.target_fraction = target_fraction
        self._prev_spread: Decimal | None = None

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        features: FeatureSet,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        del portfolio  # unused in this reference strategy
        spread = features.values.get("ema_spread")
        if spread is None:
            return []

        direction = Direction.FLAT
        if self._prev_spread is not None:
            # Bullish cross: spread flips from <=0 to >0
            if self._prev_spread <= 0 and spread > 0:
                direction = Direction.LONG
            # Bearish cross: spread flips from >=0 to <0
            elif self._prev_spread >= 0 and spread < 0:
                direction = Direction.FLAT
            elif spread > 0:
                direction = Direction.LONG
            else:
                direction = Direction.FLAT
        else:
            direction = Direction.LONG if spread > 0 else Direction.FLAT

        self._prev_spread = spread
        exposure = self.target_fraction if direction == Direction.LONG else Decimal(0)
        return [
            TradingSignal(
                signal_id=str(uuid4()),
                strategy_id=self.strategy_id,
                instrument=snapshot.instrument,
                direction=direction,
                confidence=Decimal("1"),
                target_exposure=exposure,
                timestamp=snapshot.timestamp,
            )
        ]
