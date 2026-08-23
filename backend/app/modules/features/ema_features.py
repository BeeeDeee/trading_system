"""Deterministic EMA feature engine for the reference strategy."""

from __future__ import annotations

from collections import deque
from decimal import Decimal

from app.domain.models.market import FeatureSet, MarketSnapshot


class EmaFeatureEngine:
    """Streaming EMA features; call calculate() once per completed bar in order."""

    feature_version = "ema-v1"

    def __init__(self, fast_period: int = 12, slow_period: int = 26) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._closes: deque[Decimal] = deque(maxlen=slow_period * 4)
        self._ema_fast: Decimal | None = None
        self._ema_slow: Decimal | None = None
        self._alpha_fast = Decimal(2) / Decimal(fast_period + 1)
        self._alpha_slow = Decimal(2) / Decimal(slow_period + 1)
        self._fast_ready = 0
        self._slow_ready = 0

    def calculate(self, snapshot: MarketSnapshot) -> FeatureSet:
        price = snapshot.close
        self._closes.append(price)

        if self._ema_fast is None:
            self._fast_ready += 1
            if self._fast_ready >= self.fast_period:
                window = list(self._closes)[-self.fast_period :]
                self._ema_fast = sum(window, Decimal(0)) / Decimal(self.fast_period)
        else:
            self._ema_fast = (price * self._alpha_fast) + (
                self._ema_fast * (Decimal(1) - self._alpha_fast)
            )

        if self._ema_slow is None:
            self._slow_ready += 1
            if self._slow_ready >= self.slow_period:
                window = list(self._closes)[-self.slow_period :]
                self._ema_slow = sum(window, Decimal(0)) / Decimal(self.slow_period)
        else:
            self._ema_slow = (price * self._alpha_slow) + (
                self._ema_slow * (Decimal(1) - self._alpha_slow)
            )

        values: dict[str, Decimal] = {"close": price}
        if self._ema_fast is not None:
            values["ema_fast"] = self._ema_fast
        if self._ema_slow is not None:
            values["ema_slow"] = self._ema_slow
        if self._ema_fast is not None and self._ema_slow is not None:
            values["ema_spread"] = self._ema_fast - self._ema_slow

        return FeatureSet(
            instrument=snapshot.instrument,
            timestamp=snapshot.timestamp,
            feature_version=self.feature_version,
            values=values,
        )
