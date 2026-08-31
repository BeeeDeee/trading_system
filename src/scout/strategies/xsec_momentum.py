"""Cross-sectional 12-1 momentum. Primary v1 strategy (ADR-019)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from scout.domain.enums import Direction, MarketRegime, Regime
from scout.domain.features import FeatureRow
from scout.domain.setup import Setup
from scout.utils.errors import ScoutConfigError

_PARAM_NAMES = frozenset(
    {"xs_threshold", "exit_xs_threshold", "stop_atr", "max_hold_bars"}
)


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScoutConfigError(f"{name}: expected a finite number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ScoutConfigError(f"{name}: must be finite")
    return out


def _as_hold_bars(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoutConfigError(f"max_hold_bars: expected an int, got {type(value).__name__}")
    if value < 1:
        raise ScoutConfigError("max_hold_bars: must be >= 1")
    return value


@dataclass(frozen=True, slots=True)
class XSecMomentum:
    strategy_id: ClassVar[str] = "xsec_momentum_v1"
    # Empty: the engine skips the per-symbol regime check (14-CONFIG §5 row 10
    # is Donchian only). Market regime is still gated via allowed_market_regimes.
    allowed_regimes: ClassVar[frozenset[Regime]] = frozenset()
    allowed_market_regimes: ClassVar[frozenset[MarketRegime]] = frozenset(
        {MarketRegime.RISK_ON, MarketRegime.NEUTRAL}
    )
    required_warmup_bars: ClassVar[int] = 273

    xs_threshold: float = 0.90
    exit_xs_threshold: float = 0.70
    stop_atr: float = 5.0
    max_hold_bars: int = 21

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> XSecMomentum:
        unknown = sorted(set(params) - _PARAM_NAMES)
        if unknown:
            raise ScoutConfigError(
                f"unknown param {unknown[0]!r} for {cls.strategy_id}; "
                f"known: {sorted(_PARAM_NAMES)}"
            )
        kwargs: dict[str, Any] = {}
        if "xs_threshold" in params:
            kwargs["xs_threshold"] = _as_float("xs_threshold", params["xs_threshold"])
        if "exit_xs_threshold" in params:
            kwargs["exit_xs_threshold"] = _as_float(
                "exit_xs_threshold", params["exit_xs_threshold"]
            )
        if "stop_atr" in params:
            stop_atr = _as_float("stop_atr", params["stop_atr"])
            if stop_atr <= 0:
                raise ScoutConfigError("stop_atr: must be > 0")
            kwargs["stop_atr"] = stop_atr
        if "max_hold_bars" in params:
            kwargs["max_hold_bars"] = _as_hold_bars(params["max_hold_bars"])
        return cls(**kwargs)

    def detect(self, row: FeatureRow) -> Setup | None:
        if not row.is_warm:
            return None
        if not math.isfinite(row.close) or not math.isfinite(row.atr_14) or row.atr_14 <= 0:
            return None
        if not math.isfinite(row.mom_252_xs_pct):
            return None

        if row.mom_252_xs_pct >= self.xs_threshold:
            stop = row.close - self.stop_atr * row.atr_14
            return Setup(
                symbol=row.symbol,
                ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.LONG,
                regime=row.regime,
                reference_price=row.close,
                stop_price=stop,
                target_price=None,
                max_hold_bars=self.max_hold_bars,
                trigger_note="mom_252_xs_pct>=thr",
            )

        if row.mom_252_xs_pct <= (1.0 - self.xs_threshold):
            stop = row.close + self.stop_atr * row.atr_14
            return Setup(
                symbol=row.symbol,
                ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.SHORT,
                regime=row.regime,
                reference_price=row.close,
                stop_price=stop,
                target_price=None,
                max_hold_bars=self.max_hold_bars,
                trigger_note="mom_252_xs_pct<=1-thr",
            )

        return None
