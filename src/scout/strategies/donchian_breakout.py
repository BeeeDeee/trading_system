"""55-session Donchian breakout. Secondary v1 strategy; contrast shape."""

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
    {
        "entry_buffer_atr",
        "stop_atr",
        "target_rr",
        "max_hold_bars",
        "min_ema_spread_atr",
    }
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
class DonchianBreakout:
    strategy_id: ClassVar[str] = "donchian_breakout_v1"
    allowed_regimes: ClassVar[frozenset[Regime]] = frozenset(
        {Regime.TREND_UP, Regime.TREND_DOWN}
    )
    allowed_market_regimes: ClassVar[frozenset[MarketRegime]] = frozenset(
        {MarketRegime.RISK_ON, MarketRegime.NEUTRAL}
    )
    required_warmup_bars: ClassVar[int] = 260

    entry_buffer_atr: float = 0.10
    stop_atr: float = 3.0
    target_rr: float = 2.0
    max_hold_bars: int = 40
    min_ema_spread_atr: float = 0.20

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> DonchianBreakout:
        unknown = sorted(set(params) - _PARAM_NAMES)
        if unknown:
            raise ScoutConfigError(
                f"unknown param {unknown[0]!r} for {cls.strategy_id}; "
                f"known: {sorted(_PARAM_NAMES)}"
            )
        kwargs: dict[str, Any] = {}
        if "entry_buffer_atr" in params:
            entry_buffer_atr = _as_float("entry_buffer_atr", params["entry_buffer_atr"])
            if entry_buffer_atr < 0:
                raise ScoutConfigError("entry_buffer_atr: must be >= 0")
            kwargs["entry_buffer_atr"] = entry_buffer_atr
        if "stop_atr" in params:
            stop_atr = _as_float("stop_atr", params["stop_atr"])
            if stop_atr <= 0:
                raise ScoutConfigError("stop_atr: must be > 0")
            kwargs["stop_atr"] = stop_atr
        if "target_rr" in params:
            target_rr = _as_float("target_rr", params["target_rr"])
            if target_rr <= 0:
                raise ScoutConfigError("target_rr: must be > 0")
            kwargs["target_rr"] = target_rr
        if "max_hold_bars" in params:
            kwargs["max_hold_bars"] = _as_hold_bars(params["max_hold_bars"])
        if "min_ema_spread_atr" in params:
            min_ema_spread_atr = _as_float("min_ema_spread_atr", params["min_ema_spread_atr"])
            if min_ema_spread_atr < 0:
                raise ScoutConfigError("min_ema_spread_atr: must be >= 0")
            kwargs["min_ema_spread_atr"] = min_ema_spread_atr
        return cls(**kwargs)

    def detect(self, row: FeatureRow) -> Setup | None:
        if not row.is_warm:
            return None
        if not math.isfinite(row.close) or not math.isfinite(row.atr_14) or row.atr_14 <= 0:
            return None
        if not math.isfinite(row.ema_spread_atr):
            return None

        if row.regime is Regime.TREND_UP:
            if not math.isfinite(row.donchian_high_55):
                return None
            if row.ema_spread_atr < self.min_ema_spread_atr:
                return None
            breakout_level = row.donchian_high_55 + self.entry_buffer_atr * row.atr_14
            if row.close <= breakout_level:
                return None
            stop = row.close - self.stop_atr * row.atr_14
            target = row.close + self.target_rr * self.stop_atr * row.atr_14
            return Setup(
                symbol=row.symbol,
                ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.LONG,
                regime=row.regime,
                reference_price=row.close,
                stop_price=stop,
                target_price=target,
                max_hold_bars=self.max_hold_bars,
                trigger_note="close>dc_high_55+buf",
            )

        if row.regime is Regime.TREND_DOWN:
            if not math.isfinite(row.donchian_low_55):
                return None
            if row.ema_spread_atr > -self.min_ema_spread_atr:
                return None
            breakdown_level = row.donchian_low_55 - self.entry_buffer_atr * row.atr_14
            if row.close >= breakdown_level:
                return None
            stop = row.close + self.stop_atr * row.atr_14
            target = row.close - self.target_rr * self.stop_atr * row.atr_14
            return Setup(
                symbol=row.symbol,
                ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.SHORT,
                regime=row.regime,
                reference_price=row.close,
                stop_price=stop,
                target_price=target,
                max_hold_bars=self.max_hold_bars,
                trigger_note="close<dc_low_55-buf",
            )

        return None
