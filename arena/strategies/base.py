"""Rozhraní strategie (SPEC §5.2). Strategie je čistá funkce: žádné I/O, čas zdi ani náhoda (kromě ctl)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, Mapping, Protocol

from arena.features import BarArrays

Side = Literal["LONG", "SHORT"]
SlotStatus = Literal["flat", "in_position", "cooldown"]


@dataclass(frozen=True)
class Intent:
    side: Side
    callback_pct: float | None        # 0.1..10, zaokrouhleno na 0.1; None = bez trailingu (jen ll.v09)
    activate_offset_pct: float | None  # None = trailing hned od vstupu; jinak aktivace při zisku X %
    hard_stop_pct: float              # vzdálenost tvrdého stopu od vstupu, v %
    take_profit_px: float | None      # absolutní cena LIMIT reduceOnly, nebo None
    max_hold_bars: int
    reason: str


@dataclass(frozen=True)
class BarContext:
    symbol: str
    bar_time: int                     # open_time uzavřeného baru
    features: Mapping[str, float]
    slot_id: str
    slot_status: SlotStatus = "flat"
    bars: BarArrays | None = None     # read-only pohled na posledních warmup_bars barů symbolu
    btc: BarArrays | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)   # např. empirická frekvence pro ctl


class Strategy(Protocol):
    family: str
    variant_id: str
    params: Mapping[str, Any]
    warmup_bars: int
    symbols: tuple[str, ...]

    def on_bar(self, ctx: BarContext) -> Intent | None: ...

    def is_stale(self, ctx: BarContext) -> bool: ...

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None: ...


def round1(x: float) -> float:
    return float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def common_callback(atr14_pct: float, k_cb: float, cb_min: float, cb_max: float) -> float | None:
    """`callback_pct = clip(round1(k_cb × atr14_pct), cb_min, cb_max)` (SPEC §5.2)."""
    if not math.isfinite(atr14_pct) or atr14_pct <= 0:
        return None
    return min(max(round1(k_cb * atr14_pct), cb_min), cb_max)


def common_hard_stop(callback_pct: float, hs_mult: float, hs_min_pct: float) -> float:
    """`hard_stop_pct = max(hs_mult × callback_pct, hs_min_pct)` (SPEC §5.2)."""
    return max(hs_mult * callback_pct, hs_min_pct)


def finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)
