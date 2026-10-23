"""Intent → konkrétní úrovně příkazů. JEDINÁ funkce pro paper i live (nástraha 12), jinak dvojčata nejsou
srovnatelná: zaokrouhlení callbacku, cena tvrdého stopu, activatePrice a TP se počítají tady."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from arena.binance.filters import SymbolFilters, round_callback_rate
from arena.strategies.base import Intent

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class ExitPlan:
    side: Side
    qty: Decimal
    entry_px: Decimal
    callback_rate: Decimal | None     # v procentech, krok 0.1 (None = bez trailingu)
    activate_px: Decimal | None       # None = trailing hned
    hard_stop_px: Decimal
    tp_px: Decimal | None
    max_hold_bars: int
    notes: tuple[str, ...] = field(default_factory=tuple)   # zaokrouhlení a úpravy k zalogování

    @property
    def close_side(self) -> Literal["BUY", "SELL"]:
        return "SELL" if self.side == "LONG" else "BUY"


def entry_qty(f: SymbolFilters, notional_usdt: float | Decimal, ref_px: float | Decimal) -> Decimal:
    """Množství pro vstup: notional / referenční cena, dolů na MARKET_LOT_SIZE step (vyhodí pod filtry)."""
    return f.qty_for_notional(Decimal(str(notional_usdt)), Decimal(str(ref_px)), market=True)


def plan_exits(intent: Intent, f: SymbolFilters, entry_px: float | Decimal, qty: Decimal) -> ExitPlan:
    """Úrovně výstupů od skutečné (live) nebo simulované (paper) vstupní ceny."""
    e = Decimal(str(entry_px))
    sign = Decimal(1) if intent.side == "LONG" else Decimal(-1)
    notes: list[str] = []

    cb = round_callback_rate(intent.callback_pct) if intent.callback_pct is not None else None
    if cb is not None and float(cb) != round(intent.callback_pct or 0, 1):
        notes.append(f"callback {intent.callback_pct} → {cb}")

    raw_stop = e * (1 - sign * Decimal(str(intent.hard_stop_pct)) / 100)
    stop, changed = f.stop_price(raw_stop, intent.side)
    if changed:
        notes.append(f"hard stop {raw_stop} → {stop} (konzervativně, blíž k ceně)")

    act = None
    if intent.activate_offset_pct is not None and cb is not None:
        raw_act = e * (1 + sign * Decimal(str(intent.activate_offset_pct)) / 100)
        # aktivace dál od trhu = později; zaokrouhlení k bližší straně (konzervativně pro ochranu zisku)
        act = f.round_price(raw_act, "down" if intent.side == "LONG" else "up")

    tp = None
    if intent.take_profit_px is not None:
        tp = f.take_profit_price(Decimal(str(intent.take_profit_px)), intent.side)

    return ExitPlan(intent.side, qty, e, cb, act, stop, tp, intent.max_hold_bars, tuple(notes))
