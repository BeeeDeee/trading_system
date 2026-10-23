"""Filtry symbolů z `exchangeInfo` a zaokrouhlování cen a množství (jen `Decimal`, nástraha 6)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any, Literal

PositionSide = Literal["LONG", "SHORT"]


def _d(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def quantize_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    """Zaokrouhlí na násobek `step` (step nemusí být mocnina 10). Výsledek má exponent stepu."""
    units = (value / step).to_integral_value(rounding=rounding)
    return units * step


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    market_step_size: Decimal
    market_min_qty: Decimal
    market_max_qty: Decimal
    min_notional: Decimal
    trigger_protect: Decimal | None
    underlying_type: str | None
    status: str

    @classmethod
    def from_exchange_info(cls, s: dict[str, Any]) -> SymbolFilters:
        f = {x["filterType"]: x for x in s["filters"]}
        lot = f["LOT_SIZE"]
        mlot = f.get("MARKET_LOT_SIZE", lot)
        tp = s.get("triggerProtect")
        return cls(
            symbol=s["symbol"],
            tick_size=_d(f["PRICE_FILTER"]["tickSize"]),
            step_size=_d(lot["stepSize"]),
            min_qty=_d(lot["minQty"]),
            max_qty=_d(lot["maxQty"]),
            market_step_size=_d(mlot["stepSize"]),
            market_min_qty=_d(mlot["minQty"]),
            market_max_qty=_d(mlot["maxQty"]),
            min_notional=_d(f["MIN_NOTIONAL"]["notional"]),
            trigger_protect=_d(tp) if tp is not None else None,
            underlying_type=s.get("underlyingType"),
            status=s.get("status", ""),
        )

    # --- ceny ---------------------------------------------------------------

    def round_price(self, px: Decimal | float | str, mode: Literal["down", "up", "nearest"] = "nearest") -> Decimal:
        rounding = {"down": ROUND_FLOOR, "up": ROUND_CEILING, "nearest": ROUND_HALF_UP}[mode]
        return quantize_to_step(_d(px), self.tick_size, rounding)

    def stop_price(self, raw_px: Decimal | float | str, position: PositionSide) -> tuple[Decimal, bool]:
        """Cena tvrdého stopu zaokrouhlená konzervativně: nikdy dál od trhu, než je požadováno.

        LONG stop (SELL pod cenou) se zaokrouhlí nahoru, SHORT stop (BUY nad cenou) dolů.
        Vrací (cena, změněno) – volající změnu zaloguje (SPEC §6)."""
        raw = _d(raw_px)
        px = self.round_price(raw, "up" if position == "LONG" else "down")
        return px, px != raw

    def take_profit_price(self, raw_px: Decimal | float | str, position: PositionSide) -> Decimal:
        """TP LIMIT se zaokrouhlí od trhu (LONG nahoru, SHORT dolů), aby nebyl agresivnější než intent."""
        return self.round_price(raw_px, "up" if position == "LONG" else "down")

    # --- množství -----------------------------------------------------------

    def floor_qty(self, qty: Decimal | float | str, market: bool = True) -> Decimal:
        step = self.market_step_size if market else self.step_size
        return quantize_to_step(_d(qty), step, ROUND_FLOOR)

    def qty_for_notional(self, notional: Decimal | float | str, price: Decimal | float | str,
                         market: bool = True) -> Decimal:
        """Množství pro cílový notional, zaokrouhlené dolů na step. Vyhodí ValueError pod filtry."""
        n, p = _d(notional), _d(price)
        qty = self.floor_qty(n / p, market)
        lo = self.market_min_qty if market else self.min_qty
        hi = self.market_max_qty if market else self.max_qty
        if qty < lo:
            raise ValueError(f"{self.symbol}: qty {qty} < minQty {lo} (notional {n} @ {p})")
        if qty > hi:
            raise ValueError(f"{self.symbol}: qty {qty} > maxQty {hi}")
        if qty * p < self.min_notional:
            raise ValueError(f"{self.symbol}: notional {qty * p} < MIN_NOTIONAL {self.min_notional}")
        return qty


def round_callback_rate(pct: float) -> Decimal:
    """`callbackRate` je v PROCENTECH (0.5 = 0,5 %), krok 0.1, rozsah 0.1–10 (SPEC D3, nástraha 5)."""
    d = Decimal(str(pct)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return min(max(d, Decimal("0.1")), Decimal("10"))


def parse_exchange_info(info: dict[str, Any], symbols: list[str] | tuple[str, ...] | None = None) -> dict[str, SymbolFilters]:
    out: dict[str, SymbolFilters] = {}
    for s in info["symbols"]:
        if symbols is None or s["symbol"] in symbols:
            out[s["symbol"]] = SymbolFilters.from_exchange_info(s)
    return out
