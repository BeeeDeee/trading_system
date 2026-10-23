"""Společné typy executorů (SPEC §7, §8, §12)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class FillQuote:
    px: float
    source: Literal["book", "ticker", "trade", "outage"]
    complete: bool = True      # False = kniha neměla dost hloubky


class FillModel(Protocol):
    def market(self, symbol: str, side: OrderSide, qty: float, t_ms: int,
               trade: dict[str, Any] | None = None) -> FillQuote | None: ...

    def mid(self, symbol: str) -> float | None: ...

    def last_price(self, symbol: str) -> float | None: ...


@dataclass
class TradeRecord:
    """Jeden uzavřený obchod = řádek tabulky `trades` (SPEC §12)."""

    trade_id: str
    slot_id: str
    mode: str
    signal_id: int | None
    side: str
    qty: float
    notional: float
    decision_px: float
    arrival_mid: float | None
    entry_px: float
    entry_time: int
    entry_fee: float
    exit_trigger_px: float
    exit_px: float
    exit_time: int
    exit_fee: float
    exit_reason: str
    funding: float
    gross_pnl: float
    net_pnl: float
    gross_bps: float
    net_bps: float
    entry_slip_bps: float
    exit_slip_bps: float
    mfe_bps: float | None
    mae_bps: float | None
    bars_held: int
    tainted: int
    cost_model_version: int
    config_hash: str
    twin_trade_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # nejde do DB tabulky trades

    def row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra")
        return d


def side_sign(side: str) -> int:
    return 1 if side == "LONG" else -1


def bps(x: float, ref: float) -> float:
    return x / ref * 1e4 if ref else 0.0
