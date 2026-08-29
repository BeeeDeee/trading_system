from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from scout.domain._checks import require_aware
from scout.domain.enums import Direction, OrderType


@dataclass(frozen=True, slots=True)
class OrderIntent:
    client_order_id: str  # idempotency key; deterministic
    symbol: str
    side: Direction
    order_type: OrderType
    qty: Decimal
    limit_price: Decimal | None
    stop_price: Decimal | None
    reduce_only: bool
    created_ts: datetime
    correlation_id: str  # links entry, stop, target, and the decision

    def __post_init__(self) -> None:
        require_aware(self.created_ts, "created_ts")


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    symbol: str
    side: Direction
    qty: Decimal
    price: Decimal
    fee_usd: Decimal
    ts: datetime
    is_maker: bool
    exit_reason: str | None  # "TARGET" | "STOP" | "TIME" | "DELISTED" | "KILL_SWITCH" | None

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")
