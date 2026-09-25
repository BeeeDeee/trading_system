from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware
from scout.domain.enums import RejectionReason


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    eligible: bool
    reason: RejectionReason | None  # None iff eligible
    adv_usd_30: float  # trailing 30-bar median quote volume, scaled to a daily figure
    spread_bps_est: float  # causal estimate; see 04-DATA doc
    bars_available: int
    listed_days: float

    def __post_init__(self) -> None:
        if self.eligible and self.reason is not None:
            raise ValueError("reason must be None iff eligible")
        if not self.eligible and self.reason is None:
            raise ValueError("ineligible entry requires a RejectionReason")


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    ts: datetime
    entries: Mapping[str, UniverseEntry]  # keyed by symbol

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        """Sorted, for determinism. Never rely on dict insertion order."""
        return tuple(sorted(symbol for symbol, entry in self.entries.items() if entry.eligible))
