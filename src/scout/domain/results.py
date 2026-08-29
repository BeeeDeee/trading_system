from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from scout.domain._checks import require_aware
from scout.domain.enums import Direction, Regime, RejectionReason, RunMode, SetupOutcome, VolBucket


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A completed round trip. Money fields are Decimal: this feeds the ledger
    and P&L accumulation.
    """

    symbol: str
    strategy_id: str
    direction: Direction
    entry_ts: datetime
    exit_ts: datetime
    entry_price: Decimal
    exit_price: Decimal
    qty: Decimal
    bars_held: int
    outcome: SetupOutcome
    gross_pnl_usd: Decimal
    fees_usd: Decimal
    dividends_usd: Decimal
    borrow_usd: Decimal
    funding_usd: Decimal  # crypto (M7); zero for equities
    net_pnl_usd: Decimal
    realised_r: float  # net_pnl / risk_at_entry. float: analytics.
    mae_r: float  # max adverse excursion, in R
    mfe_r: float  # max favourable excursion, in R
    regime: Regime
    vol_bucket: VolBucket
    cluster: str
    ev_net_r_at_entry: float  # what we predicted. Enables calibration checks.

    def __post_init__(self) -> None:
        require_aware(self.entry_ts, "entry_ts")
        require_aware(self.exit_ts, "exit_ts")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: str
    config_hash: str
    started_at: datetime
    period_start: datetime
    period_end: datetime
    mode: RunMode

    trades: tuple[ClosedTrade, ...]
    equity_curve: pd.Series  # UTC DatetimeIndex, float equity
    benchmark_curve: pd.Series  # buy-and-hold SPY (total return), same index
    metrics: Mapping[str, float]
    n_decisions_considered: int
    n_rejections_by_reason: Mapping[RejectionReason, int]

    def __post_init__(self) -> None:
        require_aware(self.started_at, "started_at")
        require_aware(self.period_start, "period_start")
        require_aware(self.period_end, "period_end")
