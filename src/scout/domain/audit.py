from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware
from scout.domain.enums import Direction, Regime, RejectionReason, VolBucket


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One row per (decision_ts, symbol, strategy_id) CONSIDERED, accepted or
    not. This is the system's most valuable output. Written for every cycle,
    including cycles with zero trades.

    Flat and denormalised on purpose: it becomes one Parquet table that
    answers research questions with a groupby instead of a join.
    """

    run_id: str
    decision_ts: datetime
    symbol: str
    strategy_id: str | None  # None for pre-strategy rejections

    stage: str  # "GATE"|"STRATEGY"|"EDGE"|"RANK"|"PORTFOLIO"|"SIZE"
    accepted: bool
    rejection_reason: RejectionReason | None

    direction: Direction | None
    regime: Regime | None
    vol_bucket: VolBucket | None

    reference_price: float | None
    stop_price: float | None
    target_price: float | None
    reward_risk_ratio: float | None

    bin_key: str | None  # BinKey rendered as "strategy|dir|vol"
    bin_n: int | None
    ev_r_point: float | None
    ev_r_lcb: float | None
    cost_r: float | None
    ev_net_r: float | None
    ev_per_bar_r: float | None
    rank: int | None

    sentiment_score: float | None
    sentiment_confidence: float | None
    sentiment_multiplier: float | None

    portfolio_heat_pct: float | None
    cluster: str | None
    size_multiplier: float | None
    qty: float | None  # float here; this row is analytics, not a ledger
    notional_usd: float | None

    adv_usd_30: float | None
    spread_bps_est: float | None
    features_json: str | None  # compact JSON of the FeatureRow, for forensics

    def __post_init__(self) -> None:
        require_aware(self.decision_ts, "decision_ts")
