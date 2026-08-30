from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scout.domain.enums import RunMode

KNOWN_STRATEGY_PARAMS: dict[str, frozenset[str]] = {
    "xsec_momentum_v1": frozenset(
        {"xs_threshold", "exit_xs_threshold", "stop_atr", "max_hold_bars"}
    ),
    "donchian_breakout_v1": frozenset(
        {
            "entry_buffer_atr",
            "stop_atr",
            "target_rr",
            "max_hold_bars",
            "min_ema_spread_atr",
        }
    ),
}


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class PeriodSplit(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    HOLDOUT = "HOLDOUT"
    FORWARD = "FORWARD"


class DataVendor(str, Enum):
    NORGATE = "norgate"
    SHARADAR = "sharadar"
    POLYGON = "polygon"
    FIXTURE = "fixture"


class TieRule(str, Enum):
    STOP = "stop"
    TARGET = "target"


class LcbMethod(str, Enum):
    NORMAL = "normal"
    BOOTSTRAP = "bootstrap"


class FeaturesJsonPolicy(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_AND_RANKED = "accepted_and_ranked"
    ALL = "all"
    NONE = "none"


class LoggingFormat(str, Enum):
    JSON = "json"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Frozen):
    seed: int = 20260827
    strategy_slug: str = "xsec-momentum-donchian"
    mode: RunMode = RunMode.BACKTEST
    asset_class: AssetClass = AssetClass.EQUITY


class PeriodConfig(_Frozen):
    start: datetime = datetime(1998, 1, 1, tzinfo=UTC)
    warmup_end: datetime = datetime(2004, 1, 1, tzinfo=UTC)
    end: datetime = datetime(2017, 12, 31, 23, 59, 59, tzinfo=UTC)
    split: PeriodSplit = PeriodSplit.DEVELOPMENT

    @field_validator("start", "warmup_end", "end", mode="before")
    @classmethod
    def _aware_utc(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class DataConfig(_Frozen):
    vendor: DataVendor = DataVendor.SHARADAR
    calendar: str = "XNYS"
    decision_timeframe: str = "1d"
    raw_dir: str = "data/raw/equity"
    processed_dir: str = "data/processed"
    snapshot_id_path: str = "data/raw/equity/SNAPSHOT.json"
    max_bar_staleness_bars: int = 2


class SpreadFloorConfig(_Frozen):
    tier_500m: float = 0.5
    tier_100m: float = 1.0
    tier_20m: float = 3.0
    tier_below: float = 8.0


class UniverseConfig(_Frozen):
    candidates_file: str = "config/universe_candidates.txt"
    clusters_file: str = "config/clusters.yaml"
    snapshot_frequency: str = "weekly"
    snapshots_path: str = "data/universe/snapshots.parquet"
    universe_size: int = 1000
    min_history_bars: int = 400
    min_bars_since_gap: int = 10
    min_price_usd: float = 5.00
    min_adv_usd: float = 5_000_000
    max_spread_bps: float = 15.0
    max_suspect_lookback: int = 5
    delisting_grace_bars: int = 3
    spread_window_bars: int = 30
    spread_floor_bps_by_tier: SpreadFloorConfig = Field(default_factory=SpreadFloorConfig)


class FeaturesConfig(_Frozen):
    atr_n: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    slope_lookback: int = 20
    donchian_n: int = 55
    keltner_k: float = 2.0
    er_n: int = 20
    er_long_n: int = 60
    vol_window_bars: int = 504
    vol_min_periods: int = 126
    beta_window: int = 90
    beta_reference_symbol: str = "SPY"
    mom_skip_bars: int = 21
    mom_lookback_bars: int = 252


class RegimeConfig(_Frozen):
    er_trend_min: float = 0.30
    er_long_trend_min: float = 0.20
    slope_min: float = 0.25
    er_range_max: float = 0.15
    vol_range_max: float = 0.67
    vol_low_pct: float = 0.33
    vol_high_pct: float = 0.67


class MarketRegimeConfig(_Frozen):
    benchmark_symbol: str = "SPY"
    sma_n: int = 200
    dd_warn: float = 0.10
    dd_stress: float = 0.15
    vol_stress_pct: float = 0.80


class GatesConfig(_Frozen):
    require_warm: bool = True
    require_universe_eligible: bool = True
    require_regime_allowed: bool = True
    max_bar_staleness_bars: int = 2
    earnings_blackout_sessions: int = 2
    skip_hard_to_borrow: bool = True


class StrategyConfig(_Frozen):
    strategy_id: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


def _default_strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            strategy_id="xsec_momentum_v1",
            enabled=True,
            params={
                "xs_threshold": 0.90,
                "exit_xs_threshold": 0.70,
                "stop_atr": 5.0,
                "max_hold_bars": 21,
            },
        ),
        StrategyConfig(
            strategy_id="donchian_breakout_v1",
            enabled=True,
            params={
                "entry_buffer_atr": 0.10,
                "stop_atr": 3.0,
                "target_rr": 2.0,
                "max_hold_bars": 40,
                "min_ema_spread_atr": 0.20,
            },
        ),
    ]


class LabelingConfig(_Frozen):
    tie_rule: TieRule = TieRule.STOP
    labels_dir: str = "data/labels"


class EdgeConfig(_Frozen):
    bin_dimensions: list[str] = Field(
        default_factory=lambda: ["strategy_id", "direction", "vol_bucket"]
    )
    min_bin_samples: int = 100
    z: float = 1.28
    lcb_method: LcbMethod = LcbMethod.NORMAL
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260827
    as_of_grid: str = "monthly"
    table_path: str = "data/edge/edge_table.parquet"


class ScoringConfig(_Frozen):
    min_ev_net_r: float = 0.05


class CostsConfig(_Frozen):
    commission_per_share_usd: float = 0.0035
    commission_min_usd: float = 0.35
    commission_max_pct_of_notional: float = 0.01
    slippage_fixed_bps: float = 1.0
    slippage_vol_coef: float = 0.02
    impact_coef: float = 1.0
    borrow_bps_per_year_default: float = 30.0
    hard_to_borrow_max_bps_per_year: float = 300.0
    dividend_yield_source: str = "historical"
    dividend_yield_default: float = 0.015
    cost_multiplier: float = 1.0
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    funding_source: str = "default"
    funding_rate_default_per_8h: float = 0.0001


class PortfolioConfig(_Frozen):
    initial_equity_usd: float = 100000
    risk_fraction_per_trade: float = 0.004
    max_positions: int = 6
    max_positions_per_cluster: int = 2
    top_n: int = 3
    max_portfolio_heat_pct: float = 0.020
    max_cluster_risk_pct: float = 0.010
    max_net_beta_pct: float = 0.015
    max_gross_exposure_pct: float = 1.50
    max_position_notional_pct: float = 0.25
    max_pct_of_adv: float = 0.005
    heat_taper_start: float = 0.60
    heat_min_multiplier: float = 0.35
    liquidity_full_size_adv_usd: float = 200_000_000
    liquidity_min_multiplier: float = 0.50


class RiskConfig(_Frozen):
    trading_enabled: bool = True
    max_daily_loss_pct: float = 0.030
    max_drawdown_pct: float = 0.150
    max_trades_per_day: int = 8
    breaker_cooldown_bars: int = 10


class SentimentSourceConfig(_Frozen):
    source_weight: float
    broadcast: bool
    lag_minutes: int


def _default_sentiment_sources() -> dict[str, SentimentSourceConfig]:
    return {
        "vix_term": SentimentSourceConfig(source_weight=0.50, broadcast=True, lag_minutes=0),
        "short_interest": SentimentSourceConfig(
            source_weight=1.00, broadcast=False, lag_minutes=1440
        ),
        "funding_skew": SentimentSourceConfig(source_weight=1.00, broadcast=False, lag_minutes=0),
        "fear_greed": SentimentSourceConfig(source_weight=0.30, broadcast=True, lag_minutes=0),
        "cryptopanic": SentimentSourceConfig(source_weight=1.00, broadcast=False, lag_minutes=60),
    }


def _default_source_ids() -> list[str | None]:
    return [None]


class SentimentConfig(_Frozen):
    enabled: bool = False
    source_ids: list[str | None] = Field(default_factory=_default_source_ids)
    observations_path: str = "data/sentiment/observations.parquet"
    ingest_lag_minutes: int = 60
    half_life_hours: int = 24
    max_age_hours: int = 120
    max_freshness_bars: int = 5
    freshness_half_life_bars: int = 2
    n_eff_full: int = 20
    min_n_eff: float = 3.0
    min_confidence: float = 0.25
    veto_threshold: float = 0.70
    penalty_slope: float = 0.85
    min_multiplier: float = 0.40
    sources: dict[str, SentimentSourceConfig] = Field(default_factory=_default_sentiment_sources)


class AuditConfig(_Frozen):
    results_dir: str = "results"
    features_json_policy: FeaturesJsonPolicy = FeaturesJsonPolicy.ACCEPTED_AND_RANKED
    flush_every_cycles: int = 500


class LoggingConfig(_Frozen):
    level: str = "INFO"
    format: LoggingFormat = LoggingFormat.JSON
    dir: str = "logs"


class ResearchConfig(_Frozen):
    registry_path: str = "experiments/registry.csv"
    lockbox_path: str = "experiments/holdout_lockbox.json"
    holdout_budget: int = 3
    plot_dpi: int = 150
    bootstrap_iterations: int = 2000


class ScoutConfig(_Frozen):
    run: RunConfig = Field(default_factory=RunConfig)
    period: PeriodConfig = Field(default_factory=PeriodConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    market_regime: MarketRegimeConfig = Field(default_factory=MarketRegimeConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    strategies: list[StrategyConfig] = Field(default_factory=_default_strategies)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
