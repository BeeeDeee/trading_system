from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware
from scout.domain.edge import BinKey
from scout.domain.enums import Direction, Regime, VolBucket
from scout.domain.setup import Setup


@dataclass(frozen=True, slots=True)
class Opportunity:
    """A scored candidate. Exactly one ranking statistic; everything else is a
    diagnostic or provenance.
    """

    # --- identity ---
    symbol: str
    ts: datetime
    strategy_id: str
    direction: Direction
    setup: Setup

    # --- the ranking statistic ---
    ev_net_r: float  # ev_r_lcb - cost_r. Units: R. THE number.
    ev_per_bar_r: float  # ev_net_r / max(expected_bars_held, 1.0)

    # --- components, for auditing the number above ---
    ev_r_lcb: float
    ev_r_point: float  # bin mean_r, un-penalised
    cost_r: float
    expected_bars_held: float

    # --- provenance ---
    bin_key: BinKey
    bin_n: int
    bin_std_r: float
    bin_win_rate: float
    regime: Regime
    vol_bucket: VolBucket

    # --- context recorded for research, not used in ranking ---
    adv_usd_30: float
    spread_bps_est: float
    beta_bench_90: float
    cluster: str

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")
