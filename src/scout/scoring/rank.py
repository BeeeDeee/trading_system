"""Build and rank Opportunity records.

One ranking statistic: ev_net_r = ev_r_lcb - cost_r. Sort key is ev_per_bar_r.
"""

from __future__ import annotations

from collections.abc import Sequence

from scout.config.schema import ScoringConfig
from scout.domain.costs import CostEstimate
from scout.domain.edge import BinStats
from scout.domain.features import FeatureRow
from scout.domain.market import Asset
from scout.domain.opportunity import Opportunity
from scout.domain.setup import Setup
from scout.domain.universe import UniverseEntry


def build_opportunity(
    setup: Setup,
    row: FeatureRow,
    stats: BinStats,
    cost: CostEstimate,
    asset: Asset,
    universe_entry: UniverseEntry,
) -> Opportunity:
    """Populate every Opportunity field from the decision-time inputs."""
    ev_net_r = stats.ev_r_lcb - cost.cost_r
    expected_bars_held = max(stats.mean_bars_held, 1.0)
    return Opportunity(
        symbol=setup.symbol,
        ts=setup.ts,
        strategy_id=setup.strategy_id,
        direction=setup.direction,
        setup=setup,
        ev_net_r=ev_net_r,
        ev_per_bar_r=ev_net_r / expected_bars_held,
        ev_r_lcb=stats.ev_r_lcb,
        ev_r_point=stats.mean_r,
        cost_r=cost.cost_r,
        expected_bars_held=expected_bars_held,
        bin_key=stats.key,
        bin_n=stats.n,
        bin_std_r=stats.std_r,
        bin_win_rate=stats.win_rate,
        regime=row.regime,
        vol_bucket=row.vol_bucket,
        adv_usd_30=universe_entry.adv_usd_30,
        spread_bps_est=universe_entry.spread_bps_est,
        beta_bench_90=row.beta_bench_90,
        cluster=asset.cluster,
    )


def rank(
    candidates: Sequence[Opportunity],
    cfg: ScoringConfig,
) -> tuple[Opportunity, ...]:
    """Keep ev_net_r >= min_ev_net_r; sort by ev_per_bar_r desc, then (symbol, strategy_id)."""
    eligible = [c for c in candidates if c.ev_net_r >= cfg.min_ev_net_r]
    return tuple(
        sorted(
            eligible,
            key=lambda c: (-c.ev_per_bar_r, c.symbol, c.strategy_id),
        )
    )
