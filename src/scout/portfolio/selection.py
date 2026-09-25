"""Greedy portfolio selection: one TradeDecision per ranked Opportunity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from types import MappingProxyType

from scout.config.schema import CostsConfig, PortfolioConfig, RiskConfig, SentimentConfig
from scout.costs.model import estimate_cost
from scout.domain.costs import CostEstimate
from scout.domain.enums import OrderType, RejectionReason
from scout.domain.market import Asset
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState, Position, TradeDecision
from scout.domain.sentiment import SentimentView
from scout.domain.universe import UniverseEntry
from scout.portfolio.breakers import breaker_tripped, trading_enabled
from scout.portfolio.sizing import (
    heat_multiplier,
    liquidity_multiplier,
    size_position,
    to_raw_distance,
)
from scout.sentiment.multiplier import sentiment_multiplier
from scout.utils.decimals import to_decimal
from scout.utils.errors import ScoutError

_ZERO = Decimal("0")


def select_and_size(
    ranked: Sequence[Opportunity],
    state: PortfolioState,
    sentiment: Mapping[str, SentimentView],
    assets: Mapping[str, Asset],
    cfg: PortfolioConfig,
    risk_cfg: RiskConfig,
    sentiment_cfg: SentimentConfig,
    cost_cfg: CostsConfig,
    min_ev_net_r: float,
    price_raw_by_symbol: Mapping[str, float] | None = None,
) -> tuple[TradeDecision, ...]:
    """Greedy in rank order. One decision per input, accepted or rejected.

    `price_raw_by_symbol` is `close_raw` at the decision bar (ADR-017). When
    omitted, sizing falls back to `setup.reference_price`.
    """
    decisions: list[TradeDecision] = []
    provisional = state

    for rank, opp in enumerate(ranked, start=1):
        if not trading_enabled(risk_cfg):
            decisions.append(_rejected(opp, rank, RejectionReason.KILL_SWITCH))
            continue
        if breaker_tripped(provisional, risk_cfg) is not None:
            decisions.append(_rejected(opp, rank, RejectionReason.CIRCUIT_BREAKER))
            continue

        if opp.symbol in provisional.positions:
            decisions.append(_rejected(opp, rank, RejectionReason.ALREADY_IN_POSITION))
            continue
        if len(provisional.positions) >= cfg.max_positions:
            decisions.append(_rejected(opp, rank, RejectionReason.MAX_POSITIONS))
            continue
        if rank > cfg.top_n:
            decisions.append(_rejected(opp, rank, RejectionReason.BELOW_TOP_N))
            continue

        view = sentiment.get(opp.symbol)
        mult_sentiment = sentiment_multiplier(view, opp.direction, sentiment_cfg)
        if mult_sentiment <= 0.0:
            decisions.append(_rejected(opp, rank, RejectionReason.SENTIMENT_VETO))
            continue

        if opp.symbol not in assets:
            raise ScoutError(f"no Asset for {opp.symbol!r}")

        mult = (
            mult_sentiment
            * liquidity_multiplier(opp, cfg)
            * heat_multiplier(provisional, cfg)
        )
        raw = _price_raw(opp, price_raw_by_symbol)
        qty, risk_usd = size_position(
            opp, provisional, assets[opp.symbol], mult, cfg, price_raw=raw
        )

        notional = qty * to_decimal(raw)
        if qty <= 0 or notional < assets[opp.symbol].min_notional_usd:
            decisions.append(_rejected(opp, rank, RejectionReason.SIZE_BELOW_MIN_NOTIONAL))
            continue

        after = simulate_add(provisional, opp, qty, risk_usd, price_raw=raw)
        if after.open_risk_pct > cfg.max_portfolio_heat_pct:
            decisions.append(_rejected(opp, rank, RejectionReason.PORTFOLIO_HEAT_CAP))
            continue
        if after.cluster_risk_pct(opp.cluster) > cfg.max_cluster_risk_pct:
            decisions.append(_rejected(opp, rank, RejectionReason.CLUSTER_CAP))
            continue
        if _cluster_count(after, opp.cluster) > cfg.max_positions_per_cluster:
            decisions.append(_rejected(opp, rank, RejectionReason.CLUSTER_CAP))
            continue
        if abs(after.net_beta_exposure_pct) > cfg.max_net_beta_pct:
            decisions.append(_rejected(opp, rank, RejectionReason.BETA_CAP))
            continue
        if after.gross_exposure_pct > cfg.max_gross_exposure_pct:
            decisions.append(_rejected(opp, rank, RejectionReason.GROSS_EXPOSURE_CAP))
            continue

        final_cost = estimate_cost(
            opp.setup,
            _universe_entry(opp),
            float(notional),
            float(risk_usd),
            opp.expected_bars_held,
            cost_cfg,
            atr_pct=opp.atr_pct,
            price_raw=raw,
        )
        if opp.ev_r_lcb - final_cost.cost_r < min_ev_net_r:
            decisions.append(_rejected(opp, rank, RejectionReason.BELOW_EV_THRESHOLD))
            continue

        decisions.append(
            _accepted(opp, rank, qty, risk_usd, mult, mult_sentiment, final_cost, raw)
        )
        provisional = after

    return tuple(decisions)


def simulate_add(
    state: PortfolioState,
    opp: Opportunity,
    qty: Decimal,
    risk_usd: Decimal,
    *,
    price_raw: float | None = None,
) -> PortfolioState:
    """Provisional state after accepting `opp` at `qty`. Caps see this, not `state`."""
    del risk_usd  # implied by qty and stop distance at entry; Position recomputes it
    entry = to_decimal(
        opp.setup.reference_price if price_raw is None else price_raw
    )
    risk = to_raw_distance(
        opp.setup.reference_price,
        opp.setup.stop_price,
        raw_price=entry,
    )
    stop = entry - Decimal(opp.direction.sign) * risk
    if opp.setup.target_price is None:
        target = entry
    else:
        reward = to_raw_distance(
            opp.setup.reference_price,
            opp.setup.target_price,
            raw_price=entry,
        )
        target = entry + Decimal(opp.direction.sign) * reward
    position = Position(
        symbol=opp.symbol,
        direction=opp.direction,
        qty=qty,
        entry_price=entry,
        entry_ts=opp.ts,
        stop_price=stop,
        target_price=target,
        max_hold_bars=opp.setup.max_hold_bars,
        bars_held=0,
        strategy_id=opp.strategy_id,
        cluster=opp.cluster,
        beta_bench_90=opp.beta_bench_90,
        client_order_id=_client_order_id(opp),
        realised_fees_usd=_ZERO,
        dividends_usd=_ZERO,
        borrow_usd=_ZERO,
        funding_paid_usd=_ZERO,
    )
    positions = dict(state.positions)
    positions[opp.symbol] = position
    return PortfolioState(
        ts=state.ts,
        equity_usd=state.equity_usd,
        cash_usd=state.cash_usd,
        positions=MappingProxyType(positions),
        peak_equity_usd=state.peak_equity_usd,
        realised_pnl_today_usd=state.realised_pnl_today_usd,
        day_start_equity_usd=state.day_start_equity_usd,
        trades_today=state.trades_today + 1,
        bars_since_breaker=state.bars_since_breaker,
    )


def _price_raw(
    opp: Opportunity, price_raw_by_symbol: Mapping[str, float] | None
) -> float:
    if price_raw_by_symbol is not None and opp.symbol in price_raw_by_symbol:
        return float(price_raw_by_symbol[opp.symbol])
    return opp.setup.reference_price


def _universe_entry(opp: Opportunity) -> UniverseEntry:
    return UniverseEntry(
        symbol=opp.symbol,
        eligible=True,
        reason=None,
        adv_usd_30=opp.adv_usd_30,
        spread_bps_est=opp.spread_bps_est,
        bars_available=0,
        listed_days=0.0,
    )


def _cluster_count(state: PortfolioState, cluster: str) -> int:
    return sum(1 for pos in state.positions.values() if pos.cluster == cluster)


def _client_order_id(opp: Opportunity) -> str:
    # Engine (M3.4) prefixes run_id[:8] when building OrderIntent.
    return f"{opp.symbol}-{int(opp.ts.timestamp())}-e"


def _rejected(opp: Opportunity, rank: int, reason: RejectionReason) -> TradeDecision:
    return TradeDecision(
        opportunity=opp,
        accepted=False,
        rejection_reason=reason,
        rank=rank,
        qty=None,
        entry_order_type=None,
        intended_notional_usd=None,
        risk_usd=None,
        size_multiplier=None,
        sentiment_multiplier=None,
        client_order_id=None,
        final_cost=None,
    )


def _accepted(
    opp: Opportunity,
    rank: int,
    qty: Decimal,
    risk_usd: Decimal,
    size_multiplier: float,
    sentiment_multiplier: float,
    final_cost: CostEstimate,
    price_raw: float,
) -> TradeDecision:
    return TradeDecision(
        opportunity=opp,
        accepted=True,
        rejection_reason=None,
        rank=rank,
        qty=qty,
        entry_order_type=OrderType.MARKET,
        intended_notional_usd=qty * to_decimal(price_raw),
        risk_usd=risk_usd,
        size_multiplier=size_multiplier,
        sentiment_multiplier=sentiment_multiplier,
        client_order_id=_client_order_id(opp),
        final_cost=final_cost,
    )
