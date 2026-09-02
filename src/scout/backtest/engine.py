"""Timestamp-outer backtest loop. Never loop symbols on the outside."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from scout.backtest import ledger as ledger_mod
from scout.config.hashing import config_hash
from scout.config.schema import FeaturesJsonPolicy, ScoutConfig
from scout.costs.model import estimate_cost
from scout.data.ingest import read_candidate_pairs
from scout.data.store import read_parquet
from scout.domain.audit import DecisionRecord
from scout.domain.edge import BinKey, BinStats, EdgeTable
from scout.domain.enums import (
    Direction,
    OrderType,
    Regime,
    RejectionReason,
    RunMode,
    VolBucket,
)
from scout.domain.execution import Fill, OrderIntent
from scout.domain.features import FeaturePanel, FeatureRow, feature_row_from_record
from scout.domain.market import (
    BENCHMARK_COLUMNS,
    Asset,
    BenchmarkPanel,
    CorporateAction,
    EarningsEvent,
    MarketPanel,
)
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState, TradeDecision
from scout.domain.ports import Broker, CandleSource, DecisionSink, SentimentSource, Strategy
from scout.domain.results import BacktestResult, ClosedTrade
from scout.domain.sentiment import SentimentObservation, SentimentView
from scout.domain.setup import Setup
from scout.domain.universe import UniverseEntry, UniverseSnapshot
from scout.features.engine import compute_features
from scout.gates.eligibility import evaluate_gates
from scout.portfolio.selection import select_and_size
from scout.scoring.rank import build_opportunity, rank
from scout.sentiment.aggregate import build_view
from scout.storage.run_outputs import compute_headline_metrics, make_run_id
from scout.utils.clock import BarClock, Clock, WallClock
from scout.utils.decimals import to_decimal
from scout.utils.errors import ScoutConfigError, ScoutDataError


class BacktestEngine:
    """Panel loop: timestamps outside, symbols inside. Exits before entries."""

    def __init__(
        self,
        *,
        candles: CandleSource,
        sentiment: SentimentSource,
        broker: Broker,
        strategies: Sequence[Strategy],
        edge_table: EdgeTable,
        sink: DecisionSink,
        universe: Any,
        assets: Mapping[str, Asset],
        cfg: ScoutConfig,
        calendar: pd.DataFrame | None = None,
        earnings_by_asset: Mapping[str, Sequence[EarningsEvent]] | None = None,
        benchmark: BenchmarkPanel | None = None,
        actions: Sequence[CorporateAction] = (),
        run_id: str = "",
        clock: Clock | None = None,
        features: FeaturePanel | None = None,
    ) -> None:
        self.candles = candles
        self.sentiment = sentiment
        self.broker = broker
        self.strategies = tuple(strategies)
        self.edge_table = edge_table
        self.sink = sink
        self.universe = universe
        self.assets = dict(assets)
        self.cfg = cfg
        self.calendar = calendar if calendar is not None else pd.DataFrame()
        self.earnings_by_asset: Mapping[str, Sequence[EarningsEvent]] = (
            earnings_by_asset if earnings_by_asset is not None else {}
        )
        self._benchmark = benchmark
        self._actions = tuple(actions)
        self.run_id = run_id
        self._wall = clock if clock is not None else WallClock()
        self._bar_clock = BarClock()
        self._entry_meta: dict[str, tuple[Regime, VolBucket, float]] = {}
        self._used_bins: list[BinStats] = []
        self._last_close: dict[str, Decimal] = {}
        self._last_close_raw: dict[str, float] = {}
        self._last_ts_by_symbol: dict[str, datetime] = {}
        self._session_pos: dict[datetime, int] = {}
        self._asset_id_by_symbol: dict[str, str] = {
            asset.symbol: asset.asset_id for asset in self.assets.values()
        }
        self._observations: tuple[SentimentObservation, ...] = ()
        self._features = features

    def run(self) -> BacktestResult:
        started_at = self._wall.now()
        digest = config_hash(self.cfg)
        if self.edge_table.config_hash and self.edge_table.config_hash != digest:
            raise ScoutConfigError(
                "edge table config_hash does not match the resolved config; "
                "rebuild the edge table under this config"
            )
        if not self.run_id:
            self.run_id = make_run_id(self.cfg.run.strategy_slug, self._wall)

        candidates = read_candidate_pairs(Path(self.cfg.universe.candidates_file))
        symbols = [str(x) for x in candidates["asset_id"].tolist()]
        if not symbols:
            symbols = [a.asset_id for a in self.assets.values()]
        panel = self.candles.load_panel(
            symbols,
            self.cfg.data.decision_timeframe,
            self.cfg.period.start,
            self.cfg.period.end,
        )
        attach = getattr(self.broker, "attach_panel", None)
        if callable(attach):
            attach(panel)
        attach_actions = getattr(self.broker, "attach_actions", None)
        if callable(attach_actions) and self._actions:
            attach_actions(self._actions)

        benchmark = self._benchmark if self._benchmark is not None else _load_benchmark(self.cfg)
        snapshots_frame = self.universe.load_all()
        if self._features is not None:
            features = self._features
        else:
            features = compute_features(panel, benchmark, snapshots_frame, self.cfg.features)
        self._session_pos = {_as_utc(t): i for i, t in enumerate(panel.timestamps)}

        self._observations = self.sentiment.observations(
            tuple(sorted(self.assets)),
            self.cfg.period.start,
            self.cfg.period.end,
        )

        first_ts = (
            _as_utc(panel.timestamps[0])
            if len(panel.timestamps)
            else self.cfg.period.start
        )
        state = ledger_mod.initial_state(self.cfg.portfolio.initial_equity_usd, first_ts)
        equity_points: list[tuple[datetime, float]] = []
        trades: list[ClosedTrade] = []
        n_considered = 0
        reason_counts: Counter[RejectionReason] = Counter()
        records: list[DecisionRecord] = []

        for ts in panel.timestamps:
            py_ts = _as_utc(ts)
            if py_ts < self.cfg.period.warmup_end:
                continue
            self._bar_clock.set(py_ts)

            bar_block = panel.rows_at(py_ts)
            self._update_close_cache(bar_block)

            marks = self._marks_for(state, py_ts)
            exit_kwargs = self._exit_kwargs_from_brackets(state)
            state = _broker_mark(self.broker, state, panel, py_ts, marks)

            fills = self.broker.poll_fills(py_ts)
            state, closed = ledger_mod.apply_exits(
                state,
                fills,
                py_ts,
                last_marks=marks,
                **exit_kwargs,
            )
            trades.extend(closed)
            self._forget_closed(closed)

            state, closed = self.apply_time_stops(state, panel, py_ts)
            trades.extend(closed)
            state, closed = self.apply_delistings(state, panel, py_ts)
            trades.extend(closed)

            snapshot = self.universe.snapshot_at(py_ts)
            candidates_opp, recs = self.evaluate(py_ts, features, snapshot, state)
            records.extend(recs)
            n_considered += len(recs)
            for rec in recs:
                if rec.rejection_reason is not None:
                    reason_counts[rec.rejection_reason] += 1

            ranked = rank(candidates_opp, self.cfg.scoring)
            sviews = self.sentiment_views(py_ts, [c.symbol for c in ranked])
            decisions = select_and_size(
                ranked,
                state,
                sviews,
                self.assets,
                self.cfg.portfolio,
                self.cfg.risk,
                self.cfg.sentiment,
                self.cfg.costs,
                self.cfg.scoring.min_ev_net_r,
            )
            decision_recs = to_records(
                decisions,
                py_ts,
                self.run_id,
                state,
                sviews,
                self.cfg.audit.features_json_policy,
            )
            records.extend(decision_recs)
            n_considered += len(decision_recs)
            for rec in decision_recs:
                if rec.rejection_reason is not None:
                    reason_counts[rec.rejection_reason] += 1

            for decision in (d for d in decisions if d.accepted):
                entry, stop, target = build_bracket(decision, py_ts, self.run_id)
                fill, corr = _submit_bracket(
                    self.broker, entry, stop, target, decision
                )
                if fill is None:
                    records.append(data_gap_record(decision, py_ts, self.run_id))
                    n_considered += 1
                    reason_counts[RejectionReason.DATA_GAP] += 1
                    continue
                state = ledger_mod.apply_entry(state, decision, fill, corr)
                opp = decision.opportunity
                self._entry_meta[fill.symbol] = (opp.regime, opp.vol_bucket, opp.ev_net_r)

            equity_points.append((py_ts, float(state.equity_usd)))
            self.sink.write(records)
            records.clear()

        self.sink.flush()
        close_sink = getattr(self.sink, "close", None)
        if callable(close_sink):
            close_sink()

        equity_curve = _equity_series(equity_points)
        benchmark_curve = _benchmark_curve(benchmark, equity_curve)
        metrics = compute_headline_metrics(tuple(trades), equity_curve)
        return BacktestResult(
            run_id=self.run_id,
            config_hash=digest,
            started_at=started_at,
            period_start=self.cfg.period.start,
            period_end=self.cfg.period.end,
            mode=self.cfg.run.mode if isinstance(self.cfg.run.mode, RunMode) else RunMode.BACKTEST,
            trades=tuple(trades),
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            metrics=metrics,
            n_decisions_considered=n_considered,
            n_rejections_by_reason=dict(reason_counts),
        )

    def evaluate(
        self,
        ts: datetime,
        features: FeaturePanel,
        snapshot: UniverseSnapshot | None,
        state: PortfolioState,
    ) -> tuple[tuple[Opportunity, ...], list[DecisionRecord]]:
        """Gates → detect → edge → cost → Opportunity. Records every rejection."""
        recs: list[DecisionRecord] = []
        opps: list[Opportunity] = []
        feat_block = features.rows_at(ts)
        feat_by_symbol = _frame_by_symbol(feat_block)
        if snapshot is None:
            recs.append(_blank_record(self.run_id, ts, RejectionReason.NOT_IN_UNIVERSE))
            return (), recs

        symbols = tuple(sorted(snapshot.entries))
        equity = float(state.equity_usd)
        risk_capital = equity * self.cfg.portfolio.risk_fraction_per_trade
        for symbol in symbols:
            entry = snapshot.entries.get(symbol)
            feat_rec = feat_by_symbol.get(symbol)
            feature_row: FeatureRow | None = None
            if feat_rec is not None and (entry is None or entry.eligible):
                feature_row = feature_row_from_record(feat_rec)
            asset = self.assets.get(symbol)
            is_etf = asset.is_etf if asset is not None else False
            asset_id = (
                asset.asset_id
                if asset is not None
                else self._asset_id_by_symbol.get(symbol, symbol)
            )
            events = self.earnings_by_asset.get(asset_id, ())
            close_raw = self._last_close_raw.get(symbol)
            for strategy in self.strategies:
                reason, setup, opp = self._evaluate_one(
                    ts=ts,
                    symbol=symbol,
                    strategy=strategy,
                    feature_row=feature_row,
                    universe_entry=entry,
                    is_etf=is_etf,
                    events=events,
                    asset=asset,
                    risk_capital=risk_capital,
                    close_raw=close_raw,
                )
                if opp is not None:
                    opps.append(opp)
                    continue
                recs.append(
                    _rejection_record(
                        run_id=self.run_id,
                        ts=ts,
                        symbol=symbol,
                        strategy_id=strategy.strategy_id,
                        reason=reason if reason is not None else RejectionReason.NO_SETUP,
                        feature_row=feature_row,
                        setup=setup,
                        policy=self.cfg.audit.features_json_policy,
                    )
                )
        return tuple(opps), recs

    def _evaluate_one(
        self,
        *,
        ts: datetime,
        symbol: str,
        strategy: Strategy,
        feature_row: FeatureRow | None,
        universe_entry: UniverseEntry | None,
        is_etf: bool,
        events: Sequence[EarningsEvent],
        asset: Asset | None,
        risk_capital: float,
        close_raw: float | None,
    ) -> tuple[RejectionReason | None, Setup | None, Opportunity | None]:
        max_hold = int(getattr(strategy, "max_hold_bars", 21))
        gate = evaluate_gates(
            feature_row,
            universe_entry,
            self.cfg.gates,
            is_etf=is_etf,
            earnings=events,
            max_hold_bars=max_hold,
            calendar=self.calendar,
            min_bars_since_gap=self.cfg.universe.min_bars_since_gap,
            bar_age_bars=0 if feature_row is not None else 10_000,
        )
        if gate is not None:
            return gate, None, None
        assert feature_row is not None
        if not math.isfinite(feature_row.atr_14) or feature_row.atr_14 <= 0:
            return RejectionReason.INSUFFICIENT_HISTORY, None, None
        if feature_row.bars_available < strategy.required_warmup_bars:
            return RejectionReason.INSUFFICIENT_HISTORY, None, None
        if self.cfg.gates.require_regime_allowed:
            allowed_market = getattr(strategy, "allowed_market_regimes", None)
            if (
                allowed_market is not None
                and feature_row.market_regime not in allowed_market
            ):
                return RejectionReason.MARKET_REGIME_BLOCKED, None, None
            allowed = strategy.allowed_regimes
            if allowed and feature_row.regime not in allowed:
                return RejectionReason.REGIME_BLOCKED, None, None
        setup = strategy.detect(feature_row)
        if setup is None:
            return RejectionReason.NO_SETUP, None, None
        if (
            not self.cfg.gates.skip_hard_to_borrow
            and setup.direction is Direction.SHORT
        ):
            htb = evaluate_gates(
                feature_row,
                universe_entry,
                self.cfg.gates,
                is_etf=is_etf,
                earnings=events,
                max_hold_bars=max_hold,
                calendar=self.calendar,
                min_bars_since_gap=self.cfg.universe.min_bars_since_gap,
                direction=setup.direction,
                borrow_bps_per_year=self.cfg.costs.borrow_bps_per_year_default,
                hard_to_borrow_max_bps_per_year=(
                    self.cfg.costs.hard_to_borrow_max_bps_per_year
                ),
            )
            if htb is RejectionReason.HARD_TO_BORROW:
                return htb, setup, None
        key = BinKey(setup.strategy_id, setup.direction, feature_row.vol_bucket)
        stats = self.edge_table.lookup(key, ts)
        if stats is None or not stats.is_usable:
            return RejectionReason.INSUFFICIENT_BIN_SAMPLES, setup, None
        if asset is None or universe_entry is None or close_raw is None or close_raw <= 0:
            return RejectionReason.COST_UNAVAILABLE, setup, None
        if risk_capital <= 0 or setup.risk_per_unit <= 0:
            return RejectionReason.COST_UNAVAILABLE, setup, None
        notional = risk_capital / setup.risk_per_unit * setup.reference_price
        cost = estimate_cost(
            setup,
            universe_entry,
            notional,
            risk_capital,
            max(stats.mean_bars_held, 1.0),
            self.cfg.costs,
            atr_pct=feature_row.atr_pct,
            price_raw=close_raw,
        )
        if not cost.is_valid:
            return RejectionReason.COST_UNAVAILABLE, setup, None
        opp = build_opportunity(setup, feature_row, stats, cost, asset, universe_entry)
        if opp.ev_net_r < self.cfg.scoring.min_ev_net_r:
            return RejectionReason.BELOW_EV_THRESHOLD, setup, None
        self._used_bins.append(stats)
        return None, setup, opp

    def sentiment_views(
        self, ts: datetime, symbols: Sequence[str]
    ) -> dict[str, SentimentView]:
        if not self.cfg.sentiment.enabled:
            return {}
        views: dict[str, SentimentView] = {}
        for symbol in symbols:
            views[symbol] = build_view(self._observations, symbol, ts, self.cfg.sentiment)
        return views

    def apply_time_stops(
        self, state: PortfolioState, panel: MarketPanel, ts: datetime
    ) -> tuple[PortfolioState, tuple[ClosedTrade, ...]]:
        del panel
        closed: list[ClosedTrade] = []
        current = state
        for symbol in sorted(current.positions):
            pos = current.positions[symbol]
            if pos.bars_held < pos.max_hold_bars:
                continue
            kwargs = self._one_exit_kwargs(symbol)
            fill = self.broker.close_position(symbol, "TIME")
            if fill is None:
                continue
            mark_px = self._last_close.get(symbol)
            current, trade = ledger_mod.apply_exit(
                current,
                pos,
                fill,
                ts,
                last_mark=mark_px,
                **kwargs,
            )
            closed.append(trade)
            self._entry_meta.pop(symbol, None)
        return current, tuple(closed)

    def apply_delistings(
        self, state: PortfolioState, panel: MarketPanel, ts: datetime
    ) -> tuple[PortfolioState, tuple[ClosedTrade, ...]]:
        closed: list[ClosedTrade] = []
        current = state
        bars_today: set[str] = set()
        block = panel.rows_at(ts)
        if not block.empty:
            bars_today = {str(s) for s in block["symbol"].tolist()}
        grace = self.cfg.universe.delisting_grace_bars
        i_ts = self._session_pos.get(ts)
        for symbol in sorted(current.positions):
            if not self._is_delisted(symbol, ts, bars_today, i_ts, grace):
                continue
            pos = current.positions[symbol]
            kwargs = self._one_exit_kwargs(symbol)
            fill = self.broker.close_position(symbol, "DELISTED")
            if fill is None:
                price = self._last_close.get(symbol)
                if price is None:
                    continue
                fill = Fill(
                    client_order_id=pos.client_order_id,
                    symbol=symbol,
                    side=(
                        Direction.SHORT if pos.direction is Direction.LONG else Direction.LONG
                    ),
                    qty=pos.qty,
                    price=price,
                    fee_usd=Decimal("0"),
                    ts=ts,
                    is_maker=False,
                    exit_reason="DELISTED",
                )
            mark_px = self._last_close.get(symbol)
            current, trade = ledger_mod.apply_exit(
                current,
                pos,
                fill,
                ts,
                last_mark=mark_px,
                **kwargs,
            )
            closed.append(trade)
            self._entry_meta.pop(symbol, None)
        return current, tuple(closed)

    def _is_delisted(
        self,
        symbol: str,
        ts: datetime,
        bars_today: set[str],
        i_ts: int | None,
        grace: int,
    ) -> bool:
        asset = self.assets.get(symbol)
        if asset is not None and asset.delisted_at is not None and asset.delisted_at <= ts:
            return True
        if symbol in bars_today:
            return False
        last = self._last_ts_by_symbol.get(symbol)
        if last is None or i_ts is None:
            return False
        i_last = self._session_pos.get(last)
        if i_last is None:
            return False
        return (i_ts - i_last) > grace

    def used_bins_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str, str, datetime]] = set()
        for stats in self._used_bins:
            if not isinstance(stats, BinStats):
                continue
            key = (*stats.key.as_tuple(), stats.as_of)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "strategy_id": stats.key.strategy_id,
                    "direction": stats.key.direction.value,
                    "vol_bucket": stats.key.vol_bucket.value,
                    "as_of": stats.as_of.isoformat(),
                    "n": stats.n,
                    "mean_r": stats.mean_r,
                    "std_r": stats.std_r,
                    "ev_r_lcb": stats.ev_r_lcb,
                    "mean_bars_held": stats.mean_bars_held,
                    "win_rate": stats.win_rate,
                }
            )
        return pd.DataFrame(rows)

    def _update_close_cache(self, bar_block: pd.DataFrame) -> None:
        if bar_block.empty:
            return
        for rec in bar_block.to_dict("records"):
            symbol = str(rec["symbol"])
            self._last_close[symbol] = to_decimal(float(rec["close"]))
            self._last_close_raw[symbol] = float(rec["close_raw"])
            ts_raw = rec.get("ts")
            if ts_raw is not None:
                self._last_ts_by_symbol[symbol] = _as_utc(ts_raw)

    def _marks_for(self, state: PortfolioState, ts: datetime) -> dict[str, Decimal]:
        del ts
        out: dict[str, Decimal] = {}
        for symbol in state.positions:
            px = self._last_close.get(symbol)
            if px is not None:
                out[symbol] = px
        return out

    def _exit_kwargs_from_brackets(self, state: PortfolioState) -> dict[str, Any]:
        evs: dict[str, float] = {}
        maes: dict[str, float] = {}
        mfes: dict[str, float] = {}
        regimes: dict[str, Regime] = {}
        buckets: dict[str, VolBucket] = {}
        bracket_fn = getattr(self.broker, "bracket", None)
        for symbol in state.positions:
            meta = self._entry_meta.get(symbol)
            if meta is not None:
                regimes[symbol] = meta[0]
                buckets[symbol] = meta[1]
                evs[symbol] = meta[2]
            if callable(bracket_fn):
                br = bracket_fn(symbol)
                if br is not None:
                    maes[symbol] = float(br.mae_r)
                    mfes[symbol] = float(br.mfe_r)
                    evs[symbol] = float(br.ev_net_r_at_entry)
        return {
            "ev_net_r_at_entry": evs,
            "mae_r": maes,
            "mfe_r": mfes,
            "regime": regimes,
            "vol_bucket": buckets,
        }

    def _one_exit_kwargs(self, symbol: str) -> dict[str, Any]:
        meta = self._entry_meta.get(symbol)
        mae = 0.0
        mfe = 0.0
        ev = float("nan")
        regime = Regime.UNKNOWN
        bucket = VolBucket.UNKNOWN
        bracket_fn = getattr(self.broker, "bracket", None)
        if callable(bracket_fn):
            br = bracket_fn(symbol)
            if br is not None:
                mae = float(br.mae_r)
                mfe = float(br.mfe_r)
                ev = float(br.ev_net_r_at_entry)
        if meta is not None:
            regime, bucket, ev_meta = meta
            if not math.isfinite(ev):
                ev = ev_meta
        return {
            "ev_net_r_at_entry": ev,
            "mae_r": mae,
            "mfe_r": mfe,
            "regime": regime,
            "vol_bucket": bucket,
        }

    def _forget_closed(self, closed: Sequence[ClosedTrade]) -> None:
        for trade in closed:
            self._entry_meta.pop(trade.symbol, None)


def build_bracket(
    decision: TradeDecision, ts: datetime, run_id: str
) -> tuple[OrderIntent, OrderIntent, OrderIntent]:
    """Three OrderIntents. Engine prefixes run_id[:8] on client_order_id."""
    if decision.qty is None:
        raise ScoutConfigError("build_bracket requires an accepted decision with qty")
    opp = decision.opportunity
    setup = opp.setup
    prefix = run_id[:8]
    stamp = int(ts.timestamp())
    corr = f"{prefix}-{setup.symbol}-{stamp}"
    qty = decision.qty
    side = opp.direction
    close_side = Direction.SHORT if side is Direction.LONG else Direction.LONG
    stop_px = to_decimal(setup.stop_price)
    target_px = (
        None if setup.target_price is None else to_decimal(setup.target_price)
    )
    entry = OrderIntent(
        client_order_id=f"{prefix}-{setup.symbol}-{stamp}-e",
        symbol=setup.symbol,
        side=side,
        order_type=OrderType.MARKET,
        qty=qty,
        limit_price=None,
        stop_price=None,
        reduce_only=False,
        created_ts=ts,
        correlation_id=corr,
    )
    stop = OrderIntent(
        client_order_id=f"{prefix}-{setup.symbol}-{stamp}-s",
        symbol=setup.symbol,
        side=close_side,
        order_type=OrderType.STOP_MARKET,
        qty=qty,
        limit_price=None,
        stop_price=stop_px,
        reduce_only=True,
        created_ts=ts,
        correlation_id=corr,
    )
    target = OrderIntent(
        client_order_id=f"{prefix}-{setup.symbol}-{stamp}-t",
        symbol=setup.symbol,
        side=close_side,
        order_type=OrderType.TAKE_PROFIT_MARKET,
        qty=qty,
        limit_price=target_px,
        stop_price=None,
        reduce_only=True,
        created_ts=ts,
        correlation_id=corr,
    )
    return entry, stop, target


def to_records(
    decisions: Sequence[TradeDecision],
    ts: datetime,
    run_id: str,
    state: PortfolioState,
    views: Mapping[str, SentimentView],
    policy: FeaturesJsonPolicy,
) -> list[DecisionRecord]:
    recs: list[DecisionRecord] = []
    heat = state.open_risk_pct
    for decision in decisions:
        opp = decision.opportunity
        view = views.get(opp.symbol)
        recs.append(
            DecisionRecord(
                run_id=run_id,
                decision_ts=ts,
                symbol=opp.symbol,
                strategy_id=opp.strategy_id,
                stage="PORTFOLIO" if not decision.accepted else "SIZE",
                accepted=decision.accepted,
                rejection_reason=decision.rejection_reason,
                direction=opp.direction,
                regime=opp.regime,
                vol_bucket=opp.vol_bucket,
                reference_price=opp.setup.reference_price,
                stop_price=opp.setup.stop_price,
                target_price=opp.setup.target_price,
                reward_risk_ratio=(
                    None
                    if opp.setup.target_price is None
                    else opp.setup.reward_risk_ratio
                ),
                bin_key=_bin_key_str(opp.bin_key),
                bin_n=opp.bin_n,
                ev_r_point=opp.ev_r_point,
                ev_r_lcb=opp.ev_r_lcb,
                cost_r=opp.cost_r,
                ev_net_r=opp.ev_net_r,
                ev_per_bar_r=opp.ev_per_bar_r,
                rank=decision.rank,
                sentiment_score=None if view is None else view.score,
                sentiment_confidence=None if view is None else view.confidence,
                sentiment_multiplier=decision.sentiment_multiplier,
                portfolio_heat_pct=heat,
                cluster=opp.cluster,
                size_multiplier=decision.size_multiplier,
                qty=None if decision.qty is None else float(decision.qty),
                notional_usd=(
                    None
                    if decision.intended_notional_usd is None
                    else float(decision.intended_notional_usd)
                ),
                adv_usd_30=opp.adv_usd_30,
                spread_bps_est=opp.spread_bps_est,
                features_json=None,
            )
        )
    return recs


def data_gap_record(decision: TradeDecision, ts: datetime, run_id: str) -> DecisionRecord:
    opp = decision.opportunity
    return DecisionRecord(
        run_id=run_id,
        decision_ts=ts,
        symbol=opp.symbol,
        strategy_id=opp.strategy_id,
        stage="SIZE",
        accepted=False,
        rejection_reason=RejectionReason.DATA_GAP,
        direction=opp.direction,
        regime=opp.regime,
        vol_bucket=opp.vol_bucket,
        reference_price=opp.setup.reference_price,
        stop_price=opp.setup.stop_price,
        target_price=opp.setup.target_price,
        reward_risk_ratio=(
            None if opp.setup.target_price is None else opp.setup.reward_risk_ratio
        ),
        bin_key=_bin_key_str(opp.bin_key),
        bin_n=opp.bin_n,
        ev_r_point=opp.ev_r_point,
        ev_r_lcb=opp.ev_r_lcb,
        cost_r=opp.cost_r,
        ev_net_r=opp.ev_net_r,
        ev_per_bar_r=opp.ev_per_bar_r,
        rank=decision.rank,
        sentiment_score=None,
        sentiment_confidence=None,
        sentiment_multiplier=decision.sentiment_multiplier,
        portfolio_heat_pct=None,
        cluster=opp.cluster,
        size_multiplier=decision.size_multiplier,
        qty=None if decision.qty is None else float(decision.qty),
        notional_usd=(
            None
            if decision.intended_notional_usd is None
            else float(decision.intended_notional_usd)
        ),
        adv_usd_30=opp.adv_usd_30,
        spread_bps_est=opp.spread_bps_est,
        features_json=None,
    )


def _submit_bracket(
    broker: Broker,
    entry: OrderIntent,
    stop: OrderIntent,
    target: OrderIntent,
    decision: TradeDecision,
) -> tuple[Fill | None, str]:
    extra: Any = broker
    try:
        result = extra.submit_bracket(
            entry, stop, target, cost=decision.final_cost, decision=decision
        )
    except TypeError:
        result = broker.submit_bracket(entry, stop, target)
    return cast(tuple[Fill | None, str], result)


def _broker_mark(
    broker: Broker,
    state: PortfolioState,
    panel: MarketPanel,
    ts: datetime,
    marks: Mapping[str, Decimal],
) -> PortfolioState:
    mark_fn = getattr(broker, "mark", None)
    if not callable(mark_fn):
        return ledger_mod.mark(state, panel, ts, marks=dict(marks))
    try:
        out = mark_fn(state, panel, ts, closes=dict(marks))
    except TypeError:
        out = mark_fn(state, panel, ts)
    return cast(PortfolioState, out)


def _load_benchmark(cfg: ScoutConfig) -> BenchmarkPanel:
    path = Path(cfg.data.processed_dir).parent / "reference" / "benchmark_1d.parquet"
    if not path.is_file():
        raise ScoutDataError(
            f"benchmark not found: {path}; expected data/reference/benchmark_1d.parquet"
        )
    frame = read_parquet(path)
    missing = [c for c in BENCHMARK_COLUMNS if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"benchmark missing columns: {missing}")
    return BenchmarkPanel(frame.loc[:, list(BENCHMARK_COLUMNS)])


def _frame_by_symbol(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    out: dict[str, dict[str, object]] = {}
    for rec in frame.to_dict("records"):
        out[str(rec["symbol"])] = rec
    return out


def _as_utc(value: object) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = ts.to_pydatetime()
    if not isinstance(out, datetime):
        raise TypeError(f"ts is not a datetime: {type(out)!r}")
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out


def _equity_series(points: Sequence[tuple[datetime, float]]) -> pd.Series:
    if not points:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex([p[0] for p in points], tz="UTC")
    return pd.Series([p[1] for p in points], index=idx, dtype="float64")


def _benchmark_curve(benchmark: BenchmarkPanel, equity: pd.Series) -> pd.Series:
    if equity.empty or benchmark.frame.empty:
        return pd.Series(dtype="float64")
    close = benchmark.frame.set_index("ts")["close"].astype("float64")
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    aligned = close.reindex(equity.index, method="ffill")
    first = aligned.dropna()
    if first.empty:
        return pd.Series(dtype="float64")
    base = float(first.iloc[0])
    if base == 0.0 or not math.isfinite(base):
        return pd.Series(dtype="float64")
    return aligned / base


def _bin_key_str(key: BinKey) -> str:
    return f"{key.strategy_id}|{key.direction.value}|{key.vol_bucket.value}"


def _features_json(row: FeatureRow | None) -> str | None:
    if row is None:
        return None
    payload = row.to_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _include_features(policy: FeaturesJsonPolicy, *, accepted: bool, ranked: bool) -> bool:
    if policy is FeaturesJsonPolicy.NONE:
        return False
    if policy is FeaturesJsonPolicy.ALL:
        return True
    if policy is FeaturesJsonPolicy.ACCEPTED:
        return accepted
    return ranked or accepted


def _rejection_record(
    *,
    run_id: str,
    ts: datetime,
    symbol: str,
    strategy_id: str | None,
    reason: RejectionReason,
    feature_row: FeatureRow | None,
    setup: Setup | None,
    policy: FeaturesJsonPolicy,
) -> DecisionRecord:
    stage = _stage_for_reason(reason)
    ranked = stage in {"EDGE", "RANK"}
    include = _include_features(policy, accepted=False, ranked=ranked)
    return DecisionRecord(
        run_id=run_id,
        decision_ts=ts,
        symbol=symbol,
        strategy_id=strategy_id,
        stage=stage,
        accepted=False,
        rejection_reason=reason,
        direction=None if setup is None else setup.direction,
        regime=None if feature_row is None else feature_row.regime,
        vol_bucket=None if feature_row is None else feature_row.vol_bucket,
        reference_price=None if setup is None else setup.reference_price,
        stop_price=None if setup is None else setup.stop_price,
        target_price=None if setup is None else setup.target_price,
        reward_risk_ratio=(
            None
            if setup is None or setup.target_price is None
            else setup.reward_risk_ratio
        ),
        bin_key=None,
        bin_n=None,
        ev_r_point=None,
        ev_r_lcb=None,
        cost_r=None,
        ev_net_r=None,
        ev_per_bar_r=None,
        rank=None,
        sentiment_score=None,
        sentiment_confidence=None,
        sentiment_multiplier=None,
        portfolio_heat_pct=None,
        cluster=None,
        size_multiplier=None,
        qty=None,
        notional_usd=None,
        adv_usd_30=None if feature_row is None else None,
        spread_bps_est=None,
        features_json=_features_json(feature_row) if include else None,
    )


def _blank_record(run_id: str, ts: datetime, reason: RejectionReason) -> DecisionRecord:
    return DecisionRecord(
        run_id=run_id,
        decision_ts=ts,
        symbol="_UNIVERSE_",
        strategy_id=None,
        stage="GATE",
        accepted=False,
        rejection_reason=reason,
        direction=None,
        regime=None,
        vol_bucket=None,
        reference_price=None,
        stop_price=None,
        target_price=None,
        reward_risk_ratio=None,
        bin_key=None,
        bin_n=None,
        ev_r_point=None,
        ev_r_lcb=None,
        cost_r=None,
        ev_net_r=None,
        ev_per_bar_r=None,
        rank=None,
        sentiment_score=None,
        sentiment_confidence=None,
        sentiment_multiplier=None,
        portfolio_heat_pct=None,
        cluster=None,
        size_multiplier=None,
        qty=None,
        notional_usd=None,
        adv_usd_30=None,
        spread_bps_est=None,
        features_json=None,
    )


def _stage_for_reason(reason: RejectionReason) -> str:
    if reason in {
        RejectionReason.INSUFFICIENT_BIN_SAMPLES,
        RejectionReason.COST_UNAVAILABLE,
    }:
        return "EDGE"
    if reason is RejectionReason.BELOW_EV_THRESHOLD:
        return "RANK"
    if reason is RejectionReason.NO_SETUP:
        return "STRATEGY"
    if reason in {
        RejectionReason.ALREADY_IN_POSITION,
        RejectionReason.PORTFOLIO_HEAT_CAP,
        RejectionReason.CLUSTER_CAP,
        RejectionReason.BETA_CAP,
        RejectionReason.GROSS_EXPOSURE_CAP,
        RejectionReason.MAX_POSITIONS,
        RejectionReason.BELOW_TOP_N,
        RejectionReason.SENTIMENT_VETO,
        RejectionReason.KILL_SWITCH,
        RejectionReason.CIRCUIT_BREAKER,
    }:
        return "PORTFOLIO"
    if reason is RejectionReason.SIZE_BELOW_MIN_NOTIONAL:
        return "SIZE"
    return "GATE"
