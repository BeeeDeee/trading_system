"""Simulated broker: next-session MOO entries, conservative stop/target fills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from scout.backtest import ledger as ledger_mod
from scout.config.schema import CostsConfig, PortfolioConfig
from scout.domain.costs import CostEstimate
from scout.domain.enums import ActionType, Direction, RejectionReason
from scout.domain.execution import Fill, OrderIntent
from scout.domain.market import CorporateAction, MarketPanel
from scout.domain.portfolio import PortfolioState, TradeDecision
from scout.portfolio.sizing import to_raw_distance
from scout.utils.decimals import to_decimal
from scout.utils.errors import ScoutLookaheadError

_BPS = Decimal("10000")
_SESSIONS_PER_YEAR = Decimal("252")


@dataclass
class _Bracket:
    symbol: str
    direction: Direction
    qty: Decimal
    entry_fill: Fill
    stop_price: Decimal
    target_price: Decimal | None
    stop_intent: OrderIntent
    target_intent: OrderIntent | None
    cost: CostEstimate | None
    corr_id: str
    mae_r: float = 0.0
    mfe_r: float = 0.0
    ev_net_r_at_entry: float = float("nan")


class SimBroker:
    """Backtest broker. Fills the next session open; never the decision close."""

    def __init__(
        self,
        costs: CostsConfig,
        portfolio: PortfolioConfig,
        *,
        panel: MarketPanel | None = None,
        actions: Sequence[CorporateAction] = (),
        borrow_bps_per_year: float | None = None,
    ) -> None:
        self._costs = costs
        self._portfolio = portfolio
        self._panel = panel
        self._actions = tuple(actions)
        self._borrow_bps_per_year = (
            to_decimal(costs.borrow_bps_per_year_default)
            if borrow_bps_per_year is None
            else to_decimal(borrow_bps_per_year)
        )
        self._brackets: dict[str, _Bracket] = {}
        self._entry_fills: dict[str, tuple[Fill, str]] = {}
        self._state: PortfolioState | None = None
        self._last_ts: datetime | None = None
        self.last_entry_rejection: RejectionReason | None = None
        self._frames_by_symbol: dict[str, pd.DataFrame] = {}
        self._symbol_by_asset_id: dict[str, str] = {}
        self._indexed_panel: MarketPanel | None = None
        if panel is not None:
            self._ensure_symbol_index()

    def attach_panel(self, panel: MarketPanel) -> None:
        self._panel = panel
        self._ensure_symbol_index()

    def attach_actions(self, actions: Sequence[CorporateAction]) -> None:
        self._actions = tuple(actions)

    def submit_bracket(
        self,
        entry: OrderIntent,
        stop: OrderIntent,
        target: OrderIntent,
        *,
        cost: CostEstimate | None = None,
        decision: TradeDecision | None = None,
    ) -> tuple[Fill | None, str]:
        """Fill at the next session open, worsened by modelled slip.

        Production path (M3.4): pass `decision` (re-anchors stop/target on the
        fill, matching labeling) and `cost` (entry slip and fees). `None` fill
        is DATA_GAP — the engine records the rejection; no position is opened.
        """
        self.last_entry_rejection = None
        cached = self._entry_fills.get(entry.client_order_id)
        if cached is not None:
            return cached
        bar = self._next_bar(entry.symbol, entry.created_ts)
        if bar is None:
            self.last_entry_rejection = RejectionReason.DATA_GAP
            return None, entry.correlation_id
        fill_ts = _as_utc(bar["ts"])
        if fill_ts <= entry.created_ts:
            raise ScoutLookaheadError(
                f"entry fill ts {fill_ts.isoformat()} is not after "
                f"decision ts {entry.created_ts.isoformat()}"
            )
        raw_open = to_decimal(float(bar["open"]) * _bar_scale(bar))
        direction = entry.side
        fill_price = _entry_fill_price(raw_open, direction, cost)
        fee = _entry_fee(entry.qty, fill_price, cost, self._costs)
        fill = Fill(
            client_order_id=entry.client_order_id,
            symbol=entry.symbol,
            side=direction,
            qty=entry.qty,
            price=fill_price,
            fee_usd=fee,
            ts=fill_ts,
            is_maker=False,
            exit_reason=None,
        )
        stop_px, target_px = _anchored_barriers(fill_price, direction, stop, target, decision)
        has_target = _has_target(target, decision)
        self._brackets[entry.symbol] = _Bracket(
            symbol=entry.symbol,
            direction=direction,
            qty=entry.qty,
            entry_fill=fill,
            stop_price=stop_px,
            target_price=target_px if has_target else None,
            stop_intent=stop,
            target_intent=target if has_target else None,
            cost=cost,
            corr_id=entry.correlation_id,
            ev_net_r_at_entry=(
                decision.opportunity.ev_net_r if decision is not None else float("nan")
            ),
        )
        result = (fill, entry.correlation_id)
        self._entry_fills[entry.client_order_id] = result
        return result

    def close_position(self, symbol: str, reason: str) -> Fill | None:
        """Reduce-only market exit at the current bar's close."""
        bracket = self._brackets.get(symbol)
        if bracket is None:
            return None
        ts = self._last_ts
        if ts is None:
            return None
        bar = self._bar_at(symbol, ts)
        if bar is None:
            return None
        close_px = to_decimal(float(bar["close"]) * _bar_scale(bar))
        fee = _exit_fee(bracket.qty, close_px, bracket.cost, self._costs)
        fill = self._exit_fill(bracket, close_px, fee, ts, reason)
        del self._brackets[symbol]
        return fill

    def portfolio_state(self, ts: datetime) -> PortfolioState:
        if self._state is None:
            return ledger_mod.initial_state(self._portfolio.initial_equity_usd, ts)
        return self._state

    def poll_fills(self, ts: datetime) -> tuple[Fill, ...]:
        """Intrabar stop/target fills at `ts`, ordered by (ts, symbol)."""
        self._last_ts = ts
        if not self._brackets:
            return ()
        out: list[Fill] = []
        for symbol in sorted(self._brackets):
            fill = self._poll_one(symbol, ts)
            if fill is not None:
                out.append(fill)
        return tuple(out)

    def mark(
        self,
        state: PortfolioState,
        panel: MarketPanel,
        ts: datetime,
        closes: Mapping[str, Decimal] | None = None,
    ) -> PortfolioState:
        """Dividends, borrow, then MTM. Halts on zero equity via the ledger."""
        self._panel = panel
        self._last_ts = ts
        current = state
        session_day = ts.date()
        for action in self._actions:
            if action.ex_date != session_day:
                continue
            symbol = self._symbol_for_asset(action.asset_id, current.positions)
            if symbol is None:
                continue
            if action.action_type is ActionType.SPLIT:
                ratio = to_decimal(action.split_ratio)
                current = ledger_mod.apply_split(current, symbol, ratio, ts)
                self._rescale_bracket(symbol, ratio)
            elif action.action_type is ActionType.DIVIDEND:
                current = ledger_mod.apply_dividend(
                    current, symbol, to_decimal(action.cash_amount), ts
                )
        if closes is None:
            latest = panel.latest(ts)
            close_by_symbol = {}
            if not latest.empty:
                for rec in latest.itertuples(index=False):
                    close_by_symbol[str(rec.symbol)] = to_decimal(
                        float(rec.close) * _bar_scale(rec)
                    )
        else:
            close_by_symbol = dict(closes)
        for symbol, pos in list(current.positions.items()):
            if pos.direction is Direction.LONG:
                continue
            mark_px = close_by_symbol.get(symbol)
            if mark_px is None:
                continue
            daily = (
                pos.qty * mark_px * self._borrow_bps_per_year / _BPS / _SESSIONS_PER_YEAR
            )
            current = ledger_mod.accrue_borrow(current, symbol, daily, ts)
        current = ledger_mod.mark(current, panel, ts, marks=close_by_symbol)
        self._state = current
        return current

    def bracket(self, symbol: str) -> _Bracket | None:
        return self._brackets.get(symbol)

    def _poll_one(self, symbol: str, ts: datetime) -> Fill | None:
        bracket = self._brackets[symbol]
        if ts < bracket.entry_fill.ts:
            return None
        bar = self._bar_at(symbol, ts)
        if bar is None:
            return None
        self._update_excursions(bracket, bar)
        scale = _bar_scale(bar)
        open_px = to_decimal(float(bar["open"]) * scale)
        high = float(bar["high"]) * scale
        low = float(bar["low"]) * scale
        sign = bracket.direction.sign
        hit_stop = (low <= float(bracket.stop_price)) if sign > 0 else (
            high >= float(bracket.stop_price)
        )
        hit_target = False
        if bracket.target_price is not None:
            tgt = float(bracket.target_price)
            hit_target = (high >= tgt) if sign > 0 else (low <= tgt)
        gapped_stop = (
            (open_px <= bracket.stop_price) if sign > 0 else (open_px >= bracket.stop_price)
        )
        if not hit_stop and not gapped_stop and not hit_target:
            return None
        if gapped_stop or hit_stop:
            price = _stop_fill_price(open_px, bracket.stop_price, bracket.direction)
            reason = "STOP"
        else:
            # Targets never fill better than the target, even on a gap through.
            assert bracket.target_price is not None
            price = bracket.target_price
            reason = "TARGET"
        fee = _exit_fee(bracket.qty, price, bracket.cost, self._costs)
        fill = self._exit_fill(bracket, price, fee, ts, reason)
        del self._brackets[symbol]
        return fill

    def _exit_fill(
        self,
        bracket: _Bracket,
        price: Decimal,
        fee: Decimal,
        ts: datetime,
        reason: str,
    ) -> Fill:
        close_side = Direction.SHORT if bracket.direction is Direction.LONG else Direction.LONG
        return Fill(
            client_order_id=bracket.stop_intent.client_order_id
            if reason == "STOP"
            else (
                bracket.target_intent.client_order_id
                if bracket.target_intent is not None and reason == "TARGET"
                else bracket.entry_fill.client_order_id
            ),
            symbol=bracket.symbol,
            side=close_side,
            qty=bracket.qty,
            price=price,
            fee_usd=fee,
            ts=ts,
            is_maker=False,
            exit_reason=reason,
        )

    def _next_bar(self, symbol: str, ts: datetime) -> pd.Series | None:
        frame = self._symbol_frame(symbol)
        if frame is None or frame.empty:
            return None
        future = frame.loc[frame["ts"] > pd.Timestamp(ts)]
        if future.empty:
            return None
        return future.iloc[0]

    def _bar_at(self, symbol: str, ts: datetime) -> pd.Series | None:
        frame = self._symbol_frame(symbol)
        if frame is None or frame.empty:
            return None
        hit = frame.loc[frame["ts"] == pd.Timestamp(ts)]
        if hit.empty:
            return None
        return hit.iloc[0]

    def _ensure_symbol_index(self) -> None:
        if self._panel is self._indexed_panel:
            return
        self._indexed_panel = self._panel
        self._frames_by_symbol = {}
        self._symbol_by_asset_id = {}
        if self._panel is None or self._panel.frame.empty:
            return
        frame = self._panel.frame
        for raw, grp in frame.groupby("symbol", sort=False, observed=True):
            self._frames_by_symbol[str(raw)] = grp.sort_values("ts", kind="mergesort")
        asset_ids = frame["asset_id"].astype(str)
        symbols = frame["symbol"].astype(str)
        for aid, sym in zip(asset_ids.tolist(), symbols.tolist(), strict=False):
            self._symbol_by_asset_id.setdefault(aid, sym)

    def _symbol_frame(self, symbol: str) -> pd.DataFrame | None:
        self._ensure_symbol_index()
        return self._frames_by_symbol.get(symbol)

    def _symbol_for_asset(
        self, asset_id: str, positions: Mapping[str, object]
    ) -> str | None:
        if asset_id in positions:
            return asset_id
        self._ensure_symbol_index()
        symbol = self._symbol_by_asset_id.get(asset_id)
        if symbol is not None and symbol in positions:
            return symbol
        return None

    def _rescale_bracket(self, symbol: str, split_ratio: Decimal) -> None:
        bracket = self._brackets.get(symbol)
        if bracket is None or split_ratio <= 0 or split_ratio == Decimal("1"):
            return
        inv = Decimal("1") / split_ratio
        entry = bracket.entry_fill
        new_fill = replace(
            entry,
            qty=entry.qty * split_ratio,
            price=entry.price * inv,
        )
        target = None if bracket.target_price is None else bracket.target_price * inv
        self._brackets[symbol] = replace(
            bracket,
            qty=bracket.qty * split_ratio,
            entry_fill=new_fill,
            stop_price=bracket.stop_price * inv,
            target_price=target,
        )

    def _update_excursions(self, bracket: _Bracket, bar: pd.Series) -> None:
        risk = float(abs(bracket.entry_fill.price - bracket.stop_price))
        if risk == 0.0:
            return
        entry = float(bracket.entry_fill.price)
        sign = bracket.direction.sign
        scale = _bar_scale(bar)
        high_r = sign * (float(bar["high"]) * scale - entry) / risk
        low_r = sign * (float(bar["low"]) * scale - entry) / risk
        adverse = min(high_r, low_r)
        favorable = max(high_r, low_r)
        bracket.mae_r = min(bracket.mae_r, adverse)
        bracket.mfe_r = max(bracket.mfe_r, favorable)


def _bar_scale(bar: object) -> float:
    """`close_raw / close` for this session. 1.0 when the columns match or close is 0."""
    close = _bar_float(bar, "close")
    raw = _bar_float(bar, "close_raw")
    if close is None or close == 0.0:
        return 1.0
    if raw is None:
        return 1.0
    return raw / close


def _bar_float(bar: object, name: str) -> float | None:
    if isinstance(bar, pd.Series):
        if name not in bar.index:
            return None
        value: object = bar[name]
    else:
        if not hasattr(bar, name):
            return None
        value = getattr(bar, name)
    if value is None:
        return None
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _as_utc(value: object) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = ts.to_pydatetime()
    if not isinstance(out, datetime):
        raise TypeError(f"ts is not a datetime: {type(out)!r}")
    return out


def _entry_fill_price(
    raw_open: Decimal, direction: Direction, cost: CostEstimate | None
) -> Decimal:
    if cost is None:
        return raw_open
    slip = (
        to_decimal(cost.slippage_bps)
        + to_decimal(cost.impact_bps)
        + to_decimal(cost.spread_bps)
    )
    return raw_open * (Decimal(1) + Decimal(direction.sign) * slip / _BPS)


def _stop_fill_price(open_px: Decimal, stop: Decimal, direction: Direction) -> Decimal:
    if direction is Direction.LONG:
        return min(stop, open_px) if open_px < stop else stop
    return max(stop, open_px) if open_px > stop else stop


def _entry_fee(
    qty: Decimal,
    price: Decimal,
    cost: CostEstimate | None,
    cfg: CostsConfig,
) -> Decimal:
    if cost is not None:
        notional = qty * price
        return to_decimal(cost.entry_fee_bps) / _BPS * notional
    return _commission(qty, price, cfg)


def _exit_fee(
    qty: Decimal,
    price: Decimal,
    cost: CostEstimate | None,
    cfg: CostsConfig,
) -> Decimal:
    if cost is not None:
        notional = qty * price
        return to_decimal(cost.exit_fee_bps) / _BPS * notional
    return _commission(qty, price, cfg)


def _commission(qty: Decimal, price: Decimal, cfg: CostsConfig) -> Decimal:
    notional = qty * price
    per_share = to_decimal(cfg.commission_per_share_usd) * qty
    floor = to_decimal(cfg.commission_min_usd)
    cap = to_decimal(cfg.commission_max_pct_of_notional) * notional
    per_side = min(max(per_share, floor), cap)
    return per_side * to_decimal(cfg.cost_multiplier)


def _has_target(target: OrderIntent, decision: TradeDecision | None) -> bool:
    if decision is not None and decision.opportunity.setup.target_price is None:
        return False
    return target.limit_price is not None or target.stop_price is not None


def _anchored_barriers(
    fill_price: Decimal,
    direction: Direction,
    stop: OrderIntent,
    target: OrderIntent,
    decision: TradeDecision | None,
) -> tuple[Decimal, Decimal | None]:
    if decision is not None:
        setup = decision.opportunity.setup
        sign = Decimal(direction.sign)
        if decision.risk_usd is not None and decision.qty is not None and decision.qty > 0:
            risk = decision.risk_usd / decision.qty
        else:
            risk = to_raw_distance(
                setup.reference_price,
                setup.stop_price,
                raw_price=fill_price,
            )
        if setup.target_price is None:
            return fill_price - sign * risk, None
        rr = to_decimal(setup.reward_risk_ratio)
        return fill_price - sign * risk, fill_price + sign * risk * rr
    stop_px = stop.stop_price if stop.stop_price is not None else fill_price
    tgt = target.limit_price if target.limit_price is not None else target.stop_price
    return stop_px, tgt
