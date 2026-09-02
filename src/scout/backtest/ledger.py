"""Cash and position accounting. All arithmetic is Decimal.

Unrealised in the identity `equity == cash + unrealised` is signed mark-to-market
value (`direction.sign * qty * mark`), not P&L alone. `apply_entry` debits
signed notional from cash (cash equities, not margin), so inventory must sit in
the unrealised term or the identity cannot hold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from scout.domain.enums import Direction, Regime, SetupOutcome, VolBucket
from scout.domain.execution import Fill
from scout.domain.market import MarketPanel
from scout.domain.portfolio import PortfolioState, Position, TradeDecision
from scout.domain.results import ClosedTrade
from scout.utils.decimals import to_decimal
from scout.utils.errors import ScoutDataError, ScoutError, ScoutLookaheadError

BREAKER_SENTINEL = 10_000
_ZERO = Decimal("0")


def initial_state(equity_usd: Decimal | float | str, ts: datetime) -> PortfolioState:
    """Empty book: cash = equity, no positions."""
    equity = to_decimal(equity_usd)
    if equity <= 0:
        raise ScoutError("account blew up")
    return PortfolioState(
        ts=ts,
        equity_usd=equity,
        cash_usd=equity,
        positions=MappingProxyType({}),
        peak_equity_usd=equity,
        realised_pnl_today_usd=_ZERO,
        day_start_equity_usd=equity,
        trades_today=0,
        bars_since_breaker=BREAKER_SENTINEL,
    )


def apply_entry(
    state: PortfolioState,
    decision: TradeDecision,
    fill: Fill,
    corr_id: str,
) -> PortfolioState:
    """Debit signed notional + fee; open a Position.

    `cash -= direction.sign * qty * fill.price + fee`. Longs pay the notional;
    shorts receive it. Stop/target on the Position are re-anchored from the
    fill using the Setup's risk/reward distances (same geometry as labeling).
    `corr_id` links the bracket; Position keys by `client_order_id`.
    """
    del corr_id  # carried on OrderIntent / Fill; Position keys by client_order_id
    if not decision.accepted or decision.qty is None:
        raise ScoutError("apply_entry requires an accepted TradeDecision with qty")
    if fill.symbol in state.positions:
        raise ScoutError(f"already in position: {fill.symbol}")
    if fill.qty <= 0:
        raise ScoutError(f"fill qty must be positive, got {fill.qty}")

    opp = decision.opportunity
    setup = opp.setup
    sign = Decimal(opp.direction.sign)
    notional = fill.qty * fill.price
    cash = state.cash_usd - sign * notional - fill.fee_usd

    risk = to_decimal(setup.risk_per_unit)
    stop = fill.price - sign * risk
    if setup.target_price is None:
        # ADR-019: no take-profit. Store a barrier that poll_fills must ignore.
        target = fill.price
    else:
        target = fill.price + sign * to_decimal(setup.reward_per_unit)

    position = Position(
        symbol=fill.symbol,
        direction=opp.direction,
        qty=fill.qty,
        entry_price=fill.price,
        entry_ts=fill.ts,
        stop_price=stop,
        target_price=target,
        max_hold_bars=setup.max_hold_bars,
        bars_held=0,
        strategy_id=opp.strategy_id,
        cluster=opp.cluster,
        beta_bench_90=opp.beta_bench_90,
        client_order_id=fill.client_order_id,
        realised_fees_usd=fill.fee_usd,
        dividends_usd=_ZERO,
        borrow_usd=_ZERO,
        funding_paid_usd=_ZERO,
    )
    positions = dict(state.positions)
    positions[fill.symbol] = position
    implied = state.equity_usd - state.cash_usd
    equity = cash + implied + sign * notional
    out = replace(
        state,
        cash_usd=cash,
        equity_usd=equity,
        positions=MappingProxyType(positions),
        trades_today=state.trades_today + 1,
    )
    _assert_identity(out, implied_unrealised=implied + sign * notional)
    _assert_qty_positive(out)
    _assert_solvent(out)
    return out


def apply_exit(
    state: PortfolioState,
    position: Position,
    fill: Fill,
    ts: datetime,
    *,
    last_mark: Decimal | None = None,
    ev_net_r_at_entry: float = float("nan"),
    mae_r: float = 0.0,
    mfe_r: float = 0.0,
    regime: Regime = Regime.UNKNOWN,
    vol_bucket: VolBucket = VolBucket.UNKNOWN,
) -> tuple[PortfolioState, ClosedTrade]:
    """Close `position` at `fill`.

    `last_mark` is the price used in the last `mark()` for this symbol (this
    bar's close). The engine must pass it after marking; defaulting to
    `fill.price` is for unit tests that skip `mark()`.
    `regime` / `vol_bucket` come from the Opportunity at entry (M3.4).
    """
    if position.symbol not in state.positions:
        raise ScoutError(f"no open position for {position.symbol}")
    sign = Decimal(position.direction.sign)
    mark_px = fill.price if last_mark is None else last_mark
    implied = state.equity_usd - state.cash_usd
    this_mtm = sign * position.qty * mark_px
    cash = state.cash_usd + sign * position.qty * fill.price - fill.fee_usd
    remaining = dict(state.positions)
    del remaining[position.symbol]
    equity = cash + (implied - this_mtm)

    fees_total = position.realised_fees_usd + fill.fee_usd
    gross = sign * (fill.price - position.entry_price) * position.qty
    net = gross - fees_total - position.borrow_usd + position.dividends_usd
    risk_at_entry = abs(position.entry_price - position.stop_price) * position.qty
    realised_r = float(net / risk_at_entry) if risk_at_entry != 0 else float("nan")

    trade = ClosedTrade(
        symbol=position.symbol,
        strategy_id=position.strategy_id,
        direction=position.direction,
        entry_ts=position.entry_ts,
        exit_ts=fill.ts,
        entry_price=position.entry_price,
        exit_price=fill.price,
        qty=position.qty,
        bars_held=position.bars_held,
        outcome=_outcome_from_fill(fill),
        gross_pnl_usd=gross,
        fees_usd=fees_total,
        dividends_usd=position.dividends_usd,
        borrow_usd=position.borrow_usd,
        funding_usd=position.funding_paid_usd,
        net_pnl_usd=net,
        realised_r=realised_r,
        mae_r=mae_r,
        mfe_r=mfe_r,
        regime=regime,
        vol_bucket=vol_bucket,
        cluster=position.cluster,
        ev_net_r_at_entry=ev_net_r_at_entry,
    )
    out = replace(
        state,
        ts=ts,
        cash_usd=cash,
        equity_usd=equity,
        positions=MappingProxyType(remaining),
        realised_pnl_today_usd=state.realised_pnl_today_usd + net,
    )
    _assert_identity(out, implied_unrealised=implied - this_mtm)
    _assert_qty_positive(out)
    _assert_solvent(out)
    return out, trade


def apply_exits(
    state: PortfolioState,
    fills: tuple[Fill, ...],
    ts: datetime,
    *,
    last_marks: Mapping[str, Decimal] | None = None,
    ev_net_r_at_entry: Mapping[str, float] | None = None,
    mae_r: Mapping[str, float] | None = None,
    mfe_r: Mapping[str, float] | None = None,
    regime: Mapping[str, Regime] | None = None,
    vol_bucket: Mapping[str, VolBucket] | None = None,
) -> tuple[PortfolioState, tuple[ClosedTrade, ...]]:
    """Apply exit fills in order. Missing symbols are ignored."""
    marks = last_marks or {}
    evs = ev_net_r_at_entry or {}
    maes = mae_r or {}
    mfes = mfe_r or {}
    regimes = regime or {}
    buckets = vol_bucket or {}
    closed: list[ClosedTrade] = []
    current = state
    for fill in fills:
        pos = current.positions.get(fill.symbol)
        if pos is None:
            continue
        current, trade = apply_exit(
            current,
            pos,
            fill,
            ts,
            last_mark=marks.get(fill.symbol),
            ev_net_r_at_entry=evs.get(fill.symbol, float("nan")),
            mae_r=maes.get(fill.symbol, 0.0),
            mfe_r=mfes.get(fill.symbol, 0.0),
            regime=regimes.get(fill.symbol, Regime.UNKNOWN),
            vol_bucket=buckets.get(fill.symbol, VolBucket.UNKNOWN),
        )
        closed.append(trade)
    return current, tuple(closed)


def apply_dividend(
    state: PortfolioState,
    symbol: str,
    cash_amount: Decimal,
    ts: datetime,
) -> PortfolioState:
    """Credit (long) or debit (short) `qty * cash_amount`. Signed on the position."""
    del ts
    pos = state.positions.get(symbol)
    if pos is None:
        return state
    sign = Decimal(pos.direction.sign)
    flow = sign * pos.qty * cash_amount
    updated = replace(pos, dividends_usd=pos.dividends_usd + flow)
    positions = dict(state.positions)
    positions[symbol] = updated
    cash = state.cash_usd + flow
    equity = state.equity_usd + flow
    out = replace(
        state,
        cash_usd=cash,
        equity_usd=equity,
        positions=MappingProxyType(positions),
    )
    _assert_identity(out, implied_unrealised=out.equity_usd - out.cash_usd)
    _assert_solvent(out)
    return out


def accrue_borrow(
    state: PortfolioState,
    symbol: str,
    amount_usd: Decimal,
    ts: datetime,
) -> PortfolioState:
    """Debit borrow on a short. No-op for longs or zero amounts."""
    del ts
    if amount_usd == 0:
        return state
    if amount_usd < 0:
        raise ScoutError(f"borrow accrual must be non-negative, got {amount_usd}")
    pos = state.positions.get(symbol)
    if pos is None or pos.direction is Direction.LONG:
        return state
    updated = replace(pos, borrow_usd=pos.borrow_usd + amount_usd)
    positions = dict(state.positions)
    positions[symbol] = updated
    cash = state.cash_usd - amount_usd
    equity = state.equity_usd - amount_usd
    out = replace(
        state,
        cash_usd=cash,
        equity_usd=equity,
        positions=MappingProxyType(positions),
    )
    _assert_identity(out, implied_unrealised=out.equity_usd - out.cash_usd)
    _assert_solvent(out)
    return out


def mark(state: PortfolioState, panel: MarketPanel, ts: datetime) -> PortfolioState:
    """Recompute unrealised at this bar's close. Update peak, day-start, bars_held."""
    marks = _marks_at(panel, state.positions, ts)
    positions = dict(state.positions)
    held: dict[str, Position] = {}
    for symbol, pos in positions.items():
        held[symbol] = replace(pos, bars_held=pos.bars_held + 1)
    unrealised = _unrealised(held, marks)
    equity = state.cash_usd + unrealised
    peak = max(state.peak_equity_usd, equity)
    date_changed = state.ts.date() != ts.date()
    if date_changed:
        day_start = state.equity_usd
        realised_today = _ZERO
        trades_today = 0
    else:
        day_start = state.day_start_equity_usd
        realised_today = state.realised_pnl_today_usd
        trades_today = state.trades_today
    out = replace(
        state,
        ts=ts,
        equity_usd=equity,
        positions=MappingProxyType(held),
        peak_equity_usd=peak,
        realised_pnl_today_usd=realised_today,
        day_start_equity_usd=day_start,
        trades_today=trades_today,
    )
    _assert_identity(out, implied_unrealised=unrealised)
    _assert_qty_positive(out)
    _assert_solvent(out)
    return out


def unrealised_usd(positions: Mapping[str, Position], marks: Mapping[str, Decimal]) -> Decimal:
    """Signed mark-to-market value of open positions."""
    return _unrealised(positions, marks)


def _unrealised(positions: Mapping[str, Position], marks: Mapping[str, Decimal]) -> Decimal:
    total = _ZERO
    for symbol, pos in positions.items():
        mark_px = marks.get(symbol)
        if mark_px is None:
            raise ScoutDataError(f"no mark for open position {symbol}")
        total += Decimal(pos.direction.sign) * pos.qty * mark_px
    return total


def _marks_at(
    panel: MarketPanel,
    positions: Mapping[str, Position],
    ts: datetime,
) -> dict[str, Decimal]:
    if not positions:
        return {}
    latest = panel.latest(ts)
    by_symbol: dict[str, Decimal] = {}
    if not latest.empty:
        for rec in latest.to_dict("records"):
            by_symbol[str(rec["symbol"])] = to_decimal(float(rec["close"]))
    out: dict[str, Decimal] = {}
    for symbol in positions:
        if symbol not in by_symbol:
            raise ScoutDataError(f"no mark for open position {symbol} at {ts.isoformat()}")
        out[symbol] = by_symbol[symbol]
    return out


def _outcome_from_fill(fill: Fill) -> SetupOutcome:
    reason = fill.exit_reason
    if reason == "TARGET":
        return SetupOutcome.TARGET
    if reason == "STOP":
        return SetupOutcome.STOP
    if reason == "TIME":
        return SetupOutcome.TIME
    # DELISTED / KILL_SWITCH / unknown: treat as a forced stop for the label.
    return SetupOutcome.STOP


def _assert_identity(state: PortfolioState, *, implied_unrealised: Decimal) -> None:
    if state.equity_usd != state.cash_usd + implied_unrealised:
        raise ScoutLookaheadError(
            f"equity identity failed: equity={state.equity_usd} "
            f"cash={state.cash_usd} unrealised={implied_unrealised}"
        )


def _assert_qty_positive(state: PortfolioState) -> None:
    for pos in state.positions.values():
        if pos.qty <= 0:
            raise ScoutError(f"position qty must be positive, got {pos.qty} for {pos.symbol}")


def _assert_solvent(state: PortfolioState) -> None:
    if state.equity_usd <= 0:
        raise ScoutError("account blew up")
