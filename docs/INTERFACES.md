# Core Interfaces

> Stable contracts between the runtime, domain modules, adapters, and storage.

## 1. Interface Principles

- Interfaces use platform-level domain models.
- Interfaces do not expose exchange client types.
- Implementations are synchronous or asynchronous according to the port, not
  according to a provider library.
- A module returns an explicit result; it does not mutate another module's
  state directly.
- Safety-critical failures are represented as typed failures or rejected
  decisions, not silently ignored.

## 2. Data and Analysis Ports

```text
MarketDataPort
  get_snapshot(instrument, timestamp) -> RawMarketData

Normalizer
  normalize(raw_data) -> MarketSnapshot

FeatureEngine
  calculate(snapshot, feature_context) -> FeatureSet

Strategy
  evaluate(snapshot, features, portfolio) -> TradingSignal[]
```

`MarketDataPort` is implemented by market-data adapters. Normalization owns
conversion to canonical models; feature generation begins only after
normalization.

Regime classification is part of `FeatureSet` (produced by the Feature
Engine); there is no standalone regime port. Strategies that need regime
information read it from `features`.

Lookahead rule: a `FeatureSet` computed for bar close time `t` may only use
data with timestamp `<= t`, and the resulting decision executes no earlier
than the next bar open (see `ADR/009-cross-cutting-conventions.md`).

## 3. Decision and Execution Ports

```text
PortfolioManager
  aggregate(signals, portfolio_state) -> TargetPosition[]

RiskManager
  evaluate(target, portfolio_state, account_state) -> RiskDecision

OrderPlanner
  plan(target, risk_decision, portfolio_state) -> OrderIntent[]

TradingEngine
  execute(intent) -> ExecutionReport
  cancel(order_id) -> ExecutionReport

ExecutionPort
  submit(intent) -> ExternalOrderAcknowledgement
  cancel(order_id) -> ExternalCancellationAcknowledgement
  get_open_orders() -> ExternalOrder[]
  get_fills(since) -> ExternalFill[]
```

`RiskManager` may reduce or reject a proposed size but never increases it.
Sizing is proposed by `PortfolioManager`.

`OrderPlanner` may create intents only for `APPROVE` or `MODIFY` decisions. It
must reject or return no intents for `REJECT` and `HALT`. It assigns the
client-generated `client_order_id` to every intent.

`TradingEngine` accepts only risk-approved `OrderIntent` objects and owns local
order lifecycle state around `ExecutionPort` calls, including placement and
maintenance of exchange-native protective stop/take-profit orders.
`ExecutionPort` is the provider-facing boundary and returns provider-neutral
acknowledgements.

## 4. Account and Reconciliation Ports

```text
AccountPort
  get_account_snapshot() -> ExternalAccountSnapshot

ReconciliationService
  reconcile(local_state, external_snapshot) -> ReconciliationResult

PortfolioView
  derive(account_state, position_state, order_state) -> PortfolioState
```

`AccountPort` is implemented by the Account / Execution Adapter. Reconciliation
compares external observations with local state; it does not generate strategy
decisions.

## 5. Persistence Ports

```text
OrderRepository
  save(order_state)
  get(order_id) -> OrderState | None

PositionRepository
  save(position_state)
  get_all() -> PositionState[]

TradeRepository
  save(trade)

StrategyRepository
  get_active() -> StrategyConfiguration[]

DecisionRepository
  save(decision_record)
  get_by_correlation(correlation_id) -> DecisionRecord[]
```

`DecisionRepository` persists the decision audit trail: signals, risk
decisions (including rejections), and planned intents, linked by correlation
identifiers.

Repositories are called by application services. Domain modules do not access
PostgreSQL directly.

## 6. Runtime Ports

```text
TradingRuntime
  submit(command: RuntimeCommand) -> CommandStatus

TradingPipeline
  process(snapshot: MarketSnapshot) -> PipelineResult
```

`TradingRuntime` serializes lifecycle, order, and position state transitions.
`TradingPipeline` is a normal Python call chain and does not require an
internal message broker.

## 7. Contract Invariants

1. No `OrderIntent` exists without an `APPROVE` or `MODIFY` `RiskDecision`.
2. No execution call bypasses the Trading Engine.
3. No strategy or API route calls an exchange adapter directly.
4. `PortfolioState` is derived and is not authoritative order state.
5. Missing or stale required inputs produce no new **risk-opening** order.
   Reduce-only exits follow the asymmetric failure policy
   (`ADR/010-asymmetric-failure-policy.md`).
6. Every `OrderIntent` carries a `client_order_id`; resubmission with the
   same key must not create a duplicate order.
7. Risk never increases a proposed position size.
8. Every pipeline decision is persisted via `DecisionRepository`.
9. Exchange-specific models stop at infrastructure adapter boundaries.
