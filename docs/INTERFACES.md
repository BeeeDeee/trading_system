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

RegimeDetector
  detect(snapshot, features) -> RegimeState

Strategy
  evaluate(snapshot, features, regime, portfolio) -> TradingSignal[]
```

`MarketDataPort` is implemented by market-data adapters. Normalization owns
conversion to canonical models; feature generation begins only after
normalization.

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

`OrderPlanner` may create intents only for `APPROVE` or `MODIFY` decisions. It
must reject or return no intents for `REJECT` and `HALT`.

`TradingEngine` accepts only risk-approved `OrderIntent` objects and owns local
order lifecycle state around `ExecutionPort` calls. `ExecutionPort` is the
provider-facing boundary and returns provider-neutral acknowledgements.

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
```

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
5. Missing or stale required inputs produce no new trading order.
6. Exchange-specific models stop at infrastructure adapter boundaries.
