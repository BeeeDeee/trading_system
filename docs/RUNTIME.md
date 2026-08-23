# Runtime Model

> Lifecycle, concurrency, and command semantics of the trading runtime.

## 1. Process Model

The initial platform runs as one Python process:

```text
One Python process
│
├── FastAPI async server
│
└── TradingRuntime
     ├── lifecycle / state machine
     ├── market-data tasks
     ├── runtime command handling
     └── serialized trading pipeline
```

The runtime uses asyncio tasks for I/O, websocket connections, reconnects,
and background work. Trading logic does not use threads initially. CPU-bound
work (feature computation, ML inference) must not block the event loop; it is
offloaded to an executor when it becomes measurable.

## 1a. Decision Cadence

The runtime is bar-driven (`ADR/008-bar-driven-runtime.md`). The decision
pipeline runs once per completed bar of the configured timeframe (initially
1h). Between bar closes the runtime only:

- ingests market data,
- processes asynchronous execution events (fills, rejections, cancellations),
- maintains exchange-native protective orders,
- serves control-plane commands.

Sub-bar protection is the job of exchange-native stop/take-profit orders,
not the pipeline.

## 2. Serialized State Transitions

Trading state transitions are serialized. No two coroutines may concurrently
mutate local order, position, or runtime lifecycle state. The first version
uses direct Python coordination inside `TradingRuntime`; it does not introduce
an actor framework or an internal event bus.

## 3. Pipeline

```text
snapshot
 → features (incl. regime)
 → strategies
 → portfolio
 → risk
 → order planning
 → execution
```

The pipeline is a normal Python call chain coordinated by the runtime. Every
stage result is persisted to the decision audit trail with correlation
identifiers.

## 4. Runtime Commands

FastAPI submits a `RuntimeCommand`:

```text
command_id
type
timestamp
requested_by
payload
```

The runtime reports one of:

```text
accepted
rejected
completed
failed
```

### PAUSE

Do not accept new trading decisions. Existing orders may continue through
their normal lifecycle.

### STOP

Gracefully stop the trading runtime and generate no new orders. Existing
positions are not automatically closed.

### HALT

Enter an emergency safety state and prohibit new risk-opening orders.
Reduce-only exits remain permitted, and exchange-native protective stop
orders remain active. Cancellation of open risk-opening orders may follow
according to configuration; protective orders are not cancelled by HALT.

### RESUME

Leave a paused or stopped state only when the runtime is healthy and the
relevant safety conditions are satisfied.

### RECONCILE

Request an explicit comparison of local state with external account state.

## 5. Failure Behavior

The failure policy is asymmetric: opening risk fails closed; reducing risk
requires the minimum possible dependencies. See `ARCHITECTURE.md` and
`ADR/010-asymmetric-failure-policy.md`. n8n is outside the runtime and its
availability does not determine whether trading can continue.
