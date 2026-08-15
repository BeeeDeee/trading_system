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
and background work. Trading logic does not use threads initially.

## 2. Serialized State Transitions

Trading state transitions are serialized. No two coroutines may concurrently
mutate local order, position, or runtime lifecycle state. The first version
uses direct Python coordination inside `TradingRuntime`; it does not introduce
an actor framework or an internal event bus.

## 3. Pipeline

```text
snapshot
 → features
 → regime
 → strategies
 → portfolio
 → risk
 → order planning
 → execution
```

The pipeline is a normal Python call chain coordinated by the runtime.

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

Enter an emergency safety state and prohibit new orders. Cancellation of open
orders may follow according to configuration.

### RESUME

Leave a paused or stopped state only when the runtime is healthy and the
relevant safety conditions are satisfied.

### RECONCILE

Request an explicit comparison of local state with external account state.

## 5. Failure Behavior

Required inputs and safety dependencies fail closed. See
`ARCHITECTURE.md` for the failure policy. n8n is outside the runtime and its
availability does not determine whether trading can continue.
