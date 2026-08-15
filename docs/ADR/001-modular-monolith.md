# ADR-001: Modular Monolith

## Status

Accepted

## Decision

The initial trading platform will be implemented as a modular monolith.

All core trading modules will run inside one Python process.

## Rationale

- simpler deployment,
- lower resource usage,
- easier debugging,
- easier testing,
- no internal network communication overhead,
- sufficient for the initial scale.

## Consequence

Logical module boundaries must be maintained even though modules share the same process.

Future extraction into separate services remains possible.

The initial process contains an async FastAPI server and `TradingRuntime`.
Trading state transitions remain serialized inside the runtime.
