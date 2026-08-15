# ADR-006: No Internal Message Broker Initially

## Status

Accepted

## Decision

The initial system will not use an internal message broker.

Modules will communicate through Python interfaces and typed domain models.

The Trading Runtime coordinates the initial pipeline as a normal Python call
chain. Async tasks may perform I/O, but trading state transitions are
serialized without an actor framework or internal event bus.

## Rationale

A message broker would add infrastructure and operational complexity without a current requirement.

## Consequence

The design must keep module boundaries clean enough to allow future event-driven communication if needed.
