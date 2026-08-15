# ADR-002: FastAPI as Control Plane

## Status

Accepted

## Decision

FastAPI is the external control and observation interface of the platform.

The realtime trading loop does not run inside HTTP request handlers.

## Rationale

The API lifecycle and trading-runtime lifecycle are different concerns.

## Consequence

FastAPI sends commands to and reads state from the Trading Runtime.

The initial implementation uses `RuntimeCommand` objects within the same
Python process; it does not introduce an internal command bus.
