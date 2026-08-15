# ADR-003: n8n as Orchestration Only

## Status

Accepted

## Decision

n8n is used for orchestration, scheduling, automation, reporting, and notifications.

It does not contain core trading logic.

## Rationale

Trading logic should remain centralized, testable, version-controlled, and independent of workflow tooling.

## Consequence

The platform must remain operational without n8n.

n8n is outside the trading safety boundary. Its unavailability must not stop
the core runtime or permit unsafe decisions.
