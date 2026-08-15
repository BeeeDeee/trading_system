# ADR-005: Risk as Mandatory Execution Gate

## Status

Accepted

## Decision

No order may reach execution without passing Risk Management.

## Rationale

Risk controls must remain independent from strategy decisions.

## Consequence

Strategies and administrative interfaces cannot directly execute orders.

Risk Manager returns a `RiskDecision`; it does not create orders. Order
Planner materializes an approved or modified target into an `OrderIntent`.
