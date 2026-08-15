# ADR-004: Explicit Trading Pipeline

## Status

Accepted

## Decision

The trading system follows an explicit domain pipeline:

Data → Features → Regime → Strategy → Portfolio → Risk → Order Planning → Execution.

## Rationale

This provides clear responsibility boundaries and makes the system easier to test and reason about.

## Consequence

Each stage consumes and produces typed domain objects. Order planning creates
an `OrderIntent` only from an approved or modified `RiskDecision`.
