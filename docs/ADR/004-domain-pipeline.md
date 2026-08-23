# ADR-004: Explicit Trading Pipeline

## Status

Accepted (amended 2026-08-22: regime detection demoted from pipeline stage
to feature-layer concern)

## Decision

The trading system follows an explicit domain pipeline:

Data → Features → Strategy → Portfolio → Risk → Order Planning → Execution.

Market regime classification is a derived feature owned by the Feature
Engine, not a standalone pipeline stage. Strategies may consume regime
features or ignore them.

## Rationale

This provides clear responsibility boundaries and makes the system easier to test and reason about.

Regime classifiers are unstable and repaint-prone; making one a mandatory
stage would turn the least reliable component into a system-wide trading
prerequisite. As a feature, its unavailability affects only the strategies
that require it.

## Consequence

Each stage consumes and produces typed domain objects. Order planning creates
an `OrderIntent` only from an approved or modified `RiskDecision`.

There is no `RegimeDetector` port; `RegimeState` is an optional part of
`FeatureSet`.
