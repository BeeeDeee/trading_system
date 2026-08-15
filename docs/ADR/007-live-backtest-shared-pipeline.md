# ADR-007: Shared Live and Backtest Pipeline

## Status

Accepted

## Decision

Live trading and backtesting should reuse the same trading decision pipeline where practical.

## Rationale

Sharing logic reduces divergence between historical testing and live behavior.

## Consequence

Data acquisition and execution must be abstracted from the core decision logic.

Live and backtest modes share the decision pipeline through the Order Planner;
only data and execution adapters differ. Simulation assumptions remain an
explicit backtest concern.
