# ADR-010: Asymmetric Failure Policy

## Status

Accepted

## Decision

Failure handling distinguishes between opening risk and reducing risk:

- **Opening or increasing a position fails closed.** All required inputs
  (fresh market data, features, risk management, exchange, PostgreSQL) must
  be healthy, otherwise no new risk-opening order is created.
- **Reducing or closing a position requires the minimum possible
  dependencies.** Reduce-only exits remain permitted in degraded states.
- **Protective stops are exchange-native conditional orders by default.**
  They live on the exchange and remain active even if the platform process,
  the VPS, or the database is down.

`HALT` prohibits new risk-opening orders but does not cancel protective
orders and does not block reduce-only exits.

## Rationale

A symmetric fail-closed policy is dangerous: it can lock the system out of
exiting a losing position precisely when infrastructure is degraded. Safety
means being strict about taking on risk and maximally permissive about
shedding it.

Exchange-native stops are the only exit mechanism that survives a dead
process; runtime-managed stops are an availability liability.

## Consequence

- The Trading Engine owns placement and maintenance of exchange-native
  stop-loss / take-profit orders for every open position.
- Reconciliation verifies protective orders exist after every restart and
  restores them if missing.
- The Risk Manager provides a minimal degraded-mode check path for
  reduce-only intents.
- The failure policy table in `ARCHITECTURE.md` applies to risk-opening
  decisions.
