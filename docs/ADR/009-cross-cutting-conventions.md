# ADR-009: Cross-Cutting Conventions

## Status

Accepted

## Decision

The following conventions are fixed platform-wide from the first line of
code. They are cheap to enforce from day one and brutal to retrofit.

### Time

- All timestamps are timezone-aware UTC. `datetime.now(timezone.utc)` only;
  `datetime.now()`, `datetime.utcnow()`, and ad-hoc offset math are
  forbidden.
- In backtests the clock source is the bar timestamp, never wall-clock.
- End-of-bar convention: a decision for bar close time `t` may use data with
  timestamp `<= t` (the completed bar's own data included) and executes no
  earlier than the next bar open. Features must never use data with
  timestamp `> t` (no lookahead, no backfill from the future).

### Money and precision

- `Decimal` or integer minor units at three boundaries: order construction
  (price, quantity), balance and position accounting, and P&L / fee
  accumulation.
- Floats remain the representation for time-series data, indicators, and
  analytics (pandas/numpy).
- Prices and quantities are rounded to the instrument's tick size and step
  size, using the exchange's stated rounding rule, before every order.
- No float equality comparisons in trading logic; use tolerances or
  `Decimal`.

### Identity and idempotency

- Canonical symbol form: `BASE/QUOTE` (e.g. `BTC/USDT`). Exchange-specific
  forms are mapped inside adapters.
- Every order carries a client-generated `client_order_id`; retries reuse
  the same key.
- Account-scoped models and tables carry `account_id` from day one.

### Traceability

- Correlation identifiers flow through the pipeline:
  `signal_id → risk_decision_id → order_intent_id → exchange_order_id`.
- Structured logging with these identifiers exists from the first vertical
  slice; dashboards can come later, traceability cannot.

### Secrets

- API keys and account identifiers come from environment variables
  (gitignored `.env`); a committed `.env.example` lists variable names only.
- Secrets, signed payloads, and full account state are never logged.

## Consequence

These conventions apply to backtest, paper, and live code equally. Reviews
and tests enforce them; violations are defects, not style issues.
