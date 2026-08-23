# State and Persistence

> Ownership, reconciliation, and persistence boundaries.

## 1. State Flow

```text
Account / Execution Adapter
          │
          ▼
External Account Snapshot
          │
          ▼
Reconciliation
          │
          ├── OrderState
          ├── PositionState
          └── AccountState
                    │
                    ▼
             PortfolioState view
```

The exchange is the external source of truth for account, position, and
exchange-order state. Local state records platform intent and observed
lifecycle. Reconciliation identifies and resolves differences according to
explicit operational policy.

## 2. Ownership

- Trading Engine owns the local order lifecycle.
- Account / Execution Adapter retrieves external balances, positions, orders,
  and fills.
- Reconciliation compares local and external state.
- Portfolio Manager derives `PortfolioState` from current reconciled state.
- No module treats `PortfolioState` as the authoritative order or position
  store.

## 3. Persistence Boundary

```text
Domain
   ↓
Application Services
   ↓
Repositories
   ↓
PostgreSQL
```

Modules do not issue direct SQL or communicate through database tables. Initial
repository boundaries include:

- `OrderRepository`
- `PositionRepository`
- `StrategyRepository`
- `TradeRepository`
- `DecisionRepository`

Repository calls and their transaction boundaries are owned by application
services. PostgreSQL is durable storage, not an internal event transport.

Database schema changes are managed with versioned migrations (Alembic) from
the first table onward.

All account-scoped tables carry an `account_id` column from day one, even
while only one account exists.

## 3a. Decision Audit Trail

Every pipeline run persists its stage results — features summary, signals,
risk decisions (including rejections and halts), and planned intents — via
`DecisionRepository`, linked by correlation identifiers:

```text
signal_id → risk_decision_id → order_intent_id → exchange_order_id
```

This trail is the primary tool for debugging divergence between backtest,
paper, and live behavior. Structured logs carry the same correlation
identifiers.

## 3b. Historical Market Data

Historical market data used for research and backtesting is stored as
Parquet files on disk (gitignored), separate from PostgreSQL. PostgreSQL
holds trading state (orders, positions, trades, decisions, configuration),
not bulk candle history. See `DATA_SOURCES.md`.

## 4. Recovery and Reconciliation

After restart, the runtime restores local state from PostgreSQL and reconciles
it with the external account before allowing new orders. If PostgreSQL or the
exchange is unavailable, no new orders are generated; reconciliation is
retried later.

Order submission is idempotent: every order carries a client-generated
`client_order_id`, and retries after network failures reuse the same key so
an order is never double-submitted. Fills, cancellations, and reconciliation
preserve stable platform identifiers so replays do not silently create
duplicate local orders.

Reconciliation after restart also verifies that every open position has its
exchange-native protective stop orders in place, and restores them if
missing.
