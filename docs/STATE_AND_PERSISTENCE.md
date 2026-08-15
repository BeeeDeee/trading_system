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

Repository calls and their transaction boundaries are owned by application
services. PostgreSQL is durable storage, not an internal event transport.

## 4. Recovery and Reconciliation

After restart, the runtime restores local state from PostgreSQL and reconciles
it with the external account before allowing new orders. If PostgreSQL or the
exchange is unavailable, no new orders are generated; reconciliation is
retried later.

Order submission, fills, cancellations, and reconciliation must preserve
stable platform identifiers so retries do not silently create duplicate local
orders.
