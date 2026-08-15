# Services and Module Boundaries

> Logical responsibilities and ownership within the Python trading application.

## 1. Modular Monolith

The core application is a modular monolith.

All modules run inside one Python process.

A module is a logical architectural boundary, not necessarily a Docker container or OS process.

Modules communicate through explicit Python interfaces and typed domain models.

---

## 2. FastAPI / Control Plane

### Owns

- external REST API,
- administrative commands,
- configuration access,
- system status,
- health endpoints,
- strategy control,
- inspection of trading state.

### Does not own

- trading strategy logic,
- market analysis,
- direct exchange communication,
- realtime trading loop.

---

## 3. Data Sources

### Owns

- connection to external data providers,
- retrieval of raw data,
- provider-specific authentication,
- provider-specific rate limiting,
- provider-specific error handling.

### Does not own

- indicators,
- strategy decisions,
- risk decisions,
- orders.

---

## 4. Normalization

### Owns

- conversion of provider-specific data into canonical platform models,
- timestamp normalization,
- symbol normalization,
- basic input validation.

### Does not own

- feature calculation,
- strategy logic,
- trading decisions.

---

## 5. Feature Engine

### Owns

- technical indicators,
- statistical features,
- alternative-data features,
- deterministic feature generation.

### Does not own

- strategy decisions,
- risk rules,
- execution.

---

## 6. Regime Detection

### Owns

- market regime classification,
- regime confidence,
- regime state.

Initial regimes:

- TREND
- RANGE
- CHOP

### Does not own

- strategy selection,
- portfolio allocation,
- execution.

---

## 7. Strategy Engine

### Owns

- strategy logic,
- strategy evaluation,
- trading signals.

### Does not own

- exchange communication,
- order execution,
- portfolio-wide risk decisions.

---

## 8. Portfolio Manager

### Owns

- aggregation of strategy signals,
- capital allocation,
- strategy weighting,
- target positions.

PortfolioState is derived from current local and reconciled account/position
state. The Portfolio Manager is not the authoritative owner of order or
position lifecycle state.

### Does not own

- strategy logic,
- exchange communication,
- final risk approval.

---

## 9. Risk Manager

### Owns

- risk limits,
- position sizing,
- exposure constraints,
- drawdown protection,
- trading halts,
- final risk approval.

Risk Manager evaluates a `TargetPosition` and returns `RiskDecision`. It may
approve, modify, reject, or halt. It does not create exchange orders.

### Critical rule

No order may proceed to execution without Risk Manager approval.

---

## 10. Trading Engine

### Owns

- order lifecycle,
- execution,
- cancellation,
- partial fills,
- local order lifecycle and position-state updates.

### Does not own

- strategy logic,
- risk policy,
- market analysis.

---

## 11. Exchange Adapters

Exchange-specific client code is split into two platform-level adapter
boundaries. They may share a provider client implementation.

### Market Data Adapter

Owns candles, trades, order books, funding, open interest, and other market
data retrieval.

### Account / Execution Adapter

Owns balances, positions, open orders, order submission, cancellation, and fill
retrieval.

Both adapters own provider authentication, symbol mapping, rate limits, and
provider-specific errors for their respective operations.

The rest of the system must use platform-level interfaces instead of exchange-specific APIs.

---

## 12. Reconciliation

### Owns

- comparing local order and position state with external account snapshots,
- identifying state differences,
- coordinating recovery according to operational policy.

Reconciliation does not own strategy decisions or order creation.

## 13. Order Planner

The Order Planner converts an approved or modified target position into the
required position change and creates `OrderIntent` objects. It does not make
risk decisions, communicate with exchanges, or own order lifecycle state.

## 14. AI / ML

### Owns

- model loading,
- inference,
- model metadata,
- optional training interfaces.

### Does not own

- risk bypass,
- direct order execution.

AI availability should not be a system-wide single point of failure.

---

## 15. Notifications

### Owns

- application notification events,
- Telegram or future notification channels.

### Does not own

- trading decisions,
- orchestration logic.

---

## 16. Communication Rules

Core modules should communicate through:

- typed Python interfaces,
- typed domain models,
- explicit application services.

They should not communicate through:

- direct database coupling,
- hidden global state,
- exchange-specific objects,
- internal HTTP calls.

No internal message broker is required initially.

---

## 17. Forbidden Dependencies

The following are explicitly forbidden:

```text
Strategy → Exchange
FastAPI → Exchange
AI → Exchange
n8n → Exchange for core trading
Strategy → Trading Engine bypassing Risk
````

The normal direction is:

```text
Data
 → Analysis
 → Decision
 → Allocation
 → Risk
 → Order Planning
 → Execution
```

---

## 18. Extraction Rule

A logical module becomes a separate runtime service only when there is a concrete reason such as:

* independent scaling,
* different resource requirements,
* fault isolation,
* independent deployment,
* GPU requirements,
* external integration constraints.

The default remains a modular monolith.
