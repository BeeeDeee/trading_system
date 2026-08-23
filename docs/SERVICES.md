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

- connection to external data providers (market data, **news / sentiment**,
  and later alternative sources),
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
- alternative-data features including **news and market sentiment**
  (near-term Milestone 1b),
- market regime classification (trend/range/chop, trend direction,
  volatility state) as derived features,
- deterministic feature generation.

Regime classification and sentiment features are ordinary derived features,
not pipeline stages. Strategies may consume them or ignore them. Their
unavailability affects only strategies that require them; they are not
system-wide trading prerequisites.

### Does not own

- strategy decisions,
- risk rules,
- execution.

---

## 6. Strategy Engine

### Owns

- strategy logic,
- strategy evaluation,
- trading signals,
- proposed exit levels (stop-loss and take-profit proposals attached to
  signals).

### Does not own

- exchange communication,
- order execution,
- portfolio-wide risk decisions.

---

## 7. Portfolio Manager

### Owns

- aggregation of strategy signals,
- capital allocation,
- strategy weighting,
- **proposing target position sizes**.

Position sizing is proposed here. The Risk Manager may constrain a proposed
size (reduce or reject) but never increases it. This is the single place
where the sizing responsibility split is defined.

PortfolioState is derived from current local and reconciled account/position
state. The Portfolio Manager is not the authoritative owner of order or
position lifecycle state.

### Does not own

- strategy logic,
- exchange communication,
- final risk approval.

---

## 8. Risk Manager

### Owns

- risk limits,
- exposure constraints,
- drawdown protection,
- trading halts,
- final risk approval,
- **constraining proposed position sizes** (cap, scale down, reject — never
  increase).

Risk Manager evaluates a `TargetPosition` and returns `RiskDecision`. It may
approve, modify, reject, or halt. It does not create exchange orders and it
does not propose sizes.

### Critical rules

- No risk-opening order may proceed to execution without Risk Manager
  approval.
- Risk-reducing (reduce-only) exits are subject to a minimal, explicitly
  degraded-mode-compatible check path so that reducing risk is never blocked
  by unavailable non-essential dependencies.

---

## 9. Trading Engine

### Owns

- order lifecycle,
- execution,
- cancellation,
- partial fills,
- local order lifecycle and position-state updates,
- **position exit management**: placement and maintenance of exchange-native
  protective stop-loss and take-profit orders derived from approved exit
  levels.

Protective stops live on the exchange by default so they survive platform
restarts, network failures, and database outages. The Trading Engine ensures
that every open position has its protective orders in place and reconciles
them on restart.

Asynchronous execution events (fills, partial fills, rejections,
cancellations) arriving between decision cycles are handled by the Trading
Engine as serialized state transitions; they do not re-enter the decision
pipeline.

### Does not own

- strategy logic,
- risk policy,
- market analysis.

---

## 10. Exchange Adapters

Exchange-specific client code is split into two platform-level adapter
boundaries. They may share a provider client implementation.

### Market Data Adapter

Owns candles, trades, order books, funding, open interest, and other market
data retrieval.

### Account / Execution Adapter

Owns balances, positions, open orders, order submission, cancellation, and fill
retrieval, including exchange-native conditional (stop / take-profit) orders.

Both adapters own provider authentication, symbol mapping, rate limits,
tick-size and step-size rounding rules, and provider-specific errors for
their respective operations.

The rest of the system must use platform-level interfaces instead of exchange-specific APIs.

---

## 11. Reconciliation

### Owns

- comparing local order and position state with external account snapshots,
- verifying protective stop orders exist for every open position,
- identifying state differences,
- coordinating recovery according to operational policy.

Reconciliation does not own strategy decisions or order creation.

## 12. Order Planner

The Order Planner converts an approved or modified target position into the
required position change and creates `OrderIntent` objects, including
reduce-only intents and protective-order intents. It assigns the
client-generated idempotency key. It does not make risk decisions,
communicate with exchanges, or own order lifecycle state.

## 13. AI / ML

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

## 14. Notifications

### Owns

- application notification events,
- Telegram or future notification channels.

### Does not own

- trading decisions,
- orchestration logic.

---

## 15. Communication Rules

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

## 16. Forbidden Dependencies

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

## 17. Extraction Rule

A logical module becomes a separate runtime service only when there is a concrete reason such as:

* independent scaling,
* different resource requirements,
* fault isolation,
* independent deployment,
* GPU requirements,
* external integration constraints.

The default remains a modular monolith.
