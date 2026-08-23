# System Architecture

> High-level architecture of the AI Trading Platform.

## 1. Purpose

This document defines the overall structure and runtime behavior of the platform.

It describes:

- system boundaries,
- major architectural layers,
- control flow,
- trading flow,
- deployment model,
- key architectural principles.

Detailed component responsibilities are defined in `SERVICES.md`.

---

## 2. System at a Glance

The platform is a modular algorithmic trading system.

It consists of:

- external data sources,
- a Python trading application,
- PostgreSQL,
- FastAPI,
- n8n,
- exchange adapters,
- optional AI/ML functionality.

The core trading application is initially implemented as a **modular monolith running in one Python process**.

---

## 3. High-Level Architecture

```text
                         ┌───────────────────┐
                         │       n8n         │
                         │   Orchestration   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │      FastAPI      │
                         │   Control Plane   │
                         └─────────┬─────────┘
                                   │
═══════════════════════════════════╪══════════════════════════════
                                   │
                         Python Trading Process
                                   │
                      ┌────────────┴────────────┐
                      │     Trading Runtime     │
                      │                         │
                      │ Data Sources            │
                      │      ↓                  │
                      │ Normalization           │
                      │      ↓                  │
                      │ Feature Engine          │
                      │ (incl. regime features) │
                      │      ↓                  │
                      │ Strategy Engine         │
                      │      ↓                  │
                      │ Portfolio Manager       │
                      │      ↓                  │
                      │ Risk Manager            │
                      │      ↓                  │
                      │ Order Planner           │
                      │      ↓                  │
                      │ Trading Engine          │
                      │      ↓                  │
                      │ Account / Execution      │
                      │ Adapter                  │
                      └────────────┬────────────┘
                                   │
                                   ▼
                               Exchange

                         ┌───────────────────┐
                         │    PostgreSQL     │
                         │ Persistent State  │
                         └───────────────────┘
````

---

## 4. Control Plane

FastAPI is the control and observation interface.

It is responsible for:

* system status,
* configuration,
* administration,
* strategy control,
* portfolio inspection,
* order inspection,
* operational commands,
* health endpoints.

FastAPI does not implement the trading loop.

FastAPI and the Trading Runtime run in the same Python process. FastAPI submits
`RuntimeCommand` objects to the runtime; it does not mutate runtime state
directly.

---

## 5. Trading Runtime

The Trading Runtime contains the core decision and execution pipeline.

```text
Data Sources
     ↓
Normalization
     ↓
Feature Engine (incl. regime features)
     ↓
Strategy Engine
     ↓
Portfolio Manager
    ↓
Risk Manager
    ↓
Order Planner
    ↓
Trading Engine
    ↓
Account / Execution Adapter
```

All components initially run inside the same Python process.

The runtime is **bar-driven**: the decision pipeline runs once per completed
bar of the configured timeframe (initially 1h). Sub-bar protection is handled
by exchange-native conditional orders, not by the pipeline
(see `ADR/008-bar-driven-runtime.md`).

The runtime uses asyncio tasks for the API, market-data handling, reconnects,
and background work. Trading state transitions are serialized. The initial
design does not use threads for trading logic or an internal event bus.

Market regime classification (trend/range/chop, volatility state) is owned by
the Feature Engine as derived features. It is an optional strategy input, not
a pipeline stage and not a trading prerequisite.

---

## 6. Data and Decision Flow

The normal live trading flow is:

```text
External Data
     ↓
Canonical Data Model
     ↓
Analytical Features (incl. regime)
     ↓
Strategy Signal
     ↓
Portfolio Target
     ↓
Risk Decision
     ↓
Order Planner
     ↓
Order Intent
     ↓
Execution Report
```

Every stage result is persisted as a decision audit trail with correlation
identifiers, so any live order can be traced back to the signal and features
that produced it (see `STATE_AND_PERSISTENCE.md`).

The domain objects in this flow are defined in `DATA_MODEL.md`.

---

## 7. Backtesting and Paper Trading

The platform has three execution modes sharing the same decision pipeline.
Only configuration and the data/execution adapters differ:

```text
Backtest:  Historical Data → Same Pipeline → Simulated Execution
Paper:     Live Data       → Same Pipeline → Testnet / PaperBroker
Live:      Live Data       → Same Pipeline → Real Execution
```

The backtest vertical slice is the first implementation milestone; it
validates the pipeline and its interfaces before any live infrastructure is
built. Paper trading is a mandatory stage before live trading.

In backtests the clock source is the bar timestamp, never wall-clock.
Simulation assumptions (fees, slippage, funding, fill model) are explicit
backtest concerns.

A faithful event-driven backtest and a fast vectorized research layer serve
different purposes; the research layer may exist separately, but strategy
validation must use the shared pipeline.

---

## 8. News, Sentiment, and AI

### News and sentiment (near-term)

News and market sentiment are **planned near-term inputs** (Milestone 1b),
not optional afterthoughts. They enter through provider adapters, are
normalized into platform models, and become Feature Engine outputs that
strategies may consume.

They must:

* align to the bar-driven clock (1h) without lookahead in backtests,
* fail soft when unavailable (only strategies that require them are affected),
* never bypass Risk Management or create orders directly.

On-chain and macro data remain later expansions; see `DATA_SOURCES.md`.

### AI / ML

AI/ML is an optional analytical layer.

It may support:

* regime / sentiment scoring,
* signal scoring,
* anomaly detection,
* strategy selection,
* parameter optimization,
* prediction.

AI must remain inside the normal decision pipeline.

AI must never bypass Risk Management or directly execute orders.

---

## 9. n8n

n8n is external to the core trading runtime.

It is used for:

* scheduling,
* automation,
* reports,
* notifications,
* external integrations,
* operational workflows.

n8n is not part of realtime trading.

---

## 10. Failure Policy

The core trading runtime continues when n8n is unavailable. n8n is not part
of the safety boundary.

The failure policy is **asymmetric** (see `ADR/010-asymmetric-failure-policy.md`):

- **Opening or increasing risk fails closed.** All required inputs and safety
  dependencies must be healthy.
- **Reducing or closing risk requires the minimum possible dependencies.**
  Reduce-only exits remain permitted in degraded states, and exchange-native
  protective stops remain active even if the platform process is down.

Risk-opening decisions fail closed:

| Condition | Behavior |
|---|---|
| stale market data | no new risk-opening trade |
| missing features | no new risk-opening trade |
| unavailable risk management | no new risk-opening trade |
| unavailable AI | strategy-defined fallback or no trade |
| unavailable exchange | no new orders |
| unavailable PostgreSQL | no new risk-opening orders; reconcile later |
| unavailable n8n | trading continues |

Regime features are ordinary features: if unavailable, strategies that
require them produce no signal; strategies that do not require them are
unaffected. The same rule applies to news / sentiment features.

## 11. Persistence

PostgreSQL is the primary persistent store. Domain code accesses it through
application services and repositories; modules do not issue direct SQL or use
the database as an internal event transport.

State ownership, reconciliation, and persistence boundaries are defined in
`STATE_AND_PERSISTENCE.md`.

---

## 12. Initial Deployment

```text
Docker Host
│
├── PostgreSQL
├── n8n
├── trading-platform
│   ├── FastAPI
│   └── Trading Runtime
└── monitoring
```

Logical modules are not automatically separate Docker services.

---

## 13. Repository Structure

```text
trading-platform/
│
├── PROJECT.md
├── CHANGELOG.md
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SERVICES.md
│   ├── DATA_MODEL.md
│   ├── DATA_SOURCES.md
│   ├── API.md
│   ├── N8N.md
│   └── ADR/
│
├── backend/
│
├── services/
│
├── tests/
│
├── scripts/
│
└── docker/
```

The repository structure may evolve during implementation.

---

## 14. Server / Deployment Structure

Deployment structure is separate from repository structure.

Current server layout:

```text
~/docker/
│
├── core/
├── services/
├── monitoring/
├── scripts/
├── backups/
└── .git/
```

The logical architecture must not depend on this physical layout.

---

## 15. Architectural Philosophy

The platform follows these priorities:

1. logical separation,
2. explicit contracts,
3. deterministic behavior,
4. testability,
5. operational simplicity.

Process separation or additional infrastructure should only be introduced when a concrete requirement exists.
