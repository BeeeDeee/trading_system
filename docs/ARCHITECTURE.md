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
                      │      ↓                  │
                      │ Regime Detection        │
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
Feature Engine
     ↓
Regime Detection
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

The runtime uses asyncio tasks for the API, market-data handling, reconnects,
and background work. Trading state transitions are serialized. The initial
design does not use threads for trading logic or an internal event bus.

---

## 6. Data and Decision Flow

The normal live trading flow is:

```text
External Data
     ↓
Canonical Data Model
     ↓
Analytical Features
     ↓
Market Regime
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

The domain objects in this flow are defined in `DATA_MODEL.md`.

---

## 7. Backtesting

Backtesting should reuse the same decision pipeline.

```text
Historical Data
      ↓
Same Trading Pipeline
      ↓
Simulated Execution
```

Live trading:

```text
Live Data
      ↓
Same Trading Pipeline
      ↓
Real Execution
```

The main difference should be the data source and execution implementation, not the trading logic itself.

---

## 8. AI

AI/ML is an optional analytical layer.

It may support:

* regime detection,
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

Trading decisions fail closed:

| Condition | Behavior |
|---|---|
| stale market data | no trade |
| missing features | no trade |
| unavailable regime detection | no trade |
| unavailable risk management | no trade |
| unavailable AI | strategy-defined fallback or no trade |
| unavailable exchange | no new orders |
| unavailable PostgreSQL | no new orders; reconcile later |
| unavailable n8n | trading continues |

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
