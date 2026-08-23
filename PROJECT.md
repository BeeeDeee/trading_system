# AI Trading Platform

> Living project overview and navigation document.

## Vision

Build a professional, modular algorithmic trading platform for automated trading, research, backtesting, and AI-assisted decision making.

Price data is the foundation. **News and market sentiment** are first-class
near-term inputs: fetched via adapters, turned into features, and available
to strategies that opt in — never as a bypass of risk management.

The platform should remain understandable, testable, and maintainable over the long term.

The core trading system must not depend on n8n or AI.

---

## Core Principles

- Modular monolith first.
- Single Python process for the core trading application.
- Clear logical boundaries between trading components.
- Trading logic stays in Python.
- n8n is orchestration only.
- FastAPI is the external control plane.
- Risk management is a mandatory execution gate.
- Exchange-specific logic is isolated behind adapters.
- Live trading and backtesting should reuse the same decision pipeline.
- Avoid unnecessary technologies and infrastructure.
- Prefer safe failure over uncontrolled trading.

---

## Current Stage

**Architecture Definition**

The infrastructure is already operational.

Next: Milestone 1 (price backtest vertical slice), then Milestone 1b
(news/sentiment ingestion and features). Paper and live follow after those.

---

## Current Infrastructure

- Ubuntu VPS
- Docker
- Docker Compose
- PostgreSQL
- n8n
- Git
- Automated database backup
- Update and status scripts

Detailed infrastructure information is maintained separately.

---

## Target System

```text
External Data
     ↓
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
     ↓
Exchange
````

The runtime is bar-driven. Initial scope: BTC/USDT on 1h bars
(see `docs/ADR/008-bar-driven-runtime.md`).

FastAPI provides the control interface.

n8n provides external orchestration.

PostgreSQL provides persistent storage.

---

## Documentation Map

### Architecture

* `docs/ARCHITECTURE.md`
* `docs/SERVICES.md`

### Domain Model

* `docs/DATA_MODEL.md`
* `docs/DATA_SOURCES.md`
* `docs/STATE_AND_PERSISTENCE.md`
* `docs/PACKAGE_STRUCTURE.md`
* `docs/INTERFACES.md`

### Interfaces

* `docs/API.md`

### Orchestration

* `docs/N8N.md`
* `docs/RUNTIME.md`

### Decisions

* `docs/ADR/`

---

## Roadmap

### Infrastructure

* [x] VPS
* [x] Docker
* [x] PostgreSQL
* [x] n8n
* [x] Git
* [x] Backup
* [x] Update tooling
* [x] Status tooling

### Milestone 1 — Backtest Vertical Slice

* [x] Python project structure
* [ ] Domain models and core interfaces
* [ ] Historical data layer (download, storage, loaders)
* [ ] Feature engine (incl. regime features)
* [ ] Strategy engine + one reference strategy
* [ ] Portfolio manager
* [ ] Risk manager
* [ ] Order planner
* [ ] Backtest engine with simulated execution
* [ ] Decision audit trail and structured logging

### Milestone 1b — News & Sentiment (near-term)

Price path comes first; this milestone follows immediately so alternative
data is in the system before paper trading.

* [ ] News / sentiment provider adapters (e.g. news API, Fear & Greed)
* [ ] Normalization into canonical sentiment / news domain models
* [ ] Historical storage for sentiment series (Parquet alongside OHLCV)
* [ ] Feature Engine: sentiment features aligned to 1h bar timestamps
* [ ] Optional strategy consumption (strategies that ignore sentiment still run)
* [ ] Stale / missing sentiment fails soft (no system-wide halt)

### Milestone 2 — Paper Trading

* [ ] Trading runtime (live loop)
* [ ] Exchange adapters (market data + account/execution)
* [ ] FastAPI control plane
* [ ] Reconciliation
* [ ] Paper trading via testnet / PaperBroker (config-only difference)

### Milestone 3 — Live Trading

* [ ] Live execution with kill switch and circuit breakers
* [ ] Exchange-native protective stop orders

### Research and AI

* [ ] ML models
* [ ] AI inference
* [ ] Model management

### Operations

* [ ] Monitoring
* [ ] Metrics
* [ ] Dashboard
* [ ] Alerting

---

## Current Architectural Decisions

* Core application is a modular monolith.
* Core trading components run in one Python process.
* The runtime is bar-driven; initial scope is BTC/USDT on 1h bars.
* News and market sentiment are near-term data sources (Milestone 1b),
  consumed as optional features — not a trading prerequisite.
* FastAPI is the control plane.
* n8n is orchestration only.
* Trading modules communicate through Python interfaces and typed domain models.
* No internal message broker is required initially.
* Risk management is mandatory before execution.
* Regime detection is a feature-layer concern, not a pipeline stage.
* Portfolio Manager proposes position sizes; Risk Manager constrains them
  (it may reduce or reject, never increase).
* The failure policy is asymmetric: opening risk fails closed, reducing
  risk requires the minimum possible dependencies.
* Protective stops are exchange-native conditional orders by default.
* Every order carries a client-generated idempotency key.
* Every pipeline decision is persisted as an audit trail.
* Exchange-specific behavior is isolated behind adapters.
* Trading state transitions are serialized inside the single Python process.

The runtime, state ownership, and persistence boundaries are defined in
`docs/RUNTIME.md` and `docs/STATE_AND_PERSISTENCE.md`.

Full rationale is documented in `docs/ADR/`.
