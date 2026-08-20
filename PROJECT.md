# AI Trading Platform

> Living project overview and navigation document.

## Vision

Build a professional, modular algorithmic trading platform for automated trading, research, backtesting, and AI-assisted decision making.

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

The next implementation milestone is the Python backend and trading runtime.

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
     ↓
Exchange
````

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

### Core Application

* [x] Python project structure
* [ ] FastAPI control plane
* [ ] Trading runtime
* [ ] Domain models
* [ ] Data source layer
* [ ] Feature engine
* [ ] Regime detection
* [ ] Strategy engine
* [ ] Portfolio manager
* [ ] Risk manager
* [ ] Trading engine
* [ ] Exchange adapter

### Research and AI

* [ ] Backtesting
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
* FastAPI is the control plane.
* n8n is orchestration only.
* Trading modules communicate through Python interfaces and typed domain models.
* No internal message broker is required initially.
* Risk management is mandatory before execution.
* Exchange-specific behavior is isolated behind adapters.
* Trading state transitions are serialized inside the single Python process.
* Trading decisions fail closed when required inputs or safety dependencies are unavailable.

The runtime, state ownership, and persistence boundaries are defined in
`docs/RUNTIME.md` and `docs/STATE_AND_PERSISTENCE.md`.

Full rationale is documented in `docs/ADR/`.
