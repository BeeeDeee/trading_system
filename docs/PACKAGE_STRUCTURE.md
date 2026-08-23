# Python Package Structure

> Proposed package boundaries for the initial Python application.

## 1. Repository Layout

```text
backend/
└── app/
    ├── api/
    │   ├── routes/
    │   └── schemas/
    ├── application/
    │   ├── commands/
    │   ├── queries/
    │   └── services/
    ├── domain/
    │   ├── models/
    │   ├── ports/
    │   └── services/
    ├── runtime/
    │   ├── commands.py
    │   ├── lifecycle.py
    │   ├── pipeline.py
    │   └── runtime.py
    ├── modules/
    │   ├── data_sources/
    │   ├── normalization/
    │   ├── features/          # incl. regime features
    │   ├── strategies/
    │   ├── portfolio/
    │   ├── risk/
    │   ├── order_planning/
    │   ├── execution/
    │   ├── backtest/          # simulated execution, backtest clock, fill model
    │   └── reconciliation/
    ├── infrastructure/
    │   ├── persistence/       # repositories + Alembic migrations
    │   ├── exchanges/
    │   ├── configuration/
    │   └── notifications/
    └── main.py

data/                          # gitignored Parquet market data
├── raw/
└── processed/

tests/
├── unit/
├── integration/
└── contract/
```

## 2. Package Responsibilities

### `domain`

Contains exchange-independent models, enums, value objects, and stable ports.
It must not import FastAPI, PostgreSQL, exchange clients, or n8n code.

### `application`

Coordinates use cases and transaction boundaries. It connects domain ports to
repositories and runtime operations without containing strategy or exchange
logic.

### `runtime`

Owns lifecycle state, asyncio task coordination, runtime commands, and the
serialized trading pipeline.

### `modules`

Contains the logical trading modules. Modules implement domain ports and use
typed domain models rather than infrastructure-specific objects.

### `infrastructure`

Contains PostgreSQL repositories, exchange clients and adapters, configuration
providers, and notification channels.

### `api`

Contains FastAPI routes, request/response schemas, authentication, and API
dependency wiring. API schemas must not become domain models.

## 3. Dependency Direction

```text
api ────────────────┐
runtime ────────────┼──> application ───> domain
modules ────────────┘          ▲             ▲
                               │             │
                         infrastructure ────┘
```

Infrastructure may implement domain ports. Domain code must not depend on
infrastructure. Modules must not import FastAPI or call repositories directly.

## 4. Initial Scope Rule

The first implementation should create only the packages needed for the
backtest vertical slice: domain models, historical data loading, features,
one reference strategy, portfolio, risk, order planning, and `modules/backtest`
simulated execution. Milestone 1b then adds news/sentiment adapters and
features under `modules/data_sources/` and `modules/features/` — no new
top-level package is required. Empty extension packages for AI, model
management, and on-chain/macro providers should not be created until their
contracts are required. Live exchange adapters and the runtime loop belong
to Milestone 2.
