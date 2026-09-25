# ADR-001: Research-first — CLI and files, no service infrastructure before M5

**Status:** Accepted

## Context

A VPS with Docker, PostgreSQL, and n8n is already running, and the previous
project's architecture placed FastAPI and n8n in the diagram from day one. The
temptation to reuse that infrastructure immediately is strong.

The current question, however, is "does this hypothesis have an edge". Answering
it requires historical bars, a deterministic loop, and files.

## Decision

Milestones M0–M4 are **CLI plus Parquet plus DuckDB**. No database server, no web
framework, no orchestrator, no containers.

Each deferred component has an explicit trigger:

| Component | Introduced when |
|---|---|
| PostgreSQL | Live order state needs durable crash-safe storage (M5) |
| FastAPI | A running loop exists whose state someone must inspect (M5) |
| n8n | Scheduled reports and alerts are needed (M5) |
| Docker | Deploying to the VPS (M5) |

## Consequences

- Roughly 60% fewer moving parts before live trading.
- A full backtest runs in five minutes from a cold start with no services up.
- New-machine setup is one PowerShell script.
- No work is wasted: `DecisionSink` and `Broker` are Protocols, so
  `PostgresDecisionSink` and `LiveBroker` slot in without touching decision code.
- Cost: the eventual M5 integration is one larger step rather than several small
  ones. Accepted, because most of that integration would otherwise be built
  against requirements that do not exist yet.

## Alternatives rejected

**Postgres from the start.** 5.3M bars in Parquet queried by DuckDB is faster
than the same data in Postgres over a network, needs no schema migrations, and no
daemon. Postgres earns its place when there is durable *transactional* state,
which appears at M5.

**FastAPI from the start.** There is no loop to control. A control plane over a
batch CLI is a REST wrapper around `subprocess`.

**n8n scheduling the research runs.** Research runs are triggered by a human
forming a hypothesis. Scheduling them would encourage running many
configurations, which is precisely the multiple-testing behaviour ADR-011 exists
to constrain.

## Revisit trigger

M5, or an earlier need for multi-machine coordination.
