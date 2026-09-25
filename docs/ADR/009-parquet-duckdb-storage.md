# ADR-009: Parquet and DuckDB, not PostgreSQL, for research

**Status:** Accepted

## Context

PostgreSQL is already running on the VPS and the previous project used it as the
primary store. The research workload, however, is: write once, read whole
columns, scan by time range, group by category. That is an analytical workload,
not a transactional one.

## Decision

- **Parquet** for all research data: raw bars, processed panels, universe
  snapshots, labeled setups, edge tables, sentiment observations, decision audit
  trails.
- **DuckDB** for ad-hoc queries over those files.
- **PostgreSQL** only at M5, only for live order state, behind the
  `DecisionSink` Protocol and a future `OrderStore`.

Layout, partitioning, and schemas:
[`04-DATA_AND_UNIVERSE.md`](../04-DATA_AND_UNIVERSE.md).

## Consequences

- 5.3M 1h bars across 120 symbols is roughly 150 MB with dictionary-encoded
  symbols, and DuckDB scans it in under a second.
- No daemon, no schema migrations, no connection handling, no network hop.
- Panels load into memory in one call. The engine needs the whole panel anyway
  (ADR-006), so streaming buys nothing.
- Files are inspectable: `duckdb -c "select * from 'x.parquet' limit 5"` needs no
  running service.
- Reprocessing is a rerun, not a migration. Raw data is immutable, so any
  processing change is replayable.
- Cost: no ACID transactions. Irrelevant for append-only research artifacts, and
  mitigated by atomic write-then-rename.
- Cost: no concurrent writers. Also irrelevant — one process writes.

## Alternatives rejected

**PostgreSQL for everything.** Slower for column scans, requires migrations for
every schema change during a phase where schemas change weekly, and introduces a
network hop for data that lives on the same disk. It also encourages storing
derived data that should be recomputed, which is how a research repo accumulates
stale tables nobody trusts.

**SQLite.** Row-oriented; poor for wide analytical scans. DuckDB is SQLite's
analytical counterpart and reads Parquet directly.

**CSV.** No types, no compression, no column pruning. Fine for the small
human-readable outputs (`trades.csv`, `equity.csv`, `funnel.csv`), which is
exactly where it is used.

**A time-series database (TimescaleDB, InfluxDB, ClickHouse).** Real advantages at
tick granularity and at scale. At 4h bars and 150 MB the operational cost exceeds
the benefit by a wide margin.

**Feather or HDF5.** Feather is not a long-term format. HDF5 has a fragile Windows
story and poor tooling compared to Parquet.

## Revisit trigger

M5 for live order state (Postgres), or if a dataset exceeds roughly 50 GB, at
which point partitioned Parquet plus DuckDB still works but the layout needs
revisiting.
