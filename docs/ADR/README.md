# Architecture Decision Records

Short records of decisions that are expensive to reverse. Each states the
decision, why, what was rejected, and what would justify revisiting it.

Format: context, decision, consequences, alternatives rejected, revisit trigger.
Keep them under one page. If a decision needs more than a page, it belongs in a
numbered document in `docs/` and the ADR should link to it.

| # | Decision | Status |
|---|---|---|
| [001](001-research-first-no-infra.md) | Research-first: CLI and files, no service infrastructure before M5 | Accepted |
| [002](002-single-ranking-statistic.md) | One ranking statistic in R units; no weighted score | Accepted |
| [003](003-empirical-walk-forward-edge.md) | Binned empirical walk-forward edge estimator with a lower confidence bound | Accepted |
| [004](004-regime-as-feature.md) | Regime is a causal feature plus a gate, not a pipeline stage | Accepted |
| [005](005-point-in-time-universe.md) | Point-in-time universe snapshots | Accepted |
| [006](006-panel-loop-engine.md) | Single time-ordered panel loop | Accepted |
| [007](007-cluster-caps-not-covariance.md) | Static cluster caps and index beta instead of a covariance matrix | Accepted — amended for equity sectors |
| [008](008-bracket-exits.md) | Broker-native bracket exits; no pipeline-driven stops | Accepted |
| [009](009-parquet-duckdb-storage.md) | Parquet and DuckDB, not PostgreSQL, for research | Accepted |
| [010](010-4h-decision-timeframe.md) | 4h decision bars with 1d context | **Superseded by 016** (crypto only) |
| [011](011-holdout-lockbox.md) | Trial registry and holdout lockbox | Accepted |
| [012](012-float-decimal-boundary.md) | Float/Decimal boundary | Accepted |
| [013](013-sentiment-penalty-only.md) | Sentiment as a penalty-only size modifier behind a promotion gate | Accepted — equity sources added |
| [014](014-no-self-reported-confidence.md) | Strategies do not report confidence | Accepted |

### The equity pivot

ADRs 001–014 were written for a crypto-first scope. ADR-015 reverses that
ordering, and 016–020 work through the consequences. Where an earlier ADR is
affected, its status column above says so and the ADR itself carries a banner.

| # | Decision | Status |
|---|---|---|
| [015](015-equities-first.md) | US equities and ETFs first; crypto last and optional | Accepted |
| [016](016-daily-decision-bars.md) | Daily decision bars, index context, next-open execution | Accepted |
| [017](017-unadjusted-prices-pinned-snapshot.md) | Store unadjusted prices plus an actions table; pin the data snapshot | Accepted |
| [018](018-liquidity-rank-universe.md) | Universe by causal liquidity rank, not index membership | Accepted |
| [019](019-cross-sectional-momentum-primary.md) | Cross-sectional momentum is primary; R generalises to a sizing unit | Accepted |
| [020](020-earnings-gate.md) | Exclude positions whose holding window contains an earnings date | Accepted |

**Unaffected by the pivot**, because they are about method rather than market:
002 (single ranking statistic), 003 (empirical walk-forward edge), 004 (regime as
feature), 006 (panel loop), 011 (holdout lockbox), 012 (float/Decimal boundary),
014 (no self-reported confidence). These are the load-bearing decisions and they
survive an asset-class change unchanged, which is a reasonable sign that they were
drawn at the right level.
