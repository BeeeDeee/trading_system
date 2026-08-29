# Scout — Multi-Asset Trading Opportunity Engine

> Documentation set. Read this file first.

Scout scans a liquidity-ranked universe of US stocks and ETFs on the daily cash
session clock, and on most cycles decides to do nothing. It only trades when a
setup's estimated expected value, measured in units of risk and net of costs,
clears a hard threshold.

The design principle is stated once and applies everywhere:

> **Every number that influences a trading decision must have a unit, and every
> unit must be either "R" (one unit of trade risk), "bars", or "fraction of
> equity". If a proposed number has no unit, it is not allowed into the decision.**

That rule is what removes most of the complexity from this system.

---

## Reading order

### For a human reviewing the design

| # | Document | What it answers |
|---|---|---|
| 00 | [`00-REVIEW.md`](00-REVIEW.md) | What was wrong or risky in the original brief, and what was changed |
| 01 | [`01-ARCHITECTURE.md`](01-ARCHITECTURE.md) | Components, data flow, boundaries, what each part must not know |
| 07 | [`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md) | The core math. If you read only one doc, read this one |
| 15 | [`15-ROADMAP.md`](15-ROADMAP.md) | Milestones, task list, acceptance criteria |

### For an implementer (including AI models)

Read in this exact order and do not skip:

1. [`17-IMPLEMENTER_GUIDE.md`](17-IMPLEMENTER_GUIDE.md) — hard rules, forbidden patterns, definition of done
2. [`13-PROJECT_LAYOUT.md`](13-PROJECT_LAYOUT.md) — exact file tree; where each thing goes
3. [`02-DOMAIN_MODEL.md`](02-DOMAIN_MODEL.md) — every dataclass, field, type, unit
4. [`03-INTERFACES.md`](03-INTERFACES.md) — every Protocol with exact signatures
5. [`14-CONFIG.md`](14-CONFIG.md) — full YAML schema and defaults
6. [`15-ROADMAP.md`](15-ROADMAP.md) — pick the lowest-numbered unfinished task and do only that
7. The doc for the specific module you were assigned

---

## Full document list

| Document | Contents |
|---|---|
| [`00-REVIEW.md`](00-REVIEW.md) | Critique of the original brief; simplification decisions; what was cut |
| [`01-ARCHITECTURE.md`](01-ARCHITECTURE.md) | Component map, decision cycle, layer rules, dependency direction |
| [`02-DOMAIN_MODEL.md`](02-DOMAIN_MODEL.md) | All dataclasses and enums, with units and invariants |
| [`03-INTERFACES.md`](03-INTERFACES.md) | The five Protocols; everything else is a pure function |
| [`04-DATA_AND_UNIVERSE.md`](04-DATA_AND_UNIVERSE.md) | Storage layout, Parquet schemas, point-in-time universe snapshots |
| [`05-FEATURES_AND_REGIME.md`](05-FEATURES_AND_REGIME.md) | Exact indicator formulas and the regime classifier |
| [`06-STRATEGIES.md`](06-STRATEGIES.md) | The two v1 strategies, rule by rule, plus how to add a third |
| [`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md) | Setup labeling, edge estimation, the ranking statistic, thresholds |
| [`08-COSTS.md`](08-COSTS.md) | Fee, spread, slippage, funding and impact model; conversion to R |
| [`09-PORTFOLIO_AND_RISK.md`](09-PORTFOLIO_AND_RISK.md) | Gates, caps, portfolio heat, position sizing, Decimal boundary |
| [`10-SENTIMENT.md`](10-SENTIMENT.md) | Point-in-time sentiment store, penalty-only integration, promotion gate |
| [`11-BACKTEST_ENGINE.md`](11-BACKTEST_ENGINE.md) | Panel loop pseudocode, fill rules, corner cases, run outputs |
| [`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md) | Data splits, walk-forward, trial budget, metrics, robustness suite |
| [`13-PROJECT_LAYOUT.md`](13-PROJECT_LAYOUT.md) | Exact directory tree and per-module responsibility |
| [`14-CONFIG.md`](14-CONFIG.md) | Complete config schema with defaults and validation rules |
| [`15-ROADMAP.md`](15-ROADMAP.md) | Milestones M0–M6 with per-task acceptance criteria |
| [`16-TESTING.md`](16-TESTING.md) | Required tests, fixtures, property tests, lookahead tripwires |
| [`17-IMPLEMENTER_GUIDE.md`](17-IMPLEMENTER_GUIDE.md) | Rules for whoever writes the code |
| [`ADR/`](ADR/) | Short decision records with rationale and alternatives rejected |

---

## Non-negotiable conventions

These appear in several documents; they are collected here so there is one
authoritative statement of each.

**Bar convention.** A bar labeled `t` is *closed* at `t`. For US equities that
is the session close (typically 21:00 UTC during EST, 20:00 UTC during EDT —
always taken from the calendar, never hardcoded). Features computed for
timestamp `t` may read any bar with `close_time <= t`. A decision made at `t` is
executed at the **open of the next session**, `t + 1`. No exceptions.

**Time.** All timestamps are timezone-aware UTC. In backtests the clock is the
bar timestamp. `datetime.now()`, `datetime.utcnow()` and `time.time()` are
forbidden inside `src/scout/` except in `src/scout/utils/clock.py`.

**Units.** `_r` suffix means "in units of one trade's risk". `_bps` means basis
points. `_pct` means fraction of equity in `[0, 1]`. `_bars` means bar counts.
Any float field influencing a decision carries one of these suffixes.

**Floats vs Decimal.** Floats everywhere in features, indicators, and analytics.
`Decimal` at exactly three boundaries: order price/quantity construction, the
cash/position ledger, and P&L/fee accumulation. See
[`09-PORTFOLIO_AND_RISK.md`](09-PORTFOLIO_AND_RISK.md#7-the-floatdecimal-boundary).

**No weighted score.** There is exactly one ranking number and it is an expected
value in R, net of costs. Nothing is combined by tunable additive weights. See
[`ADR/002-single-ranking-statistic.md`](ADR/002-single-ranking-statistic.md).

**Deferred, deliberately.** Machine learning, live execution, FastAPI, n8n,
PostgreSQL, real sentiment data, and **crypto** are all out of scope until their
milestone in [`15-ROADMAP.md`](15-ROADMAP.md). Crypto is M7 and optional.
US equities and ETFs are the v1 market. Do not add deferred pieces early.
