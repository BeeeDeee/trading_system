# Scout — Multi-Asset Trading Opportunity Engine

> Living project overview. Documentation lives in [`docs/`](docs/README.md).

## What this is

A bar-driven, cross-sectional opportunity scanner. After each US cash-session
close it evaluates a liquidity-ranked universe of **US stocks and ETFs**, ranks
candidate trades by a single expected-value statistic net of costs, applies
portfolio-level risk constraints, and on most cycles does nothing.

"No trade" is the normal outcome.

**Crypto is not in scope until M7**, and only after an equity holdout result
exists. See [ADR-015](docs/ADR/015-equities-first.md).

## Current stage

**M3.6 development run recorded** in
[`docs/results/m3_development.md`](docs/results/m3_development.md).
Trial 1 (`20260906-203830-…`) is the CHKAQ accounting invalidation. Trial 2
(`20260907-154935-xsec-momentum-donchian`) is the measurement: 687 trades,
`mean_r` 0.064 R, inverted calibration, CAGR 0.81%. Do not tune. Do not
start M3.7 / holdout / M4. Fast labeling is now `scout label` (checkpoints
in `data/labels`); the slow sibling modules are gone.
New chats: paste the handoff block in
[`docs/17-IMPLEMENTER_GUIDE.md` §8](docs/17-IMPLEMENTER_GUIDE.md#8-starting-a-new-chat).

## The one-line design

```text
ev_net_r = ev_r_lcb - cost_r          [units: R, one unit of intended risk]
```

One ranking number, with a unit, derived from measured historical outcomes rather
than assembled from weighted sub-scores. Trade if it clears a threshold. Rank by
expected value per bar held.

There are **no scoring weights**. Every other input — regime, liquidity,
sentiment, portfolio context — enters as a gate, a conditioning variable, or a
one-sided size penalty. Rationale:
[`ADR-002`](docs/ADR/002-single-ranking-statistic.md).

## Core principles

- Optimise expectancy and robustness. Never win rate.
- Every decision-relevant number has a unit: R, bars, or fraction of equity.
- Point-in-time everything. Universe, features, sentiment, edge statistics.
- Costs modelled pessimistically. A strategy that dies at 1.5× cost is not
  deployable.
- Correlated positions are one position.
- Count your trials. Lock the holdout.
- Rule-based baseline before any machine learning.
- Ask "why should this edge persist" before admiring a backtest.

## Scope

**In v1 (M0–M6):** US-listed common stocks and ETFs, **daily** decision bars,
next-open execution, two strategies (`xsec_momentum_v1` primary, Donchian
breakout secondary), liquidity-ranked point-in-time universe (top 1,000 by
trailing ADV, delisted included), empirical walk-forward edge estimation,
GICS/ETF cluster caps, SPY beta, full equity cost model (commission, spread,
slippage, impact, borrow, dividends), earnings gate, backtest engine, research
and robustness tooling.

**Deliberately deferred, each with a trigger:** real sentiment data (M2 plumbing,
M4 promotion test), live execution (M5), FastAPI / PostgreSQL / n8n / Docker
(M5), machine learning (M6), **crypto (M7, optional)**, additional strategies
and timeframes (M4/M6).

Full list with reasoning:
[`docs/00-REVIEW.md §11`](docs/00-REVIEW.md#11-full-list-of-what-was-cut-from-v1-and-why)
and [`§14`](docs/00-REVIEW.md#14-the-equity-pivot).

## Documentation map

| Purpose | Document |
|---|---|
| Start here | [`docs/README.md`](docs/README.md) |
| What changed from the original brief, and why | [`docs/00-REVIEW.md`](docs/00-REVIEW.md) |
| **Probability of an edge** | [`docs/00-REVIEW.md §14.3`](docs/00-REVIEW.md#143-probability-of-finding-an-edge) |
| Architecture and data flow | [`docs/01-ARCHITECTURE.md`](docs/01-ARCHITECTURE.md) |
| **The core math** | [`docs/07-EDGE_AND_SCORING.md`](docs/07-EDGE_AND_SCORING.md) |
| Milestones and acceptance criteria | [`docs/15-ROADMAP.md`](docs/15-ROADMAP.md) |
| Rules for implementers | [`docs/17-IMPLEMENTER_GUIDE.md`](docs/17-IMPLEMENTER_GUIDE.md) |
| Decisions and rejected alternatives | [`docs/ADR/`](docs/ADR/README.md) |

## Roadmap summary

| Milestone | Scope | Estimate | Status |
|---|---|---|---|
| M0 | Repo skeleton, config, domain model | 3 days | Done |
| M1 | Equity data pipeline, calendar, features, PIT universe | 7 days | Done |
| M2 | Strategies, setup labeling, edge table | 4 days | Done (M2.2a recorded) |
| M3 | Backtest engine, metrics, **the holdout answer** | 6 days | In progress (M3.1–M3.4 done) |
| M4 | Robustness, parameters, sentiment, paper prep | 10 days | Gated on M3 |
| M5 | Paper trading, then live | 15 days | Gated on M4 + 60-day soak |
| M6 | ML evaluation | 10 days | Gated on M3 criteria |
| M7 | Optional crypto sleeve | — | Gated on an equity holdout result |

**M3 is the milestone that matters.** Everything before it is plumbing; everything
after it is conditional on its result.

## Go / no-go

The criteria for proceeding past M3 are fixed **now**, before implementation, so
the bar cannot move later:
[`docs/12-RESEARCH_PROTOCOL.md §9`](docs/12-RESEARCH_PROTOCOL.md#9-go--no-go-criteria-for-m3).

If the holdout shows no edge after costs, the correct action is to change the
hypothesis — not to add sentiment, ML, crypto, and two more timeframes until the
number turns positive.

## Honest expectation

Equities are the right market for this architecture (~7× lower `cost_r` than
crypto, and a documented 12–1 momentum prior). A **deployable live edge is still
the minority outcome** — roughly 15–25% if you are honest about crowding, gap
risk, and the 2018–present holdout. A clean, well-measured negative is the
default, and it is a successful project. Full table:
[`docs/00-REVIEW.md §14.3`](docs/00-REVIEW.md#143-probability-of-finding-an-edge).

## Integrity artifacts

Two committed files are the project's research record. Never delete or hand-edit
them:

- `experiments/registry.csv` — every backtest run ever executed, with its trial id
- `experiments/holdout_lockbox.json` — holdout evaluations used, out of 3
