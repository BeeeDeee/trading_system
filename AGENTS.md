# Scout — Project Agent Instructions

## Collaboration Preferences

- Tell it like it is; do not sugar-coat responses.
- Take a forward-thinking view.
- Prioritize practical outcomes.
- Be innovative and think outside the box.

---

## Before writing any code

Read [`docs/17-IMPLEMENTER_GUIDE.md`](docs/17-IMPLEMENTER_GUIDE.md), then
[`docs/15-ROADMAP.md`](docs/15-ROADMAP.md). Pick the lowest-numbered unfinished
task and do only that task.

If the documentation is ambiguous or self-contradictory, **stop and ask**. Do not
resolve it by guessing. An ambiguity resolved silently in the scoring or labeling
code produces a system that looks correct and is wrong, and the error will not
surface until it costs money.

---

## Overrides to the global trading-bot conventions

The global conventions apply as written except where restated below. Each
override is required to be declared here by the global rules themselves.

### 1. End-of-bar convention (global §4, "No lookahead bias")

The global rule permits an end-of-bar convention if the project states it
explicitly. **This project uses one:**

> A bar labeled `t` is **closed** at `t` (the US cash-session close, XNYS
> calendar). Features computed for timestamp `t` may read any bar with
> `close_time <= t`. A decision made at `t` is executed at the **open of the
> next session**, `t + 1` (market-on-open).

Consequences that follow and are enforced elsewhere:

- Rolling windows on `high`/`low` used for breakout levels must be `.shift(1)` so
  the channel excludes the current bar
  ([`docs/05-FEATURES_AND_REGIME.md §2.4`](docs/05-FEATURES_AND_REGIME.md#24-donchian-channel)).
- There is **no context timeframe** in v1. Market-regime columns are computed
  from the SPY/VIX benchmark panel and merged on `ts`. The crypto sleeve (M7)
  reintroduces a 1d context join onto 4h bars, with a production assertion that
  `ctx_age_bars >= 0`.
- Entry fills are at the next session's open, never the decision bar's close.
- Split/dividend adjustment is causal: a split at `t` must not change adjusted
  prices at `t-1`. `close_raw` is pinned and never rewritten.
- In backtests the clock source is the bar timestamp. `datetime.now`,
  `datetime.utcnow`, and `time.time` are forbidden everywhere in `src/scout/`
  except `src/scout/utils/clock.py`, enforced by
  `tests/unit/test_no_wallclock.py`.

### 2. Decimal scope clarification (global §1, "Money and precision")

The global rule's three boundaries are honoured. This project draws the line at a
single named function so it is unambiguous:

- **The boundary is `src/scout/portfolio/sizing.py::size_position`.** It receives
  `float` inputs and returns `Decimal` `qty` and `risk_usd`. Everything upstream
  is `float`; everything downstream is `Decimal`.
- `Decimal` is **required** in: sizing, the ledger, `Position`, `PortfolioState`,
  `OrderIntent`, `Fill`, `ClosedTrade` money fields, and everything sent to an
  exchange.
- `Decimal` is **forbidden** in: `MarketPanel`, `FeaturePanel`, every indicator,
  `Setup` prices, `BinStats`, `Opportunity`, `CostEstimate`, `DecisionRecord`,
  and all metrics and plots. These are time series and analytics, which the global
  rule already exempts.
- `tick_size`, `step_size`, and `min_notional_usd` are **stored as strings** in
  Parquet and parsed to `Decimal` on load. Storing them as doubles reintroduces
  the exact error the boundary exists to prevent.

Full table: [`docs/ADR/012-float-decimal-boundary.md`](docs/ADR/012-float-decimal-boundary.md).

### 3. Project structure (global §8, "Suggested project structure")

The layout follows the global default with these documented deviations:

| Deviation | Reason |
|---|---|
| Package is `src/scout/`, not `src/trading_system/` | Short imports across every line of the codebase; lower typo rate |
| `risk/` merged into `portfolio/` | Sizing and risk caps operate on the same state in one pass |
| `metrics/` and `plotting/` merged into `research/` | Both exist only to serve research output |
| Added `universe/`, `gates/`, `scoring/` | Point-in-time universe, eligibility gating, and edge estimation are first-class here and have no home in the default layout |
| No `notebooks/` | Research is reproducible CLI runs writing to `results/`. Nothing may import from a notebook and no reported result may originate in one. |
| Added `experiments/` at repo root | Holds the two **committed** integrity artifacts (`registry.csv`, `holdout_lockbox.json`), which cannot live in gitignored `results/` |
| Added `docs/results/` | Committed research write-ups, including negative results |
| No `backend/` | There is no frontend. The directory from the previous project is to be deleted in M0.1. |

Full tree: [`docs/13-PROJECT_LAYOUT.md`](docs/13-PROJECT_LAYOUT.md).

### 4. Run outputs (global §7, "Backtest outputs and plots")

The global requirements are met in full: `results/<run-id>/` with `config.yaml`,
`metrics.json`, `trades.csv`, `equity.csv`, and `plots/` containing `equity.png`,
`drawdown.png`, and `summary.png`, all at ≥150 DPI with timezone-aware axes.

This project **adds** the following required per-run artifacts:

- `decisions.parquet` — every candidate considered, accepted or rejected, with the
  reason. The primary research artifact.
- `funnel.csv` — stage × rejection-reason counts.
- `bins.csv` — the edge statistics used, with realised outcomes joined.
- `config_hash.txt`, `run.log`.
- `plots/calibration.png` — predicted `ev_net_r` versus realised R. **The most
  important plot the system produces**; a flat line means the ranking carries no
  information.
- `plots/monthly_returns.png`, `plots/regime_breakdown.png`.

Also: `config.yaml` in the run directory is the **fully resolved** config with
every default filled in, not a copy of the input file. Reproducibility requires
knowing the value that was used, not the value that was written down.

### 5. Configuration (global §20)

Satisfied, with one addition: the resolved config is **hashed** (sha256 over the
canonical sorted-key JSON) and the hash is recorded on every run and every derived
artifact. A backtest whose `config_hash` does not match its edge table's
`config_hash` **fails at startup**. This catches the most dangerous silent error
available: edge statistics computed under different rules than the ones currently
making decisions.

---

## Project-specific hard rules

These are not style preferences. Each has cost someone money in a real system.

1. **No weighted scores.** There is exactly one ranking statistic and it has a
   unit. See [`docs/ADR/002-single-ranking-statistic.md`](docs/ADR/002-single-ranking-statistic.md).
2. **Never assume the favourable intrabar path.** A bar touching both the stop and
   the target resolves as a **stop**. Always.
3. **Never fill a data gap.** Not forward, not backward, not by interpolation. An
   absent row is visible; a filled row is an invisible lie.
4. **Never let a missing input make a trade more likely.** Every fallback must be
   the conservative one. Table in
   [`docs/17-IMPLEMENTER_GUIDE.md §2`](docs/17-IMPLEMENTER_GUIDE.md#rule-4--never-let-a-missing-input-make-a-trade-more-likely).
5. **Never round in your own favour.** Quantities round down. Stops round away
   from entry.
6. **Never delete a symbol from `config/universe_candidates.txt`.** That file is
   append-only. Deleting a delisted symbol is survivorship bias applied by hand.
7. **Never run the holdout to "check progress."** Budget is 3 evaluations for the
   project's lifetime, enforced by `experiments/holdout_lockbox.json`.
8. **Never catch `ScoutLookaheadError`.** It must crash.
9. **The backtest outer loop is over timestamps, never symbols.**
10. **No new dependencies** beyond `pyproject.toml`, and no loosening of version
    pins. Determinism depends on exact versions.
11. **No TA-Lib, `pandas_ta`, `vectorbt`, or `backtrader`.** Build pain on
    Windows, and silent formula changes between versions invalidate every stored
    edge statistic.
12. **Equities first.** Do not implement crypto ingest, funding, 4h bars, or a
    Binance client before M7. Crypto is optional and gated on a completed equity
    holdout write-up. Yahoo Finance is not a data vendor.

Full list, with the reasoning:
[`docs/17-IMPLEMENTER_GUIDE.md`](docs/17-IMPLEMENTER_GUIDE.md).

---

## Environment

- Windows development machine, PowerShell. Use `pathlib.Path`, explicit UTF-8 on
  every text read and write, and atomic write-then-rename for all file output.
- Virtual environment at repo root (`.venv`), created by
  `scripts/setup_dev.ps1`. Not in `backend/`.
- Python 3.11+.
- Primary market: **US-listed common stocks and ETFs**, daily XNYS sessions.
  Paid survivorship-free data (Norgate or Sharadar).
- A VPS with Docker, PostgreSQL, and n8n exists but is **not used before M5**.
  See [`docs/ADR/001-research-first-no-infra.md`](docs/ADR/001-research-first-no-infra.md).
