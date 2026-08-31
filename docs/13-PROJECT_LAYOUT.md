# Project Layout

> Exact directory tree. Put files where this says. Module: everything.

---

## 1. Repository tree

```text
trading_system/
├── AGENTS.md                       # collaboration prefs + project rule overrides
├── PROJECT.md                      # living project overview, status, roadmap link
├── README.md                       # quickstart
├── pyproject.toml                  # deps + ruff + mypy + pytest config
├── .env.example
├── .gitignore
├── .cursorignore
│
├── docs/                           # this documentation set
│   ├── README.md … 17-IMPLEMENTER_GUIDE.md
│   ├── ADR/
│   └── results/                    # COMMITTED research write-ups
│       ├── m3_holdout.md
│       └── sentiment_promotion.md
│
├── config/
│   ├── base.yaml                   # every default, in one place
│   ├── development.yaml            # base + development period
│   ├── holdout.yaml                # base + holdout period
│   ├── paper.yaml                  # M5
│   ├── live.yaml                   # M5
│   ├── universe_candidates.txt     # append-only; includes delisted symbols
│   └── clusters.yaml
│
├── src/
│   └── scout/
│       ├── __init__.py
│       ├── domain/                 # dataclasses, enums, Protocols. No deps.
│       │   ├── enums.py
│       │   ├── market.py           # Asset, Bar, MarketPanel
│       │   ├── features.py         # FeatureRow, FeaturePanel
│       │   ├── universe.py
│       │   ├── setup.py
│       │   ├── edge.py             # BinKey, BinStats, EdgeTable
│       │   ├── costs.py
│       │   ├── opportunity.py
│       │   ├── portfolio.py        # Position, PortfolioState, TradeDecision
│       │   ├── execution.py        # OrderIntent, Fill
│       │   ├── sentiment.py
│       │   ├── audit.py            # DecisionRecord
│       │   ├── results.py          # ClosedTrade, BacktestResult
│       │   └── ports.py            # the five Protocols
│       │
│       ├── config/
│       │   ├── schema.py           # Pydantic models
│       │   ├── loader.py           # YAML + env overrides + resolution
│       │   └── hashing.py
│       │
│       ├── data/
│       │   ├── parquet_source.py   # ParquetCandleSource
│       │   ├── norgate_source.py   # ingest only
│       │   ├── sharadar_source.py  # ingest only (alt vendor)
│       │   ├── calendar.py         # XNYS via exchange_calendars
│       │   ├── actions.py          # splits, dividends, spinoffs — causal adjust
│       │   ├── quality.py          # gap / suspect-session detection
│       │   └── store.py            # atomic parquet read/write
│       │
│       ├── universe/
│       │   ├── build.py            # snapshot construction
│       │   ├── eligibility.py      # the ordered rules
│       │   └── spread.py           # Corwin-Schultz
│       │
│       ├── features/
│       │   ├── indicators.py       # every formula, standalone pure functions
│       │   ├── regime.py           # classify_regime (per-symbol)
│       │   ├── market.py           # classify_market_regime (SPY)
│       │   ├── cross_sectional.py  # ranks within the eligible set at t
│       │   └── engine.py           # compute_features
│       │
│       ├── gates/
│       │   └── eligibility.py      # evaluate_gates (14-CONFIG §5 rows 1–8)
│       │
│       ├── strategies/
│       │   ├── xsec_momentum.py    # primary
│       │   ├── donchian_breakout.py
│       │   └── registry.py
│       │
│       ├── scoring/
│       │   ├── labeling.py         # triple barrier
│       │   ├── edge.py             # bins, BinStats, LCB, EdgeTable build
│       │   └── rank.py             # build_opportunity, rank
│       │
│       ├── costs/
│       │   └── model.py            # estimate_cost. One implementation.
│       │
│       ├── portfolio/
│       │   ├── selection.py        # select_and_size
│       │   ├── sizing.py
│       │   └── breakers.py         # kill switch, circuit breakers
│       │
│       ├── sentiment/
│       │   ├── null_source.py
│       │   ├── parquet_source.py
│       │   ├── vix_term.py         # M2
│       │   ├── short_interest.py   # M2
│       │   ├── aggregate.py        # build_view
│       │   └── multiplier.py       # sentiment_multiplier
│       │
│       ├── backtest/
│       │   ├── engine.py           # BacktestEngine
│       │   ├── sim_broker.py
│       │   └── ledger.py
│       │
│       ├── execution/              # M5
│       │   ├── paper_broker.py
│       │   ├── live_broker.py
│       │   ├── ibkr_client.py      # or alpaca_client.py — pick one at M5
│       │   └── reconciliation.py
│       │
│       ├── storage/
│       │   ├── decision_sink.py    # ParquetDecisionSink
│       │   └── run_outputs.py      # write_run_outputs
│       │
│       ├── research/
│       │   ├── metrics.py
│       │   ├── splits.py           # warm-up / dev / holdout constants
│       │   ├── stability.py
│       │   ├── robustness.py
│       │   ├── diagnostics.py      # calibration, bin tables
│       │   ├── plots.py
│       │   ├── registry.py         # trial registry
│       │   └── lockbox.py          # holdout budget
│       │
│       ├── utils/
│       │   ├── clock.py            # THE only wall-clock module
│       │   ├── logging.py          # JSON formatter
│       │   ├── errors.py
│       │   ├── decimals.py         # rounding helpers, tick/step
│       │   └── stats.py            # bootstrap, wilson, welch
│       │
│       └── cli/
│           ├── __main__.py         # `scout` entry point, subcommand dispatch
│           ├── ingest.py
│           ├── adjust.py           # causal split/dividend adjust
│           ├── build_universe.py
│           ├── label_setups.py
│           ├── build_edge_table.py
│           ├── run_backtest.py
│           ├── robustness.py
│           └── report.py
│
├── tests/                          # see 16-TESTING.md
├── experiments/
│   ├── registry.csv                # COMMITTED trial log
│   └── holdout_lockbox.json        # COMMITTED peek budget
├── scripts/
│   └── setup_dev.ps1               # Windows dev bootstrap
│
├── data/                           # gitignored
├── results/                        # gitignored
└── logs/                           # gitignored
```

---

## 2. Deviations from the global structure rule, and why

The global convention is `src/<project_name>/` with `strategies/`, `data/`,
`execution/`, `backtest/`, `risk/`, `metrics/`, `plotting/`, `utils/`. This layout
follows it, with these differences — all restated in `AGENTS.md` per the override
requirement:

| Difference | Reason |
|---|---|
| Package is `scout`, not `trading_system` | Short imports. `from scout.scoring.rank import rank` beats `from trading_system.scoring.rank import rank` on every line of the codebase, and typo rate matters when cheaper models write the code. |
| `risk/` merged into `portfolio/` | Sizing and risk caps operate on the same state and are called in one pass. Splitting them creates a circular dependency or a pointless indirection. |
| `metrics/` and `plotting/` merged into `research/` | Both exist only to serve research output. |
| Added `universe/`, `gates/`, `scoring/` | Point-in-time universe construction, eligibility gating, and edge estimation are first-class in this design and have no home in the default layout. |
| No `notebooks/` | Research is reproducible CLI runs writing to `results/`. A notebook cannot be replayed and its outputs cannot be trusted six weeks later. Add one for ad-hoc exploration if you like, but nothing may import from it and no result may originate there. |
| Added `experiments/` at repo root | Holds the two committed integrity artifacts (registry, lockbox). Keeping them out of `results/` is deliberate: `results/` is gitignored and these must be committed. |
| Added `docs/results/` | Committed research write-ups, including negative results. |
| No `backend/` | The previous project's `backend/` directory should be deleted (M0). There is no frontend, so there is no backend. |

---

## 3. `pyproject.toml`

```toml
[project]
name = "scout"
version = "0.1.0"
description = "Multi-asset trading opportunity engine"
requires-python = ">=3.11"
dependencies = [
    "pandas==2.2.3",
    "numpy==2.1.3",
    "pyarrow==18.1.0",
    "duckdb==1.1.3",
    "pydantic==2.10.3",
    "PyYAML==6.0.2",
    "matplotlib==3.9.3",
    "scipy==1.14.1",
    "httpx==0.28.1",
    "exchange-calendars==4.13.2",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.4",
    "pytest-cov==6.0.0",
    "hypothesis==6.122.3",
    "ruff==0.8.4",
    "mypy==1.13.0",
]

[project.scripts]
scout = "scout.cli.__main__:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF", "DTZ", "PD"]
# DTZ: flake8-datetimez -- catches naive datetimes. Non-negotiable here.
# PD:  pandas-vet -- catches inplace=True and other footguns.

[tool.mypy]
python_version = "3.11"
strict = true
plugins = []
[[tool.mypy.overrides]]
module = ["scout.research.*", "scout.cli.*"]
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
filterwarnings = ["error::FutureWarning"]
```

Notes:

- **Exact pins, not ranges.** A pandas minor release has changed `resample`
  boundary behaviour, which silently shifts every bar by one period and
  invalidates every stored edge statistic. Determinism requires exact versions.
- **`DTZ` in ruff** catches naive datetimes at lint time, which is cheaper than
  catching them in a backtest.
- **`filterwarnings = error::FutureWarning`** turns pandas deprecations into
  failures. A `FutureWarning` about `resample` semantics is exactly the warning
  you cannot afford to ignore.
- **mypy strict everywhere except `research/` and `cli/`.** The decision path
  must be fully typed; plotting code need not be.
- **No TA-Lib, no `pandas_ta`, no `vectorbt`, no `backtrader`.** Reasons in
  [`05-FEATURES_AND_REGIME.md §1`](05-FEATURES_AND_REGIME.md#1-rules) and
  [`17-IMPLEMENTER_GUIDE.md`](17-IMPLEMENTER_GUIDE.md).

---

## 4. CLI surface

```text
scout ingest          --config <cfg> [--symbols-file F] [--start D] [--end D]
scout adjust          --config <cfg>
scout build-universe  --config <cfg>
scout label           --config <cfg> [--strategy ID]
scout build-edge      --config <cfg>
scout backtest        --config <cfg> [--notes TEXT] [--force-holdout]
scout robustness      --config <cfg> [--suite NAME]
scout report          --run-id ID
```

Full pipeline from nothing:

```powershell
scout ingest         --config config/development.yaml
scout adjust         --config config/development.yaml
scout build-universe --config config/development.yaml
scout label          --config config/development.yaml
scout build-edge     --config config/development.yaml
scout backtest       --config config/development.yaml --notes "baseline"
```

Steps 1–2 run once. Steps 3–5 rerun when config, features, or strategies change.
Step 6 is the iteration loop.

Every command: `--config` required, prints a resolved-config summary and the
`config_hash` on startup, exits non-zero on any failure. No interactive prompts —
these must run unattended.

---

## 5. Windows notes

The development machine is Windows. Choices made accordingly:

- **No TA-Lib.** Its C build on Windows is a reliable half-day of pain.
- **`pathlib.Path` everywhere.** No string path concatenation, no hardcoded `/`.
- **UTF-8 explicit** on every text read and write. Windows defaults to cp1252 and
  a symbol name or log message with a non-ASCII character will crash a run at
  hour three.
- **Atomic writes** via `tempfile` in the target directory plus `os.replace`.
  Windows will not overwrite an open file, so `os.replace` onto a path another
  process has open fails — close before replacing.
- **`scripts/setup_dev.ps1`** creates `.venv` at the repo root, installs
  `-e .[dev]`, and runs the test suite.
- The venv lives at repo root, not in `backend/`.

---

## 6. Where a new thing goes

| Adding | Location | Also update |
|---|---|---|
| An indicator | `features/indicators.py` | `FeatureRow`, `compute_features`, `test_features.py` |
| A strategy | `strategies/<name>.py` | `registry.py`, `config/base.yaml`, `test_strategies.py`, then relabel |
| A rejection reason | `domain/enums.py` | wherever it is raised, plus a test producing it |
| A cost component | `costs/model.py` | `CostEstimate`, `CostConfig`, `test_costs.py` |
| A portfolio cap | `portfolio/selection.py` | `PortfolioConfig`, a `RejectionReason`, `test_selection.py` |
| A sentiment source | `sentiment/<name>_source.py` | `sentiment.sources` config, `test_sentiment.py` |
| A metric | `research/metrics.py` | `test_metrics.py` with a hand-computed value |
| A plot | `research/plots.py` | `test_run_outputs.py` file list |
| A CLI command | `cli/<name>.py` | `cli/__main__.py` dispatch, a smoke test |
| An exchange | `execution/<name>_client.py` | nothing in the decision path — that is the point |

If a change does not fit any row, the design did not anticipate it. Write an ADR
before writing the code.
