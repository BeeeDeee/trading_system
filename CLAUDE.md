# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`qlab`: a local research framework for honest, pre-registered evaluation of long-only systematic
strategies on daily US stock/ETF data (Sharadar snapshot `sharadar_2026-09-25`). The project is a
series of numbered studies (research 1–7, all concluded; see the README results table). Most
results are negative, and the framework's job is to make that verdict trustworthy.

## Ground rules (from AGENTS.md; binding)

- `docs/PROJECT_SPECIFICATION.md` is the source of truth for research 1 methodology. Methodology
  changes go through the spec first, with a row in its decision log (§18).
- **Documentation is in Czech; code, identifiers and commit messages are in English.**
- Never commit data, run outputs or secrets. `data/` and `runs/` are git-ignored, and so are
  `*.csv`, `*.parquet` and `*.sqlite`. Committed results are JSON under `docs/research*/results/`.
- Holdout protection: data after 2019-12-31 is guarded by the OOS vault (`qlab.validation.vault`).
  Do not read or evaluate on holdout data outside a logged final evaluation.
- Every change that affects strategy selection is a new trial in the trial registry.
- Collaboration style: blunt and practical, no sugar-coating.

## Commands

```bash
uv sync                                   # install (Python 3.12, uv-managed)
uv run pytest                             # full suite (~35 s; one test skips without data/)
uv run pytest tests/test_research5.py -k split        # single test
uv run pytest -p no:warnings -o addopts="" -s <test>  # see print output (addopts sets -q)
```

There is no linter config. Match the surrounding style: dense numpy/polars, type hints, and short
docstrings that state the point-in-time contract.

Pipeline scripts all take the snapshot id (`sharadar_2026-09-25`) and run from the repo root:
`raw_to_parquet → build_bars → audit_sharadar → build_universe → build_panel`, then per study
`run_grid/run_wfo/final_evaluation` (research 1), `r2_*`, `r3_*`, `r4_*`, `r5_build_panel →
r5_grid → r5_evaluate → r5_final validation|late`, `r6_evaluate`, `r7_evaluate`. The README has the full list.

## Architecture (the parts that span files)

**Data flow.** Sharadar zips → `data/parquet/<snap>/` → normalized `bars`
(`qlab.data.normalize`, schema in `qlab.data.schema`) → point-in-time universes (`qlab.universe`)
→ dense date×asset **Panel** (`qlab.data.panel`), saved as one memory-mapped `.npy` per matrix under
`data/derived/<snap>/panel_*/`. Extra matrices are stored as `extra_<name>.npy` and come back as a
dict from `load_panel`.

**Bars/Panel semantics that everything relies on:**
- Returns are total returns split into `ret_co` (prev close→open) and `ret_oc` (open→close). The
  TR index is `cumprod((1+ret_co)(1+ret_oc))`. Prices (`close_u`, …) are **unadjusted** and are used
  only for filters such as price ≥ 5 USD, because adjusted prices leak future splits.
- A delisted security gets one extra `DELISTED` row whose `ret_co` carries the terminal return
  (policy in `normalize.terminal_return`: acquisition = exact consideration, bankruptcy −100 %,
  performance −30 %, SPAC liquidation 0). Engines pay the position out at that open.
- `tradable` = a usable open exists. Orders on non-tradable days do not fill.

**Point-in-time contract.** Row *t* of any feature/signal may use data only up to the close of
*t*. Decisions are made after the close of *t* and executed at the open of *t+1*.
`qlab.validation.leakage.check_point_in_time` enforces this by truncation plus perturbation of
future rows. Use it in a test for every new signal, and for whole pipelines (see
`tests/test_research5.py::test_whole_pipeline_is_point_in_time`). Rebalance schedules use the
*first* trading day of a period, because "last day" needs tomorrow's date.

**Engines.**
- `engine.vector.simulate`: target weights → daily net returns (fast, NAV units).
- `engine.ledger.simulate_ledger`: USD, units, per-order fees. Written independently on purpose.
  The vector/ledger parity test (10⁻¹⁵) validates both.
- `research5.sim.simulate`: event-driven (size fixed at entry, hold until an exit event, ADV cap,
  optional cash that carries another asset's returns via `cash_ret`/`cash_ret_oc`). Parity with the
  vector engine is tested on a single trade.

**Validation stack (`qlab.validation`).** `registry.TrialRegistry` is an append-only SQLite store
(triggers block UPDATE/DELETE). Its totals feed the Deflated Sharpe (`stats.deflated_sharpe`).
`vault.Vault` holds the boundary, the methodology freeze (hash lock) and a one-time open per
snapshot bound to the git commit. `stats` also has PSR/DSR, PBO (CSCV), the stationary bootstrap and
SPA. Research 5 has its own vault (`runs/vault_r5/`) and registries (`runs/registry/research5..7.sqlite`).

**Research 5–7 layout.** `qlab.research5` holds STR-TF: `universe`, `signals`, `costs`, `sim`,
`strategy` (Config dataclass, feature cache, candidates, `run`, random benchmark, VIX gate, SPY
core), `setup` (stage loading with vault boundaries: `dev` ≤2014, `validation` ≤2019, `late` via
vault; `load_full` only for studies that declare all history seen) and `report`. Features are
cached as float32 memmaps in `data/derived/<snap>/r5_cache/<stage>/`, computed in 1000-column chunks.

## How a study is run (follow this for any new hypothesis)

1. Write `docs/researchN/PREREGISTRATION.md` (Czech) with fixed rules, criteria and a decision log.
   **Commit it before computing anything.**
2. Implement and test, then commit the code before the evaluation run.
3. Record every configuration in the study's registry before computing it. For smoke runs, redirect
   the registry to a scratch path (monkeypatch `TrialRegistry` / `setup.REGISTRY`), otherwise the
   trial count N used by the DSR gets inflated.
4. Run once. If a run crashes after registering, log the rerun in the registry and the decision log.
5. Write `REPORT.md`, add a row to the README results table, and append the verdict to the decision log.

The 2020–2026 period and all earlier history are **already consumed**. Any new hypothesis is post hoc
on this snapshot, and only a forward test on newer data can confirm it. The Sharadar subscription
ended 2026-09-27, so a new snapshot needs renewal (`scripts/download_sharadar.sh`, API key in
`~/.config/sharadar/api_key`).

## Environment constraints

The machine has ~3 GB RAM, 3 cores and limited disk (panels are 1–3 GB each). Load panels with
`mmap_mode="r"`, compute features by column chunks, and never materialize several float64
(T×N) matrices at once. DuckDB queries set `memory_limit='1200MB'`.
