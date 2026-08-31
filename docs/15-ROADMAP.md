# Roadmap

> Milestones M0–M7 with per-task acceptance criteria.
>
> **For implementers:** pick the lowest-numbered unfinished task. Do only that
> task. A task is done when every acceptance criterion is objectively verifiable
> and its listed tests pass. Do not start a later task because an earlier one
> looks boring.

Estimates assume one focused developer plus AI assistance. They are calibration,
not commitments.

---

## Status

| Milestone | Scope | Estimate | Status |
|---|---|---|---|
| M0 | Repo skeleton, config, domain model | 3 days | Done |
| M1 | Equity data, calendar, features, PIT universe | 7 days | Done |
| M2 | Strategies, labeling, edge table | 4 days | Not started |
| M3 | Backtest engine, metrics, **the answer** | 6 days | Not started |
| M4 | Robustness, parameters, sentiment, paper prep | 10 days | Gated on M3 |
| M5 | Paper then live trading | 15 days | Gated on M4 + 60-day soak |
| M6 | ML evaluation | 10 days | Gated on M3 criteria |
| M7 | Optional crypto sleeve | — | Gated on an equity holdout result |

**M3 is the milestone that matters.** Everything before it is plumbing;
everything after it is conditional on its result. **Do not start crypto work
before M7 is unlocked.**

---

## M0 — Foundation

### M0.1 Repo cleanup — done
- Delete `backend/` entirely (empty `pyproject.toml`, empty `README.md`, and a
  stale `.venv`).
- Create `pyproject.toml` at repo root exactly as in
  [`13-PROJECT_LAYOUT.md §3`](13-PROJECT_LAYOUT.md#3-pyprojecttoml).
- Create `src/scout/` with the full package tree; every package has
  `__init__.py`, and no `__init__.py` contains logic or re-exports.
- `scripts/setup_dev.ps1` creates `.venv`, installs `-e .[dev]`, runs pytest.

**Accept:** `.\scripts\setup_dev.ps1` succeeds from a clean clone on Windows.
`python -c "import scout"` works. `ruff check .` and `mypy src/scout/domain`
pass. `backend/` does not exist.

### M0.2 Domain model — done
Implement all of [`02-DOMAIN_MODEL.md`](02-DOMAIN_MODEL.md): every enum,
dataclass, `MarketPanel`, `FeaturePanel`, `EdgeTable`. Validators in
`Setup.__post_init__` and the UTC assertions.

**Accept:** `tests/unit/test_layering.py` passes (proving `domain` imports
nothing from `scout` except `utils`). Every dataclass is
`frozen=True, slots=True`. `Setup` rejects a stop on the wrong side, a
non-positive risk, and a naive datetime. `MarketPanel.as_of` returns only rows
with `ts <= requested`, verified on a fixture. `mypy --strict src/scout/domain`
clean.

### M0.3 Config — done
Pydantic schema, loader with layered resolution, hashing, cross-field validation
from [`14-CONFIG.md §8`](14-CONFIG.md#8-cross-field-validation). `config/base.yaml`
with every default.

**Accept:** every test in `test_config.py`. An unknown YAML key raises with the
key path in the message. Key reordering does not change the hash. Loading
`config/base.yaml` produces a fully-populated `ScoutConfig`.

### M0.4 Utils — done
`clock.py` (`BarClock`, `WallClock`), `logging.py` (JSON formatter with `run_id`),
`errors.py`, `decimals.py` (tick/step rounding), `stats.py` (bootstrap CI,
Welch's t-test, deflated Sharpe).

**Accept:** `test_no_wallclock.py` passes. Bootstrap is deterministic under a
fixed seed. Step rounding always rounds down, verified by a property test.
`stats.deflated_sharpe` matches a hand-computed value from the Bailey–López de
Prado paper's example.

---

## M1 — Data and features

Paid vendor required. Yahoo is fixtures-only.
[`04-DATA_AND_UNIVERSE.md §1`](04-DATA_AND_UNIVERSE.md#1-data-source-this-decision-is-load-bearing).

### M1.1 Ingest — done
`data/norgate_source.py` or `data/sharadar_source.py`, `cli/ingest.py`. Unadjusted
OHLCV, actions table, earnings dates, sector, delisted included. Writes
`data/raw/equity/SNAPSHOT.json`.

**Accept:** ingests daily bars for ≥50 symbols including ≥5 delisted names over
2015–2016 into `data/raw/equity/...`. Re-running is idempotent. Killing the
process mid-write leaves no truncated file. `SNAPSHOT.json` is written last.
`cli/build_universe.py` later refuses to run if the delisted fraction of the
candidate list is below 15%.

### M1.2 Calendar, actions, and quality — done
`data/calendar.py` (`exchange_calendars`, XNYS), `data/actions.py` (causal
adjustment), `data/quality.py`. **No resample step** — v1 ingests daily bars
directly.

**Accept:** session grid matches NYSE holidays for 2008, 2012 (Hurricane Sandy),
and a known half-day. Adjustment of a 2-for-1 split on a fixture produces
unchanged *adjusted* series after the ex-date and unchanged *raw* series.
Dividends are applied on the ex-date, never the pay date. **Nothing is
forward-filled.** `close` (adjusted) and `close_raw` are both present on every
row.

### M1.3 `ParquetCandleSource` — done
**Accept:** satisfies the `CandleSource` Protocol. `load_panel` returns a
`(ts, asset_id)`-sorted frame, omits missing symbols without raising, and is
deterministic. Loading 1,000 symbols × 25 years of daily bars takes under 15
seconds and under 600 MB.

### M1.4 Indicators — done
`features/indicators.py`: every formula in
[`05-FEATURES_AND_REGIME.md`](05-FEATURES_AND_REGIME.md) for daily bars,
including cross-sectional ranks over the eligible set at `t`.

**Accept:** every golden test in `test_features.py`. Critically,
`test_donchian_excludes_current_bar` passes. Truncating the panel does not
change any cross-sectional rank at earlier `t`. NaN during warm-up, never zero.

### M1.5 Regime — done
`features/regime.py` (per-symbol) and `features/market.py` (SPY).

**Accept:** every test in `test_regime.py`. Pure and total. Straight line ⇒ ER of
exactly 1.0; perfect zigzag ⇒ exactly 0.0. SPY below its 200-day MA with a 20%
drawdown ⇒ `RISK_OFF`.

### M1.6 Feature engine — done
`features/engine.py`: `compute_features(panel, snapshot, benchmark, cfg)`.

**Accept:** column set equals `FeatureRow`'s field set exactly. SPY's own
`beta_bench_90` is 1.0. Symbol-order invariance. 1,000 symbols × 25 years in
under 90 seconds.

### M1.7 Universe — done
`universe/spread.py`, `universe/eligibility.py`, `universe/build.py`,
`cli/build_universe.py`. Liquidity rank, not index membership
([ADR-018](ADR/018-liquidity-rank-universe.md)).

**Accept:** every test in `test_universe.py`.
**`test_snapshot_uses_only_trailing_data` passes.** Ineligible rows recorded with
reasons. Candidate list delisted fraction ≥ 15%. Eligible count per year printed.
`adv_rank <= 1000` is the binding size constraint.

### M1.8 Gates — done
`gates/eligibility.py`, including the earnings blackout
([ADR-020](ADR/020-earnings-gate.md)).

**Accept:** each of 14-CONFIG §5 rows 1–8 produces its documented reason
(rows 9–10 are engine, M3). Order is stable. ETFs are exempt from the earnings
gate. A stock with earnings in 2 sessions is blocked (`2 <= max_hold_bars`);
the same stock 5 sessions after earnings is not (window is open on the left).
Announcement in `(t, t+max_hold_bars]` blocks; `t+max_hold_bars+1` does not.
A stock with no earnings row is blocked. `earnings_blackout_sessions` does not
exist — the window is the strategy's `max_hold_bars`.

---

## M2 — Strategies and edge

### M2.1 The two strategies
`strategies/xsec_momentum.py`, `strategies/donchian_breakout.py`, `registry.py`.
**No `range_fade` in v1.**

**Accept:** both satisfy the `Strategy` Protocol. Momentum may return
`target_price=None`. Neither imports `costs`, `portfolio`, `execution`,
`sentiment`, or `data`. Each has at least one fixture where it fires and one
where it must not. An unknown `strategy_id` raises `ScoutConfigError` at config
load.

### M2.2 Labeling
`scoring/labeling.py`, `cli/label_setups.py`.

**Accept:** every test in `test_labeling.py`. **`test_both_barriers_same_bar_gives_stop`**
and **`test_stop_target_reanchored_on_entry`** pass. Produces
`data/labels/setups_<id>.parquet` with the full §3 schema.

**Then run the master sanity check and record the result in
`docs/results/m2_labeling_sanity.md`:** pooled `mean(realised_r_gross)` on a
seeded geometric random walk of the same length and volatility must be within the
bootstrap CI of zero, and `win_rate` must be near `1/(1+rr)`. **If pooled
`mean_r` on real data exceeds about 0.4 R, stop and find the bug** — the likely
culprits are the missing Donchian shift, the tie rule, and stop/target
re-anchoring, in that order. Do not proceed to M2.3 until this is recorded.

### M2.3 Edge table
`scoring/edge.py`, `cli/build_edge_table.py`. Bins, `BinStats`, both LCB methods,
the monthly `as_of` grid, `EdgeTable` with backward lookup.

**Accept:** every test in `test_edge.py`. **`test_only_resolved_before_as_of_included`
is the critical one.** Bootstrap is deterministic. Lookup is O(log n), verified by
timing 100,000 lookups in under one second. Table saves and loads with its
`config_hash` in the metadata.

### M2.4 Costs
`costs/model.py`.

**Accept:** every test in `test_costs.py`. Reproduces the large-cap long worked
example in [`08-COSTS.md §3`](08-COSTS.md#3-worked-examples) to within 0.001 R.
Impact follows the square-root law. `cost_multiplier < 1.0` is rejected. Shorts
accrue borrow; longs do not.

### M2.5 Ranking
`scoring/rank.py`.

**Accept:** `build_opportunity` populates every field. `rank` filters by
`min_ev_net_r`, sorts by `ev_per_bar_r` descending, and breaks ties
alphabetically. Verified deterministic under shuffled input order.

---

## M3 — Backtest and the answer

### M3.1 SimBroker and ledger
`backtest/sim_broker.py`, `backtest/ledger.py`.

**Accept:** every test in `test_sim_broker.py` and `test_ledger.py`. The equity
identity holds after every operation. All ledger fields are `Decimal`.
**`test_gap_through_stop_loses_more_than_one_r` passes.** Targets never fill
better than the target. Zero equity halts the run.

### M3.2 Portfolio
`portfolio/sizing.py`, `portfolio/selection.py`, `portfolio/breakers.py`.

**Accept:** every test in `test_sizing.py` and `test_selection.py`.
**`test_caps_checked_against_provisional_state` passes.** Every input
`Opportunity` yields a `TradeDecision`. Every multiplier is in `(0, 1]`, property
tested. Breakers block entries but never exits. `SCOUT_TRADING_ENABLED=0` blocks
all entries.

### M3.3 Sentiment plumbing
`sentiment/null_source.py`, `aggregate.py`, `multiplier.py`. **No real sources.**

**Accept:** `NullSentimentSource` returns `()`. `build_view` on an empty
observation list yields `is_neutral == True`. The multiplier is exactly 1.0 when
disabled, neutral, or aligned. Every test in `test_sentiment.py` **and**
`test_sentiment_pit.py`.
**`test_sentiment_pit.py::test_filters_on_available_ts_not_event_ts` passes** even
though no real source exists yet — the plumbing must be correct before data arrives, because retrofitting
point-in-time discipline never happens.

### M3.4 Engine
`backtest/engine.py`, `storage/decision_sink.py`, `storage/run_outputs.py`,
`cli/run_backtest.py`.

**Accept:** **`test_engine_structure.py` passes** (timestamp-outer loop).
`test_pipeline_smoke.py` and `test_determinism.py` pass. Exits are applied before
entries. Every corner case in
[`11-BACKTEST_ENGINE.md §7`](11-BACKTEST_ENGINE.md#7-corner-cases) has a test.
All output files from §8 are produced. `registry.csv` gains exactly one row per
run. 6,300 sessions × 1,000 symbols completes in under 3 minutes.

### M3.5 Metrics, plots, diagnostics
`research/metrics.py`, `plots.py`, `diagnostics.py`, `stability.py`,
`registry.py`, `lockbox.py`.

**Accept:** every metric in
[`12-RESEARCH_PROTOCOL.md §4`](12-RESEARCH_PROTOCOL.md#4-metrics), each tested
against a hand-computed value on a 20-point curve. All six plots at ≥150 DPI with
timezone-aware axes. `deflated_sharpe` reads the trial count from the registry.
**The lockbox refuses a fourth holdout evaluation and refuses any holdout run
from a dirty git tree.**

### M3.6 Development run and iteration
Run the full pipeline on `config/development.yaml`. Follow the inspection order in
[`12-RESEARCH_PROTOCOL.md §7`](12-RESEARCH_PROTOCOL.md#7-required-workflow):
funnel, then trade count, then cost drag, then calibration, then stability, and
only then Sharpe.

**Accept:** written to `docs/results/m3_development.md`: the funnel table, trade
count, `cost_drag_pct`, the calibration plot with commentary, the per-year and
per-regime stability tables, the SPY-beta regression, and the momentum-crash
windows. Every anomaly explained
or filed as a bug. Every run in the registry.

**Do not tune toward a target here.** Fix bugs, explain anomalies. If the funnel
shows a gate rejecting 99.9% of candidates, that is a bug. If calibration is flat,
that is the answer, not a tuning problem.

### M3.7 The holdout run — **lockbox evaluation 1 of 3**

Preconditions, all mandatory:
- M3.6 complete and written up.
- Historical borrow/dividend inputs ingested (`costs.dividend_yield_source: historical`).
- `edge.lcb_method: bootstrap`.
- Git tree clean and committed.
- Sentiment disabled.
- `asset_class: equity`.

Run `config/holdout.yaml` once. Evaluate against the go/no-go criteria in
[`12-RESEARCH_PROTOCOL.md §9`](12-RESEARCH_PROTOCOL.md#9-go--no-go-criteria-for-m3).

**Accept:** `docs/results/m3_holdout.md` committed, containing the full metrics
table, all nine criteria with pass/fail, the plots, and an explicit **PROCEED** or
**STOP** decision.

**If criterion 2, 3, or 4 fails: STOP.** Do not proceed to M4. Change the
hypothesis, per
[`12-RESEARCH_PROTOCOL.md §9`](12-RESEARCH_PROTOCOL.md#if-2-3-or-4-fail). Adding
sentiment or ML to rescue a failed baseline manufactures a fake edge, and doing
so is the single worst available action at this point.

---

## M4 — Robustness and paper preparation

**Gated on M3.7 = PROCEED.**

### M4.1 Robustness suite
`research/robustness.py`, `cli/robustness.py`. All six suites from
[`12-RESEARCH_PROTOCOL.md §6`](12-RESEARCH_PROTOCOL.md#6-robustness-suite).

**Accept:** `docs/results/m4_robustness.md` with every table. Parameter surfaces
plotted. Cost sensitivity at 1.0×/1.5×/2.0×/3.0×. Tie-rule gap reported.
Universe bootstrap over 20 seeds. **All on development data only.**

### M4.2 Parameter finalisation
Choose final parameters from the **plateau centres** in M4.1, not the peaks. Write
down the reasoning per parameter.

**Accept:** `docs/results/m4_parameters.md` justifying each value and naming any
parameter whose surface was flat (candidate for deletion).

### M4.3 Real sentiment sources
`vix_term` first (broadcast), then `short_interest` if the vendor provides it.
**Not** crypto sources.

**Accept:** `observations.parquet` populated, append-only enforced.
`available_ts >= event_ts` on every row. `test_sentiment_pit.py` passes against
real data. Ingest is resumable and idempotent.

### M4.4 Sentiment promotion — **lockbox evaluation 2 or 3**
The paired test from
[`10-SENTIMENT.md §6`](10-SENTIMENT.md#6-the-promotion-gate).

**Accept:** `docs/results/sentiment_promotion.md` with all four criteria and a
PROMOTE or REJECT decision. **A negative result is a successful outcome** — record
it and set `enabled: false`.

### M4.5 Optional strategy or exit-rule work
At most **one** of: a third strategy, a trailing stop, or a second timeframe.
Development data only, registered trials.

**Accept:** hypothesis written before implementation. Development result recorded.
Only promoted if the improvement is large enough to justify spending the last
lockbox evaluation.

---

## M5 — Paper then live

**Gated on M4 complete and a final holdout PROCEED.**

### M5.1 Live infrastructure
Now, and not before, the deferred infrastructure arrives:
`execution/ibkr_client.py` (or Alpaca), `paper_broker.py`, `live_broker.py`,
`reconciliation.py`, PostgreSQL for order state, FastAPI for inspection, Docker
for the VPS, n8n for reports and alerts.

**Accept:** `PaperBroker` satisfies the `Broker` Protocol. Idempotency verified by
double-submitting the same `client_order_id` and confirming one position.
Protective stops are broker-native (or simulated if the broker cannot park a stop
on a MOO entry — documented either way). **If the stop cannot be placed, the
entry is closed immediately.** Reconciliation recovers full state after a process
kill. `SCOUT_TRADING_ENABLED=0` blocks sends within one cycle.

### M5.2 Paper soak — 60 days minimum
Run paper trading on the live universe. Weekly: compare paper decisions against a
backtest of the same period.

**Accept:** ≥60 calendar days. ≥30 closed trades. `docs/results/m5_paper.md`
containing:
- Paper versus backtest divergence, explained.
- **Realised versus modelled cost per fill.** If realised exceeds modelled,
  update the cost model and **re-run the M3 holdout** — an optimistic cost model
  invalidates the go-live decision, not just the cost model.
- Zero unprotected positions, ever.
- Zero duplicate fills.
- Reconciliation correct after at least one deliberate restart.

### M5.3 Go-live
Follow the `go-live-checklist` skill. Start at **10% of intended size** for the
first 30 days.

**Accept:** kill switch tested in production. Alerting on breaker trips, stale
data, and reconciliation mismatches. Rollback procedure documented and rehearsed.
Daily reconciliation report. Size increase only after 30 clean days.

---

## M6 — ML evaluation

**Gated on the M3 criteria in
[`07-EDGE_AND_SCORING.md §11`](07-EDGE_AND_SCORING.md#11-when-ml-is-allowed-to-replace-the-estimator).**
If the baseline has no edge, ML will find the same absence plus an overfit, and
it will take three months.

### M6.1 `EdgeEstimator` Protocol
Refactor `BinnedEmpiricalEstimator` behind the Protocol. **No behaviour change.**

**Accept:** byte-identical results before and after the refactor, verified by
`test_determinism.py` against a stored baseline `metrics.json`.

### M6.2 Gradient-boosted estimator
Predict realised R from features. Purged, embargoed walk-forward
cross-validation — a plain time-series split leaks through overlapping label
windows, because a setup detected at `t` resolves up to `max_hold_bars` later.

**Accept:** out-of-fold predictions only. The embargo is at least
`max_hold_bars`. `ev_r_lcb` is an honest out-of-fold prediction interval, not a
training-set residual. Calibration plot at least as monotone as the baseline's.

### M6.3 Paired comparison — **lockbox evaluation 3 of 3**
**Accept:** `docs/results/m6_ml.md` with a paired bootstrap CI on Δ mean R. Adopt
only if the CI excludes zero **and** the calibration is no worse. Otherwise
record the negative result and keep the baseline.

---

## M7 — Optional crypto sleeve

**Gated on a completed equity holdout write-up**, whether PROCEED or STOP.
Crypto is diversification, not a rescue.

Reuse the same engine, scoring, portfolio, and research protocol. Differences
are documented in the M7 appendices of `04`, `05`, `06`, and `08`, and in
ADR-010. New ingest adapter only. Do **not** reimplement the decision path.

**Accept:** a separate `asset_class: crypto` config runs end-to-end on fixtures.
A holdout crypto run consumes lockbox budget like any other holdout. If the
equity holdout already used the three evaluations, crypto holdout is refused
unless the budget is explicitly extended by a committed ADR — which it should
not be.

---

## Dependency graph

```text
M0.1 ─► M0.2 ─► M0.3 ─► M0.4
                 │
                 ▼
M1.1 ─► M1.2 ─► M1.3 ─► M1.4 ─► M1.5 ─► M1.6 ─► M1.7 ─► M1.8
                                                 │
                                                 ▼
                          M2.1 ─► M2.2 ─► M2.3 ─► M2.5
                                    │       ▲
                                    │     M2.4
                                    ▼
                          M3.1 ─► M3.2 ─► M3.3 ─► M3.4 ─► M3.5 ─► M3.6 ─► M3.7
                                                                            │
                                                                    ┌───────┴───────┐
                                                                 PROCEED          STOP
                                                                    │               │
                                                                   M4        change hypothesis
                                                                    │
                                                                   M5, M6
                                                                    │
                                                                   M7 (optional, after equity holdout)
```

Parallelisable: M2.4 (costs) alongside M2.1–M2.3. M3.5 (metrics) alongside
M3.1–M3.4. Everything else is sequential.

---

## What "done" means

A task is done when:

1. Every acceptance criterion is objectively verifiable — a passing test, a
   committed file, or a measured number.
2. Its listed tests exist and pass.
3. `ruff check .` and `mypy src/scout` are clean.
4. Nothing outside the task's stated scope was changed.

A task is **not** done because the code looks right, because a manual run
produced plausible output, or because the remaining criteria "are just tests".
The tests in this project are not verification of the implementation; several of
them are the only thing standing between you and a backtest that lies.
