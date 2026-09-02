# Testing

> Module: `tests/`. Every test listed here is required. An implementer's task is
> not done until its listed tests exist and pass.

---

## 1. Layout

```text
tests/
├── conftest.py                 # shared fixtures
├── fixtures/
│   ├── candles.py              # synthetic deterministic bar generators
│   ├── features.py             # hand-built FeatureRow factories
│   └── golden/                 # small CSVs with hand-computed expected values
├── unit/                       # mirrors src/scout/, fast, no I/O
│   ├── test_layering.py
│   ├── test_no_lookahead.py
│   ├── test_no_wallclock.py
│   ├── test_ports.py
│   ├── test_features.py
│   ├── test_regime.py
│   ├── test_strategies.py
│   ├── test_labeling.py
│   ├── test_edge.py
│   ├── test_costs.py
│   ├── test_sizing.py
│   ├── test_selection.py
│   ├── test_sentiment.py
│   ├── test_sentiment_pit.py
│   ├── test_universe.py
│   ├── test_gates.py
│   ├── test_calendar.py
│   ├── test_actions.py
│   ├── test_delisting.py
│   ├── test_ledger.py
│   ├── test_sim_broker.py
│   ├── test_engine_structure.py
│   ├── test_metrics.py
│   └── test_config.py
└── integration/
    ├── test_pipeline_smoke.py
    ├── test_determinism.py
    ├── test_synthetic_edge.py
    ├── test_synthetic_no_edge.py
    └── test_run_outputs.py
```

Unit tests must run in under 30 seconds total. Slow tests get skipped, and
skipped tests do not catch anything.

---

## 2. The five structural tests

These four-to-fifteen-line tests catch more real bugs than the rest combined,
because each one guards an entire class of error rather than one function.

### `test_layering.py`

Walks the AST of every module in `src/scout/`, extracts imports, and asserts the
dependency direction from
[`01-ARCHITECTURE.md §4`](01-ARCHITECTURE.md#4-dependency-direction).

```python
FORBIDDEN = {
    "scout.domain": {"scout.data", "scout.features", "scout.strategies",
                     "scout.scoring", "scout.portfolio", "scout.execution",
                     "scout.backtest", "scout.research", "scout.config"},
    "scout.strategies": {"scout.costs", "scout.portfolio", "scout.execution",
                         "scout.sentiment", "scout.data", "scout.backtest"},
    "scout.features": {"scout.strategies", "scout.scoring", "scout.portfolio"},
    "scout.scoring": {"scout.portfolio", "scout.execution"},
}
```

Also asserts that `httpx` and any exchange SDK are imported only inside
`scout.execution` and `scout.data.ingest`.

### `test_no_lookahead.py`

Greps the source of `features/`, `scoring/`, and `strategies/` for forbidden
patterns:

```python
FORBIDDEN_PATTERNS = [
    r"\.shift\(\s*-",          # negative shift
    r"\.bfill\(",
    r'fillna\([^)]*method\s*=\s*["\']bfill',
    r"\.interpolate\(",
    r"center\s*=\s*True",
    r"\.iloc\[::-1\]",
    r"limit_direction\s*=\s*['\"]backward",
]
```

Crude, and it will occasionally need an explicit `# noqa: lookahead` with a
justification comment. That friction is the feature: it forces a reviewer to see
and approve every use.

### `test_no_wallclock.py`

Greps all of `src/scout/` for `datetime.now`, `datetime.utcnow`, `time.time`,
`pd.Timestamp.now`, `date.today`. Allows only `src/scout/utils/clock.py`.

### `test_engine_structure.py`

Parses `BacktestEngine.run` and asserts its outermost `for` iterates over
timestamps, not symbols
([`11-BACKTEST_ENGINE.md §1`](11-BACKTEST_ENGINE.md#1-the-hard-constraint)).

### `test_ports.py`

For every concrete implementation, asserts `isinstance(obj, TheProtocol)`.
Catches signature drift without inheritance.

---

## 3. Golden-value tests

Every formula gets a hand-computed expected value. `pytest.approx` with
`rel=1e-9`.

### Features (`test_features.py`)

| Test | Assertion |
|---|---|
| `test_true_range_golden` | 5-bar fixture, hand-computed TR |
| `test_atr_wilder_golden` | 20-bar fixture; also asserts `adjust=False` recursion by comparing against a manual loop |
| `test_ema_golden` | 10-bar fixture |
| **`test_donchian_excludes_current_bar`** | On a monotonically rising series, `close > donchian_high` on **every** bar after warm-up. Without the mandatory `.shift(1)` it is true on **no** bar. This test is the tripwire for the single most common bug in breakout systems. |
| `test_efficiency_ratio_monotone_series` | Straight line ⇒ ER == 1.0 exactly |
| `test_efficiency_ratio_sawtooth` | Perfect zigzag returning to start ⇒ ER == 0.0 |
| `test_atr_percentile_bounds` | Result in `[0, 1]`, NaN before `min_periods` |
| `test_warmup_produces_nan_not_zero` | First `REQUIRED_WARMUP_BARS - 1` rows have NaN and `is_warm == False` |
| `test_feature_columns_match_dataclass` | `set(panel.columns) == {f.name for f in fields(FeatureRow)}` |
| `test_market_regime_broadcast_no_lost_rows` | Merge on `ts` keeps every decision row; warm rows have a market regime |
| `test_beta_spy_self_is_one` | SPY's own `beta_bench_90 == 1.0` |
| `test_cross_sectional_truncation` | Panel over `[start, end]` vs `[start, end-100]` is bit-identical on overlap |
| `test_symbol_order_invariance` | Features for `["A","B"]` equal features for `["B","A"]` after sorting |

### Regime (`test_regime.py`)

| Test | Assertion |
|---|---|
| `test_nan_gives_unknown` | Any NaN input ⇒ `UNKNOWN` |
| `test_strong_uptrend` | High ER-20 and ER-60, positive slope ⇒ `TREND_UP` |
| `test_strong_downtrend` | Mirror ⇒ `TREND_DOWN` |
| `test_low_er_low_vol_is_range` | ⇒ `RANGE` |
| `test_low_er_high_vol_is_chop` | ⇒ `CHOP`, not `RANGE` |
| `test_high_er_flat_slope_is_chop` | Efficient but directionless ⇒ `CHOP` |
| `test_classifier_is_pure` | 1000 random inputs, called twice, identical results |

### Costs (`test_costs.py`)

| Test | Assertion |
|---|---|
| `test_worked_example` | Reproduces [`08-COSTS.md §3`](08-COSTS.md#worked-example) to within 0.001 R |
| `test_cost_r_invariant_to_risk_fraction` | Halving `risk_fraction` changes `cost_r` by < 1% (impact term only) |
| `test_borrow_sign` | Short pays borrow; long pays zero |
| `test_dividend_sign` | Long receives; short pays |
| `test_impact_sqrt_law` | 4× notional ⇒ 2× impact bps |
| `test_spread_floor_applied` | A negative Corwin–Schultz estimate ⇒ the tier floor |
| `test_cost_multiplier_below_one_rejected` | `ScoutConfigError` |

### Labeling (`test_labeling.py`)

| Test | Assertion |
|---|---|
| `test_stop_hit_gives_minus_one_r` | Clean stop ⇒ `realised_r ≈ -1.0` |
| `test_target_hit_gives_rr` | Clean target ⇒ `realised_r ≈ reward_risk_ratio` |
| **`test_both_barriers_same_bar_gives_stop`** | The conservative tie rule |
| `test_timeout_exits_at_close` | ⇒ `TIME`, R computed from that close |
| `test_insufficient_forward_data_gives_open` | ⇒ `OPEN`, excluded from stats |
| **`test_stop_target_reanchored_on_entry`** | Gap up between decision and entry ⇒ labeled R/R equals intended R/R, not an inflated one |
| `test_mae_mfe_bounds` | `mae_r <= 0 <= mfe_r`; consistent with the outcome |
| `test_resolution_ts_is_exit_bar` | Exact equality |
| **`test_no_edge_series_gives_near_zero_mean_r`** | On a seeded geometric random walk, pooled `mean_r` is within the bootstrap CI of 0 and `win_rate ≈ 1/(1+rr)`. **This is the master sanity test.** If it fails, something leaks. |

### Edge (`test_edge.py`)

| Test | Assertion |
|---|---|
| **`test_only_resolved_before_as_of_included`** | A setup resolving after `as_of` is excluded. The causality test. |
| `test_open_setups_excluded` | `resolution_ts` null ⇒ excluded |
| `test_lcb_below_mean` | `ev_r_lcb < mean_r` whenever `z > 0` and `n` finite |
| `test_lcb_tightens_with_n` | Same mean and std, 10× n ⇒ `ev_r_lcb` closer to `mean_r` |
| `test_lcb_widens_with_std` | 2× std ⇒ lower `ev_r_lcb` |
| `test_insufficient_samples_unusable` | `n < min_bin_samples` ⇒ `is_usable == False` |
| `test_bootstrap_matches_normal_large_n` | At n=5000 with normal data, the two methods agree within 5% |
| `test_bootstrap_deterministic` | Fixed seed ⇒ identical result across calls |
| `test_as_of_lookup_rounds_backward` | 2023-04-17 resolves to the 2023-04-01 grid point |

### Portfolio (`test_sizing.py`, `test_selection.py`)

| Test | Assertion |
|---|---|
| `test_risk_based_size` | `qty * risk_per_unit ≈ equity * risk_fraction` before rounding |
| `test_step_rounding_always_down` | Never rounds up into extra risk |
| `test_notional_cap_binds_in_low_vol` | Tiny `atr_pct` ⇒ notional cap binds, not the risk formula |
| `test_adv_cap_binds` | Thin symbol ⇒ `max_pct_of_adv` binds |
| `test_multiplier_never_above_one` | Property test over random inputs |
| **`test_caps_checked_against_provisional_state`** | Three candidates that individually pass the heat cap but jointly breach it ⇒ only the first two accepted |
| `test_cluster_cap` | Three `L1` candidates ⇒ at most `max_positions_per_cluster` accepted |
| `test_beta_cap_credits_hedge` | A short with positive beta reduces `net_beta_exposure_pct` |
| `test_heat_taper` | Multiplier decreases monotonically past `heat_taper_start` |
| `test_every_opportunity_gets_a_decision` | `len(decisions) == len(ranked)` |
| `test_rejection_before_top_n_recorded` | Rank 4 with `top_n=3` ⇒ `BELOW_TOP_N` |
| `test_breaker_blocks_entries_not_exits` | Tripped breaker ⇒ no entries, `close_position` still works |
| `test_kill_switch_from_env` | `SCOUT_TRADING_ENABLED=0` overrides config |

### Sentiment (`test_sentiment.py`, `test_sentiment_pit.py`)

| Test | Assertion |
|---|---|
| **`test_filters_on_available_ts_not_event_ts`** | Observation with `event_ts = T`, `available_ts = T+6h`; a view at `T+1h` sees nothing |
| `test_aligned_sentiment_gives_no_bonus` | Positive score + LONG ⇒ multiplier exactly 1.0 |
| `test_opposed_sentiment_penalises` | Negative score + LONG ⇒ multiplier < 1.0 |
| `test_severe_opposition_vetoes` | severity ≥ `veto_threshold` ⇒ 0.0 |
| `test_stale_view_is_neutral` | ⇒ 1.0 |
| `test_no_observations_is_neutral` | ⇒ 1.0, no exception |
| `test_disabled_is_neutral` | ⇒ 1.0 |
| `test_decay_halves_at_half_life` | Weight at `half_life_hours` is half the weight at 0 |
| `test_multiplier_range` | Property test: always in `[0, 1]` |
| `test_null_source_returns_empty` | ⇒ `()` |

### Ledger and broker (`test_ledger.py`, `test_sim_broker.py`)

| Test | Assertion |
|---|---|
| `test_equity_identity` | `equity == cash + signed MTM`, after every operation |
| `test_decimal_throughout` | No `float` in any ledger field; asserted by type inspection |
| `test_round_trip_pnl` | Hand-computed net P&L including fees, dividends, and borrow |
| `test_entry_fills_next_session_open` | Never the decision bar's close; MOO of the next XNYS session |
| **`test_gap_through_stop_loses_more_than_one_r`** | Bar opens beyond the stop ⇒ `realised_r < -1.0` |
| `test_target_never_fills_better_than_target` | Bar opens beyond the target ⇒ fill exactly at the target |
| `test_stop_target_reanchored_on_entry` | Gap between decision close and fill ⇒ working stop/target shift with the fill; intended R unchanged |
| `test_exit_fill_has_no_extra_slippage` | Modelled slip is on the entry only; a stop at the stop level fills at the stop |
| `test_no_next_bar_no_entry` | ⇒ `DATA_GAP`, no position |
| `test_dividend_credits_long` | Ex-date cash credit on a long |
| `test_zero_equity_halts` | Raises `ScoutError` and marks the run failed |
| `test_apply_exit_forwards_regime_and_vol_bucket` | `ClosedTrade.regime` / `vol_bucket` come from `apply_exit` kwargs |

### Universe (`test_universe.py`, `test_delisting.py`)

| Test | Assertion |
|---|---|
| `test_snapshot_uses_only_trailing_data` | Eligibility at `t` is unchanged when bars after `t` are altered. The core PIT test. |
| `test_ineligible_rows_recorded` | Ineligible symbols appear with a reason |
| `test_snapshot_lookup_rounds_backward` | Wednesday resolves to Monday |
| `test_eligible_symbols_sorted` | Determinism |
| **`test_delisted_position_force_closed`** | Data ends mid-position ⇒ force-close, loss taken, trade in `trades.csv` |
| `test_delisted_symbol_not_reentered` | No new entry after the delisting |
| `test_min_price_uses_close_raw` | A reverse-split-adjusted close above $5 with `close_raw` below $5 is `LOW_PRICE` |

### Strategies (`test_strategies.py`)

| Test | Assertion |
|---|---|
| `test_xsec_momentum_satisfies_protocol` | `isinstance(XSecMomentum(), Strategy)` and no Protocol in `__mro__` |
| `test_donchian_breakout_satisfies_protocol` | Same for `DonchianBreakout` |
| `test_xsec_momentum_long_fires` | `mom_252_xs_pct >= 0.90` ⇒ LONG, `target_price is None`, stop = close − 5 ATR |
| `test_xsec_momentum_short_fires` | `mom_252_xs_pct <= 0.10` ⇒ SHORT, `target_price is None`, stop = close + 5 ATR |
| `test_xsec_momentum_mid_rank_does_not_fire` | Rank 0.50 ⇒ `None` |
| `test_donchian_long_fires` | Close through 55-session high + buffer, TREND_UP, ema spread ok ⇒ LONG with target at 2 R |
| `test_donchian_short_fires` | Mirror |
| `test_donchian_close_on_channel_does_not_fire` | Close equal to the buffered level ⇒ `None` |
| `test_from_params_rejects_unknown_key` | Misspelled param ⇒ `ScoutConfigError` |
| `test_build_strategies_unknown_id_raises` | Unknown `strategy_id` ⇒ `ScoutConfigError` |
| `test_no_range_fade_in_v1` | Registry keys are exactly the two v1 ids |

### Gates (`test_gates.py`)

| Test | Assertion |
|---|---|
| `test_not_in_universe_when_entry_missing` | `universe_entry is None` ⇒ `NOT_IN_UNIVERSE` |
| `test_ineligible_returns_snapshot_reason` | Ineligible entry ⇒ that entry's reason, not a later gate |
| `test_insufficient_history` | `is_warm=False` ⇒ `INSUFFICIENT_HISTORY` |
| `test_data_gap` | `bars_since_gap < min_bars_since_gap` ⇒ `DATA_GAP` |
| `test_stale_data` | `bar_age_bars > max_bar_staleness_bars` ⇒ `STALE_DATA` |
| `test_thin_cross_section` | `xs_population < min_xs_population` ⇒ `THIN_CROSS_SECTION` |
| **`test_earnings_in_window_blocks`** | Announcement in `(t, t+max_hold_bars]` ⇒ `EARNINGS_IN_WINDOW`; ETFs exempt even with no earnings row |
| `test_earnings_two_sessions_ahead_blocks` | Earnings in 2 sessions, `max_hold_bars=21` ⇒ blocked |
| `test_earnings_five_sessions_after_does_not_block` | Earnings 5 sessions ago ⇒ not blocked |
| `test_earnings_window_is_max_hold_not_two` | Earnings at `t+10`: hold 5 ⇒ pass; hold 21 ⇒ `EARNINGS_IN_WINDOW` |
| `test_missing_earnings_blocks_stock` | No earnings rows, `is_etf=False` ⇒ `EARNINGS_IN_WINDOW` |
| `test_hard_to_borrow_shorts_only` | Short + borrow above cap ⇒ `HARD_TO_BORROW`; long does not |
| `test_gate_order_stable` | Several failures at once ⇒ the earliest reason in §5 |

### Calendar and corporate actions (`test_calendar.py`, `test_actions.py`)

| Test | Assertion |
|---|---|
| `test_session_index_skips_holidays` | A Friday-to-Monday pair is 1 session, not 3 calendar days |
| `test_half_day_is_a_session` | Early close still increments `session_index` |
| `test_split_adjust_is_causal` | A 2:1 split at `t` does not change adjusted prices at `t-1` |
| `test_unadjusted_close_pinned` | `close_raw` is never altered by later splits |
| `test_yahoo_not_a_vendor` | Config `vendor: yahoo` raises `ScoutConfigError` |

### Config (`test_config.py`)

| Test | Assertion |
|---|---|
| `test_unknown_key_rejected` | Typo in YAML ⇒ `ScoutConfigError` |
| `test_defaults_complete` | An empty YAML plus defaults validates |
| `test_config_hash_stable` | Key reordering ⇒ same hash |
| `test_config_hash_changes_on_value_change` | |
| `test_warmup_at_least_six_months` | Violation ⇒ `ScoutConfigError` |
| `test_unmapped_symbol_warns` | Symbol absent from every cluster ⇒ warning naming it |
| `test_unknown_strategy_id_rejected` | |

---

## 4. Integration tests

### `test_pipeline_smoke.py`

3 synthetic symbols, 2,000 bars, both strategies, full pipeline. Asserts: the run
completes, `n_trades > 0`, all output files exist, `funnel.csv` sums to the
considered count, and the equity identity holds at the end.

### `test_determinism.py`

Run twice, assert byte-identical `metrics.json` and `trades.csv`.

### `test_synthetic_edge.py`

Generate bars with a **known, injected** trend-persistence edge: a Markov process
where an up-bar is followed by an up-bar with probability 0.62.

Assert the system **finds** it: `n_trades > 50`, `mean_r > 0`, and the bootstrap
CI lower bound is positive.

This is the positive control. Without it, a system that finds nothing is
indistinguishable from a system that is broken, and you would never know which
one you had.

### `test_synthetic_no_edge.py`

Generate a seeded geometric random walk with the same volatility. Assert the
system finds **nothing** exploitable: after costs, `mean_r` is within the
bootstrap CI of zero, and the CI lower bound is negative.

This is the negative control, and it is the more important of the two. A system
that reports an edge on a random walk has a leak, and this test finds it before
your money does. Run it with **at least 20 different seeds** — a single seed can
pass by luck.

### `test_run_outputs.py`

Asserts the exact file list from
[`11-BACKTEST_ENGINE.md §8`](11-BACKTEST_ENGINE.md#8-run-outputs) exists, that
every plot is at least 150 DPI, that `config.yaml` round-trips to the same
`config_hash`, and that `registry.csv` gained exactly one row.

---

## 5. Property-based tests

`hypothesis`, for invariants that must hold over ranges rather than at points.

```python
@given(
    equity=st.decimals(min_value=1000, max_value=10_000_000, places=2),
    atr_pct=st.floats(min_value=0.001, max_value=0.20),
    risk_frac=st.floats(min_value=0.001, max_value=0.02),
)
def test_sizing_never_exceeds_risk_mandate(equity, atr_pct, risk_frac):
    qty, risk = size_position(...)
    assert risk <= equity * Decimal(str(risk_frac)) * Decimal("1.0001")
```

Required property tests:

| Property | Applies to |
|---|---|
| Position risk never exceeds the mandate | `size_position` |
| Multiplier always in `[0, 1]` | every `*_multiplier` |
| `ev_r_lcb <= mean_r` | the LCB, for all `z >= 0` |
| `Setup.__post_init__` rejects a stop on the wrong side | `Setup` |
| Regime classifier total and pure | `classify_regime` |
| `open_risk_usd >= 0` for any mark | `Position` |
| Cost components non-negative except dividends (longs receive) | `estimate_cost` |

---

## 6. Runtime tripwires

Assertions that stay in **production** code, not just tests. Causality
failures raise `ScoutLookaheadError` and are never caught. Zero equity raises
`ScoutError` (not an `assert`: those are stripped under `python -O`).

| Location | Assertion |
|---|---|
| `features/engine.py` after the market-regime merge | no lost rows; warm rows have a market regime |
| `MarketPanel.as_of` | returned max `ts <= requested ts` |
| `scoring/edge.py` sample construction | `resolution_ts < as_of` for all rows |
| `sentiment/aggregate.py` | `available_ts <= ts` for all contributing observations |
| `backtest/sim_broker.py` entry | fill bar's `ts > decision ts` |
| `universe/build.py` | every bar used has `ts <= snapshot ts` |
| `backtest/ledger.py` mark | `equity == cash + signed MTM` (`ScoutLookaheadError`) |
| `backtest/ledger.py` mark | `equity > 0` (`ScoutError`, not an assert — those vanish under `-O`) |

Cost: microseconds per bar. Benefit: a lookahead bug crashes instead of producing
a profitable backtest you believe.

---

## 7. Coverage expectations

Not a percentage target — a per-module requirement, because 90% coverage
concentrated in the wrong places is worthless.

| Module | Requirement |
|---|---|
| `domain/` | Every validator and derived property |
| `features/` | Every indicator with a golden value |
| `scoring/` | Every branch of labeling; every LCB path |
| `costs/` | Every component; the worked example |
| `portfolio/` | Every cap; every rejection reason reachable |
| `backtest/` | Every fill rule; every corner case in §7 of that doc |
| `sentiment/` | Every multiplier branch; the PIT test |
| `research/` | Metrics against hand-computed values on a 20-point curve |
| `cli/` | Smoke test per command |

Rule of thumb: **every `RejectionReason` member must be produced by at least one
test.** An unreachable reason is either dead code or a gate that never fires, and
both are worth knowing about. A parametrised test iterating the enum and
asserting each is covered by the suite enforces it.
