# Configuration

> Complete schema, defaults, and validation rules. Module: `src/scout/config/`

---

## 1. Rules

1. **Typed, validated, hashed.** Pydantic v2 models. `extra="forbid"` on every
   model, so a typo in a YAML key is a startup error rather than a silently
   ignored setting. A misspelled `stop_atr` that defaults to 2.0 costs you a day
   of analysing the wrong run.
2. **One file with every default** (`config/base.yaml`). Environment files
   override it by deep merge. Defaults live in the Pydantic models too, and
   `test_config.py::test_defaults_complete` asserts the two agree.
3. **Secrets never in YAML.** API keys come from the environment, loaded from a
   gitignored `.env`. `config.yaml` written into a run directory is scrubbed of
   any field whose name contains `key`, `secret`, or `token`.
4. **The resolved config is hashed** (sha256 over the canonical JSON dump with
   sorted keys) and recorded on every run and every artifact. A backtest whose
   `config_hash` does not match its edge table's `config_hash` **fails at
   startup**.
5. **No config access below `cli/`.** Config objects are passed as parameters. A
   module that imports a global `settings` is untestable and hides its
   dependencies.

---

## 2. Layered resolution

```text
Pydantic model defaults
        ↓ overridden by
config/base.yaml
        ↓ overridden by
config/<environment>.yaml     (development | holdout | paper | live)
        ↓ overridden by
environment variables         SCOUT_<SECTION>__<FIELD>, double underscore = nesting
        ↓ overridden by
CLI flags                     only --notes, --force-holdout, --strategy
```

```python
def load_config(path: Path) -> ScoutConfig:
    """Deep-merge, validate, then run cross-field validation (§8).
    Raises ScoutConfigError with the offending key path on any failure."""
```

Env vars exist for two things only: secrets and `SCOUT_TRADING_ENABLED`. Putting
strategy parameters in env vars destroys reproducibility, because the env is not
captured in `config.yaml`. The loader logs a warning naming any env override it
applied, so a surprising result can be traced.

---

## 3. `config/base.yaml` — the complete default

```yaml
# ─── identity ────────────────────────────────────────────────────────────────
run:
  seed: 20260827
  strategy_slug: xsec-momentum-donchian  # used in run-id
  mode: BACKTEST                         # BACKTEST | PAPER | LIVE
  asset_class: equity                    # equity | crypto (crypto is M7 only)

# ─── period ──────────────────────────────────────────────────────────────────
# Canonical splits: 04-DATA_AND_UNIVERSE.md §8. Do not widen the holdout in YAML.
period:
  start: 1998-01-01T00:00:00Z
  warmup_end: 2004-01-01T00:00:00Z       # trading begins; >= start + 6 months
  end: 2017-12-31T23:59:59Z
  split: DEVELOPMENT                     # DEVELOPMENT | HOLDOUT | FORWARD

# ─── data ────────────────────────────────────────────────────────────────────
data:
  vendor: norgate                        # norgate | sharadar | polygon | fixture
  calendar: XNYS                         # exchange_calendars code
  decision_timeframe: 1d                 # sessions. The only v1 timeframe.
  raw_dir: data/raw/equity
  processed_dir: data/processed
  snapshot_id_path: data/raw/equity/SNAPSHOT.json
  max_bar_staleness_bars: 2              # sessions

# ─── universe ────────────────────────────────────────────────────────────────
universe:
  candidates_file: config/universe_candidates.txt
  clusters_file: config/clusters.yaml
  snapshot_frequency: monthly
  snapshots_path: data/universe/snapshots.parquet
  universe_size: 1000                    # top-N by trailing ADV rank
  min_history_bars: 400                  # sessions
  min_bars_since_gap: 10
  min_price_usd: 5.00                    # on close_raw, never adjusted close
  min_adv_usd: 5000000
  max_spread_bps: 15.0
  max_suspect_lookback: 5
  delisting_grace_bars: 3
  spread_window_bars: 30
  spread_floor_bps_by_tier:
    tier_500m: 0.5
    tier_100m: 1.0
    tier_20m: 3.0
    tier_below: 8.0

# ─── features ────────────────────────────────────────────────────────────────
features:
  atr_n: 14
  ema_fast: 20
  ema_slow: 50
  slope_lookback: 20
  donchian_n: 55                         # ~one quarter of sessions
  keltner_k: 2.0
  er_n: 20
  er_long_n: 60
  vol_window_bars: 504                   # ≈2y of sessions
  vol_min_periods: 126
  beta_window: 90
  beta_reference_symbol: SPY
  mom_skip_bars: 21
  mom_lookback_bars: 252

# ─── regime (per-symbol) ─────────────────────────────────────────────────────
regime:
  er_trend_min: 0.30
  er_long_trend_min: 0.20
  slope_min: 0.25
  er_range_max: 0.15
  vol_range_max: 0.67
  vol_low_pct: 0.33
  vol_high_pct: 0.67

# ─── market regime (SPY, once per timestamp) ─────────────────────────────────
market_regime:
  benchmark_symbol: SPY
  sma_n: 200
  dd_warn: 0.10
  dd_stress: 0.15
  vol_stress_pct: 0.80

# ─── gates (ORDER MATTERS — see §5) ──────────────────────────────────────────
gates:
  require_warm: true
  require_universe_eligible: true
  require_regime_allowed: true           # engine (M3), not evaluate_gates
  max_bar_staleness_bars: 2
  min_xs_population: 100
  skip_hard_to_borrow: true              # no historical borrow file in v1

# ─── strategies ──────────────────────────────────────────────────────────────
strategies:
  - strategy_id: xsec_momentum_v1
    enabled: true
    params:
      xs_threshold: 0.90
      exit_xs_threshold: 0.70
      stop_atr: 5.0
      max_hold_bars: 21

  - strategy_id: donchian_breakout_v1
    enabled: true
    params:
      entry_buffer_atr: 0.10
      stop_atr: 3.0
      target_rr: 2.0
      max_hold_bars: 40
      min_ema_spread_atr: 0.20

# ─── labeling ────────────────────────────────────────────────────────────────
labeling:
  tie_rule: stop                         # stop | target (target = robustness only)
  labels_dir: data/labels

# ─── edge ────────────────────────────────────────────────────────────────────
edge:
  bin_dimensions: [strategy_id, direction, vol_bucket]
  min_bin_samples: 100
  z: 1.28                                # one-sided 90%
  lcb_method: normal                     # normal | bootstrap
  bootstrap_iterations: 2000
  bootstrap_seed: 20260827
  as_of_grid: monthly
  table_path: data/edge/edge_table.parquet

# ─── scoring ─────────────────────────────────────────────────────────────────
scoring:
  min_ev_net_r: 0.05
  # There is no `rank_by` option. Ranking is always ev_per_bar_r; the threshold
  # is always on ev_net_r. See 07-EDGE_AND_SCORING.md §8.

# ─── costs ───────────────────────────────────────────────────────────────────
costs:
  commission_per_share_usd: 0.0035
  commission_min_usd: 0.35
  commission_max_pct_of_notional: 0.01
  slippage_fixed_bps: 1.0
  slippage_vol_coef: 0.02
  impact_coef: 1.0
  borrow_bps_per_year_default: 30.0
  hard_to_borrow_max_bps_per_year: 300.0
  dividend_yield_source: historical
  dividend_yield_default: 0.015
  cost_multiplier: 1.0                   # >= 1.0 enforced
  # crypto keys exist for M7 only; unused while asset_class=equity
  taker_fee_bps: 5.0
  maker_fee_bps: 2.0
  funding_source: default
  funding_rate_default_per_8h: 0.0001

# ─── portfolio ───────────────────────────────────────────────────────────────
portfolio:
  initial_equity_usd: 100000
  risk_fraction_per_trade: 0.004
  max_positions: 6
  max_positions_per_cluster: 2
  top_n: 3
  max_portfolio_heat_pct: 0.020
  max_cluster_risk_pct: 0.010
  max_net_beta_pct: 0.015
  max_gross_exposure_pct: 1.50
  max_position_notional_pct: 0.25
  max_pct_of_adv: 0.005
  heat_taper_start: 0.60
  heat_min_multiplier: 0.35
  liquidity_full_size_adv_usd: 200000000
  liquidity_min_multiplier: 0.50

# ─── risk / breakers ─────────────────────────────────────────────────────────
risk:
  trading_enabled: true                  # env: SCOUT_TRADING_ENABLED
  max_daily_loss_pct: 0.030
  max_drawdown_pct: 0.150
  max_trades_per_day: 8
  breaker_cooldown_bars: 10              # sessions (~2 weeks)

# ─── sentiment ───────────────────────────────────────────────────────────────
sentiment:
  enabled: false
  source_ids: [null]
  observations_path: data/sentiment/observations.parquet
  ingest_lag_minutes: 60
  half_life_hours: 24                # daily clock: a day, not 12 hours
  max_age_hours: 120
  max_freshness_bars: 5                  # sessions
  freshness_half_life_bars: 2
  n_eff_full: 20
  min_n_eff: 3.0
  min_confidence: 0.25
  veto_threshold: 0.70
  penalty_slope: 0.85
  min_multiplier: 0.40
  sources:
    vix_term:     { source_weight: 0.50, broadcast: true,  lag_minutes: 0 }
    short_interest:{ source_weight: 1.00, broadcast: false, lag_minutes: 1440 }
    # M7 only:
    funding_skew: { source_weight: 1.00, broadcast: false, lag_minutes: 0 }
    fear_greed:   { source_weight: 0.30, broadcast: true,  lag_minutes: 0 }
    cryptopanic:  { source_weight: 1.00, broadcast: false, lag_minutes: 60 }

# ─── audit / output ──────────────────────────────────────────────────────────
audit:
  results_dir: results
  features_json_policy: accepted_and_ranked   # accepted | accepted_and_ranked | all | none
  flush_every_cycles: 500

logging:
  level: INFO
  format: json
  dir: logs

research:
  registry_path: experiments/registry.csv
  lockbox_path: experiments/holdout_lockbox.json
  holdout_budget: 3
  plot_dpi: 150
  bootstrap_iterations: 2000
```

---

## 4. Parameter count audit

The most important table in this document. Count what can actually be tuned:

| Category | Tunable | Notes |
|---|---|---|
| Strategy params | **9** | 4 (momentum) + 5 (Donchian) |
| Regime thresholds | **5** per-symbol + **3** market | market regime is a documented filter, not a fitted score |
| Edge | **2** | `min_bin_samples`, `z` |
| Scoring | **1** | `min_ev_net_r` |
| Score weights | **0** | The whole point |
| **Total decision-shaping** | **~20** | |

Fixed by reasoning, not tuned:

- Universe thresholds — set from tradeability, not from results
- Cost parameters — set from the fee schedule and literature; only revised
  against real fills
- Portfolio caps — set from the drawdown you can tolerate
- `risk_fraction_per_trade` — scales the curve without changing risk-adjusted
  performance, so sweeping it teaches nothing
- Sentiment parameters — 11 of them, and none are swept; one paired test decides

Seventeen against roughly 30,000 resolved setups per strategy is a defensible
ratio. The brief's design had 12–20 score weights *plus* everything above, on top
of 6 strategies and 4 timeframes. That is the difference between a research
project and a random-number generator with a diagram.

---

## 5. Gate order

**Fixed.** Do not reorder without updating this table, because funnel analysis
compares reason counts across runs and reordering silently changes what those
counts mean.

`evaluate_gates` (`gates/eligibility.py`) owns rows 1–8. Rows 9–10 are checked
by the engine before `detect()`, gated by `require_regime_allowed`. They are
not inside `evaluate_gates`.

| # | Gate | Reason on failure | Where |
|---|---|---|---|
| 1 | Symbol in the universe snapshot at `t` (`universe_entry is None`) | `NOT_IN_UNIVERSE` | `evaluate_gates` |
| 2 | `universe_entry.eligible` | the snapshot's own reason | `evaluate_gates` |
| 3 | `feature_row` present and `is_warm` | `INSUFFICIENT_HISTORY` | `evaluate_gates` |
| 4 | `bars_since_gap >= min_bars_since_gap` (threshold from `universe` config, passed in) | `DATA_GAP` | `evaluate_gates` |
| 5 | Latest bar age ≤ `max_bar_staleness_bars` | `STALE_DATA` | `evaluate_gates` |
| 6 | `xs_population >= min_xs_population` | `THIN_CROSS_SECTION` | `evaluate_gates` |
| 7 | Earnings not in `(t, t+max_hold_bars]` (equities only; ETFs exempt). Window is the strategy's `max_hold_bars`, not a separate config knob. | `EARNINGS_IN_WINDOW` | `evaluate_gates` |
| 8 | Not hard-to-borrow when going short | `HARD_TO_BORROW` | `evaluate_gates` |
| 9 | Market regime in `strategy.allowed_market_regimes` | `MARKET_REGIME_BLOCKED` | engine (M3) |
| 10 | Per-symbol `regime` in `strategy.allowed_regimes` (Donchian only) | `REGIME_BLOCKED` | engine (M3) |

Rows 1–6 are per symbol (same answer for every strategy). Rows 7–8 take
`max_hold_bars` / `direction` from the strategy under consideration, so the
engine calls `evaluate_gates` per `(symbol, strategy)`. Rows 9–10 are also per
`(symbol, strategy)`. Cheapest and broadest first.

A missing earnings row for an individual equity is `EARNINGS_IN_WINDOW`, never
a pass. See [ADR-020](ADR/020-earnings-gate.md).

Two later gates are not in this table because they occur after scoring:
`INSUFFICIENT_BIN_SAMPLES` and `COST_UNAVAILABLE`.

---

## 6. Environment variables

`.env.example`:

```bash
# Data vendor — required for M1 ingest (not Yahoo)
SCOUT_NORGATE_USER=
SCOUT_NORGATE_PASSWORD=
# or
SCOUT_SHARADAR_API_KEY=

# Broker credentials — required for M5 paper/live only
SCOUT_BROKER_API_KEY=
SCOUT_BROKER_API_SECRET=

# Kill switch. Set to 0 to block all new risk-opening orders immediately.
SCOUT_TRADING_ENABLED=1

# Optional overrides
SCOUT_LOGGING__LEVEL=INFO
```

Never logged, never written to `results/`, never in any YAML. `LiveBroker`
re-reads `SCOUT_TRADING_ENABLED` before **every** order send — a kill switch that
requires a restart is not a kill switch.

---

## 7. Environment files

`config/development.yaml`:

```yaml
period:
  start: 1998-01-01T00:00:00Z
  warmup_end: 2004-01-01T00:00:00Z
  end: 2017-12-31T23:59:59Z
  split: DEVELOPMENT
```

`config/holdout.yaml`:

```yaml
period:
  start: 1998-01-01T00:00:00Z      # history is still needed to warm the bins
  warmup_end: 2018-01-01T00:00:00Z # no trade before the holdout begins
  end: 2026-08-01T00:00:00Z
  split: HOLDOUT
edge:
  lcb_method: bootstrap            # mandatory for holdout
```

The holdout config loads history from 1998 but sets `warmup_end` to 2018-01-01,
so bin statistics are warm while no trade occurs before the holdout window. The
holdout *range itself* is a constant in `research/splits.py`, not a free YAML
knob — editing `end` cannot silently eat more of the lockbox period.

---

## 8. Cross-field validation

Beyond per-field types, `validate_config` enforces:

| Rule | Error |
|---|---|
| `warmup_end >= start + 6 months` | `ScoutConfigError` |
| `end > warmup_end` | `ScoutConfigError` |
| `costs.cost_multiplier >= 1.0` | `ScoutConfigError` |
| `portfolio.max_cluster_risk_pct <= max_portfolio_heat_pct` | `ScoutConfigError` |
| `portfolio.risk_fraction_per_trade * max_positions >= max_portfolio_heat_pct` | warning: heat cap unreachable |
| `edge.z >= 0` | `ScoutConfigError` |
| `regime.er_range_max < regime.er_trend_min` | `ScoutConfigError` — overlapping definitions |
| `regime.vol_low_pct < regime.vol_high_pct` | `ScoutConfigError` |
| `features.ema_fast < features.ema_slow` | `ScoutConfigError` |
| every `strategies[].strategy_id` in the registry | `ScoutConfigError` |
| every strategy `params` key known to its factory | `ScoutConfigError` |
| `sentiment.enabled` ⇒ `source_ids != [null]` | `ScoutConfigError` |
| `sentiment.min_multiplier` in `(0, 1)` | `ScoutConfigError` |
| every candidate symbol appears in `clusters.yaml` | warning, naming them |
| `mode == LIVE` ⇒ credentials present and `risk.trading_enabled` explicitly set | `ScoutConfigError` |
| `mode == LIVE` ⇒ git tree clean | `ScoutConfigError` |
| `split == HOLDOUT` ⇒ lockbox budget available and git tree clean | `ScoutConfigError` |
| `asset_class == crypto` ⇒ `split` is not HOLDOUT unless M7 is explicitly unlocked | `ScoutConfigError` |

The last two HOLDOUT checks are the enforcement points for
[`12-RESEARCH_PROTOCOL.md §3`](12-RESEARCH_PROTOCOL.md#3-the-holdout-lockbox).
Both are checked in the config layer, before any expensive work, so a violation
fails in under a second rather than at the end of a five-minute run.

---

## 9. Config hashing

```python
def config_hash(cfg: ScoutConfig) -> str:
    payload = cfg.model_dump(mode="json", exclude={"run": {"mode"}})
    payload["data_snapshot_id"] = read_snapshot_id(cfg.data.snapshot_id_path)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

`run.mode` is excluded so the same logical configuration hashes identically in
backtest and paper, which is what lets you assert that the paper run uses the
validated configuration.

Recorded in `results/<run-id>/config_hash.txt`, in `registry.csv`, in the edge
table's metadata, and in every log line's `extra`. The startup check comparing
the run's hash to the edge table's hash is what catches the most dangerous silent
error in the system: statistics computed under different rules than the ones
currently making decisions.
