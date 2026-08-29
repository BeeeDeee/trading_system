# Backtest Engine

> Panel loop, fill rules, ledger, corner cases, run outputs.
>
> Module: `src/scout/backtest/`

---

## 1. The hard constraint

**The outer loop is over timestamps. Never over symbols.**

```python
# CORRECT
for ts in decision_timestamps:
    for symbol in eligible_at(ts):
        ...

# FORBIDDEN
for symbol in symbols:
    for ts in timestamps:
        ...
```

The per-symbol outer loop is the natural instinct and it is structurally
incapable of expressing this system: it cannot rank assets against each other at
an instant, cannot enforce a portfolio cap, and cannot know what else was
available. Discovering this after writing the engine means rewriting the engine.

`tests/unit/test_engine_structure.py` asserts that `BacktestEngine.run` iterates
`panel.timestamps` as its outermost loop, by inspecting the AST of the method.
That test exists because this mistake is easy to make and expensive to fix.

---

## 2. Engine skeleton

```python
class BacktestEngine:
    def __init__(self, *, candles, sentiment, broker, strategies,
                 edge_table, sink, universe, assets, cfg): ...

    def run(self) -> BacktestResult:
        candidates = read_candidates(self.cfg.universe.candidates_file)
        panel = self.candles.load_panel(candidates,
                                        self.cfg.data.decision_timeframe,
                                        self.cfg.period.start, self.cfg.period.end)
        benchmark = load_benchmark(self.cfg)          # SPY + VIX, not from universe
        snapshots = self.universe.load_all()
        features = compute_features(panel, benchmark, snapshots, self.cfg.features)

        state = initial_state(self.cfg.portfolio.initial_equity_usd)
        equity_points, trades, records = [], [], []

        for ts in panel.timestamps:
            if ts < self.cfg.period.warmup_end:
                continue                                     # warm-up: no trading

            # --- 0. mark to market on this bar's close ---
            state = self.broker.mark(state, panel, ts)

            # --- 1. intrabar exits: stops, targets (before anything else) ---
            fills = self.broker.poll_fills(ts)
            state, closed = apply_exits(state, fills, ts)
            trades.extend(closed)

            # --- 2. time stops and delistings ---
            state, closed = self.apply_time_stops(state, panel, ts)
            trades.extend(closed)
            state, closed = self.apply_delistings(state, panel, ts)
            trades.extend(closed)

            # --- 3. new entries ---
            snapshot = self.universe.snapshot_at(ts)
            candidates, recs = self.evaluate(ts, features, snapshot, state)
            records.extend(recs)

            ranked = rank(candidates, self.cfg.scoring)
            sviews = self.sentiment_views(ts, [c.symbol for c in ranked])
            decisions = select_and_size(
                ranked, state, sviews, self.assets,
                self.cfg.portfolio, self.cfg.risk, self.cfg.sentiment,
                self.cfg.costs, self.cfg.scoring.min_ev_net_r,
            )
            records.extend(to_records(decisions, ts))

            for d in (d for d in decisions if d.accepted):
                fill, corr = self.broker.submit_bracket(*build_bracket(d, ts))
                state = apply_entry(state, d, fill, corr)

            # --- 4. record ---
            equity_points.append((ts, float(state.equity_usd)))
            self.sink.write(records); records.clear()

        self.sink.flush()
        return assemble_result(...)
```

### Why the order of steps 1–3 matters

Exits before entries, always. If entries came first, a symbol whose stop was hit
this bar would still be "in position" and would be rejected with
`ALREADY_IN_POSITION`, deferring a legitimate re-entry by one bar. Worse, a
position closed this bar would still consume portfolio heat, understating
available capacity.

Marking to market before exits means `equity_usd` reflects the bar's close before
any exit is applied, which is what the equity curve should show.

---

## 3. The clock

`ts` is always a bar close time from the panel. There is no wall-clock anywhere.

```python
# src/scout/utils/clock.py — the ONLY module allowed to touch wall-clock time
class Clock(Protocol):
    def now(self) -> datetime: ...

class BarClock:
    """Backtest clock. `now()` returns the current bar's close time."""

class WallClock:
    """Live clock. The only place datetime.now(timezone.utc) appears."""
```

`tests/unit/test_no_wallclock.py` greps `src/scout/` for `datetime.now`,
`datetime.utcnow`, `time.time`, and `pd.Timestamp.now`, allowing only
`utils/clock.py`. Mixing wall-clock into a backtest is the mechanism behind a
whole family of lookahead bugs, and grepping is cheaper than auditing.

---

## 4. Warm-up and period boundaries

```yaml
period:
  start: 2019-01-01T00:00:00Z        # data load begins
  warmup_end: 2019-07-01T00:00:00Z   # trading begins
  end: 2026-01-01T00:00:00Z
```

Two separate warm-ups, both required:

1. **Feature warm-up** — 360 bars, per
   [`05-FEATURES_AND_REGIME.md §8`](05-FEATURES_AND_REGIME.md#8-warm-up-accounting).
   Handled per symbol by `is_warm`.
2. **Edge-table warm-up** — bins need `min_bin_samples` resolved setups, which
   takes 12–18 months of history. Handled naturally by the
   `INSUFFICIENT_BIN_SAMPLES` rejection.

`warmup_end` must be at least 6 months after `start`. Config validation enforces
this. Expect near-zero trades for the first year even so; that is correct, and
[`07-EDGE_AND_SCORING.md §5`](07-EDGE_AND_SCORING.md#cold-start) explains why
"fixing" it is leakage.

---

## 5. Fill rules

`SimBroker`, in `backtest/sim_broker.py`. Every rule below is conservative on
purpose; where reality is ambiguous, the simulation takes the worse side.

### 5.1 Entry

```python
raw_entry = bar[ts + 1].open
slip = cost_estimate.slippage_bps + cost_estimate.impact_bps + cost_estimate.spread_bps
fill_price = raw_entry * (1 + direction.sign * (slip / 1e4))
```

Decision at the close of `t`, fill at the open of `t+1`, worsened by the modelled
half-spread, slippage, and impact. If `t+1` does not exist (end of data, or a
gap), **the entry does not happen** and the decision is recorded with a
`DATA_GAP` rejection. Never fill at the decision bar's close: that is a
one-bar lookahead worth several percent per year in a trending market.

### 5.2 Stop and target within a bar

For each open position, on each bar after entry:

```python
hit_stop   = bar.low  <= stop   if long else bar.high >= stop
hit_target = bar.high >= target if long else bar.low  <= target

if hit_stop and hit_target:
    exit at STOP                      # conservative tie rule
elif hit_stop:
    exit at STOP
elif hit_target:
    exit at TARGET
```

Fill prices:

```python
# Stops: assume the stop level, plus a gap penalty when the bar OPENED beyond it.
if long:
    stop_fill = min(stop, bar.open) if bar.open < stop else stop
else:
    stop_fill = max(stop, bar.open) if bar.open > stop else stop
stop_fill *= (1 - direction.sign * exit_slippage_bps / 1e4)

# Targets: assume exactly the target level, never better.
target_fill = target
```

The gap handling on stops is important and frequently omitted. If a bar opens
below your long stop, you do not get filled at the stop — you get filled at the
open, and the loss exceeds 1 R. **For US equities this is the common case, not
the tail:** overnight and weekend gaps, earnings prints, and halt reopenings
all produce opens beyond the stop. A simulation that caps every loss at exactly
1 R systematically understates tail risk and overstates Sharpe.

Targets are never filled better than the target level, even if the bar opened
past it. That is the mirror-image conservatism.

### 5.3 Dividends and borrow (equities)

Cash dividends on a long credit `Position.dividends_usd` at the ex-date (session
close of the last cum-dividend bar, paid as cash at the next open — match the
vendor's ex-date convention and document it in the ingest adapter). Shorts pay
the dividend. Borrow accrues daily on open short notional at the snapshot
borrow rate ([`08-COSTS.md`](08-COSTS.md)). Both are `Decimal`.

Crypto funding (8-hour intervals, `funding_paid_usd`) is **M7 only**.

### 5.4 Time stop

At `bars_held >= max_hold_bars`, exit at that bar's close plus exit slippage.
Evaluated at step 2 of the cycle, before new entries.

### 5.5 Delisting

Per [`04-DATA_AND_UNIVERSE.md §5.5`](04-DATA_AND_UNIVERSE.md#55-the-delisting-rule):
force-close at the last available close, `exit_reason = "DELISTED"`, full loss
taken. The trade appears in `trades.csv`.

---

## 6. The ledger

`backtest/ledger.py`. All `Decimal`.

```python
def apply_entry(state, decision, fill, corr_id) -> PortfolioState:
    """cash -= notional + fee (cash equities, not margin perps). Open a Position.
    Track notional through the Position for exposure caps."""

def apply_exit(state, position, fill, ts) -> tuple[PortfolioState, ClosedTrade]:
    """gross_pnl = sign * (exit - entry) * qty
       net_pnl   = gross_pnl - fees_total - borrow_total + dividends_total
       cash     += notional proceeds + net_pnl
       equity    = cash + Σ unrealised"""

def mark(state, panel, ts) -> PortfolioState:
    """Recompute unrealised P&L at this bar's close. Update peak_equity_usd,
    day_start_equity_usd (on UTC date change), and reset trades_today."""
```

Invariants asserted every bar (cheap, and they catch ledger drift immediately):

```python
assert state.equity_usd == state.cash_usd + sum(unrealised)
assert state.equity_usd > 0, "account blew up"        # halt the run
assert all(p.qty > 0 for p in state.positions.values())
```

An account reaching zero equity **halts the run** rather than continuing with
negative equity and producing meaningless metrics. It is reported as a failed run,
which it is.

---

## 7. Corner cases

| Case | Handling |
|---|---|
| No bar at `t+1` for an accepted decision | Entry does not happen; `DATA_GAP` recorded |
| Symbol delisted while in position | Force-close at last close, full loss |
| Stop and target both touched in one bar | STOP (conservative tie rule) |
| Bar opens beyond the stop | Fill at the open; loss exceeds 1 R |
| Bar opens beyond the target | Fill at the target; gain does not exceed the target |
| `atr_14` is zero or NaN | Symbol ineligible (`INSUFFICIENT_HISTORY`); never divide |
| Equity reaches zero | Halt, mark the run failed |
| Two candidates with identical `ev_per_bar_r` | Alphabetical tie-break by `(symbol, strategy_id)` |
| Two strategies fire on the same symbol at the same `ts` | Both ranked; the second is rejected `ALREADY_IN_POSITION` if the first is accepted |
| Universe snapshot missing at `t` | No entries this cycle; recorded |
| Edge table has no entry for a bin | `INSUFFICIENT_BIN_SAMPLES` |
| A position's stop is beyond the current mark | `open_risk_usd` floors at 0 |
| Config hash mismatch with the edge table | **Fail at startup** |

---

## 8. Run outputs

`results/<run-id>/` where `run-id` is `YYYYMMDD-HHMMSS-<strategy-slug>`, per the
global convention.

```text
results/20260827-231500-xsec-momentum-donchian/
├── config.yaml              # fully resolved config, including all defaults
├── config_hash.txt          # sha256 of the resolved config
├── metrics.json             # all metrics from 12-RESEARCH_PROTOCOL.md
├── trades.csv               # every ClosedTrade
├── equity.csv               # ts, equity, benchmark, drawdown_pct
├── decisions.parquet        # every candidate considered (the audit trail)
├── funnel.csv               # stage x rejection_reason counts
├── bins.csv                 # BinStats used, with realised outcomes joined
├── run.log                  # JSON lines
└── plots/
    ├── equity.png           # strategy vs buy-and-hold SPY (total return),
    │                        # normalised to 1.0, legend, final return % in title
    ├── drawdown.png         # underwater curve, shaded, max DD and date annotated
    ├── summary.png          # equity top, drawdown bottom, shared x-axis
    ├── calibration.png      # predicted ev_net_r vs realised R, with error bars
    ├── monthly_returns.png  # heatmap, year x month
    └── regime_breakdown.png # performance per regime and volatility bucket
```

`config.yaml` is the **resolved** config with every default filled in, not the
input file. Reproducibility requires knowing the value that was used, not the
value that was written down.

`calibration.png` is the most important plot in the directory
([`07-EDGE_AND_SCORING.md §10.1`](07-EDGE_AND_SCORING.md#10-required-m3-diagnostics)).
`equity.png`, `drawdown.png`, and `summary.png` follow the global plotting rule:
timezone-aware x-axis, 150 DPI minimum.

---

## 9. Determinism

Required. `tests/integration/test_determinism.py` runs the same config twice and
asserts byte-identical `metrics.json` and `trades.csv`.

Rules:

1. Seed everything from `cfg.run.seed`. Numpy generators are created explicitly as
   `np.random.default_rng(cfg.run.seed)` and passed in; never use the global
   `np.random` state.
2. No `set` iteration where order affects output. Sort first.
3. No dict-order dependence. `UniverseSnapshot.eligible_symbols` sorts.
4. Explicit tie-breaks in every sort.
5. No parallelism in the decision path. If features get parallelised for speed,
   results are concatenated and re-sorted by `(ts, symbol)` before use.
6. No wall-clock. Enforced by grep test.
7. Pin `pandas`, `numpy`, and `pyarrow` to exact versions in `pyproject.toml`.
   A pandas minor upgrade has changed `resample` boundary behaviour before, which
   silently shifts every bar by one period.

---

## 10. Performance targets

| Scope | Target |
|---|---|
| Feature computation, 1,000 symbols × 25 years of daily bars | < 90 s |
| Labeling pass, both strategies | < 180 s |
| Edge table build, monthly grid | < 60 s |
| Backtest loop, ~6,300 sessions × 1,000 symbols | < 180 s |
| Full pipeline from processed data | < 8 min |

Eight minutes matters: it is the difference between iterating on a hypothesis in
an afternoon and iterating once a day. It is also, less obviously, a
*discipline* mechanism — a 4-hour backtest tempts you into changing several
things at once, which destroys attribution.

If the loop misses target, the fix is almost always one of:

- Recomputing features inside the loop. Compute once, before it.
- Boolean-mask filtering the panel per bar. Precompute the `searchsorted`
  boundary index per timestamp in `MarketPanel.__init__`.
- Constructing `FeatureRow` dataclasses for all 1,000 symbols on every bar when
  most are gated out. Filter on the numpy arrays first, construct only for
  survivors.

---

## 11. Paper and live differences

The engine class is shared. Only these differ:

| Concern | Backtest | Paper / Live |
|---|---|---|
| Loop driver | `for ts in panel.timestamps` | scheduler wakes shortly after each bar close |
| Bar source | Parquet, loaded once | REST/WS, appended incrementally |
| Fill knowledge | computed by `SimBroker` | `poll_fills` from the exchange |
| Universe | precomputed snapshots | recomputed each cycle, appended |
| Edge table | rebuilt as-of each `ts` | loaded from file, refreshed weekly |
| Reconciliation | not applicable | mandatory on startup and hourly |
| Kill switch | config | env var, re-read before every send |

`LiveEngine` subclasses `BacktestEngine` and overrides only the loop driver and
the panel-refresh step. If any decision logic needs overriding, the abstraction
is wrong — fix the abstraction rather than the override. This is checked by
review, not by a test, and it is the most important review question at M5.
