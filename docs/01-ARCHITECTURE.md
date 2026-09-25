# Architecture

> Components, data flow, and boundaries. Read [`00-REVIEW.md`](00-REVIEW.md)
> first if you want to know why this differs from the original brief.

---

## 1. What the system is

A **bar-driven, cross-sectional opportunity scanner**. Once per US cash-session
close it evaluates every eligible **stock and ETF** in a liquidity-ranked
universe, produces at most a handful of candidate trades, ranks them by a
single expected-value statistic net of costs, applies portfolio-level
constraints, and usually does nothing. Crypto is M7 and optional.

Three properties define the architecture more than anything else:

1. **Cross-sectional.** Decisions compare assets against each other at the same
   instant. Therefore the outer loop is over *time*, and all assets are processed
   inside one timestamp. A per-asset outer loop is structurally incapable of
   expressing this system and is forbidden.
2. **Deterministic.** Same config plus same data equals same output, byte for
   byte, including the run's `metrics.json`. No wall-clock, no unseeded
   randomness, no dict-ordering dependence, no network calls inside the loop.
3. **One code path.** Backtest, paper, and live differ only in which adapters are
   injected and which config file is loaded. There is no separate "live logic".

---

## 2. Component map

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  OFFLINE / PREPARATION  (run rarely, writes files, never inside the loop) │
 └──────────────────────────────────────────────────────────────────────────┘

   ingest ──────────► data/raw/equity/…           daily unadjusted bars + actions
      │
      └──► adjust ──► data/processed/panel/1d/…    causal split/dividend adjust
                │
                ├──► build_universe ─► data/universe/snapshots.parquet
                │                      point-in-time eligibility per (t, symbol)
                │
                └──► label_setups ──► data/labels/setups_<strategy_id>.parquet
                          │            resolved historical setup outcomes
                          │
                          └──► build_edge ─► data/edge/edge_table.parquet
                                              BinStats on the monthly as_of grid


 ┌──────────────────────────────────────────────────────────────────────────┐
 │  DECISION CYCLE  (pure; identical in backtest, paper, and live)           │
 └──────────────────────────────────────────────────────────────────────────┘

              MarketPanel (all symbols, bars with close_time <= t)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐            ┌─────────────────────┐
            │  1. FEATURES  │            │  UniverseSnapshot(t) │
            │  per symbol,  │            │  eligibility + ADV   │
            │  causal, float│            │  + spread estimate   │
            │  incl. regime │            └──────────┬──────────┘
            └───────┬───────┘                       │
                    └───────────────┬───────────────┘
                                    ▼
                          ┌───────────────────┐
                          │  2. GATES         │   binary, ordered, each
                          │  eligibility      │   rejection logged with reason
                          └─────────┬─────────┘
                                    ▼
                          ┌───────────────────┐
                          │  3. STRATEGIES    │   Setup | None
                          │  detect setups    │   direction, entry, stop,
                          │  regime-gated     │   target, horizon
                          └─────────┬─────────┘
                                    ▼
                          ┌───────────────────┐   ┌──────────────────────┐
                          │  4. EDGE          │◄──┤  EdgeTable (as of t) │
                          │  ev_r_lcb         │   │  bin statistics from │
                          │  from bin stats   │   │  resolved-before-t   │
                          └─────────┬─────────┘   │  setups only         │
                                    │             └──────────────────────┘
                                    ▼
                          ┌───────────────────┐
                          │  5. COSTS         │   cost_r for this symbol,
                          │  fees+spread+slip │   size, and holding horizon
                          │  +funding+impact  │
                          └─────────┬─────────┘
                                    ▼
                          ┌───────────────────┐
                          │  6. RANK          │   ev_net_r = ev_r_lcb - cost_r
                          │  threshold + sort │   sort by ev_net_r / bars
                          └─────────┬─────────┘
                                    ▼
                    ┌───────────────────────────────┐   ┌──────────────────┐
                    │  7. PORTFOLIO GATE            │◄──┤  PortfolioState  │
                    │  heat, cluster, beta, top-N   │   │  open positions  │
                    └───────────────┬───────────────┘   └──────────────────┘
                                    ▼
                          ┌───────────────────┐   ┌──────────────────────┐
                          │  8. SIZING        │◄──┤  SentimentView(t)    │
                          │  risk-based, then │   │  penalty multiplier  │
                          │  Decimal rounding │   │  in (0, 1]           │
                          └─────────┬─────────┘   └──────────────────────┘
                                    ▼
                          ┌───────────────────┐
                          │  9. BROKER        │   Sim | Paper | Live
                          │  bracket orders   │   same interface
                          └─────────┬─────────┘
                                    ▼
                       DecisionRecord  ──►  results/<run-id>/
                       every candidate, accepted or rejected,
                       with the reason and all inputs
```

Nine boxes. The original brief's diagram had six sequential stages between
"signal" and "ranking"; boxes 4–6 above replace all of them because they compute
one number instead of six partial scores.

---

## 3. Component responsibilities

The table format is deliberate: for each component, what it owns, consumes,
produces, and — most importantly — what it must **not** know.

### 3.1 `data` — market data access

| | |
|---|---|
| **Exists because** | Everything downstream needs bars, and nothing downstream should know where bars come from. |
| **Owns** | Parquet layout, symbol naming normalisation, resampling, gap detection, the `MarketPanel` object. |
| **Consumes** | Exchange REST responses (offline only); Parquet files (in-loop). |
| **Produces** | `MarketPanel` — an immutable long-format frame of all symbols' bars up to and including `t`. |
| **Must not know** | Strategies, features, scoring, positions, or that trading exists. |

### 3.2 `universe` — point-in-time eligibility

| | |
|---|---|
| **Exists because** | Survivorship bias is the most damaging and least visible backtest error, and it can only be prevented by recording what was tradeable *at the time*. |
| **Owns** | `snapshots.parquet`; the eligibility rules; the causal ADV and spread estimates. |
| **Consumes** | `MarketPanel` history strictly before each snapshot timestamp. |
| **Produces** | `UniverseSnapshot(t)` — the eligible symbol set plus liquidity metadata plus per-symbol rejection reasons. |
| **Must not know** | Strategies, edge, or the portfolio. Eligibility is about the *market*, not about whether we want the trade. |

### 3.3 `features` — causal derived values, including regime

| | |
|---|---|
| **Exists because** | Indicator computation is the most duplicated and most lookahead-prone code in any trading system, so it exists exactly once. |
| **Owns** | Every indicator formula, the regime classifier, feature naming, warm-up requirements. |
| **Consumes** | `MarketPanel`. Nothing else. |
| **Produces** | `FeaturePanel` — one row per `(t, symbol)`, float columns only. |
| **Must not know** | Strategies, costs, portfolio, sentiment, orders. A feature function that takes a config threshold used for a trading decision is a strategy in disguise and belongs in `strategies`. |

Regime lives here, as two continuous features plus one derived label, per
[`ADR/004-regime-as-feature.md`](ADR/004-regime-as-feature.md).

### 3.4 `gates` — ordered binary eligibility

| | |
|---|---|
| **Exists because** | Hard constraints must be separated from graded evidence. Mixing them was the original brief's core error. |
| **Owns** | The ordered gate list (14-CONFIG §5 rows 1–8), each gate's rejection reason. |
| **Consumes** | `FeatureRow` or None, `UniverseEntry` or None, earnings events known at `t`, the session calendar, the strategy's `max_hold_bars`, config. |
| **Produces** | `RejectionReason` or None per `(t, symbol, strategy)` — the first failing gate, or `None` if all pass. There is no `GateResult` wrapper; the reason *is* the result. Called per strategy because the earnings window is that strategy's `max_hold_bars`. |
| **Must not know** | Edge statistics or portfolio state. Gates answer "may we consider this at all", not "is it good". Regime membership (rows 9–10) is the engine's check before `detect()`, not this module's. |

### 3.5 `strategies` — setup detection

| | |
|---|---|
| **Exists because** | Signal logic must be swappable, pure, and unit-testable without any I/O. |
| **Owns** | Entry trigger rules, stop placement rule, target placement rule, max holding period, the set of regimes the strategy may fire in. |
| **Consumes** | One `FeaturePanel` row plus its own typed params. |
| **Produces** | `Setup | None`. |
| **Must not know** | Costs, position size, equity, other assets, other strategies, sentiment, the portfolio, whether it is running in backtest or live. A strategy never sees money. |

A strategy answers exactly two questions: *is there a setup here* and *where are
the entry, stop, and target*. It does **not** answer "how confident am I" — that
is the edge estimator's job, computed from data rather than asserted by the
strategy author. This is a deliberate departure from the brief, which asked each
strategy to report expected reward, expected risk, and confidence. Self-reported
confidence is an unfalsifiable free parameter.

### 3.6 `scoring` — labeling, edge estimation, ranking

| | |
|---|---|
| **Exists because** | It is the only place where "how good is this" is decided, so it is the only place that can be blamed. |
| **Owns** | Triple-barrier labeling, bin definition, the `EdgeTable`, the lower confidence bound, `ev_net_r`, thresholds, the sort order. |
| **Consumes** | `Setup`, `FeaturePanel` row, `EdgeTable` as of `t`, `cost_r`. |
| **Produces** | `Opportunity` — one ranking statistic plus its diagnostic components plus provenance. |
| **Must not know** | Portfolio state or position size. Ranking is per-asset and context-free; portfolio context is applied afterwards, in `portfolio`. |

### 3.7 `costs` — execution cost model

| | |
|---|---|
| **Exists because** | Backtest and live must use the same cost assumptions or the backtest is worthless, and that is only guaranteed if there is one implementation. |
| **Owns** | Fee schedule, spread model, slippage model, funding accrual, market-impact estimate, conversion of quote-currency cost to R. |
| **Consumes** | `Setup`, liquidity metadata, intended notional, expected holding bars, config. |
| **Produces** | `CostEstimate` with a `cost_r` field and an itemised breakdown. |
| **Must not know** | Edge, strategies, ranking. |

### 3.8 `portfolio` — constraints and sizing

| | |
|---|---|
| **Exists because** | Correlated positions are one position, and per-asset logic cannot see that. |
| **Owns** | Portfolio heat, cluster caps, beta cap, gross/net exposure caps, top-N selection, position sizing, tick/step rounding, the kill switch and circuit breakers. |
| **Consumes** | Ranked `Opportunity` list, `PortfolioState`, `SentimentView`, config. |
| **Produces** | `TradeDecision` list — accepted with size, or rejected with reason. |
| **Must not know** | How a setup was detected or how its edge was estimated. It sees `ev_net_r`, `direction`, `cluster`, `beta`, and risk distance. Nothing strategy-specific. This is the brief's "avoid strategy-specific portfolio logic", enforced by the type signature. |

### 3.9 `execution` — broker adapters

| | |
|---|---|
| **Exists because** | Vendor APIs must not leak into decision logic. |
| **Owns** | `SimBroker` (backtest fills), `PaperBroker` (testnet), `LiveBroker` (real), idempotency keys, order-state reconciliation. |
| **Consumes** | `TradeDecision`. |
| **Produces** | `Fill` events, `PortfolioState` updates. |
| **Must not know** | Why the trade was chosen. |

### 3.10 Supporting, non-decision components

| Component | Responsibility |
|---|---|
| `sentiment` | Point-in-time observation store; aggregation into `SentimentView(t)`; the penalty multiplier. Inert until it passes the promotion test in [`10-SENTIMENT.md`](10-SENTIMENT.md). |
| `backtest` | The panel loop, `SimBroker`, the ledger, run-output writing. |
| `research` | Metrics, walk-forward driver, robustness suite, plots, the trial registry, the holdout lockbox. |
| `config` | Typed config objects, YAML loading, validation, hashing for reproducibility. |
| `storage` | Parquet and DuckDB read/write helpers, `DecisionRecord` sink. |
| `utils` | Logging, the clock (the only module allowed to know wall-clock time), math helpers. |

---

## 4. Dependency direction

Strictly one-way. An import that points upward in this list is a bug and a test
enforces it.

```text
 utils, config          ← may be imported by anything
      ▲
 domain (dataclasses, enums, protocols)   ← imports only stdlib + utils
      ▲
 data, universe, features, costs, sentiment
      ▲
 strategies, scoring
      ▲
 portfolio
      ▲
 execution, backtest
      ▲
 research, cli          ← imports anything; imported by nothing
```

Concretely forbidden, and checked by `tests/unit/test_layering.py`:

- `domain` importing anything from `src/scout/` except `utils`.
- `strategies` importing `costs`, `portfolio`, `execution`, `sentiment`, or `data`.
- `features` importing `strategies` or `scoring`.
- `scoring` importing `portfolio` or `execution`.
- Anything except `execution` and `data/ingest` importing `httpx` or an exchange SDK.
- Anything except `utils/clock.py` calling `datetime.now`, `datetime.utcnow`, or `time.time`.

---

## 5. The decision cycle, precisely

For decision timestamp `t`:

```text
 1. panel        = MarketPanel of all bars with close_time <= t
 2. snapshot     = universe.snapshot_at(t)                  # from file, causal
 3. features     = features.compute(panel, t)               # per eligible symbol
 4. for each symbol in snapshot (all candidates, not only eligible):
      for each enabled strategy:
          reason = gates.evaluate_gates(
              features.get(symbol), snapshot.entries.get(symbol), cfg.gates,
              is_etf=assets[symbol].is_etf,
              earnings=earnings_known_at(t, symbol),
              max_hold_bars=strategy.max_hold_bars,
              calendar=calendar,
              min_bars_since_gap=cfg.universe.min_bars_since_gap,
          )
          if reason is not None: record(reason); continue
          if cfg.gates.require_regime_allowed:
              if market_regime not in strategy.allowed_market_regimes:
                  record(MARKET_REGIME_BLOCKED); continue
              if strategy.allowed_regimes
                 and features[symbol].regime not in strategy.allowed_regimes:
                  record(REGIME_BLOCKED); continue
          setup = strategy.detect(features[symbol])
          if setup is None: record(NO_SETUP); continue
          stats = edge_table.lookup(bin_key(setup, features[symbol]), as_of=t)
          if stats is None or not stats.is_usable:
              record(INSUFFICIENT_BIN_SAMPLES); continue
          cost = costs.estimate_cost(setup, snapshot[symbol], provisional_notional,
                                     risk_capital_usd, stats.mean_bars_held, cfg.costs)
          if not cost.is_valid: record(COST_UNAVAILABLE); continue
          opp = scoring.build_opportunity(setup, features[symbol], stats, cost,
                                          assets[symbol])
          if opp.ev_net_r < cfg.scoring.min_ev_net_r:
              record(BELOW_EV_THRESHOLD); continue
          candidates.append(opp)
 5. ranked    = scoring.rank(candidates, cfg.scoring)
 6. decisions = portfolio.select_and_size(ranked, portfolio_state, sentiment_views,
                                          assets, cfg.portfolio, cfg.risk,
                                          cfg.sentiment, cfg.costs,
                                          cfg.scoring.min_ev_net_r)
 7. broker.submit_bracket(...) for accepted decisions   # executes at open of t+1
 8. sink.write(DecisionRecord for every candidate and every rejection)
```

Steps 1–6 are pure functions over immutable inputs. Step 7 is the only side
effect. Step 8 always runs, including when zero trades are made — the log of
*why nothing was traded* is one of the most useful research artifacts the system
produces.

**Exits are not part of this cycle.** Every position is opened with a bracket:
protective stop and target submitted with the entry. Exits therefore happen
intrabar without the pipeline running, which is both realistic and what makes a
daily decision clock safe: stops are live overnight, when most equity risk
arrives. The only pipeline-driven exit is the time stop at
`max_hold_bars`, evaluated at step 0 of each cycle before new entries are
considered. Rationale in
[`ADR/008-bracket-exits.md`](ADR/008-bracket-exits.md).

---

## 6. Execution modes

| | Backtest | Paper | Live |
|---|---|---|---|
| Clock | bar timestamp from data | wall-clock, aligned to bar close | same as paper |
| Bars | Parquet panel | exchange REST/WS, appended to panel | same as paper |
| Universe | `snapshots.parquet` | recomputed each cycle, appended to snapshots | same as paper |
| `EdgeTable` | rebuilt as-of each `t` | loaded, refreshed on schedule | same as paper |
| Sentiment | point-in-time store | live provider, written with `available_at` | same as paper |
| Broker | `SimBroker` | `PaperBroker` (testnet) | `LiveBroker` |
| Ledger | in-memory Decimal | persisted | persisted, reconciled |
| Kill switch | not applicable | active | active, mandatory |

Everything in rows "Universe" through "Sentiment" is the *same code* reading
different adapters. If a change requires an `if mode == "backtest"` branch inside
`features`, `strategies`, `scoring`, or `portfolio`, the change is wrong.

---

## 7. Failure policy

Carried over from the previous project's ADR-010, which was correct.

**Asymmetric.** Opening or increasing risk fails closed; reducing or closing risk
requires the fewest possible dependencies.

| Condition | Behaviour |
|---|---|
| Bars stale beyond `max_bar_staleness_bars` | No new entries. Existing brackets remain at the exchange. |
| Feature computation fails for a symbol | That symbol is ineligible this cycle. Others unaffected. |
| `EdgeTable` missing or unloadable | No new entries at all. This is a hard stop, not a fallback. |
| Universe snapshot missing for `t` | No new entries. |
| Sentiment unavailable | Penalty multiplier is `1.0` (neutral). Trading continues. Logged. |
| Cost model inputs missing | That symbol is ineligible. Never substitute an optimistic default. |
| Broker unreachable | No new orders. Retry with the same idempotency key. |
| Kill switch off, or a circuit breaker tripped | No new entries. Reduce-only remains permitted. |

The rule behind every row: a missing input never resolves to a value that makes
a trade *more* likely.

---

## 8. Observability

Two artifacts, both written every run.

**Structured logs.** JSON lines to `logs/`. Every record carries `run_id`,
`decision_ts`, and where applicable `symbol`, `strategy_id`, and
`correlation_id`. One log line per accepted decision; rejections go to the
decision sink rather than the log to keep log volume proportional to *activity*
rather than to universe size.

**Decision audit trail.** `results/<run-id>/decisions.parquet`, one row per
`(decision_ts, symbol, strategy_id)` considered, whether accepted or rejected,
with every input that mattered: the feature values used, the regime, the bin key,
`n`, `mean_r`, `std_r`, `ev_r_lcb`, `cost_r`, `ev_net_r`, the rank, the
portfolio-gate outcome, the sentiment multiplier, the final size, and the
rejection reason if any.

This table is the primary research input. Analysing *rejections* is how you find
out that a gate is throwing away every profitable trade, and no other artifact
can tell you that. Schema in
[`04-DATA_AND_UNIVERSE.md`](04-DATA_AND_UNIVERSE.md#7-decision-audit-schema).

---

## 9. What this architecture deliberately does not have

| Absent | Why |
|---|---|
| Message bus or event queue | One process, one thread, one ordered loop. A queue would only add nondeterminism. |
| Plugin discovery / dynamic registry | Strategies are registered in one explicit dict in `strategies/registry.py`. Import-time magic breaks determinism and confuses implementers. |
| Base classes with inherited behaviour | Protocols only. No `AbstractStrategy` with 200 lines of shared logic that four subclasses partially override. |
| A `Signal` object distinct from `Setup` | They were the same thing wearing two names in the brief's model. |
| Separate live and backtest pipelines | Guarantees divergence. Forbidden. |
| A service layer / API in v1 | Nothing to serve. See [`00-REVIEW.md`](00-REVIEW.md#10-infrastructure-none-of-it-yet). |
| Async anywhere in v1 | The loop is CPU-bound and sequential. Async arrives with live WebSocket feeds in M5 and stays inside `execution`. |
