# Interfaces

> Five Protocols. Everything else is a pure function.
>
> Location: `src/scout/domain/ports.py` for the Protocols; the pure-function
> signatures listed here live in their own modules.

---

## 1. Why five and not fifteen

The brief listed fifteen things to abstract. An interface is only worth its cost
when there will be **more than one implementation** and the caller must not know
which. Applying that test:

| Brief's proposed interface | Verdict |
|---|---|
| `MarketDataProvider` | **Protocol.** Vendor / Parquet / test fixture. Yes. |
| `SentimentProvider` | **Protocol.** Mock / VIX term / short interest (crypto sources are M7). Yes. |
| `ExecutionProvider` | **Protocol.** Sim / Paper / Live. Yes. |
| `StorageProvider` | **Protocol**, but narrowed to `DecisionSink` (§7). A wide "storage provider" always grows into a god object; the only thing that needs substituting is the audit-trail sink. |
| Strategy | **Protocol.** Several implementations, selected by config. Yes. |
| Feature engine | Pure function `compute_features(panel, cfg) -> FeaturePanel`. One implementation, ever. An interface would add indirection and no substitutability. |
| Regime detector | Pure function. Not even a module of its own — it is two feature columns and one classifier function. |
| Scoring / opportunity ranking | Pure functions. Exactly one implementation by design; multiple would defeat [`ADR/002`](ADR/002-single-ranking-statistic.md). |
| Edge estimator | Pure functions plus the `EdgeTable` data structure. A Protocol arrives only if ML is promoted at M6, and then it is `EdgeEstimator` with the same `(BinKey, as_of) -> BinStats` shape. |
| Cost model | Pure function. One implementation is the *point* — backtest and live must share it. |
| Portfolio / risk | Pure functions over `PortfolioState`. |
| Backtesting interface | The backtest *is* the caller, not a dependency. |
| Logging | stdlib `logging` with a JSON formatter. Do not abstract the standard library. |
| Monitoring | Deferred to M5. |
| Config | Typed objects, not an interface. |

Result: five Protocols. Fewer interfaces means less indirection for an
implementer to get lost in, and pure functions are strictly easier to test than
mocked interfaces — a pure function needs no mock at all.

---

## 2. Protocol conventions

```python
from typing import Protocol, runtime_checkable
```

- `@runtime_checkable` on all five, so `tests/unit/test_ports.py` can assert
  every concrete class satisfies its Protocol without importing a base class.
- **No inheritance.** Implementations do not subclass the Protocol. Structural
  typing only. There is no `AbstractStrategy` and there never will be.
- Protocol methods raise nothing on the happy path and raise
  `ScoutDataError` / `ScoutExecutionError` (from `scout.utils.errors`) on
  failure. No returning `None` to mean "error"; `None` means "no result", which
  is a different thing.
- All Protocol methods are synchronous in v1. `LiveBroker` gets an async
  internals in M5 but keeps the sync facade, so the decision loop never becomes
  async.

---

## 3. `CandleSource`

```python
@runtime_checkable
class CandleSource(Protocol):
    """Supplies closed OHLCV bars. The only source of price truth."""

    def load_panel(
        self,
        symbols: Sequence[str],
        timeframe: str,          # "1d" in v1; "1h" | "4h" for crypto M7
        start: datetime,         # UTC, inclusive
        end: datetime,           # UTC, inclusive
    ) -> MarketPanel:
        """Return all CLOSED bars in [start, end] for the given symbols.

        Contract:
          - Every returned bar has close_time <= end. A bar that is still
            forming MUST NOT be returned. Violating this is a lookahead bug.
          - Missing symbols are omitted silently; missing bars are simply
            absent (no forward-fill, no zero-fill, no interpolation).
          - Result is sorted by (ts, symbol).
          - Deterministic: two calls with identical arguments return identical
            frames.
        """

    def available_range(self, symbol: str, timeframe: str) -> tuple[datetime, datetime] | None:
        """First and last available bar close times, or None if unknown."""
```

Implementations:

| Class | Module | Notes |
|---|---|---|
| `ParquetCandleSource` | `data/parquet_source.py` | Primary. Reads `data/processed/panel/`. |
| `NorgateCandleSource` | `data/norgate_source.py` | Offline ingest only. Never called inside the loop. |
| `SharadarCandleSource` | `data/sharadar_source.py` | Alternative vendor. Offline ingest only. |
| `FixtureCandleSource` | `tests/fixtures/candles.py` | Synthetic deterministic bars for tests. |
| `BinanceCandleSource` | `data/binance_source.py` | **M7 only.** Offline ingest. Never called inside the loop. |

**No forward-fill, ever.** A forward-filled bar looks like a real observation and
is indistinguishable downstream. Gaps must remain gaps so the data-quality gate
can see them. This is stated three times in these docs on purpose.

---

## 4. `Strategy`

```python
@runtime_checkable
class Strategy(Protocol):
    """Pure setup detection. Sees one symbol's features at one instant."""

    @property
    def strategy_id(self) -> str:
        """Stable, versioned registry key, e.g. "donchian_breakout_v1".
        Changing the rules REQUIRES bumping the version suffix, because
        edge statistics are keyed on it and must not pool across rule
        changes."""

    @property
    def allowed_regimes(self) -> frozenset[Regime]:
        """Regimes in which this strategy may produce a setup. The engine
        checks this BEFORE calling detect(), and records REGIME_BLOCKED."""

    @property
    def required_warmup_bars(self) -> int:
        """Minimum bars_available before detect() may be called. The ENGINE
        checks this alongside row.is_warm and records INSUFFICIENT_HISTORY;
        detect() may also guard on row.is_warm defensively, but must not be
        relied upon as the only check."""

    def detect(self, row: FeatureRow) -> Setup | None:
        """Return a Setup if the entry rule fires at row.ts, else None.

        Contract:
          - Pure. No I/O, no logging, no randomness, no mutable state, no
            clock access. Calling twice with the same row returns equal
            Setups.
          - Uses ONLY fields of `row`. It cannot reach history, so it cannot
            look ahead. This is the primary reason FeatureRow exists.
          - Returned Setup must have symbol == row.symbol and ts == row.ts.
          - Never sees equity, size, costs, other symbols, or the portfolio.
        """
```

Registration, in `strategies/registry.py`:

```python
STRATEGY_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Strategy]] = {
    "xsec_momentum_v1": CrossSectionalMomentum.from_params,
    "donchian_breakout_v1": DonchianBreakout.from_params,
}

def build_strategies(cfg: Sequence[StrategyConfig]) -> tuple[Strategy, ...]:
    """Explicit dict lookup. No entry points, no module scanning, no
    __init_subclass__ magic. An unknown id raises at config-load time."""
```

Explicit over automatic: import-time registration makes strategy availability
depend on import order, which is exactly the kind of nondeterminism this system
cannot tolerate.

---

## 5. `SentimentSource`

```python
@runtime_checkable
class SentimentSource(Protocol):
    """Supplies point-in-time sentiment observations."""

    @property
    def source_id(self) -> str: ...

    def observations(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> tuple[SentimentObservation, ...]:
        """All observations with start <= available_ts <= end.

        Contract:
          - Filtering is on `available_ts`, NEVER on `event_ts`. An
            implementation that filters on event_ts is a lookahead bug and
            will be caught by tests/unit/test_sentiment_pit.py.
          - available_ts >= event_ts for every returned observation.
          - Deduplicated by (symbol, source, payload_hash).
          - Returning an empty tuple is valid and must not raise.
        """
```

Implementations:

| Class | Module | Milestone |
|---|---|---|
| `NullSentimentSource` | `sentiment/null_source.py` | M1. Returns `()` always. `SentimentView.is_neutral` is then True and the multiplier is exactly 1.0. |
| `ParquetSentimentSource` | `sentiment/parquet_source.py` | M2. Reads `data/sentiment/observations.parquet`. |
| `VixTermSource` | `sentiment/vix_term.py` | M2. Market-wide; broadcast. VIX vs VIX3M. |
| `ShortInterestSource` | `sentiment/short_interest.py` | M2. Per-symbol; lagged to publication, not as-of. |
| `FearGreedSource` | `sentiment/feargreed_source.py` | **M7.** Market-wide crypto Fear & Greed. |
| `CryptoPanicSource` | `sentiment/cryptopanic_source.py` | **M7**, offline ingest. Sets `available_ts = event_ts + configured_lag`. |

### The `available_ts` lag rule

For any historical source without a trustworthy first-observed timestamp:

```text
available_ts = event_ts + sentiment.ingest_lag_minutes    # default 60
```

Sixty minutes is not a guess about API latency; it is an admission that scraped
archives have unreliable publication times and that assuming you saw a headline
the second it was written is the most common sentiment-backtest lie. If a result
depends on that lag being small, the result is not real. `12-RESEARCH_PROTOCOL.md`
requires re-running with `ingest_lag_minutes = 240` as a robustness check.

---

## 6. `Broker`

```python
@runtime_checkable
class Broker(Protocol):
    """Order placement and account state. The only component that talks to an
    exchange."""

    def submit_bracket(
        self,
        entry: OrderIntent,
        stop: OrderIntent,
        target: OrderIntent,
    ) -> tuple[Fill | None, str]:
        """Submit entry plus protective stop plus target as one logical unit.

        Returns (entry_fill_or_None, correlation_id).
        `None` means accepted-but-not-yet-filled (live limit orders). Backtest
        `SimBroker` fills on the next session open, or returns `None` when that
        bar does not exist (`DATA_GAP` — the engine records the rejection).

        SimBroker additionally takes keyword-only `cost: CostEstimate | None`
        and `decision: TradeDecision | None`. The engine must pass both: `cost`
        worsens the entry fill by modelled spread/slip/impact; `decision`
        re-anchors stop and target on the actual fill (same geometry as
        labeling). Omitting them is a unit-test path only.

        Contract:
          - Idempotent on entry.client_order_id. Re-submitting the same id
            must not create a second position.
          - Stop and target are reduce_only.
          - If the stop cannot be placed, the entry MUST be cancelled or
            closed immediately. A naked position is never acceptable.
        """

    def close_position(self, symbol: str, reason: str) -> Fill | None:
        """Reduce-only market exit. Used for the time stop and the kill
        switch. Must work under the maximum possible degradation."""

    def portfolio_state(self, ts: datetime) -> PortfolioState:
        """Current equity, cash, and open positions."""

    def poll_fills(self, ts: datetime) -> tuple[Fill, ...]:
        """Fills since the previous poll, including stop/target exits that
        happened intrabar. Ordered by ts."""
```

Implementations: `SimBroker` (`backtest/sim_broker.py`), `PaperBroker` and
`LiveBroker` (`execution/`, M5). The live client is IBKR or Alpaca, not a
crypto exchange. A Binance client is M7 only.

---

## 7. `DecisionSink`

```python
@runtime_checkable
class DecisionSink(Protocol):
    """Persists the audit trail. Narrow on purpose: this is not a general
    'StorageProvider'. A wide storage interface always grows into a god
    object."""

    def write(self, records: Sequence[DecisionRecord]) -> None:
        """Append. Called once per decision cycle with every candidate
        considered. Must tolerate an empty sequence."""

    def flush(self) -> None:
        """Persist buffered records. Called at run end and every
        `flush_every_cycles` cycles."""
```

Implementations: `ParquetDecisionSink` (buffers, writes row groups),
`NullDecisionSink` (tests), `PostgresDecisionSink` (M5).

---

## 8. Pure-function contracts

Not Protocols. These are the signatures to implement; they carry the same
contractual weight as the Protocols above.

### Features — `features/engine.py`

```python
def compute_features(
    panel: MarketPanel,
    benchmark: BenchmarkPanel,
    snapshots: pd.DataFrame,
    cfg: FeatureConfig,
) -> FeaturePanel:
    """Vectorised per symbol, then concatenated. Signature and steps:
    05-FEATURES_AND_REGIME.md §11.

    Contract:
      - Value at (ts, symbol) depends only on bars with close_time <= ts for
        that symbol, plus SPY/VIX bars with close_time <= ts for market and
        beta columns. Cross-sectional ranks use only symbols eligible at ts.
      - No bfill, no interpolate, no centered windows, no .shift(-n).
      - Insufficient history yields NaN and is_warm=False, never a filled value.
      - Deterministic and order-independent: computing for symbols
        ["A","B"] gives the same rows as ["B","A"].
    """

def classify_regime(er_20: float, er_60: float, slope_atr: float,
                    vol_pct: float, cfg: RegimeConfig) -> Regime:
    """Pure, stateless, no history. Formula in 05-FEATURES_AND_REGIME.md §6."""
```

### Gates — `gates/eligibility.py`

```python
def evaluate_gates(
    feature_row: FeatureRow | None,
    universe_entry: UniverseEntry | None,
    cfg: GatesConfig,
    *,
    is_etf: bool,
    earnings: Sequence[EarningsEvent],
    max_hold_bars: int,
    calendar: pd.DataFrame,
    min_bars_since_gap: int,
    bar_age_bars: int = 0,
    direction: Direction | None = None,
    borrow_bps_per_year: float | None = None,
    hard_to_borrow_max_bps_per_year: float = 300.0,
) -> RejectionReason | None:
    """Return the FIRST failing gate's reason, or None if all pass.

    Owns 14-CONFIG.md §5 rows 1–8 (universe through HARD_TO_BORROW, including
    THIN_CROSS_SECTION). Rows 9–10 (market / per-symbol regime) are the
    engine's, not this function's.

    `universe_entry is None` is gate 1 (`NOT_IN_UNIVERSE`). `max_hold_bars` is
    the strategy's planned hold — the earnings window is `(t, t+max_hold_bars]`,
    never a separate config knob. `earnings_in_window` is an internal helper,
    not a second public gate API.

    Gate order is FIXED. Order matters because the reason recorded is the first
    failure, and funnel analysis depends on that being stable across runs.
    """
```

### Scoring — `scoring/`

```python
# scoring/labeling.py
def resolve_setup(setup: Setup, forward_bars: pd.DataFrame, cfg: LabelConfig) -> ResolvedSetup:
    """Triple-barrier resolution. Rules in 07-EDGE_AND_SCORING.md §2.
    forward_bars: this symbol's bars with ts > setup.ts, ascending, at least
    max_hold_bars rows (fewer means OPEN)."""

def build_edge_table(resolved: pd.DataFrame, cfg: EdgeConfig) -> EdgeTable:
    """Bin, then compute BinStats at each as_of grid point using only setups
    with resolution_ts < as_of."""

# scoring/rank.py
def build_opportunity(setup: Setup, row: FeatureRow, stats: BinStats,
                      cost: CostEstimate, asset: Asset) -> Opportunity: ...

def rank(candidates: Sequence[Opportunity], cfg: ScoringConfig) -> tuple[Opportunity, ...]:
    """Filter by ev_net_r >= min_ev_net_r, then sort by ev_per_bar_r
    descending. Ties broken by (symbol, strategy_id) ascending — never by
    input order."""
```

### Costs — `costs/model.py`

```python
def estimate_cost(
    setup: Setup,
    universe_entry: UniverseEntry,
    notional_usd: float,
    risk_capital_usd: float,
    expected_bars_held: float,
    cfg: CostConfig,
    *,
    atr_pct: float,
    price_raw: float,
    borrow_bps_per_year: float | None = None,
    dividend_yield_annual: float | None = None,
) -> CostEstimate:
    """Formulas in 08-COSTS.md. Same function in backtest and live — there is
    no second implementation and no mode flag. Ranking and the portfolio
    re-check both pass close_raw (ADR-017), never setup.reference_price."""
```

### Portfolio — `portfolio/selection.py`, `portfolio/sizing.py`

```python
def select_and_size(
    ranked: Sequence[Opportunity],
    state: PortfolioState,
    sentiment: Mapping[str, SentimentView],
    assets: Mapping[str, Asset],
    cfg: PortfolioConfig,
    risk_cfg: RiskConfig,
    sentiment_cfg: SentimentConfig,
    cost_cfg: CostConfig,
    min_ev_net_r: float,          # from scoring config; ONE threshold, not two
) -> tuple[TradeDecision, ...]:
    """Greedy in rank order. Returns a decision for EVERY input opportunity,
    accepted or rejected — the rejections are needed for the audit trail.

    Contract:
      - Never increases size relative to the risk-based baseline. Multipliers
        are in (0, 1].
      - Caps are checked against state PLUS already-accepted decisions from
        this same cycle. Checking only against `state` allows three
        simultaneous trades to jointly breach a cap.
      - Deterministic given identical inputs.
    """

def size_position(
    opp: Opportunity, state: PortfolioState, asset: Asset,
    multiplier: float, cfg: PortfolioConfig, *, price_raw: float | None = None,
) -> tuple[Decimal, Decimal]:
    """Returns (qty, risk_usd). qty is step-rounded DOWN, in close_raw shares
    (ADR-017). Decimal boundary starts here. `price_raw` is the unadjusted
    decision close; omitted only when tests set adj == raw."""
```

---

## 9. Errors

`src/scout/utils/errors.py`

```python
class ScoutError(Exception): ...
class ScoutConfigError(ScoutError): ...        # invalid config; fail at startup
class ScoutDataError(ScoutError): ...          # missing/corrupt/stale data
class ScoutExecutionError(ScoutError): ...     # broker rejected or unreachable
class ScoutLookaheadError(ScoutError): ...     # a causality assertion tripped
```

`ScoutLookaheadError` is raised by assertions embedded in the engine, not only by
tests. It is a **hard crash, never caught**. A silent lookahead bug produces a
profitable backtest and a losing live account; crashing is strictly preferable.
Assertion sites listed in [`16-TESTING.md`](16-TESTING.md#6-runtime-tripwires).

---

## 10. Dependency injection

Constructor injection, wired in exactly one place per entry point. No container,
no framework, no service locator.

```python
# src/scout/cli/run_backtest.py  — the ONLY place these are assembled
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)                       # config/*.yaml
    run_id = make_run_id(cfg.run.strategy_slug)

    candles = ParquetCandleSource(cfg.data.processed_dir)
    sentiment = build_sentiment_source(cfg.sentiment)     # Null or Parquet
    sink = ParquetDecisionSink(results_dir(cfg.audit, run_id))
    broker = SimBroker(cfg.costs, cfg.portfolio)
    strategies = build_strategies(cfg.strategies)
    edge_table = EdgeTable.load(cfg.edge.table_path)      # asserts config_hash match

    engine = BacktestEngine(
        candles=candles, sentiment=sentiment, broker=broker,
        strategies=strategies, edge_table=edge_table, sink=sink, cfg=cfg,
    )
    result = engine.run()
    write_run_outputs(result, run_id, cfg)
    return 0
```

Everything below `cli/` receives its dependencies and constructs none of its own.
A module that instantiates `ParquetCandleSource` internally is untestable and
violates this rule.
