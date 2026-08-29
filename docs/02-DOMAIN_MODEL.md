# Domain Model

> Every dataclass, enum, field, type, and unit. This is a specification:
> implement these exactly. Do not add fields "for later". Do not rename.
>
> Location: `src/scout/domain/`.

**Scope: US-listed common stocks and ETFs on daily bars**
([ADR-015](ADR/015-equities-first.md), [ADR-016](ADR/016-daily-decision-bars.md)).
Fields that exist only for the optional crypto sleeve are marked
`# crypto (M7)`. They are retained rather than deleted so the M7 sleeve needs no
schema migration, and they are `0.0`, `None`, or `False` in equity runs.

Two vocabulary rules that apply throughout and that the crypto version did not
have:

- **`asset_id` is the join key everywhere. `symbol` is a display label.** Tickers
  change and are reused ([ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md)).
  Any dict keyed by `symbol`, any merge on `symbol`, and any position lookup by
  `symbol` is a bug.
- **"bars" means trading sessions**, and all bar arithmetic goes through
  `session_index`
  ([`04-DATA_AND_UNIVERSE.md §5`](04-DATA_AND_UNIVERSE.md#5-the-trading-calendar)).

---

## 0. Rules for this layer

1. **`@dataclass(frozen=True, slots=True)`** for every model. Immutability is
   what makes the pipeline safe to reason about; `slots` keeps 100 symbols ×
   50k bars affordable.
2. **No behaviour beyond derivation.** Methods are allowed only if they are pure
   functions of the object's own fields (e.g. `Setup.risk_per_unit`). No I/O,
   no config access, no logging, no other domain objects.
3. **No imports from `src/scout/` except `utils`.** Not `data`, not `config`.
   Checked by `tests/unit/test_layering.py`.
4. **Units in names.** `_r`, `_bps`, `_pct`, `_bars`, `_usd`. A decision-relevant
   float without a unit suffix is a review failure.
5. **`Decimal` only where marked.** Everything else is `float`. See
   [`09-PORTFOLIO_AND_RISK.md`](09-PORTFOLIO_AND_RISK.md#7-the-floatdecimal-boundary).
6. **All datetimes are timezone-aware UTC.** `__post_init__` asserts
   `ts.tzinfo is not None`.
7. **Plain `@dataclass`, not Pydantic**, inside the decision path. Pydantic
   validation cost per object is significant at panel scale and buys nothing when
   the producers are all internal. Pydantic *is* used for config
   ([`14-CONFIG.md`](14-CONFIG.md)) and for external API payload parsing in
   `execution`, where the input is untrusted.

---

## 1. Enums

`src/scout/domain/enums.py`

```python
from enum import Enum


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class Regime(str, Enum):
    """PER-SYMBOL derived label. See 05-FEATURES_AND_REGIME.md SS6."""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"      # warm-up incomplete; blocks all strategies


class MarketRegime(str, Enum):
    """MARKET-WIDE label, computed once per timestamp from the benchmark index
    and broadcast to every symbol's row. See 05-FEATURES_AND_REGIME.md SS7.

    Distinct from `Regime`, and the distinction is load-bearing: `Regime` is one
    symbol's own trend structure, `MarketRegime` is the environment. A stock can
    be TREND_UP in a RISK_OFF market, and that combination is precisely what the
    market gate is there to refuse."""
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    UNKNOWN = "UNKNOWN"      # benchmark warm-up incomplete; blocks everything


class VolBucket(str, Enum):
    """ATR percentile vs the symbol's own trailing 2-year distribution."""
    LOW = "LOW"              # percentile < 0.33
    MID = "MID"              # 0.33 <= percentile < 0.67
    HIGH = "HIGH"            # percentile >= 0.67
    UNKNOWN = "UNKNOWN"


class SetupOutcome(str, Enum):
    TARGET = "TARGET"        # target touched first (or gapped through, in our favour)
    STOP = "STOP"            # stop touched first, gapped through, or both in one bar
    TIME = "TIME"            # max_hold_bars reached; exit at that session's close
    OPEN = "OPEN"            # not yet resolved; excluded from edge statistics


class RejectionReason(str, Enum):
    """Closed set. Every rejected candidate carries exactly one of these.
    Never write a free-text reason: the decision log is queried by this column."""
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"     # warm-up bars missing
    DATA_GAP = "DATA_GAP"
    DATA_SUSPECT = "DATA_SUSPECT"                     # failed a quality check
    STALE_DATA = "STALE_DATA"
    LOW_PRICE = "LOW_PRICE"                           # close_raw below min_price_usd
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    WIDE_SPREAD = "WIDE_SPREAD"
    MARKET_REGIME_BLOCKED = "MARKET_REGIME_BLOCKED"
    REGIME_BLOCKED = "REGIME_BLOCKED"                 # per-symbol regime
    EARNINGS_IN_WINDOW = "EARNINGS_IN_WINDOW"
    HARD_TO_BORROW = "HARD_TO_BORROW"                 # shorts only
    THIN_CROSS_SECTION = "THIN_CROSS_SECTION"         # xs_population too small
    NO_SETUP = "NO_SETUP"
    INSUFFICIENT_BIN_SAMPLES = "INSUFFICIENT_BIN_SAMPLES"
    BELOW_EV_THRESHOLD = "BELOW_EV_THRESHOLD"
    COST_UNAVAILABLE = "COST_UNAVAILABLE"
    ALREADY_IN_POSITION = "ALREADY_IN_POSITION"
    PORTFOLIO_HEAT_CAP = "PORTFOLIO_HEAT_CAP"
    CLUSTER_CAP = "CLUSTER_CAP"
    BETA_CAP = "BETA_CAP"
    GROSS_EXPOSURE_CAP = "GROSS_EXPOSURE_CAP"
    MAX_POSITIONS = "MAX_POSITIONS"
    BELOW_TOP_N = "BELOW_TOP_N"
    SENTIMENT_VETO = "SENTIMENT_VETO"
    SIZE_BELOW_MIN_NOTIONAL = "SIZE_BELOW_MIN_NOTIONAL"
    KILL_SWITCH = "KILL_SWITCH"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


class ActionType(str, Enum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"
    SPINOFF = "SPINOFF"
    MERGER = "MERGER"
    TICKER_CHANGE = "TICKER_CHANGE"


class OrderType(str, Enum):
    MARKET = "MARKET"                          # market-on-open for entries
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class RunMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"
```

### Why `RejectionReason` is an enum and not a string

The decision log is the primary research artifact and it is queried by grouping
on this column. Free-text reasons produce 40 spellings of the same thing and make
the log unusable within a week. Adding a new reason requires adding an enum
member, which is a visible, reviewable act.

---

## 2. Market data

`src/scout/domain/market.py`

```python
@dataclass(frozen=True, slots=True)
class Asset:
    """Static and slowly-changing symbol metadata. Loaded from the vendor's
    ticker table; never derived inside the loop."""
    asset_id: str            # vendor PERMANENT id. The join key everywhere
    symbol: str              # display ticker as of the pinned data snapshot
    exchange: str            # "NYSE" | "NASDAQ" | "NYSEARCA" | "BATS"
    quote_currency: str      # "USD" for all equities
    is_etf: bool
    cluster: str             # GICS sector, or an ETF cluster. See ADR-007
    tick_size: Decimal       # "0.01" above $1.00
    step_size: Decimal       # "1" for whole shares; "0.0001" if fractional
    min_notional_usd: Decimal  # "0" for US equities
    listed_at: datetime | None    # UTC; None if unknown
    delisted_at: datetime | None  # UTC; None if still listed
    delist_reason: str | None     # "MERGER" | "BANKRUPTCY" | "OTHER" | None.
                                  # Drives the exit rule; see 04-DATA SS7.5
    is_perpetual: bool = False    # crypto (M7)


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One row of data/raw/equity/actions.parquet. Consumed by data/adjust.py
    to build back-adjustment factors, and by SimBroker to apply dividend cash
    flows. See ADR-017."""
    asset_id: str
    ex_date: date
    action_type: ActionType
    split_ratio: float       # new shares per old; 4.0 for 4-for-1. 1.0 otherwise
    cash_amount: float       # dividend per share USD. 0.0 otherwise
    new_symbol: str | None   # TICKER_CHANGE only


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """One announcement. `available_ts` is what makes the earnings gate causal;
    see ADR-020 for the conservative fallback when the vendor omits it."""
    asset_id: str
    earnings_date: date
    is_confirmed: bool           # False means a vendor estimate
    available_ts: datetime | None  # when WE could first have known. None => fallback
    timing: str                  # "BMO" | "AMC" | "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Session:
    """One row of the materialised trading calendar. `session_index` is the
    ONLY correct basis for bar arithmetic: adding calendar days across a
    holiday is an off-by-one that no test will catch by accident."""
    session: date
    open_utc: datetime           # DST-correct; never a fixed offset from `session`
    close_utc: datetime          # this is the panel `ts`
    is_half_day: bool
    session_index: int


@dataclass(frozen=True, slots=True)
class Bar:
    """A closed session bar. `ts` is the CLOSE time. See the bar convention in
    docs/README.md. Used for single-bar work and tests; the panel is a
    DataFrame, not a list of these."""
    asset_id: str
    symbol: str
    ts: datetime             # UTC, session close
    session_index: int
    open: float              # ADJUSTED
    high: float
    low: float
    close: float
    close_raw: float         # UNADJUSTED. Dollar-price gates and share counts only
    volume: float            # shares, adjusted
    dollar_volume: float     # close_raw * volume_raw; used for ADV
    is_suspect: bool         # failed a data-quality check; see 04-DATA SS6.7
```

### The `close` / `close_raw` split

The most consequential field pair in this document
([ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md)):

| Use | Column |
|---|---|
| Anything scale-free: returns, ATR, ratios, momentum, channels, features | `close` (adjusted) |
| Anything denominated in dollars or shares: price gates, share counts, tick rounding, commission | `close_raw` |

A `$5.00` minimum-price filter applied to the adjusted series excludes stocks that
traded at `$40` before a 10-for-1 split and admits stocks that traded at `$0.80`.
Both errors are invisible in the output and both are large. Enforced by naming and
by `tests/unit/test_adjustment.py`.

### `MarketPanel`

Not a dataclass. A thin wrapper over one pandas DataFrame, because the panel is
the hot object and per-row dataclasses would be 100× slower.

```python
class MarketPanel:
    """Immutable long-format panel of closed session bars.

    Backing frame columns, exactly, in this order:
        asset_id: str (category)
        symbol: str (category)         # display only
        ts: datetime64[ns, UTC]        # session close time
        session_index: int32
        open, high, low, close: float64          # ADJUSTED
        close_raw: float64                       # UNADJUSTED
        volume, dollar_volume: float64
        is_suspect: bool

    Index: RangeIndex. Sorted by (ts, asset_id). Never mutated in place.
    float64 throughout: float32 has ~7 significant digits, and a $400 stock
    with a $0.01 tick needs 6 -- too close to the edge for comparisons against
    stop levels.
    """

    def __init__(self, frame: pd.DataFrame) -> None: ...

    @property
    def frame(self) -> pd.DataFrame: ...        # returns a read-only view
    @property
    def asset_ids(self) -> tuple[str, ...]: ...
    @property
    def timestamps(self) -> pd.DatetimeIndex: ...  # unique, sorted

    def as_of(self, ts: datetime) -> "MarketPanel":
        """All rows with self.frame.ts <= ts. THE causality primitive."""

    def slice_symbol(self, asset_id: str) -> pd.DataFrame:
        """Single asset, ts-indexed, sorted ascending."""

    def latest(self, ts: datetime) -> pd.DataFrame:
        """One row per asset: the row with the greatest ts <= given ts."""


class BenchmarkPanel:
    """The benchmark series, loaded from data/reference/benchmark_1d.parquet and
    NOT from the universe panel. Separate because market-regime features are
    computed once per timestamp rather than once per symbol, and because SPY must
    be available even in periods when the universe is empty or SPY would fail a
    data-quality check. Making that dependency explicit rather than implicit is
    the point.

    Columns: ts, session_index, open, high, low, close,
             vix_close, vix9d_close, vix3m_close   (VIX columns nullable; see
             05-FEATURES_AND_REGIME.md SS7 for why they are NOT in the regime
             definition)
    """

    def as_of(self, ts: datetime) -> "BenchmarkPanel": ...
```

`as_of` is the single chokepoint for causality. Every consumer of history goes
through it, so there is one place to test and one place to break if someone gets it
wrong.

---

## 3. Features

`src/scout/domain/features.py`

```python
@dataclass(frozen=True, slots=True)
class FeatureRow:
    """All features for one (ts, symbol). Fixed schema — adding a feature means
    adding a field here and a column in FeaturePanel. No dict-of-floats: a
    typo in a dict key is a silent zero, and silent zeros in trading code cost
    money."""

    symbol: str
    ts: datetime

    # --- price / reference ---
    close: float
    atr_14: float                # Wilder ATR, absolute price units
    atr_pct: float               # atr_14 / close
    vol_20: float                # annualised log-return vol, 20 sessions
    vol_60: float

    # --- trend ---
    ema_fast: float              # length from config, default 20
    ema_slow: float              # default 50
    ema_spread_atr: float        # (ema_fast - ema_slow) / atr_14, signed
    slope_atr_20: float          # (close - close[-20]) / (atr_14 * sqrt(20))

    # --- regime backbone ---
    efficiency_ratio_20: float   # Kaufman ER, 20 sessions, [0, 1]
    efficiency_ratio_60: float   # same, 60 sessions (longer-horizon agreement)
    atr_percentile_1y: float     # [0, 1], rank of atr_pct in trailing ~2y
    regime: Regime
    vol_bucket: VolBucket

    # --- channels / levels ---
    donchian_high_20: float
    donchian_low_20: float
    donchian_high_55: float      # Donchian v1 uses 55; 20 is recorded for M6
    donchian_low_55: float
    keltner_upper: float         # ema_slow + k * atr_14
    keltner_lower: float
    dist_to_high_atr: float      # (donchian_high_55 - close) / atr_14
    dist_to_low_atr: float       # (close - donchian_low_55) / atr_14

    # --- momentum (primary strategy inputs) ---
    mom_252_skip21: float        # 12-1 total return; NOT a tunable
    mom_126_skip21: float        # recorded, not used in v1 decisions
    mom_21: float                # recorded, not used in v1 decisions
    mom_252_xs_pct: float        # rank of mom_252_skip21 in the eligible set at t
    vol_xs_pct: float            # cross-sectional vol rank
    xs_population: int           # eligible-set size used for the ranks above

    # --- gaps (recorded; not used in v1 decisions) ---
    gap_atr: float
    gap_abs_mean_20: float
    overnight_var_share_60: float

    # --- market (broadcast from BenchmarkPanel) ---
    market_regime: MarketRegime
    spy_dd_252: float
    spy_above_ma: bool
    vix_close: float             # nullable; NOT in the regime definition

    # --- beta / correlation vs SPY ---
    beta_bench_90: float         # rolling 90-session beta vs SPY
    corr_bench_90: float         # rolling 90-session correlation vs SPY
                                 # crypto (M7): same fields, BTC as the benchmark

    # --- data quality ---
    bars_available: int          # bars with close_time <= ts for this symbol
    bars_since_gap: int          # bars since the last detected gap; large is good
    is_warm: bool                # bars_available >= required warm-up

    def to_dict(self) -> dict[str, float | str | int | bool]: ...
```

`FeaturePanel` mirrors `MarketPanel`: a DataFrame wrapper whose columns are
exactly `FeatureRow`'s fields, with `row(ts, symbol) -> FeatureRow` for the
few places that want the typed object (strategies, tests).

**Naming rule enforced by test:** any feature expressed in price units is
normalised by ATR and carries the `_atr` suffix. Raw price differences are not
comparable across assets, and this system's entire premise is cross-asset
comparison. `ema_spread_atr` is comparable between BTC at $60,000 and DOGE at
$0.12; `ema_spread` is not.

---

## 4. Universe

`src/scout/domain/universe.py`

```python
@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    eligible: bool
    reason: RejectionReason | None       # None iff eligible
    adv_usd_30: float                    # trailing 30-bar median quote volume,
                                         # scaled to a daily figure
    spread_bps_est: float                # causal estimate; see 04-DATA doc
    bars_available: int
    listed_days: float


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    ts: datetime
    entries: Mapping[str, UniverseEntry]     # keyed by symbol

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        """Sorted, for determinism. Never rely on dict insertion order."""
```

`eligible_symbols` sorts. Iteration order affects tie-breaking in ranking, and
tie-breaking affects results, so it must not depend on dict internals.

---

## 5. Setups

`src/scout/domain/setup.py`

```python
@dataclass(frozen=True, slots=True)
class Setup:
    """A strategy's complete statement of an intended trade, in price space.
    Contains no money, no size, no probability, no score. A Setup is a
    geometric claim, nothing more."""

    symbol: str
    ts: datetime                 # decision bar close; entry is at ts+1 open
    strategy_id: str             # registry key, e.g. "donchian_breakout_v1"
    direction: Direction
    regime: Regime               # regime at detection; recorded, not re-derived

    reference_price: float       # close at ts; entry is estimated from this
    stop_price: float            # protective stop, absolute
    target_price: float          # take profit, absolute
    max_hold_bars: int           # time stop

    trigger_note: str            # short, stable label for the exact rule that
                                 # fired, e.g. "close>dc_high_20". For analysis,
                                 # never parsed by code.

    def __post_init__(self) -> None:
        """Validate, hard. A malformed Setup must fail at construction, not
        produce a nonsensical trade 400 bars later."""
        # stop must be on the losing side, target on the winning side
        # risk_per_unit must be > 0
        # reward_risk_ratio must be > 0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.reference_price - self.stop_price)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.target_price - self.reference_price)

    @property
    def reward_risk_ratio(self) -> float:
        return self.reward_per_unit / self.risk_per_unit


@dataclass(frozen=True, slots=True)
class ResolvedSetup:
    """A Setup carried forward through the triple-barrier labeling pass. One
    row of data/labels/setups_<strategy_id>.parquet; schema and semantics in
    07-EDGE_AND_SCORING.md §3. `realised_r_gross` is BEFORE costs — the
    labeling pass has no account and no size, so it cannot compute them."""

    setup: Setup
    entry_ts: datetime
    entry_price: float
    resolution_ts: datetime | None      # None iff outcome is OPEN
    exit_price: float | None
    outcome: SetupOutcome
    bars_held: int
    realised_r_gross: float
    mae_r: float
    mfe_r: float
    vol_bucket: VolBucket               # at detection
```

### Why `Setup` has no `confidence`

The brief asked each strategy to report "how confident am I". A number a strategy
asserts about itself is not evidence; it is an extra parameter that cannot be
falsified and will be tuned until the backtest improves. Confidence in this
system is *measured*: it is the width of the confidence interval around the
historical outcome of setups like this one, computed in `scoring`. A strategy's
job is to be precise about geometry and silent about quality.

---

## 6. Edge statistics

`src/scout/domain/edge.py`

```python
@dataclass(frozen=True, slots=True)
class BinKey:
    """The conditioning cell whose history is used to estimate this setup's
    edge. Keep the cardinality small — see 07-EDGE_AND_SCORING.md §4."""
    strategy_id: str
    direction: Direction
    vol_bucket: VolBucket

    def as_tuple(self) -> tuple[str, str, str]: ...


@dataclass(frozen=True, slots=True)
class BinStats:
    """Outcome statistics for one bin, computed from setups RESOLVED strictly
    before `as_of`. All *_r fields are in units of one trade's risk."""

    key: BinKey
    as_of: datetime

    n: int                       # resolved setups in the sample
    mean_r: float                # mean realised R
    std_r: float                 # sample standard deviation of realised R
    ev_r_lcb: float              # one-sided lower confidence bound on mean_r
    mean_bars_held: float

    # diagnostics only; never enter the ranking statistic
    win_rate: float              # fraction with realised_r > 0
    avg_win_r: float
    avg_loss_r: float             # negative
    target_rate: float           # fraction resolved as TARGET
    stop_rate: float
    time_rate: float
    median_r: float
    p05_r: float
    p95_r: float

    @property
    def is_usable(self) -> bool:
        """n >= min_bin_samples AND std_r > 0 AND all values finite."""
```

```python
class EdgeTable:
    """Bin statistics indexed by (BinKey, as_of). Built once per run by the
    labeling pass; queried per decision. Lookup must be O(log n), not a scan.

    Precomputed on a coarse grid (default: monthly `as_of` points) and looked up
    with backward search, so a decision at 2023-04-17 uses the 2023-04-01 table.
    This is deliberately STALE, never fresh: staleness is safe, freshness risks
    including a setup that had not yet resolved.
    """

    def lookup(self, key: BinKey, as_of: datetime) -> BinStats | None: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "EdgeTable": ...
```

The monthly `as_of` grid is a deliberate simplification. Recomputing bin stats at
every one of ~13,000 decision bars is wasteful, and the statistics move slowly.
Rounding *backwards* means the error is always in the conservative direction.

---

## 7. Costs

`src/scout/domain/costs.py`

```python
@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Round-trip execution cost. `cost_r` is the only field the ranking uses;
    the rest exist so a bad cost estimate can be diagnosed instead of guessed
    at."""

    symbol: str
    ts: datetime
    notional_usd: float          # the notional this estimate was computed for
    expected_bars_held: float

    entry_fee_bps: float
    exit_fee_bps: float
    spread_bps: float            # full spread crossed once per side
    slippage_bps: float
    impact_bps: float
    funding_bps: float           # signed: positive = we pay
    total_bps: float             # sum of the above

    cost_usd: float              # total_bps / 1e4 * notional_usd
    cost_r: float                # cost_usd / risk_capital_usd

    @property
    def is_valid(self) -> bool:
        """All components finite, non-negative except funding, cost_r > 0."""
```

`cost_r` depends on notional and on `risk_capital_usd`, which means it depends on
intended size, which is decided later. Resolution: the ranking uses a
**provisional** notional derived from `equity * risk_fraction / risk_per_unit *
reference_price`. Since both numerator and denominator scale with `risk_fraction`,
`cost_r` is nearly invariant to it — the residual dependence is only through the
nonlinear impact term. Sizing recomputes the cost with the final notional, and
if `ev_net_r` falls below threshold at that point the trade is dropped with
`BELOW_EV_THRESHOLD`. This second check matters only for large orders in thin
markets, which is exactly where it should matter.

---

## 8. Opportunity

`src/scout/domain/opportunity.py`

```python
@dataclass(frozen=True, slots=True)
class Opportunity:
    """A scored candidate. Exactly one ranking statistic; everything else is a
    diagnostic or provenance."""

    # --- identity ---
    symbol: str
    ts: datetime
    strategy_id: str
    direction: Direction
    setup: Setup

    # --- the ranking statistic ---
    ev_net_r: float              # ev_r_lcb - cost_r. Units: R. THE number.
    ev_per_bar_r: float          # ev_net_r / max(expected_bars_held, 1.0)

    # --- components, for auditing the number above ---
    ev_r_lcb: float
    ev_r_point: float            # bin mean_r, un-penalised
    cost_r: float
    expected_bars_held: float

    # --- provenance ---
    bin_key: BinKey
    bin_n: int
    bin_std_r: float
    bin_win_rate: float
    regime: Regime
    vol_bucket: VolBucket

    # --- context recorded for research, not used in ranking ---
    adv_usd_30: float
    spread_bps_est: float
    beta_bench_90: float
    cluster: str
```

### Comparison to the brief's `Opportunity`

| Brief field | Disposition |
|---|---|
| `probability` | Kept as `bin_win_rate`, diagnostic only. Not in the ranking. |
| `expected_return`, `expected_loss` | Removed. Both are inside `ev_r_point` by construction; reporting them separately invites recombining them with new weights. |
| `expectancy` | Renamed `ev_r_point`, with a unit in the name. |
| `sentiment_score` | Removed from this object. Sentiment is a portfolio-stage size multiplier and is recorded on `TradeDecision`, not here. This keeps ranking sentiment-free by construction. |
| `final_score` | Renamed `ev_net_r`. Same role, but it now has a unit and is derived rather than weighted. |
| `confidence` | Removed as a field. It is *expressed* as the gap between `ev_r_point` and `ev_r_lcb`, which is computable and auditable. |

---

## 9. Portfolio and decisions

`src/scout/domain/portfolio.py`

```python
@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    direction: Direction
    qty: Decimal                 # base units, already step-rounded
    entry_price: Decimal
    entry_ts: datetime
    stop_price: Decimal
    target_price: Decimal
    max_hold_bars: int
    bars_held: int
    strategy_id: str
    cluster: str
    client_order_id: str         # idempotency key of the entry
    realised_fees_usd: Decimal
    dividends_usd: Decimal       # cash dividends received (longs) or paid (shorts)
    borrow_usd: Decimal          # short-stock borrow accrued
    funding_paid_usd: Decimal    # crypto (M7); zero for equities

    def open_risk_usd(self, mark: Decimal) -> Decimal:
        """Distance from mark to stop, times qty. Never negative: if the stop
        has moved through the mark, risk is 0."""

    def notional_usd(self, mark: Decimal) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class PortfolioState:
    ts: datetime
    equity_usd: Decimal         # cash + unrealised
    cash_usd: Decimal
    positions: Mapping[str, Position]

    peak_equity_usd: Decimal    # for drawdown circuit breaker
    realised_pnl_today_usd: Decimal
    day_start_equity_usd: Decimal
    trades_today: int
    bars_since_breaker: int     # large sentinel when no breaker has tripped

    @property
    def open_risk_pct(self) -> float:
        """Portfolio heat: sum of open_risk_usd / equity_usd."""

    @property
    def gross_exposure_pct(self) -> float: ...
    @property
    def net_beta_exposure_pct(self) -> float: ...
    def cluster_risk_pct(self, cluster: str) -> float: ...
    def drawdown_pct(self) -> float:
        """(peak_equity_usd - equity_usd) / peak_equity_usd, floored at 0."""
    def day_loss_pct(self) -> float:
        """(day_start_equity_usd - equity_usd) / day_start_equity_usd, floored
        at 0. Includes unrealised, so an open losing position can trip the
        daily breaker before it is closed. That is intended."""


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """The portfolio stage's verdict on one Opportunity."""

    opportunity: Opportunity
    accepted: bool
    rejection_reason: RejectionReason | None    # None iff accepted
    rank: int                                   # 1-based, pre-portfolio-gate

    # populated only when accepted
    qty: Decimal | None
    entry_order_type: OrderType | None
    intended_notional_usd: Decimal | None
    risk_usd: Decimal | None
    size_multiplier: float | None       # product of all penalties, (0, 1]
    sentiment_multiplier: float | None  # the sentiment component of the above
    client_order_id: str | None
    final_cost: CostEstimate | None
```

---

## 10. Execution

`src/scout/domain/execution.py`

```python
@dataclass(frozen=True, slots=True)
class OrderIntent:
    client_order_id: str         # idempotency key; deterministic, see below
    symbol: str
    side: Direction
    order_type: OrderType
    qty: Decimal
    limit_price: Decimal | None
    stop_price: Decimal | None
    reduce_only: bool
    created_ts: datetime
    correlation_id: str          # links entry, stop, target, and the decision


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    symbol: str
    side: Direction
    qty: Decimal
    price: Decimal
    fee_usd: Decimal
    ts: datetime
    is_maker: bool
    exit_reason: str | None      # "TARGET" | "STOP" | "TIME" | "DELISTED"
                                # | "KILL_SWITCH" | None for entries
```

### `client_order_id` construction

```python
client_order_id = f"{run_id[:8]}-{symbol}-{int(ts.timestamp())}-{leg}"
# leg ∈ {"e", "s", "t"}  (entry, stop, target)
```

Deterministic from `(run, symbol, decision bar, leg)`. A retry after a network
timeout regenerates the identical id, so the exchange rejects the duplicate
rather than double-filling. This satisfies the global idempotency rule without a
UUID table.

---

## 11. Sentiment

`src/scout/domain/sentiment.py`

```python
@dataclass(frozen=True, slots=True)
class SentimentObservation:
    """One raw, timestamped observation. Point-in-time discipline lives in the
    two timestamp fields and nowhere else."""

    symbol: str
    source: str                  # e.g. "vix_term", "short_interest"
    event_ts: datetime           # when the event/publication happened
    available_ts: datetime       # when WE could first have known it.
                                 # MUST be >= event_ts. Backtests filter on
                                 # this field, never on event_ts.
    raw_score: float             # source-native
    score: float                 # normalised to [-1, 1]
    source_weight: float         # (0, 1], static per source, from config
    sample_size: int             # articles/posts behind this observation
    payload_hash: str            # dedupe key


@dataclass(frozen=True, slots=True)
class SentimentView:
    """Aggregated sentiment for one symbol as of one decision timestamp.
    Built only from observations with available_ts <= ts."""

    symbol: str
    ts: datetime
    score: float                 # [-1, 1], decay-weighted
    confidence: float            # [0, 1], from sample size, freshness, agreement
    sample_size: int
    freshness_bars: float        # bars since the newest contributing observation
    source_count: int
    disagreement: float          # [0, 1]: weighted stdev across sources
    is_stale: bool               # freshness_bars > config max; then treat neutral

    @property
    def is_neutral(self) -> bool:
        """True when stale, empty, or confidence below the config floor. A
        neutral view produces a multiplier of exactly 1.0."""
```

---

## 12. Decision record — the audit row

`src/scout/domain/audit.py`

```python
@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One row per (decision_ts, symbol, strategy_id) CONSIDERED, accepted or
    not. This is the system's most valuable output. Written for every cycle,
    including cycles with zero trades.

    Flat and denormalised on purpose: it becomes one Parquet table that
    answers research questions with a groupby instead of a join.
    """

    run_id: str
    decision_ts: datetime
    symbol: str
    strategy_id: str | None      # None for pre-strategy rejections

    stage: str                   # "GATE"|"STRATEGY"|"EDGE"|"RANK"|"PORTFOLIO"|"SIZE"
    accepted: bool
    rejection_reason: RejectionReason | None

    direction: Direction | None
    regime: Regime | None
    vol_bucket: VolBucket | None

    reference_price: float | None
    stop_price: float | None
    target_price: float | None
    reward_risk_ratio: float | None

    bin_key: str | None          # BinKey rendered as "strategy|dir|vol"
    bin_n: int | None
    ev_r_point: float | None
    ev_r_lcb: float | None
    cost_r: float | None
    ev_net_r: float | None
    ev_per_bar_r: float | None
    rank: int | None

    sentiment_score: float | None
    sentiment_confidence: float | None
    sentiment_multiplier: float | None

    portfolio_heat_pct: float | None
    cluster: str | None
    size_multiplier: float | None
    qty: float | None            # float here; this row is analytics, not a ledger
    notional_usd: float | None

    adv_usd_30: float | None
    spread_bps_est: float | None
    features_json: str | None    # compact JSON of the FeatureRow, for forensics
```

`stage` records how far the candidate got. `groupby(["stage", "rejection_reason"])`
over a full run immediately shows where the funnel loses candidates — the concrete
version of the brief's 100 → 80 → 15 → 5 → 2 diagram, measured rather than
imagined. If 99.8% of candidates die at `GATE`, a gate is misconfigured, and this
is the only artifact that will tell you.

---

## 13. Backtest results

`src/scout/domain/results.py`

```python
@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A completed round trip. Money fields are Decimal: this feeds the ledger
    and P&L accumulation."""
    symbol: str
    strategy_id: str
    direction: Direction
    entry_ts: datetime
    exit_ts: datetime
    entry_price: Decimal
    exit_price: Decimal
    qty: Decimal
    bars_held: int
    outcome: SetupOutcome
    gross_pnl_usd: Decimal
    fees_usd: Decimal
    dividends_usd: Decimal
    borrow_usd: Decimal
    funding_usd: Decimal         # crypto (M7); zero for equities
    net_pnl_usd: Decimal
    realised_r: float            # net_pnl / risk_at_entry. float: analytics.
    mae_r: float                 # max adverse excursion, in R
    mfe_r: float                 # max favourable excursion, in R
    regime: Regime
    vol_bucket: VolBucket
    cluster: str
    ev_net_r_at_entry: float     # what we predicted. Enables calibration checks.


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: str
    config_hash: str
    started_at: datetime
    period_start: datetime
    period_end: datetime
    mode: RunMode

    trades: tuple[ClosedTrade, ...]
    equity_curve: pd.Series      # UTC DatetimeIndex, float equity
    benchmark_curve: pd.Series   # buy-and-hold SPY (total return), same index, normalised
    metrics: Mapping[str, float]
    n_decisions_considered: int
    n_rejections_by_reason: Mapping[RejectionReason, int]
```

`ev_net_r_at_entry` on every closed trade is what makes the system falsifiable.
Bucketing realised R by predicted `ev_net_r` produces a calibration curve. If
predicted 0.15 R delivers 0.02 R, the estimator is broken, and you learn it from
one plot instead of from live losses. This check is mandatory in M3 —
[`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md).
