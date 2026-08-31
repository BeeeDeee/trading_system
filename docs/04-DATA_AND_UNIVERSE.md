# Data and Universe

> Storage layout, Parquet schemas, corporate-action handling, the trading
> calendar, point-in-time universe construction, and the data-quality rules that
> everything downstream relies on.

**Scope: US-listed common stocks and ETFs** ([ADR-015](ADR/015-equities-first.md)).
The crypto equivalents are in [§10](#10-appendix-crypto-m7-optional), retained for
the optional M7 sleeve.

---

## 1. Data source: this decision is load-bearing

Read this section before writing any code. A wrong choice here invalidates
everything downstream, and it does so invisibly.

### 1.1 What the system actually requires

| Requirement | Why it is non-negotiable |
|---|---|
| **Delisted symbols included**, with delisting dates | Without them the backtest deletes every company that failed. Survivorship bias, and it is unbounded — you cannot estimate how wrong you are |
| **Unadjusted OHLCV** | Adjusted series are retroactively revised, so a backtest built on them is not reproducible ([ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md)) |
| **Corporate actions table** with ex-dates: splits, cash dividends, spinoffs | Needed to adjust causally and to charge shorts for dividends |
| **A permanent symbol id** independent of the display ticker | Tickers are reused. `FB → META`, and a new `FB` could list tomorrow |
| **Earnings dates**, ideally with an as-of | Required by the earnings gate ([ADR-020](ADR/020-earnings-gate.md)) |
| **Sector classification** | Cluster caps ([ADR-007](ADR/007-cluster-caps-not-covariance.md)) |
| **20+ years of history** | Regime variety is the point ([§7](#7-data-splits)) |

### 1.2 Recommended sources

| Source | Cost | Verdict |
|---|---|---|
| **Norgate Data** (Platinum) | ~$70/mo | **Recommended.** Survivorship-free US equities to 1990, unadjusted plus full actions, delisted included, native Windows, a documented Python API, and historical index constituents if you later want the ADR-018 robustness check |
| **Sharadar** `SEP` + `ACTIONS` + `TICKERS` + `SF1`, via Nasdaq Data Link | ~$60/mo | **Recommended alternative.** `permaticker` is a genuine permanent id, delisting dates included, actions and earnings dates included. Plain REST, easy to script |
| **Polygon.io** Stocks Starter | ~$29–199/mo | Workable. Full history, splits and dividends, delisted tickers present. Sector data and earnings dates are weaker |
| **EODHD** | ~$20–80/mo | Cheapest of the credible options. Verify delisted coverage on the specific plan before committing |
| **yfinance / Yahoo** | free | **Prototype only.** No delisted symbols, adjusted-close only and silently revised, no reliable actions table, undocumented backfills. Fine for wiring up M1 against fixtures. **A result produced on this data is not a result** |
| **Alpaca free tier** | free | IEX-only trades, thin history. Good for live execution later; not for research |

Budget about $60–70/month. If that is not acceptable, the honest options are to
abandon the project or to accept that the output is an exercise rather than a
decision input. There is no free dataset that satisfies §1.1.

### 1.3 The pinned snapshot

Every ingest writes `data/raw/equity/SNAPSHOT.json`:

```json
{
  "data_snapshot_id": "20260301-sharadar",
  "vendor": "sharadar",
  "created_utc": "2026-03-01T22:14:03Z",
  "symbols": 4812,
  "rows": 21044318,
  "date_min": "1998-01-02",
  "date_max": "2026-02-27",
  "actions_rows": 184402,
  "earnings_rows": 391155
}
```

`data_snapshot_id` is **part of the config hash**. A backtest whose snapshot id
differs from its edge table's snapshot id **fails at startup**. Re-downloading
data is therefore a deliberate, visible invalidation rather than a silent drift in
results. See [ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md).

---

## 2. Storage principles

1. **Parquet on disk, DuckDB for queries.** No database server before M5.
   1,500 candidate symbols × 25 years of daily bars is roughly 9.5M rows, about
   250 MB as Parquet with dictionary-encoded symbols. DuckDB scans that in about a
   second and requires no daemon, no migration, and no network.
2. **Raw is immutable.** `data/raw/` is what the vendor returned. It is never
   edited, cleaned, or overwritten. Every correction — including corporate-action
   adjustment — happens on the way to `data/processed/`, so reprocessing never
   requires re-downloading and never changes the snapshot id.
3. **Long format, not wide.** `(ts, asset_id)` rows rather than one column per
   symbol. Wide format breaks the moment the universe changes size, which for a
   point-in-time universe is constantly.
4. **Join on `asset_id`, never on `symbol`.** `symbol` is a display label that
   changes and is reused. Every join, every index, every dictionary key uses the
   vendor's permanent id.
5. **Everything gitignored.** `data/`, `results/`, `logs/`, `.env` are never
   committed.

---

## 3. Directory layout

```text
data/
├── raw/
│   └── equity/
│       ├── SNAPSHOT.json                    # the pinned snapshot manifest
│       ├── ohlcv/                           # UNADJUSTED. Never edited.
│       │   └── 1d/
│       │       ├── 1998.parquet             # ALL symbols, one file per year
│       │       └── ...
│       ├── actions.parquet                  # splits, dividends, spinoffs
│       ├── tickers.parquet                  # asset metadata incl. delist dates
│       └── earnings.parquet                 # announcement dates
│
├── processed/
│   └── panel/
│       └── 1d/
│           ├── 1998.parquet                 # adjusted + raw close, one file/year
│           └── ...
│
├── reference/
│   ├── calendar_xnys.parquet                # materialised session grid
│   └── benchmark_1d.parquet                 # SPY + VIX, for market regime
│
├── universe/
│   ├── snapshots.parquet                    # point-in-time eligibility
│   └── assets.parquet                       # Asset metadata, resolved
│
├── labels/
│   └── setups_<strategy_id>.parquet         # resolved historical setups
│
├── edge/
│   └── edge_table.parquet                   # BinStats on the as_of grid;
│                                            # config_hash and data_snapshot_id in
│                                            # its metadata; path from config
│                                            # `edge.table_path`
│
└── sentiment/
    └── observations.parquet                 # point-in-time observations
```

Partitioning by year keeps single-year reprocessing cheap and files small enough
to inspect. Partitioning by symbol as well would produce tens of thousands of tiny
files and make the cross-sectional read — the only read pattern that matters —
needlessly slow.

---

## 4. Corporate actions and adjustment

The largest new correctness surface in the equity pivot. Read
[ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md) first.

### 4.1 `data/raw/equity/actions.parquet`

| Column | Type | Notes |
|---|---|---|
| `asset_id` | `string` | permanent id |
| `ex_date` | `date32` | first session the price reflects the action |
| `action_type` | `string` | `SPLIT`, `DIVIDEND`, `SPINOFF`, `MERGER`, `TICKER_CHANGE` |
| `split_ratio` | `double` | new shares per old share; 4.0 for a 4-for-1. 1.0 for non-splits |
| `cash_amount` | `double` | dividend per share in USD. 0.0 for non-dividends |
| `new_symbol` | `string` | for `TICKER_CHANGE`, else empty |

### 4.2 Ratio back-adjustment

```python
def adjustment_factors(actions: pd.DataFrame, sessions: pd.DatetimeIndex,
                       last_close: float) -> pd.Series:
    """Cumulative multiplicative factor per session, ending at 1.0 on the
    final session. Applied to open/high/low/close; volume is divided by it."""
    # Per-action factor, effective from ex_date onward (i.e. applied to all
    # sessions STRICTLY BEFORE ex_date when accumulating backwards).
    #   SPLIT:    f = 1 / split_ratio
    #   DIVIDEND: f = 1 - cash_amount / close_on_session_before_ex_date
    #   SPINOFF:  f = 1 - spinoff_value / close_on_session_before_ex_date
    # Accumulate from the most recent session backwards; the final session's
    # factor is 1.0 by construction.
```

Applied in `data/adjust.py`, called by `cli/build_panel.py`.

**Why this is not lookahead.** Back-adjustment scales all history before an
ex-date by a constant. Every return, every ratio, every ATR-normalised quantity is
therefore unchanged. Nothing a strategy can observe is affected. What *would* be
lookahead is deriving an absolute price level from the adjusted series — a dollar
price filter, a round-number level, a minimum tick calculation. Hence:

> **`close_adj` for anything scale-free. `close_raw` for anything denominated in
> dollars or shares.** Enforced by column naming and by
> `tests/unit/test_adjustment.py`.

A concrete example of the bug this prevents: a `$5.00` minimum-price filter
applied to an adjusted series excludes stocks that traded at `$40` before a
10-for-1 split, and includes stocks that traded at `$0.80`. Both errors are
invisible in the output and both are large.

### 4.3 Dividends as ledger cash flows

Dividends are **not** used to inflate the price series for P&L. They are applied
as an explicit cash flow when an ex-date falls inside a holding period:

```python
# In SimBroker, at the open of the ex-date session:
cash_flow = direction.sign * qty * Decimal(str(cash_amount))   # long receives, short pays
```

`Decimal`, because this compounds ([ADR-012](ADR/012-float-decimal-boundary.md)).
Recorded on `ClosedTrade.dividend_usd` so the effect is measurable rather than
buried in the entry price.

For a short, a 2% annual yield over a 21-session hold is an expected 17 bps —
roughly seven times the modelled round-trip cost of a long. Discretely it is a
single ~50 bps event with about a one-in-three chance of landing in the window.
`10-SENTIMENT.md` is not where this belongs; it is a cost, and
[`08-COSTS.md §2.5`](08-COSTS.md) models the expectation while `SimBroker` applies
the realisation.

### 4.4 Ticker changes and reuse

Handled entirely by keying on `asset_id`. `Asset.symbol` carries the display
ticker as of the pinned snapshot and is used only for logging and plots.
`tests/unit/test_ticker_reuse.py` asserts that a fixture containing two distinct
assets sharing a display ticker at different times produces two distinct series.

---

## 5. The trading calendar

Equities do not trade continuously. This is a new dependency and writing it
yourself is a mistake — the NYSE holiday schedule has changed repeatedly, half
days are irregular, and the 1985 and 2001 closures are special cases.

```toml
exchange-calendars = "==4.5.4"
```

`cli/build_calendar.py` materialises the XNYS session grid to
`data/reference/calendar_xnys.parquet`:

| Column | Type | Notes |
|---|---|---|
| `session` | `date32` | one row per trading session |
| `open_utc` | `timestamp[ms, UTC]` | session open, DST-correct |
| `close_utc` | `timestamp[ms, UTC]` | session close; **this is the panel `ts`** |
| `is_half_day` | `bool` | early close (1:00 pm ET) |
| `session_index` | `int32` | monotonic counter. **All bar arithmetic uses this** |

Two rules:

1. **`ts` is the session close in UTC**, so the end-of-bar convention in
   `AGENTS.md` carries over unchanged. Note it is 21:00 UTC in summer and 22:00 in
   winter — never construct it by adding a fixed offset to a date.
2. **"Bars" means sessions, everywhere.** `max_hold_bars=21` is 21 sessions, not 21
   calendar days. Every window, lag, and lookback uses `session_index`
   arithmetic. Mixing calendar-day and session arithmetic across a holiday is a
   silent off-by-one, and `tests/unit/test_calendar.py` includes Thanksgiving week
   and the 2012 Sandy closure specifically to catch it.

Half days are kept as normal sessions. Their volume is roughly a third of normal,
which the volume features see; special-casing them adds a branch for no benefit.

---

## 6. Schemas

### 6.1 `data/raw/equity/ohlcv/1d/<year>.parquet`

| Column | Type | Notes |
|---|---|---|
| `asset_id` | `string` (dictionary) | permanent id |
| `symbol` | `string` | display ticker as the vendor reported it *at that time*, if available |
| `session` | `date32` | vendor's native key |
| `open`, `high`, `low`, `close` | `double` | **unadjusted** |
| `volume` | `double` | shares, unadjusted |
| `dividend`, `split_ratio` | `double` | vendor's per-row action fields, if provided |

### 6.2 `data/processed/panel/1d/<year>.parquet`

| Column | Type | Notes |
|---|---|---|
| `asset_id` | `string` (dictionary) | |
| `symbol` | `string` (dictionary) | display label only |
| `ts` | `timestamp[ms, UTC]` | **session close time**. The only timestamp here |
| `session_index` | `int32` | from the calendar |
| `open`, `high`, `low`, `close` | `double` | **split- and dividend-adjusted** |
| `close_raw` | `double` | unadjusted close. Dollar-price gates and share counts |
| `volume` | `double` | adjusted (divided by the price factor) |
| `dollar_volume` | `double` | `close_raw * volume_raw`. Unadjusted by construction |
| `is_suspect` | `bool` | failed a data-quality check; §6.6 |

Sorted by `(ts, asset_id)`. No nulls in OHLCV: a missing session is an absent
row, not a null row.

There is no `is_synthetic` column, because daily bars are not resampled from
anything. Its role is taken by `is_suspect`.

### 6.3 `data/reference/benchmark_1d.parquet`

`SPY` and `^VIX` on the same session grid, with the same columns as the panel plus
`vix_close`, `vix9d_close`, `vix3m_close`. Separated from the panel because
market-regime features are computed once per timestamp, not once per symbol, and
because `SPY` must be available even in years when it would not clear the universe
gate (it always does, but the dependency should not be implicit).

### 6.4 `data/universe/assets.parquet`

Fields of `Asset` ([`02-DOMAIN_MODEL.md`](02-DOMAIN_MODEL.md#2-market-data)).
`tick_size`, `step_size`, and `min_notional_usd` are stored as **strings** and
parsed to `Decimal` on load — storing them as doubles reintroduces exactly the
float error the Decimal boundary exists to prevent.

For US equities: `tick_size = "0.01"` above $1.00, `step_size = "1"` (whole
shares; set to `"0.0001"` only if the broker supports fractional shares and the
live broker adapter confirms it), `min_notional_usd = "0"`.

### 6.5 `data/universe/snapshots.parquet`

| Column | Type | Notes |
|---|---|---|
| `ts` | `timestamp[ms, UTC]` | snapshot timestamp, on the monthly grid |
| `asset_id` | `string` | |
| `symbol` | `string` | for human inspection |
| `eligible` | `bool` | |
| `reason` | `string` | `RejectionReason` value; empty when eligible |
| `adv_usd_60` | `double` | trailing 60-session median dollar volume |
| `adv_rank` | `int32` | rank within the candidate set, 1 = most liquid |
| `spread_bps_est` | `double` | causal estimate; §7.3 |
| `close_raw` | `double` | unadjusted close at the snapshot |
| `bars_available` | `int32` | |
| `sector` | `string` | cluster key |
| `is_etf` | `bool` | |

One row per `(ts, asset_id)` for every symbol in the **candidate** list, eligible
or not. Recording the ineligible rows is the point: it lets you ask "how much of
the universe was tradeable in March 2009" and "which gate excluded the most
assets", and it makes the survivorship argument auditable.

Snapshot grid: **monthly**, first session of each month
([ADR-018](ADR/018-liquidity-rank-universe.md)). Lookup rounds **backwards**, so a
symbol that becomes eligible mid-month waits. The error is always conservative.
Size: 1,500 candidates × 340 months ≈ 510k rows.

### 6.6 `data/raw/equity/earnings.parquet`

| Column | Type | Notes |
|---|---|---|
| `asset_id` | `string` | |
| `earnings_date` | `date32` | announcement session |
| `is_confirmed` | `bool` | False means vendor estimate |
| `available_ts` | `timestamp[ms, UTC]` | when this row became known, if the vendor provides it |
| `timing` | `string` | `BMO`, `AMC`, `UNKNOWN` |

Read only with `available_ts <= t` where available. Where it is not, the
conservative `± EARNINGS_UNCERTAINTY_SESSIONS` window (constant 4 sessions,
[ADR-020](ADR/020-earnings-gate.md)) applies. A symbol with **no** earnings row
and `is_etf=False` is treated as **blocked**, never as "no earnings, therefore
fine". ETFs are exempt. The lookforward window is the strategy's
`max_hold_bars`, not a gates config field.

### 6.7 Data-quality checks and `is_suspect`

Set `is_suspect=True`, do not modify or delete the row, when any of:

| Check | Threshold |
|---|---|
| `high < low`, or `close` outside `[low, high]` | any |
| Single-session absolute log return | `> 1.0` (≈ ±170%) with no matching action row |
| `volume == 0` | any |
| `close_raw` unchanged for `≥ 5` consecutive sessions with non-zero volume | any |
| Session present in the panel but absent from the calendar | any |

The universe gate makes a symbol ineligible while any of the last
`universe.max_suspect_lookback` sessions is suspect (default 5). Flagging rather than
deleting keeps the panel aligned and keeps the problem visible. An unexplained
±170% session is almost always a missing split, and a missing split is the most
damaging single-row error available: it manufactures a 90% one-day return that
every momentum feature will treat as the strongest signal in the universe.

`cli/check_data.py` prints the suspect count per year and per reason and **exits
non-zero if the suspect rate exceeds 0.1%**. Run it after every ingest.

### 6.8 `data/labels/setups_<strategy_id>.parquet`

Schema in [`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md#3-the-resolved-setup-table).

### 6.9 `data/sentiment/observations.parquet`

Fields of `SentimentObservation`. Sorted by `available_ts`.

### 6.10 Never fill a gap

**Not forward, not backward, not by interpolation.** A filled bar is
indistinguishable from a real one downstream, and every consumer — ATR, the regime
classifier, the labeling pass — will treat it as evidence. Absent rows are
visible; filled rows are invisible lies.

The one thing that looks like an exception and is not: aligning symbols to the
calendar. A symbol with no row on a valid session simply has no row on that
session. The panel is long-format precisely so that this needs no filling.

`bars_since_gap` is computed against the **calendar**, not against the symbol's own
rows, so a halted stock's gap is visible rather than compressed away.

---

## 7. Point-in-time universe

The most important part of this document. See
[ADR-018](ADR/018-liquidity-rank-universe.md) for why index membership is not used.

### 7.1 The candidate list

`config/universe_candidates.txt` — one `asset_id,symbol` pair per line, containing
**every symbol ever considered**, including delisted ones. This file only grows.
Deleting a delisted symbol silently deletes the worst outcomes from the backtest,
which is the mechanism of survivorship bias. Project rule 6.

Seeded from the vendor's **full** ticker table — `TICKERS` in Sharadar, the
delisted database in Norgate — filtered to:

- Exchange in `{NYSE, NASDAQ, NYSEARCA, BATS}`
- Category in `{Domestic Common Stock, ETF}` — excludes ADRs, preferreds,
  warrants, units, closed-end funds, and OTC
- Any session with dollar volume above `$2,000,000` at any point in history

That yields roughly 4,000–6,000 symbols, of which perhaps 40% are delisted. **If
your candidate list contains no delisted symbols, it is wrong**, and
`cli/build_universe.py` refuses to run if the delisted fraction is below 15%.

### 7.2 Eligibility rules

Evaluated at each snapshot `ts` using **only sessions with `ts_session <= ts`**.
Ordered; the first failure is recorded.

| # | Rule | Reason | Default |
|---|---|---|---|
| 1 | `bars_available >= min_history_bars` | `INSUFFICIENT_HISTORY` | 400 sessions |
| 2 | Session exists at `ts` and the symbol is not delisted as of `ts` | `STALE_DATA` | — |
| 3 | No suspect session in the last `max_suspect_lookback` | `DATA_SUSPECT` | 5 |
| 4 | `bars_since_gap >= min_bars_since_gap` | `DATA_GAP` | 10 |
| 5 | `close_raw >= min_price_usd` | `LOW_PRICE` | 5.00 |
| 6 | `adv_usd_60 >= min_adv_usd` | `LOW_LIQUIDITY` | 5,000,000 |
| 7 | `adv_rank <= universe_size` | `NOT_IN_UNIVERSE` | 1000 |
| 8 | `spread_bps_est <= max_spread_bps` | `WIDE_SPREAD` | 15.0 |

Rule 1's 400 sessions is the feature warm-up requirement (252 sessions, bound by
the 12-month momentum lookback — see
[`05-FEATURES_AND_REGIME.md §8`](05-FEATURES_AND_REGIME.md#8-warm-up-accounting))
plus margin, so a symbol clearing the universe gate is guaranteed to have warm
features.

Rule 5 uses **`close_raw`**, not `close`. See §4.2 for why this distinction is
load-bearing.

Rule 6's $5M floor and rule 7's top-1000 rank work together: the rank is what we
want, and the absolute floor stops a thin decade from admitting untradeable names
just because everything else was thinner. In 1999 the top 1,000 US equities by
dollar volume comfortably clear $5M, so rule 6 rarely binds — it exists as a
guard, not a filter.

Rule 8's 15 bps is loose relative to reality for the top 1,000 (typically 1–5
bps) and exists to catch specific broken names. If it is rejecting more than about
2% of otherwise-eligible symbol-months, the Corwin–Schultz estimate is
misbehaving and should be investigated rather than tuned around.

### 7.3 The spread estimate

Historical order-book data is not available, so spread is estimated with the
**Corwin–Schultz high–low estimator**, which infers the effective spread from the
relationship between two-session and one-session high–low ranges:

```python
# Corwin & Schultz (2012), on the decision timeframe. Uses ADJUSTED high/low,
# since the estimator is scale-free.
beta  = mean over trailing window of [ (ln(h_t/l_t))^2 + (ln(h_{t+1}/l_{t+1}))^2 ]
gamma = mean over trailing window of [ (ln(max(h_t,h_{t+1}) / min(l_t,l_{t+1})))^2 ]

alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
spread_bps_est = max(spread, 0.0) * 1e4
```

Trailing window: 30 sessions. Negative estimates clamp to 0 and then take the
per-liquidity-tier floor from config, so the estimate can never be optimistic:

```python
spread_bps_est = max(corwin_schultz_bps, cfg.universe.spread_floor_bps_by_tier[tier])
```

Tiers by `adv_usd_60` ([`14-CONFIG.md`](14-CONFIG.md)):

| `adv_usd_60` | Floor |
|---|---|
| `> $500M` | 1.0 bps |
| `> $100M` | 2.0 bps |
| `> $20M` | 4.0 bps |
| else | 10.0 bps |

**Two equity-specific caveats.** Corwin–Schultz was developed on daily equity data
and works better here than on crypto, which is a genuine improvement. But it
**overstates spread for overnight-gap-heavy names**, because the two-session range
it uses includes the gap. Since the bias is conservative, it is accepted. Second,
it is biased upward for low-priced stocks; rule 5's $5 floor removes the worst of
that. When M5 begins collecting real quote snapshots the estimator is replaced and
the historical estimate is recalibrated against live measurements.

### 7.4 Build command

```text
scout build-universe --config config/backtest.yaml
```

Output: `data/universe/snapshots.parquet`. Prints eligible count per year and the
top rejection reasons, which is the fastest way to notice that a threshold has
excluded the entire universe.

### 7.5 The delisting rule

A symbol whose last session is before `ts - delisting_grace_bars` (default 5
sessions), or whose `tickers.parquet` row carries a delisting date at or before
`ts`, is treated as delisted at `ts`. Its **open positions are force-closed** at
the last available close, with `exit_reason = "DELISTED"`, and the loss taken in
full.

Two equity-specific refinements:

- **Merger and acquisition delistings are not losses.** If `actions.parquet`
  carries a `MERGER` row, the position is closed at the last close, which
  approximates the deal price. Treating an acquisition as a total loss is the
  mirror-image error of survivorship bias and is equally wrong.
- **Bankruptcy delistings usually are.** No action row, price collapsing, then
  data ends. Closed at the last available close, which is typically a small number,
  and the loss taken in full.

This is the mechanism that keeps Enron, Lehman, Bear Stearns, and Wirecard in the
equity curve. Any implementation that silently drops a symbol when its data ends
is deleting the largest losses. Covered by `tests/unit/test_delisting.py`, which
asserts both cases and that the resulting trades appear in `trades.csv`.

---

## 8. Data splits

25+ years of history is the main thing equities buy you over crypto, so spend it
deliberately.

| Split | Range | Use |
|---|---|---|
| Warm-up | 1998-01-01 → 2003-12-31 | Feature warm-up and the first walk-forward bin windows. **Never scored** |
| **Development** | 2004-01-01 → 2017-12-31 | All strategy design, all parameter choices, all sensitivity runs. 14 years |
| **Holdout** | 2018-01-01 → present | **Locked.** 3 evaluations for the project's lifetime ([ADR-011](ADR/011-holdout-lockbox.md)) |

The holdout is roughly 8 years and 36% of the scored sample, and it deliberately
contains the 2018 Q4 selloff, the COVID crash and recovery, the January 2021
momentum crash, the 2022 bear market, and the 2023–25 concentration rally. That
is a genuinely hostile out-of-sample period, which is the point. Development
covers 2008 and 2011, so it is not a bull-market-only design set either.

Enforced in `research/splits.py`; the holdout range is a constant in code, not a
config value, so it cannot be widened by editing a YAML file.

---

## 9. Reading data inside the loop

```python
# ALLOWED: read once before the loop, then slice with as_of()
panel = candles.load_panel(asset_ids, "1d", start, end)
for ts in panel.timestamps:
    view = panel.as_of(ts)
    ...

# FORBIDDEN: file I/O inside the loop
for ts in timestamps:
    view = candles.load_panel(asset_ids, "1d", start, ts)   # 6,300 file reads
```

Memory: 1,000 symbols × 6,300 sessions × 10 columns × 8 bytes ≈ 500 MB. Load it
all, but note this is 6× the crypto figure, so:

- Read only the columns the run needs (`pyarrow` column projection).
- Cast `symbol` and `asset_id` to `category`.
- `float32` is **not** acceptable for prices. It has about 7 significant digits,
  and a $400 stock with a $0.01 tick needs 6 — too close to the edge for
  comparisons against stop levels.

`as_of` on a `(ts, asset_id)`-sorted frame is a `searchsorted` plus a slice — O(log
n) plus a view. Precompute the boundary index per timestamp once, in
`MarketPanel.__init__`, rather than filtering with a boolean mask per iteration,
which would be O(n) per bar and turn a two-minute backtest into an hour.

---

## 10. Decision audit schema

`results/<run-id>/decisions.parquet` — fields of `DecisionRecord`
([`02-DOMAIN_MODEL.md §12`](02-DOMAIN_MODEL.md#12-decision-record--the-audit-row)).

Written by `ParquetDecisionSink`, buffered and flushed every `flush_every_cycles`
cycles (default 500) as one row group.

Expected size: 1,000 symbols × 2 strategies × 6,300 sessions ≈ 12.6M rows worst
case — roughly 5× the crypto estimate, so the `features_json` policy matters more:

```yaml
audit:
  features_json_policy: accepted_and_ranked   # accepted | accepted_and_ranked | all | none
```

Default `accepted_and_ranked` — full feature forensics for anything that reached
the ranking stage, none for the millions of rows rejected at the first gate.
Roughly 150 MB per run.

### Standard funnel query

```sql
SELECT stage, rejection_reason, count(*) AS n
FROM 'results/<run-id>/decisions.parquet'
GROUP BY 1, 2
ORDER BY n DESC;
```

Run this after every backtest **before looking at any performance number**. It is
the fastest available detector of a misconfigured gate, and a gate that rejects
everything produces a beautiful flat equity curve that looks like discipline.

For the equity system, check two rows specifically:

- `EARNINGS_IN_WINDOW` — expected to be large ([ADR-020](ADR/020-earnings-gate.md)).
  If it is near zero, the earnings data is not loading and the gate is silently
  inert.
- `NOT_IN_UNIVERSE` — expected to be the largest by far, since it rejects most of
  the candidate list on every session. If it is near zero, the liquidity rank is
  not being applied.

---

## 11. Appendix: crypto (M7, optional)

Retained for the optional M7 sleeve ([ADR-015](ADR/015-equities-first.md)). Do not
implement before an equity holdout result exists.

- **Source:** Binance Futures klines, 1h base, resampled to 4h and 1d with
  UTC-anchored boundaries ([ADR-010](ADR/010-4h-decision-timeframe.md)).
- **Layout:** `data/raw/crypto/ohlcv/binance_futures/1h/<SYMBOL>/<year>.parquet`,
  processed to `data/processed/panel_crypto/4h/<year>.parquet`.
- **No corporate actions, no calendar.** Both concerns vanish; `is_synthetic`
  replaces `is_suspect` for 4h bars built from fewer than four complete 1h bars.
- **Ingest rule that matters most:** drop the final bar of every response if
  `close_time > now_utc`. The most common ingest lookahead bug in existence.
- **Candidate list:** the 120 symbols with the largest cumulative USDT perpetual
  volume across 2019–2026, from the full symbol list **including
  `status = "DELISTED"`**. Must include at minimum `LUNAUSDT`, `FTTUSDT`,
  `SRMUSDT`, `RAYUSDT`, `ANCUSDT`, `CVCUSDT`, `BTSUSDT`, `SCUSDT`, `TOMOUSDT`,
  `WAVESUSDT`.
- **Eligibility:** `min_adv_usd` 20,000,000 and `max_spread_bps` 8.0 — much
  tighter than the equity thresholds, because crypto costs are ~7× higher.
- **Snapshot grid:** weekly.
- **Funding rates:** `data/raw/crypto/funding.parquet`, needed by both the cost
  model and the funding-skew sentiment source.
- **Splits:** development 2019-01-01 → 2023-12-31, holdout 2024-01-01 → present,
  with a **separate lockbox budget**. The equity holdout budget may not be spent
  on crypto and vice versa.
