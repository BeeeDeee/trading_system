# ADR-017: Store unadjusted prices plus an actions table; pin the data snapshot

**Status:** Accepted

## Context

Equities have corporate actions. Crypto does not, and this is the single largest
new source of silent error in the equity pivot.

Three distinct problems, often conflated:

### 1. Adjusted prices are retroactively revised

The adjusted close for AAPL on 2019-03-14 is **not the same number today as it
was in 2019**, because every subsequent split and dividend changes it. A backtest
that reads a vendor's adjusted-close column is therefore **not reproducible**: run
it in June and again in September and the equity curve differs, with no code
change and no config change. Every stored edge statistic silently becomes invalid.

### 2. Adjustment is required for features but is forward-looking by nature

A 200-day moving average across a 4-for-1 split is meaningless on unadjusted
prices. So features need an adjusted series. But back-adjustment applies factors
derived from actions that happened *after* the bar being adjusted — which looks
like lookahead.

It is not lookahead, provided one condition holds: the adjustment must be
**price-ratio-preserving**. A back-adjusted series scales all history by a
constant factor, so every *return*, every *ratio*, and every ATR-normalised
quantity is unchanged. Nothing a strategy can see is affected. What *would* be
lookahead is using the adjusted series to derive an absolute price level — a
dollar-price filter, or a round-number level.

### 3. Total return versus price return

Dividends are a real cash flow. A long receives them; a short pays them. For a
2% yielding stock over a 10-day hold the expected effect is about 8 bps, which is
material relative to a 23 bps total cost. Discretely, an ex-dividend date inside
the holding period is a 50 bps event.

## Decision

**Store raw and adjust at load time, from a pinned snapshot.**

1. `data/raw/equity/ohlcv/` holds **unadjusted** prices exactly as the vendor
   returned them. Immutable.
2. `data/raw/equity/actions/` holds the corporate-actions table: split ratios,
   cash dividends, spinoff ratios, all with ex-dates.
3. Every ingest run writes a **snapshot id**: `data_snapshot_id`, of the form
   `YYYYMMDD-<vendor>`, recorded in `data/raw/equity/SNAPSHOT.json` together with
   row counts and per-symbol date ranges.
4. `data_snapshot_id` is **part of the config hash**. A backtest whose snapshot id
   differs from its edge table's snapshot id **fails at startup**, exactly as with
   `config_hash`.
5. `resample.py` produces the processed panel by **ratio back-adjustment** of the
   raw series using the actions table, at the pinned snapshot. Both an
   `close_adj` (split-and-dividend adjusted, for features) and `close_raw`
   (unadjusted, for dollar-price filters and share-count arithmetic) are stored.
6. **Dollar-price gates and share quantities use `close_raw`.** Features use
   `close_adj`. This distinction is enforced by column naming.
7. Dividends inside a holding period are applied as an explicit cash flow in the
   ledger, signed by direction — not folded into the price series.

## Consequences

- Backtests are reproducible. Re-downloading data creates a new snapshot id and
  therefore a visible, deliberate invalidation rather than a silent drift.
- Re-adjustment is a rerun, not a migration, because raw is immutable.
- Two close columns is mild redundancy, and it eliminates an entire class of bug:
  a $5 price floor applied to an adjusted series excludes stocks that were $40
  before a 10-for-1 split, which is both wrong and invisible.
- Dividends as explicit ledger cash flows means the equity curve is total-return
  and shorts pay dividends correctly.
- **Ticker changes and reuse** are handled by keying everything on a vendor-stable
  permanent id (`permaticker` in Sharadar, the internal id in Norgate), never on
  the display ticker. `Asset.symbol` is the display ticker; `Asset.asset_id` is
  the permanent key, and all joins use `asset_id`.
- Cost: more storage and one more processing step. Immaterial.

## Alternatives rejected

**Use the vendor's adjusted close directly.** Not reproducible (problem 1). This
is the default choice and it is wrong, which is why this ADR exists.

**Store adjusted prices, re-download never.** Works until you need to extend the
history, at which point the extension is adjusted on a different basis than the
existing data and the seam is invisible.

**Forward-adjustment instead of back-adjustment.** Keeps historical prices at
their true values, so dollar-price filters work directly. Rejected because the
most recent price then differs from the actual traded price, which makes live
reconciliation confusing and error-prone — a worse trade-off.

**Ignore dividends.** Overstates shorts and understates longs by roughly 8 bps per
10-day hold, which is one third of the total modelled cost. Not acceptable.

## Revisit trigger

None foreseen. This is the standard professional approach and the alternatives are
all worse.
