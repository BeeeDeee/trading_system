# Data Sources

> External data acquisition and normalization architecture.

## 1. Purpose

The platform treats all external information sources as data providers.

A data provider supplies raw information.

It does not contain trading logic.

---

## 2. Data Source Categories

### Market Data

- OHLCV
- trades
- order books
- ticker data
- funding rates
- open interest
- liquidations

### Account and Execution Data

- account state

### News and Sentiment (near-term — Milestone 1b)

First-class alternative data, implemented right after the price backtest
slice. Same adapter → normalize → Feature Engine path as market data.

Near-term sources (priority order):

- Fear & Greed (or equivalent crypto sentiment index)
- news APIs / headlines with timestamped relevance to BTC
- optional: RSS or curated feeds

Later expansion:

- social media / Reddit-style providers
- other commercial sentiment feeds

Sentiment is **optional for strategies**. Missing or stale sentiment must not
block strategies that only use price features. Strategies that require
sentiment fail closed for *their* signals only.

Example flow:

```text
Fear & Greed / News API
    ↓
SentimentAdapter / NewsAdapter
    ↓
Normalization → SentimentSnapshot / NewsEvent[]
    ↓
Feature Engine → sentiment features (aligned to 1h bars)
    ↓
Strategy (optional consumer)
```

Historical sentiment series are stored as Parquet under `data/` alongside
OHLCV so backtests can replay the same inputs.

### On-Chain

Potential future inputs (after Milestone 1b):

- transaction activity,
- exchange flows,
- whale activity,
- network metrics.

### Macro

Potential future inputs:

- economic indicators,
- interest rates,
- inflation,
- economic calendar events,
- other macro data.

---

## 3. Architecture

```text
External Provider
      ↓
Provider Adapter
      ↓
Normalization
      ↓
Canonical Domain Model
      ↓
Feature Engine
````

---

## 4. Provider Adapters

Provider-specific details must remain inside adapters.

Market data and account/execution concerns are separate platform interfaces:

```text
Exchange
  ├── MarketDataAdapter
  └── AccountExecutionAdapter
```

Examples:

```text
BinanceMarketDataAdapter
BybitMarketDataAdapter
BinanceAccountExecutionAdapter
BybitAccountExecutionAdapter
FearGreedAdapter          # Milestone 1b
NewsProviderAdapter       # Milestone 1b
RedditAdapter             # later expansion
OnChainProviderAdapter    # future
```

The rest of the platform must not depend on provider-specific response formats.

---

## 5. Collection Responsibilities

A data source adapter is responsible for:

* connection,
* authentication,
* retrieval,
* rate limits,
* retries,
* provider-specific errors,
* provider-specific schema handling.

---

## 6. Data Source vs Feature Engine

Data acquisition and feature generation are intentionally separate.

Example:

```text
Funding API
    ↓
Funding Rate
    ↓
Funding Feature

Fear & Greed API
    ↓
SentimentSnapshot
    ↓
Sentiment Feature (aligned to bar close)
```

The adapter retrieves the raw value.

The Feature Engine decides how to derive analytical features from it.
Sentiment features must be point-in-time correct in backtests (no lookahead
from articles published after the bar).

---

## 7. n8n and Data Sources

n8n should not normally be used as the platform's primary market-data acquisition layer.

Python remains responsible for data required by trading and analysis.

n8n may be used for non-critical external data workflows where appropriate.

---

## 8. Historical Data Storage

Backtesting and research require years of historical bars. Storage split:

```text
Parquet files (data/, gitignored)
  ├── raw/        untouched provider downloads
  └── processed/  cleaned, normalized, resampled

PostgreSQL
  └── trading state only (orders, positions, trades, decisions, config)
```

Parquet is the research and backtest data store: columnar, fast to scan,
trivially portable. PostgreSQL is not used for bulk candle history.

The live runtime and the backtest engine consume the same canonical
`MarketSnapshot` model regardless of whether data comes from a websocket or
a Parquet file.

Initial datasets:

- BTC/USDT 1h bars, several years of history, plus funding when perpetuals
  are in scope (Milestone 1).
- Sentiment / Fear & Greed (and news event) history aligned for backtest
  replay (Milestone 1b).

---

## 9. Reliability

Each source should expose enough metadata to determine:

* timestamp,
* freshness,
* source,
* validity,
* completeness.

Stale or invalid data must be detectable before it influences trading decisions.

---

## 10. Future Expansion

New data sources should be added by implementing a new adapter rather than changing downstream trading logic.
