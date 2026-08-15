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

### News and Sentiment

- news APIs
- RSS
- social media
- sentiment providers
- Fear & Greed
- other external sentiment signals

### On-Chain

Potential future inputs:

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
NewsProviderAdapter
RedditAdapter
OnChainProviderAdapter
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
```

The adapter retrieves the raw value.

The Feature Engine decides how to derive analytical features from it.

---

## 7. n8n and Data Sources

n8n should not normally be used as the platform's primary market-data acquisition layer.

Python remains responsible for data required by trading and analysis.

n8n may be used for non-critical external data workflows where appropriate.

---

## 8. Reliability

Each source should expose enough metadata to determine:

* timestamp,
* freshness,
* source,
* validity,
* completeness.

Stale or invalid data must be detectable before it influences trading decisions.

---

## 9. Future Expansion

New data sources should be added by implementing a new adapter rather than changing downstream trading logic.
