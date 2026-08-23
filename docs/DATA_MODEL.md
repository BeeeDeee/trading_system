# Domain Data Model

> Core data contracts exchanged between trading modules.

## 1. Purpose

This document defines the primary domain objects used by the trading pipeline.

These objects are internal platform contracts.

They must not depend on exchange-specific API models.

---

## 2. Core Pipeline

```text
MarketSnapshot
      ↓
FeatureSet (may include RegimeState)
      ↓
TradingSignal[]
      ↓
TargetPosition[]
      ↓
RiskDecision
      ↓
Order Planner
      ↓
OrderIntent
      ↓
ExecutionReport
````

Every object in the chain carries correlation identifiers so an
`ExecutionReport` can be traced back to the `TradingSignal` and `FeatureSet`
that produced it. Stage results are persisted as the decision audit trail
(see `STATE_AND_PERSISTENCE.md`).

---

## 3. MarketSnapshot

Represents normalized market state at a specific point in time.

Expected information may include:

* symbol,
* timestamp,
* OHLCV,
* current price,
* volume,
* order book information,
* funding,
* open interest,
* liquidation data.

The exact schema is defined during implementation.

---

## 4. FeatureSet

Represents derived analytical features.

Examples:

* EMA,
* RSI,
* MACD,
* ATR,
* ADX,
* volatility,
* momentum,
* funding features,
* **sentiment features** (near-term — Fear & Greed, news-derived scores;
  see `DATA_SOURCES.md` Milestone 1b),
* on-chain features (later).

Features should be deterministic for a given input and feature version.
Sentiment features used in backtests must be point-in-time correct relative
to the bar timestamp.

---

## 5. RegimeState

Represents current market regime as a **derived feature owned by the Feature
Engine**. It is an optional part of `FeatureSet`, not a standalone pipeline
stage. Strategies may consume it or ignore it.

Initial structure:

```text
Regime:
    TREND
    RANGE
    CHOP

Trend:
    UP
    DOWN
    NONE

Volatility:
    LOW
    MID
    HIGH
```

Additional fields may include:

* confidence,
* timestamp,
* detector version.

---

## 5a. SentimentSnapshot / NewsEvent (near-term)

Canonical inputs for Milestone 1b alternative data (exchange-independent):

**SentimentSnapshot** — point-in-time aggregate sentiment (e.g. Fear & Greed
value, source, timestamp, freshness).

**NewsEvent** — timestamped headline or article reference (source, published
at, optional relevance / polarity score). Features derived from news must
only use events with `published_at <= bar_timestamp` in backtests.

These are owned by Data / Normalization; the Feature Engine turns them into
sentiment features inside `FeatureSet`.

Represents the output of a strategy.

A signal may contain:

* strategy identifier,
* symbol,
* direction,
* confidence,
* target exposure,
* stop-loss proposal,
* take-profit proposal,
* timestamp.

A TradingSignal is not an order.

---

## 7. TargetPosition

Represents desired portfolio exposure after strategy aggregation.

Possible information:

* symbol,
* target quantity,
* target percentage,
* reason,
* contributing strategies.

A TargetPosition is not yet approved for execution.

The Order Planner converts an approved or modified target into the concrete
position change required for execution. It does not make risk decisions.

---

## 8. RiskDecision

Represents the result of risk validation.

Possible states:

```text
APPROVE
MODIFY
REJECT
HALT
```

Possible information:

* approved exposure,
* rejection reason,
* modified size,
* risk rule triggered,
* timestamp.

---

## 9. OrderIntent

Represents an execution request that has passed risk validation.

It should contain platform-level execution information rather than exchange-specific API structures.

Required fields:

* `client_order_id` — a client-generated idempotency key. Retries after
  network failures must reuse the same key so an order is never
  double-submitted.
* `account_id` — present from day one even while only one account exists.

Additional fields:

* symbol,
* side,
* quantity,
* order type,
* price,
* reduce-only flag,
* strategy identifier,
* risk decision identifier.

Price and quantity are represented as `Decimal`, rounded to the instrument's
tick size and step size before submission.

---

## 10. ExecutionReport

Represents the result of an execution attempt.

Possible states include:

* submitted,
* accepted,
* partially filled,
* filled,
* cancelled,
* rejected,
* failed.

ExecutionReport may contain:

* order identifier,
* fill quantity,
* average price,
* fees,
* exchange identifier,
* timestamps.

---

## 11. PortfolioState

Represents current portfolio-level state.

May include:

* equity,
* positions,
* exposure,
* available capital,
* realized PnL,
* unrealized PnL,
* drawdown.

---

## 12. AccountState

Represents exchange account state required for trading decisions.

May include:

* balances,
* available margin,
* leverage,
* open positions,
* open orders.

---

## 13. Ownership

| Object            | Primary Owner        |
| ----------------- | -------------------- |
| MarketSnapshot    | Data / Normalization |
| SentimentSnapshot | Data / Normalization |
| NewsEvent         | Data / Normalization |
| FeatureSet        | Feature Engine       |
| RegimeState       | Feature Engine (derived feature) |
| TradingSignal     | Strategy Engine      |
| TargetPosition  | Portfolio Manager    |
| RiskDecision    | Risk Manager         |
| OrderIntent     | Order Planner        |
| ExecutionReport | Trading Engine       |
| PortfolioState  | Derived by Portfolio Manager |
| AccountState    | Reconciliation / external account state |

---

## 14. Rules

* Domain models must remain exchange-independent.
* `OrderIntent` may only be created from an approved or modified `RiskDecision`.
* Every `OrderIntent` carries a client-generated `client_order_id`
  (idempotency key).
* Account-scoped models (orders, positions, trades, balances) carry an
  `account_id` from day one, even while only one account exists.
* Money at the order, balance, and P&L accumulation boundaries uses
  `Decimal` or integer minor units. Floats remain acceptable for
  time-series features and analytics.
* All timestamps are timezone-aware UTC.
* `PortfolioState` is a derived view, not the authoritative source of order or position truth.
* Domain models should be immutable where practical.
* Models should be strongly typed.
* Models must be serializable where persistence or API transport requires it.
* Exchange-specific models must not leak into the core domain.

---

## 15. Backtest Compatibility

The same domain model chain should be usable in both live trading and backtesting.

The execution layer is the main abstraction that changes:

```text
Live:
OrderIntent → Real Exchange

Backtest:
OrderIntent → Simulated Execution
```
