# ADR-008: Bar-Driven Runtime and Initial Scope

## Status

Accepted

## Decision

The trading runtime is bar-driven: the decision pipeline runs once per
completed bar of a configured timeframe.

Initial scope:

- Instrument: BTC/USDT
- Timeframe: 1h bars
- Higher-timeframe context features (4h, 1d) are aggregated from 1h data.
- Near-term alternative data (Milestone 1b): news and market sentiment
  (e.g. Fear & Greed, news APIs), aligned to 1h bar timestamps and consumed
  as optional features — not required for every strategy.

Sub-bar reactivity (stop-loss, take-profit) is handled by exchange-native
conditional orders, not by the decision pipeline.

## Rationale

- 1h bars give ~8,760 observations per year — enough history for meaningful
  backtests over several years while fees and slippage do not dominate the
  edge, unlike 1m–15m timeframes.
- Latency is irrelevant at this cadence, so a single VPS, a single Python
  process, and a serialized pipeline are fully adequate.
- A bar-driven model makes the shared live/backtest pipeline (ADR-007)
  straightforward: the backtester replays completed bars through the exact
  same code path.
- BTC is the most liquid crypto instrument with the longest, cleanest data
  history; it is the correct single instrument to validate the platform on.
- Sentiment and news are valuable context but secondary to a working price
  path; they follow immediately as Milestone 1b so they are available before
  paper trading.

Known trade-off: 1h swing strategies produce relatively few trades per year,
so statistical significance is thin per strategy. Mitigation: backtest over
many years and later validate across additional symbols before trusting a
strategy.

## Consequence

- The pipeline is a synchronous call chain triggered by bar close.
- Between bar closes the runtime only ingests data, processes execution
  events, maintains protective orders, and serves control commands.
- Event-driven (tick/order-book) trading is out of scope; adopting it later
  is a new ADR and a substantial architectural change.
