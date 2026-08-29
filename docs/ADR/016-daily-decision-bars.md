# ADR-016: Daily decision bars, index context, next-open execution

**Status:** Accepted — supersedes [ADR-010](010-4h-decision-timeframe.md)

## Context

ADR-010 chose 4h decision bars for crypto, on cost arithmetic. Equities change
the inputs completely.

A US equity session is 6.5 hours, so it does not divide into 4h bars at all. The
available choices are intraday bars (5m/15m/30m/60m) or daily.

## Decision

- **Decision timeframe: daily bars.** One decision per trading day, evaluated
  after the close.
- **Context: the market index (`SPY`) on the same daily bars**, not a second
  per-symbol timeframe. The context dimension is now *market regime*, which is
  more useful than a longer per-symbol lookback and costs no extra data.
- **Execution: market-on-open at the next session's open.** Same
  decision-at-`t`, fill-at-`t+1` convention as before.
- One decision timeframe. A second is an M6 experiment with its own trial budget.

## Rationale

**Cost.** Daily bars targeting 4–8% moves put round-trip cost at roughly 0.3% of
the gross target. Intraday bars targeting 0.5–1% moves put it at 1–3%. Both are
survivable — unlike crypto at 15m — but daily is strictly better and needs no
intraday data.

**Data cost and availability.** Survivorship-free daily US equity history with
corporate actions is about $60/month. The equivalent intraday history with
adjustments is several hundred per month and a much larger storage and
correctness problem.

**The opening auction is a genuine execution advantage.** A market-on-open order
participates in a single-price auction. There is no spread to cross and no drift
between decision and fill, which is why the slippage volatility coefficient for
equities is 0.02 rather than crypto's 0.05
([`08-COSTS.md §2.3`](../08-COSTS.md#23-slippage)). Intraday entries forfeit this.

**Where the evidence is.** Cross-sectional equity momentum is documented at
monthly and weekly horizons, with holding periods of one to twelve months. Daily
decision bars with 21-day holds sit inside that window. Intraday equity
strategies are a different, far more competitive game against participants with
colocation.

**Operational simplicity.** One decision per day, after the close, with orders
staged for the next open. No intraday monitoring, no latency requirement.

## Consequences

- Sample accumulates about 9× slower per symbol than crypto 4h bars (252 bars per
  year against 2,190). Offset by breadth: 1,000+ symbols against 120. Total
  resolved setups are comparable or better.
- **The context timeframe disappears** and is replaced by market regime from the
  index. `efficiency_ratio_ctx` becomes `market_*` features computed on `SPY`.
  This is a simplification, not just a substitution.
- 25 years of daily history is about 6,300 bars per symbol. Ample.
- **Overnight gaps become the dominant risk.** With daily bars every holding
  period contains an overnight gap, and stops cannot protect against them. This
  drives the wider default stop (3 ATR), the earnings gate (ADR-020), and the
  gap-through-stop fill rule.
- No intraday data storage. 1,500 symbols × 25 years of daily bars is roughly
  9.5M rows and about 250 MB of Parquet.

## Alternatives rejected

**60-minute bars.** 6.5 bars per session, so the last bar of the day is a
half-bar and every window straddles the overnight gap. Session-boundary handling
adds real complexity for a horizon with weaker published evidence.

**30-minute bars.** Same problems, plus intraday data cost, plus opening and
closing auction distortions in the first and last bars.

**Weekly bars.** Best cost ratio and closest to the momentum literature's native
frequency, but only about 1,300 bars per symbol over 25 years, and rebalancing
once a week makes the earnings gate and the market-regime filter coarse. A
reasonable M6 sensitivity run.

**Daily decision with intraday entry timing.** The brief's original
higher/medium/lower timeframe idea. Forfeits the opening auction, requires
intraday data, and adds an entry-timing rule with its own parameters — before any
baseline exists.

## Revisit trigger

M6, after an equity holdout result. First sensitivity run to try is weekly
rebalancing, which needs no new data.
