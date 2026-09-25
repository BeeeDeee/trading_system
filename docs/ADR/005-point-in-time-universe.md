# ADR-005: Point-in-time universe snapshots

**Status:** Accepted. The principle is unchanged for equities; the eligibility
criteria and snapshot cadence are specified in
[ADR-018](018-liquidity-rank-universe.md).

## Context

The brief specifies liquidity and data-quality filters but never a *point-in-time
universe*. For crypto this is not a minor gap; it determines whether the backtest
is valid at all.

- Symbols listed mid-history. Backtesting SOL from 2018 is backtesting nothing.
- Delistings and collapses: LUNA, FTT, and a long tail. A universe derived from
  "symbols on the exchange today" silently deletes every asset that went to zero.
- Ticker reuse across unrelated assets.
- Liquidity arriving late: $2M daily volume in 2021 and $400M in 2024 must be
  ineligible in 2021 despite being eligible now.

The natural implementation — "load the symbol list, apply a liquidity filter over
the whole history" — commits all four errors simultaneously, and the resulting
backtest is better than reality by an amount that is impossible to bound.

## Decision

A first-class `data/universe/snapshots.parquet` table, one row per
`(snapshot_ts, symbol)` for **every candidate symbol**, eligible or not,
recording eligibility and the reason.

- Built from trailing data only, with a production assertion that every bar used
  has `ts <= snapshot_ts`.
- Weekly grid; lookup rounds **backward**, so newly eligible symbols wait up to
  six days. The error is always conservative.
- `config/universe_candidates.txt` is **append-only** and must include delisted
  symbols. If it contains no symbols that went to zero, it is wrong.
- A symbol whose data ends is treated as delisted after a grace period, and its
  open positions are **force-closed at the last available close with the full loss
  taken**.
- The engine reads eligibility from the snapshot, never from the live symbol list.

## Consequences

- Survivorship bias is structurally prevented rather than remembered.
- LUNA-style outcomes stay in the equity curve, which is where they belong.
- The universe's size varies over time, which is why the panel is long-format.
- `snapshots.parquet` answers "how much of the universe was tradeable in March
  2020" and "which rule excluded the most assets" with a groupby.
- Cost: ingest must fetch delisted symbols, which requires the exchange's full
  symbol list including `status = DELISTED` rather than the convenient active
  list.
- Cost: eligibility lags by up to six days. Accepted, and conservative.

## Alternatives rejected

**Filter at decision time from the full history.** This is the bias. A 30-day ADV
computed over the whole series is not causal.

**Static universe list.** Either it excludes late-listed assets (missing most of
the sample) or includes them from the start (backtesting non-existent instruments).

**Daily snapshots.** 7× the rows for eligibility that changes on a monthly
timescale. Weekly is the right granularity, and backward rounding makes the
residual error safe.

**Dropping a symbol when its data ends.** This is survivorship bias applied by the
engine rather than by the config, and it is worse because it is invisible.
`tests/unit/test_delisting.py` asserts the force-close happens.

## Revisit trigger

**Triggered.** Equities are now the primary scope
([ADR-015](015-equities-first.md)), which brings corporate actions, a trading
calendar, and earnings dates — a substantially larger point-in-time problem.
Resolved by [ADR-017](017-unadjusted-prices-pinned-snapshot.md),
[ADR-018](018-liquidity-rank-universe.md), and
[ADR-020](020-earnings-gate.md). Two changes to the specifics above:

- **Snapshot cadence is monthly, not weekly**, because equity liquidity rank moves
  more slowly than crypto listing status and 25 years of weekly snapshots over
  1,500 candidates is 2M rows for no benefit. Backward rounding is retained, so the
  error remains conservative.
- **Index membership is deliberately not used**, which removes the largest
  point-in-time hazard rather than solving it. See
  [ADR-018](018-liquidity-rank-universe.md).
