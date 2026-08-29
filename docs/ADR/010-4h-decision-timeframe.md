# ADR-010: 4h decision bars with 1d context

**Status:** **Superseded by [ADR-016](016-daily-decision-bars.md)** for the
primary equity scope. Remains in force for the optional crypto sleeve (M7).

> This ADR analysed timeframe selection for crypto perpetuals. The equity pivot
> ([ADR-015](015-equities-first.md)) changes the cost inputs by roughly 7× and
> replaces the 6.5-hour trading session, which a 4h bar does not divide. The
> reasoning below is retained because it is still the correct analysis for crypto
> and because the *method* — settle the timeframe on cost arithmetic before
> testing anything — is reused verbatim in ADR-016.

## Context

The brief lists 15m, 1h, 4h, and 1d and says the choice should be tested rather
than assumed. Correct in principle, but one arithmetic fact settles most of it
before any test is run.

## The arithmetic

Round-trip cost on a liquid crypto perp, modelled honestly
([`08-COSTS.md §3`](../08-COSTS.md#worked-example)), is roughly 38 bps including
funding — of which the volatility-scaled slippage term is about 42%.

That slippage term scales with ATR, and ATR scales roughly with the square root
of the timeframe. Target size, however, scales roughly *linearly* with the
timeframe for a fixed ATR multiple. So shortening the timeframe shrinks the
target faster than it shrinks the cost:

| Timeframe | Typical target move | Round-trip cost | Cost as % of gross target |
|---|---|---|---|
| 15m | ~0.4% | ~20 bps | **~50%** |
| 1h | ~0.9% | ~25 bps | **~28%** |
| 4h | ~2.5% | ~38 bps | **~15%** |
| 1d | ~6.0% | ~60 bps | **~10%** |

At 15m, cost estimation *error* alone swamps the signal. At 4h there is room for
an edge to survive.

Two further considerations:

- **Sample size** favours shorter bars, and 1d would give roughly 2,190 bars over
  six years per symbol — thin for the bin statistics in ADR-003.
- **Operational load** favours longer bars. A 4h clock means six decisions a day
  and no need for low-latency infrastructure.

## Decision

- **Decision timeframe: 4h.** UTC-anchored, bars closing at 00:00, 04:00, 08:00,
  12:00, 16:00, 20:00.
- **Context timeframe: 1d**, for the regime agreement requirement and context
  returns.
- **Base ingest: 1h**, resampled to both. Stored so the choice is reversible
  without re-downloading.
- **One decision timeframe in v1.** A second is an M4 experiment with its own
  trial budget.

## Consequences

- Cost is roughly 15% of the gross target rather than 50%. This is the difference
  between a hypothesis that can be tested and one that is arithmetically doomed.
- The 1d context requirement provides what the brief wanted from multi-timeframe
  analysis at the cost of one threshold, not a second full pipeline.
- 13,000 4h bars per symbol over six years is enough sample for 12 bins.
- No latency-sensitive infrastructure. Deciding two minutes after a bar close is
  fine.
- Cost: intra-bar setups are invisible. Accepted.
- Cost: the 4h anchoring is arbitrary relative to market microstructure. Mitigated
  by a start-date-jitter robustness test.

## Alternatives rejected

**15m decisions.** Cost is roughly half the gross target. No edge of plausible
magnitude survives.

**1h decisions.** Cost at 28% of the gross target is survivable but tight, and it
triples the trade count and therefore the cost drag. Reasonable as an M4
sensitivity run, using data already stored.

**1d decisions.** Best cost ratio, but roughly 2,190 bars per symbol makes the bin
statistics thin, and it would need many more symbols to compensate.

**Multi-timeframe decisions from the start** (regime on 1d, setup on 4h, entry on
15m). This is the brief's suggestion and it is defensible in principle, but it
triples the implementation surface and the trial budget before any single-timeframe
baseline exists. The 1d context feature captures the highest-value part of the
idea for a fraction of the cost.

## Revisit trigger

Superseded. For crypto (M7) the first sensitivity run to try is 1h, since the data
is already stored. For equities see [ADR-016](016-daily-decision-bars.md).
