# ADR-013: Sentiment as a penalty-only size modifier behind a promotion gate

**Status:** Accepted

## Context

The brief devotes four sections to sentiment and proposes it as an additive term
in the opportunity score with a configurable weight, while also — correctly, three
paragraphs later — suggesting it could act as a "penalty" or "risk modifier".

Assessment of sentiment as a component of this system:

- **Highest data-engineering cost.** Point-in-time news archives with reliable
  first-observed timestamps are expensive or unavailable. Free APIs return
  *current* sentiment, which is useless for backtesting.
- **Lowest expected value at these horizons.** Public sentiment on liquid crypto
  at 4h-plus horizons is close to fully arbitraged. Documented value exists at
  second-to-minute horizons on surprise events, or as a *volatility* predictor
  rather than a direction predictor.
- **Highest lookahead risk.** The leakage is subtle, and the backtest improvement
  from leaking is large and encouraging, which is the worst possible combination.

Under the brief's additive design, a sufficiently large `w_sentiment` would create
trades from sentiment alone, and nothing structural would prevent it.

## Decision

Three parts.

### 1. Plumbing now, data later

The `SentimentSource` Protocol, the point-in-time observation store, and the
`available_ts` discipline are built at M1–M2. They are cheap, and retrofitting
point-in-time discipline never happens.

### 2. Penalty-only integration

Sentiment is a **multiplicative size modifier in `[0, 1]`**, applied in the
portfolio stage, after ranking. It never enters `ev_net_r`.

```python
aligned = view.score * direction.sign
if aligned >= 0.0:
    return 1.0                    # NEVER a bonus
severity = abs(aligned) * view.confidence
if severity >= veto_threshold:
    return 0.0                    # veto
return max(min_multiplier, 1.0 - penalty_slope * severity)
```

### 3. Promotion gate

`enabled: false` by default. Enabling requires a paired holdout comparison against
the identical sentiment-disabled run, passing all four criteria in
[`10-SENTIMENT.md §6`](../10-SENTIMENT.md#6-the-promotion-gate), including
robustness to a 240-minute ingest lag.

## Consequences

- The brief's own stated principle — "weak setup + positive sentiment should NOT
  magically create a trade" — is **guaranteed by construction** rather than by
  parameter choice.
- Ranking is sentiment-free, so the calibration diagnostic in
  [`07-EDGE_AND_SCORING.md §10.1`](../07-EDGE_AND_SCORING.md#10-required-m3-diagnostics)
  measures the edge estimator cleanly.
- Missing or stale sentiment is neutral, never blocking. Trading continues.
  (Contrast [ADR-020](020-earnings-gate.md), where a missing earnings date *is*
  blocking. The difference is that sentiment only ever reduces size, so absence
  cannot make a trade more likely, whereas absence of an earnings date can.)
- Cost: whatever upside sentiment-confirmation has is forgone. Accepted, because
  quantifying it in R would require a fitted number, and that number would be fitted
  against the same data used to evaluate the system.
- The likely outcome is that the promotion test fails. Two weeks spent instead of
  two months, and the plumbing remains useful for the funding and open-interest
  data the cost model needs anyway.

## The unrequested recommendation

The general principle: **prefer a quantitative, exchange- or regulator-reported
measure of crowded positioning over any text-derived polarity score.** Positioning
is a mechanism; headline polarity is not. Positioning data has exact publication
timestamps, so point-in-time correctness is trivial; news archives do not, so it
is expensive and error-prone.

### For equities (primary scope)

Build in this order:

1. **Short interest.** Reported bi-monthly by FINRA with **exact, published
   settlement and dissemination dates**, which makes it the single best
   point-in-time sentiment series available in any asset class. Short interest as a
   fraction of float, and its change, is a direct positioning measure. Available
   free from FINRA and included in Sharadar and Norgate.
   - Caveat that must be respected: the dissemination date is roughly 8 calendar
     days after the settlement date. Using the settlement date as `available_ts` is
     a two-week lookahead leak, and it is exactly the kind that would look like
     signal. `available_ts` is the **dissemination** date.
2. **VIX and the VIX term structure** (`VIX9D/VIX`, `VIX/VIX3M`) as a market-wide
   input. Free from CBOE, daily, exact timestamps. Not per-symbol, so it belongs to
   market regime rather than to `SentimentView` — and it is the more valuable of the
   two uses.
3. **Options implied-volatility skew** per symbol, if a data source is available.
   Put-call skew is the cleanest per-symbol measure of directional positioning.
   Requires paid options data; defer until 1 and 2 show something.
4. **News polarity.** Last, and only if the above show anything. It is the most
   expensive, the least reliable, and the most leak-prone.

Note that the **earnings calendar is not sentiment** — it is a gate
([ADR-020](020-earnings-gate.md)) — and the **market regime is not sentiment** —
it is a feature ([`05-FEATURES_AND_REGIME.md`](../05-FEATURES_AND_REGIME.md)).
Both were candidates for being mislabelled as sentiment inputs, and both do more
work where they are.

### For crypto (M7, optional)

**Funding-rate skew** and **open-interest change**, for the same reasons: exchange
reported with exact timestamps, quantitative, no classifier drift, already needed
for the cost model, and extreme funding is the most direct available measure of
crowded positioning. Then `fear_greed`. Then `cryptopanic`, only if the first two
show anything.

## Alternatives rejected

**Additive weighted sentiment term.** ADR-002. Dimensionally incoherent, and it
permits sentiment to dominate.

**Symmetric multiplier, boosting on alignment.** Requires answering "how much is
positive sentiment worth in R", which has no defensible answer, so the number
would be fitted.

**Sentiment as a bin dimension.** Statistically the cleanest option — it would let
the data speak. Rejected for v1 on sample size: adding a 3-level sentiment
dimension takes 12 bins to 36 and takes the earliest walk-forward bins below the
minimum. It is the right upgrade if sentiment ever passes the promotion gate.

**Skipping sentiment entirely.** Tempting given the expected value, but the
point-in-time store is needed for short-interest and VIX data regardless (and for
funding data in the crypto sleeve), and building the Protocol now costs a day.

## Revisit trigger

M4 promotion test. If it fails, revisit only with a *different* hypothesis —
sentiment as a volatility or tail-risk predictor — not with different parameters
on the same one.
