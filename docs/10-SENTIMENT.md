# Sentiment

> Point-in-time store, penalty-only integration, and a promotion gate that
> sentiment must pass before it can affect a live decision.
>
> Module: `src/scout/sentiment/`

Read [`00-REVIEW.md §7`](00-REVIEW.md#7-sentiment-right-instinct-wrong-priority)
for why sentiment is built early but promoted late.

---

## 1. Position

Sentiment is treated as a **risk modifier with unproven predictive value**, not as
alpha. Concretely:

- It can **reduce** position size. It can **never increase** it.
- It can **veto** a trade. It can never **create** one.
- It never enters `ev_net_r` and therefore never affects ranking order.
- It is inert (multiplier exactly 1.0) until it passes §6's promotion test.

This structurally implements the brief's own stated principle — "weak setup +
positive sentiment → sentiment should NOT magically create a trade" — so it
cannot be violated by a parameter choice. Under the brief's additive-weight
design, a large enough `w_sentiment` *would* create trades from sentiment alone,
and nothing in the design would have stopped it.

---

## 2. Point-in-time discipline

Two timestamps on every observation, and the distinction between them is the
whole game:

| Field | Meaning |
|---|---|
| `event_ts` | When the event or publication happened |
| `available_ts` | When **we** could first have known about it |

**Backtests filter on `available_ts`. Never on `event_ts`.** Enforced by
`tests/unit/test_sentiment_pit.py`, which builds a fixture where the two differ
by 6 hours and asserts that a decision at `event_ts + 1h` sees nothing.

### The lag rule

For any source without a trustworthy first-observed timestamp:

```yaml
sentiment:
  ingest_lag_minutes: 60
```

```python
available_ts = event_ts + timedelta(minutes=cfg.ingest_lag_minutes)
```

Sixty minutes is not an estimate of API latency. It is an admission that scraped
archives have unreliable publication times, that "publication" and "propagation"
differ, and that assuming you saw a headline the instant it was written is the
most common sentiment-backtest lie. A robustness run at
`ingest_lag_minutes: 240` is **required** in
[`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md); if the result depends on
the lag being small, the result is not real.

For sources that genuinely provide a first-seen timestamp (a polling collector
writing its own observation time), use it directly and set the lag to 0 for that
source.

### Never overwrite

Sentiment providers revise their data — an article's sentiment is recomputed, a
score is corrected. **Revisions are appended as new observations with a new
`available_ts`, never as an update to an existing row.** Overwriting a historical
score with a revised one is a lookahead bug that is invisible in the code and
whose effect grows with how much the provider revises. `observations.parquet` is
append-only; the ingest job asserts this.

---

## 3. Aggregation into a `SentimentView`

```python
def build_view(
    observations: Sequence[SentimentObservation],   # available_ts <= ts, this symbol
    symbol: str,
    ts: datetime,
    cfg: SentimentConfig,
) -> SentimentView:
```

### Decay weighting

```python
age_hours = (ts - obs.available_ts).total_seconds() / 3600.0
if age_hours > cfg.max_age_hours:                    # default 120 (daily)
    continue                                          # drop entirely
decay = 0.5 ** (age_hours / cfg.half_life_hours)     # default 24 (daily)
weight = decay * obs.source_weight * log1p(obs.sample_size)
```

Exponential decay with a 24-hour half-life on the daily clock: a headline from
four days ago carries 1/16 the weight of one from today. `log1p(sample_size)` means 200
articles count more than 20 but not ten times more — article counts are
heavy-tailed and dominated by scraping artifacts.

### Score, confidence, disagreement

```python
score = Σ(weight_i * score_i) / Σ(weight_i)                  # [-1, 1]

n_eff = Σ(weight_i)**2 / Σ(weight_i**2)                      # Kish effective n
sample_confidence = min(1.0, n_eff / cfg.n_eff_full)         # default 20

per_source = weighted mean score within each source
disagreement = weighted stdev of per_source values, clipped to [0, 1]
agreement_confidence = 1.0 - disagreement

freshness_bars = (ts - newest available_ts) / bar_duration
freshness_confidence = 0.5 ** (freshness_bars / cfg.freshness_half_life_bars)  # 2 sessions

confidence = sample_confidence * agreement_confidence * freshness_confidence
```

Multiplying the three confidence components rather than averaging them means any
one being near zero drives the whole thing to zero. That is the correct
behaviour: a single stale source with a large sample is not confident evidence,
and an average would let a strong sample mask total staleness.

```python
is_stale = freshness_bars > cfg.max_freshness_bars            # default 5 sessions
is_neutral = is_stale or n_eff < cfg.min_n_eff or confidence < cfg.min_confidence
```

A neutral view yields a multiplier of exactly `1.0`. **Missing sentiment must
never block a trade** — that would make an optional input a hard dependency, which
violates the failure policy in
[`01-ARCHITECTURE.md §7`](01-ARCHITECTURE.md#7-failure-policy).

---

## 4. The multiplier

```python
def sentiment_multiplier(view: SentimentView | None, direction: Direction,
                         cfg: SentimentConfig) -> float:
    """Returns a value in [0.0, 1.0].
       1.0        = no effect (neutral, disabled, or supportive)
       (0.0, 1.0) = size penalty
       0.0        = veto (caller rejects with SENTIMENT_VETO)
    """
    if not cfg.enabled or view is None or view.is_neutral:
        return 1.0

    # Alignment: positive when sentiment agrees with the trade direction.
    aligned = view.score * direction.sign          # [-1, 1]

    if aligned >= 0.0:
        return 1.0                                 # NEVER a bonus. See §1.

    severity = abs(aligned) * view.confidence      # [0, 1]

    if severity >= cfg.veto_threshold:             # default 0.70
        return 0.0

    return max(cfg.min_multiplier,                 # default 0.40
               1.0 - cfg.penalty_slope * severity) # slope default 0.85
```

Worked cases:

| Setup | `score` | `confidence` | `aligned` | `severity` | Multiplier |
|---|---|---|---|---|---|
| LONG, positive news | +0.60 | 0.80 | +0.60 | — | **1.00** — no bonus |
| LONG, mildly negative | −0.30 | 0.50 | −0.30 | 0.15 | **0.87** |
| LONG, strongly negative | −0.80 | 0.90 | −0.80 | 0.72 | **0.00** — veto |
| SHORT, strongly negative | −0.80 | 0.90 | +0.80 | — | **1.00** — no bonus |
| LONG, stale data | any | any | — | — | **1.00** — neutral |

The asymmetry is the design. It maps exactly onto the brief's three cases:

```text
Strong long setup + positive sentiment   → unchanged (the setup was already good)
Strong long setup + negative sentiment   → reduced or vetoed
Weak setup + positive sentiment          → still below the EV threshold; no trade
```

The third row is guaranteed rather than hoped for, because sentiment never
touches `ev_net_r`.

### Why no bonus, ever

Allowing a bonus would require answering "how much is positive sentiment worth,
in R?" There is no defensible answer, so any number chosen would be fitted, and
the fitting would happen against the same data used to evaluate the system. The
asymmetric treatment costs you whatever upside sentiment-confirmation has — and
buys immunity to the failure mode where sentiment quietly becomes the dominant
term in the decision. That trade is worth making until §6 says otherwise.

---

## 5. Sources

| Source | `source_id` | Cost | Point-in-time quality | Milestone |
|---|---|---|---|---|
| Null (always empty) | `null` | free | perfect | M1 |
| VIX term structure | `vix_term` | free (in the benchmark file) | Excellent — CBOE daily close, known publication time | M2 |
| Short interest | `short_interest` | vendor (already paid) | Fair — bi-weekly FINRA/exchange, **must lag to publication, not settlement** | M2 |
| Fear & Greed Index | `fear_greed` | free | Good — daily, published on a schedule, **crypto, M7** | M7 |
| CryptoPanic | `cryptopanic` | free tier | Fair — has `published_at`; propagation lag unknown | M7 |
| Funding-rate skew | `funding_skew` | free | Excellent — exchange data with exact timestamps | M7 |
| Twitter/X, Reddit | — | expensive | Poor — historical access is restricted and expensive; bot contamination | Not planned |

### What is worth ingesting at M2

The two equity sources that have a *mechanism*, not a headline polarity:

- **`vix_term`.** Contango vs backwardation of VIX vs VIX3M is a market-wide risk
  regime, not a cross-sectional alpha. Broadcast it. A backwardated term
  structure is the crowded-long / crash-fear state; it penalises new longs and
  does nothing for shorts (no bonus). Weight 0.50 because it has no
  name-level information.
- **`short_interest`.** Days-to-cover or short interest as % of float, lagged to
  the **publication** date (typically T+4 to T+8 after the settlement date).
  Using the settlement date is a lookahead bug. Extreme short interest opposed
  to a long is a squeeze-risk penalty, not a reason to get longer.

```python
# vix_term normalisation to [-1, 1]
# Contango (VIX3M > VIX) is normal → score near 0.
# Backwardation (VIX > VIX3M) is stress → negative score → penalises longs.
basis = (vix_close - vix3m_close) / vix3m_close
score = -np.tanh(basis / cfg.vix_term_scale)   # scale default 0.10
```

```python
# short_interest: percentile of days-to-cover in the name's own 2y history.
# High short interest → negative score → penalises new longs in that name.
# Does not create shorts (no bonus).
percentile = rank of current DTC in trailing 504-session distribution
score = -(2.0 * percentile - 1.0)
```

Headline NLP is deferred. It has no trustworthy historical `available_ts` at
retail cost, and at daily horizons on liquid names it is close to fully
arbitraged. If it is ever added, it uses the same Protocol and the same
penalty-only multiplier.

### Market-wide sources applied to individual symbols

VIX term is a single market-wide number. Broadcasting it to every symbol is
legitimate but must be discounted, because it carries no cross-sectional
information:

```yaml
sentiment:
  sources:
    vix_term:      { source_weight: 0.50, broadcast: true,  lag_minutes: 0 }
    short_interest:{ source_weight: 1.00, broadcast: false, lag_minutes: 1440 }
```

A broadcast source can only ever move the whole book in one direction, so it acts
as a market-timing filter rather than a selection input. That is a different, and
weaker, claim than per-symbol sentiment, and the weight reflects it.

---

## 6. The promotion gate

Sentiment starts **disabled** (`sentiment.enabled: false`). Turning it on
requires passing this test, and the result must be committed to
`docs/results/sentiment_promotion.md`.

### Procedure

1. Complete the M3 baseline holdout with sentiment disabled. This is the control.
2. Ingest sentiment for the full history, with the point-in-time discipline of §2.
3. Run the **identical** config, identical seed, identical period, with only
   `sentiment.enabled: true` changed. Paired comparison.
4. Compute, on the **holdout** period:
   - Δ mean net R per trade, with a paired bootstrap 90% CI
   - Δ Sharpe, with a paired bootstrap 90% CI
   - the number of trades vetoed, and their realised R had they been taken
   - the number of trades downsized, and their realised R
5. Re-run step 3 with `ingest_lag_minutes: 240`.

### Pass criteria — all four

| # | Criterion | Rationale |
|---|---|---|
| 1 | Bootstrap 90% CI on Δ mean net R excludes zero and is positive | The effect is real, not noise |
| 2 | Vetoed trades had a mean realised R materially below the unvetoed population | The veto is selecting badly, not randomly |
| 3 | Criteria 1 and 2 still hold at `ingest_lag_minutes: 240` | Not an artifact of assumed instant availability |
| 4 | At least 30 vetoed and 60 downsized trades | Enough sample to mean anything |

### If it fails

Set `enabled: false`, write the negative result into
`docs/results/sentiment_promotion.md`, and move on. **This is a successful
outcome of the experiment**, and it costs one holdout evaluation from the lockbox
budget.

The likely failure mode is criterion 1: the CI on Δ mean R will straddle zero,
because at daily horizons on liquid US names, public positioning and VIX-term
data are close to fully in the price. Expect this. Writing the negative result
down is what stops you retrying it with different parameters in three months.

The plumbing remains useful regardless: the same point-in-time store serves
short-interest and VIX data that research diagnostics need, and sentiment as a
*volatility* predictor — a different hypothesis, testable later with the same
infrastructure — is more likely to work than sentiment as a direction predictor.

---

## 7. Full configuration

```yaml
sentiment:
  enabled: false                     # promotion gate, §6
  source_ids: [null]                 # [vix_term, short_interest] at M2

  ingest_lag_minutes: 60
  half_life_hours: 24                # daily clock: a day, not 12 hours
  max_age_hours: 120
  max_freshness_bars: 5              # sessions
  freshness_half_life_bars: 2

  n_eff_full: 20
  min_n_eff: 3.0
  min_confidence: 0.25

  veto_threshold: 0.70
  penalty_slope: 0.85
  min_multiplier: 0.40

  sources:
    vix_term:      { source_weight: 0.50, broadcast: true,  lag_minutes: 0 }
    short_interest:{ source_weight: 1.00, broadcast: false, lag_minutes: 1440 }
    # M7 only:
    funding_skew:  { source_weight: 1.00, broadcast: false, lag_minutes: 0 }
    fear_greed:    { source_weight: 0.30, broadcast: true,  lag_minutes: 0 }
    cryptopanic:   { source_weight: 1.00, broadcast: false, lag_minutes: 60 }
```

These parameters are **not** swept — sweeping them against the holdout is
exactly the multiple-testing failure the promotion gate exists to prevent. Set
them once from the reasoning above, run the paired test once, accept the answer.

---

## 8. Appendix: crypto sources (M7, optional)

Do not implement before an equity holdout write-up exists.

- **`funding_skew` first**, then `fear_greed`, then `cryptopanic` if the first
  two show anything. Extreme funding is crowded-positioning data with exact
  exchange timestamps; that is a mechanism.
- Normalisation: percentile of the current funding rate in its trailing 90-day
  distribution, mapped to `[-1, 1]` with a negative sign so high funding
  penalises longs.
- Half-life 12 hours and `max_freshness_bars: 18` on the 4h clock, not the
  daily values above.
