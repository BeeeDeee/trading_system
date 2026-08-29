# Review of the Original Brief

> This document is the honest assessment you asked for. It states what in the
> original brief was sound, what was statistically or architecturally wrong, and
> what was changed. Every change is justified. Nothing was changed for taste.
>
> **Scope amendment.** After this review, the primary market was reversed from
> crypto to US equities and ETFs. Crypto is M7 and optional. See [§14](#14-the-equity-pivot)
> and [ADR-015](ADR/015-equities-first.md). Sections 1–13 below are the original
> critique; they still apply. Where a later decision supersedes one of them, the
> ADR says so.

---

## 1. Summary judgement

The brief's **philosophy** is right and unusually well-calibrated for someone
starting a trading project. Specifically these are correct and were kept intact:

- Optimise expectancy and robustness, not win rate.
- "No trade" is the normal outcome.
- Costs must be modelled or the whole exercise is fiction.
- Correlation makes "5 crypto longs" one position, not five.
- Point-in-time sentiment or no sentiment.
- Rule-based baseline before ML.
- Ask "why should this edge persist" rather than admiring a backtest.

The brief's **engineering scope** is roughly 8–15× larger than what one person
should build before knowing whether any edge exists. The brief's **scoring
design** — the weighted-sum Opportunity Score — is the single most dangerous
part of it, and it is dangerous in a way that is easy to miss because it looks
professional.

Blunt version: if you build what the brief describes, the most likely outcome is
not a losing trading system. It is an unfinished trading system with 40 modules,
no validated edge, and a scoring formula with a dozen knobs that can be tuned to
make any backtest look good. The second most likely outcome is a finished system
whose backtest is beautiful and whose live results are negative, because the
free-parameter count was never bounded.

---

## 2. The critical flaw: the weighted-sum Opportunity Score

The brief proposes, as the heart of the system:

```text
FinalScore = w_technical * TechnicalScore
           + w_regime    * RegimeScore
           + w_risk      * RiskScore
           + w_sentiment * SentimentScore
           + w_context   * ContextScore
```

and separately proposes ranking on expected value:

```text
EV = P(win) * AvgWin - P(loss) * AvgLoss - Costs
```

These two ideas are incompatible, and the brief does not notice.

### 2.1 It is dimensionally incoherent

`EV` has units of money, or of risk if normalised. `SentimentScore` is a
dimensionless number in `[-1, 1]`. `RegimeQuality` is an invented index.
Multiplying each by a weight and adding them produces a quantity with no unit.
You cannot compare a unitless quantity to a cost, and cost comparison is the
entire point of a selective system. The moment you write
`0.10 * sentiment_score`, you have lost the ability to answer "is this trade
worth more than the 14 bps it costs to take?"

### 2.2 It converts every input into a free parameter

Count the tunables in the brief's design: five-plus weights, each score's
internal normalisation (window length, clipping, scaling), plus a threshold on
the final score. Call it 12–20 effective parameters, all fitted against the same
history, all interacting. With 100 assets and 6 strategies, the number of
distinct configurations you can try before dinner is in the millions. That is
not a scoring system; it is an overfitting engine with a nice diagram. Section
16 of the brief asks for overfitting protection while section 10 builds the
machine that defeats it.

### 2.3 It double-counts evidence

Regime quality, volatility quality, and technical signal strength are largely
the same information viewed three ways. Adding them with weights over-weights
whatever they have in common — usually recent realised volatility — and
under-weights everything else. This produces a score that is mostly a
volatility bet wearing a costume.

### 2.4 It has no probabilistic meaning

There is no answer to "what does a final score of 0.87 predict?" Without that,
the threshold `min_score` is chosen by looking at the backtest, which is exactly
the loop you were trying to avoid.

### 2.5 What replaces it

One number, with a unit, derived rather than tuned. For a candidate setup:

```text
ev_net_r = ev_r_lcb - cost_r

  where ev_r_lcb = a one-sided lower confidence bound on the mean realised
                   outcome, in units of R, of historically resolved setups
                   from the same bin, using only data available before t
        cost_r   = full round-trip execution cost expressed in the same R units
```

Rank by `ev_net_r / expected_bars_held`. Trade only if `ev_net_r` exceeds a
threshold and the bin has enough samples.

This one substitution eliminates every problem above:

| Brief's separate score dimension | Where it goes instead |
|---|---|
| Technical signal strength | It *is* the setup. A setup either triggers or it does not. |
| Confidence | The `- z * stderr` term. Small samples get a worse bound automatically. |
| Volatility quality | The `std_r` inside the same term. Noisy setups get a worse bound automatically. |
| Probability | An observable component of the bin statistics, reported as a diagnostic. |
| Expected return, Risk/reward | Both already inside `mean_r`, by construction. |
| Trading costs | `cost_r`, subtracted in matching units. |
| Regime quality | A hard gate on which strategies may fire, plus optionally a bin dimension. |
| Liquidity | A hard eligibility gate, and an input to `cost_r`. |
| Market context | Same: a gate, or a bin dimension if it earns one. |
| Sentiment | A multiplicative size penalty in `(0, 1]`, never a score addend. |
| Portfolio context | A portfolio-level gate applied after ranking. |

**Weights removed: all of them.** Every input now enters as one of exactly three
things: a *gate* (binary eligibility), a *conditioning variable* (which bin's
statistics apply), or a *size penalty* (multiplicative, bounded, one-sided).
Each of those is far harder to abuse than an additive weight, and each is
individually testable.

Full derivation: [`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md).

---

## 3. Second flaw: the regime detector as a pipeline stage

The brief makes regime detection a stage between features and strategies, with
discrete states TREND / RANGE / CHOP gating strategy selection.

Discrete regime labels are among the least reliable objects in quantitative
finance. They are unstable at the boundary, they lag by construction, and if
implemented carelessly they repaint. Making them a *pipeline stage* means every
downstream component depends on a fragile categorical, and a single flapping
label flips a strategy on and off across a boundary that carries no economic
meaning.

Your previous project already got this right. Its ADR-004 stated: *"Market
regime classification is owned by the Feature Engine as derived features. It is
an optional strategy input, not a pipeline stage and not a trading
prerequisite."* That was correct. The new brief regresses on it.

**Change:** regime stays in the feature layer as (a) two continuous causal
features and (b) one derived label computed from them by a documented pure
function with two thresholds. The label is used as a strategy *gate*, which is
the one use that is defensible, and is available as a bin dimension. No hidden
Markov model, no separate detector component, no state machine.

The regime backbone is Kaufman's Efficiency Ratio — one parameter, causal, no
repainting, and it measures exactly the thing that matters (how much directional
progress per unit of path length). Details in
[`05-FEATURES_AND_REGIME.md`](05-FEATURES_AND_REGIME.md).

---

## 4. Third flaw: an unbounded multiple-testing budget

The brief simultaneously requests:

- 100+ assets
- 6 strategies
- 4 timeframes
- multi-source sentiment with tunable weights
- 5+ score weights
- per-asset calibration
- and, in section 16, protection from overfitting

Those requests are in direct conflict. The search space implied by the first six
bullets is large enough that the best-looking configuration will look excellent
on any history, edge or no edge. Train/validation/test splits do not save you
from this; they only bound the damage if the number of trials is *counted*, and
the brief has no mechanism for counting.

**Changes:**

1. v1 ships **2 strategies**, ≤4 parameters each, **1 decision timeframe**,
   **0 score weights**, and **no per-asset parameters** — pooled statistics by
   default, with a documented statistical test required before any asset is
   allowed its own calibration.
2. Every backtest run appends to `experiments/registry.csv` with a monotonic
   trial counter. Reported Sharpe is always accompanied by a **deflated Sharpe
   ratio** computed from the actual recorded trial count. You cannot honestly
   report a result without declaring how many things you tried, so the tooling
   declares it for you.
3. The holdout period lives behind a **lockbox**: `src/scout/research/lockbox.py` keeps a
   counter of holdout evaluations in a committed file and refuses to run beyond
   the configured budget (default 3) without an explicit override flag that is
   recorded in git history. This makes "I only peeked once" auditable instead of
   aspirational.

That third mechanism is the one piece of genuine innovation here, and it is
cheap: about 40 lines of code that removes the most common way solo quant
projects fool themselves.

---

## 5. Fourth flaw: survivorship bias is under-specified

The brief mentions liquidity filters but never mentions a *point-in-time
universe*. For crypto this is not a minor omission; it is the difference between
a valid and an invalid backtest. Consider:

- Symbols listed mid-history. Backtesting SOL from 2018 is backtesting nothing.
- Delistings and collapses. LUNA, FTT, and a long tail of others. A universe
  built from "symbols available on the exchange today" silently deletes every
  asset that went to zero — arguably the most important observations you have.
- Ticker reuse. Crypto exchanges reuse symbols across unrelated assets.
- Liquidity that arrives late. An asset with $2M daily volume in 2021 and $400M
  in 2024 must be ineligible in 2021 even though it is eligible now.

**Change:** a first-class `universe_snapshots` Parquet table, one row per
`(timestamp, symbol)`, storing eligibility and the reason. It is built causally
from trailing data only, and the engine reads eligibility *from the snapshot*,
never from the current symbol list. Delisted assets remain in history with their
final observations intact. Details in
[`04-DATA_AND_UNIVERSE.md`](04-DATA_AND_UNIVERSE.md).

---

## 6. Fifth flaw: the backtester implied by the brief cannot do the job

The brief's flow is cross-sectional — rank all assets, pick the top few, respect
portfolio correlation. But the default instinct when implementing is a per-asset
loop, and a per-asset loop *structurally cannot* express "at bar `t`, compare 80
assets and choose 3", nor any portfolio-level cap. If this is discovered after
the engine is written, the engine gets rewritten.

**Change, stated up front:** the engine is a **single time-ordered loop over one
panel of all assets**. The outer loop is over timestamps, never over symbols.
Per-symbol work happens inside a timestamp. This is stated as a hard
architectural constraint in
[`11-BACKTEST_ENGINE.md`](11-BACKTEST_ENGINE.md) and enforced by a test.

---

## 7. Sentiment: right instinct, wrong priority

The brief devotes four sections to sentiment. Assessment:

- **Cost:** highest data-engineering cost of any component. Point-in-time news
  archives with reliable `available_at` timestamps are expensive or absent.
  Free APIs give you *current* sentiment, which is useless for backtesting, and
  scraped archives have publication-time ambiguity measured in hours.
- **Expected value:** low at the horizons this system trades. Public sentiment on
  liquid crypto at 4h-plus horizons is close to fully arbitraged. Where sentiment
  has documented value it is at second-to-minute horizons on surprise events, or
  as a *volatility and tail-risk* predictor rather than a direction predictor.
- **Risk:** highest lookahead risk in the system, because the leakage is subtle
  and the backtest improvement from leaking is dramatic and encouraging.

The brief already contains the correct insight, in section 10: sentiment should
be allowed to act as a "penalty" or "risk modifier". That framing is much better
than the additive-weight framing three paragraphs above it.

**Changes:**

1. The `SentimentSource` Protocol and the point-in-time store are specified now
   and built in M2 — the plumbing, including `available_at` discipline, is cheap
   and must exist before any data lands.
2. Sentiment is **penalty-only by default**: it may reduce position size or veto
   a trade, never increase size and never create a trade. This implements the
   brief's own stated principle ("weak setup + positive sentiment should NOT
   magically create a trade") *structurally*, so it cannot be violated by
   parameter choice.
3. Sentiment cannot affect live decisions until it passes a **documented
   incremental-value test**: paired comparison against the identical run with
   sentiment disabled, on the holdout, requiring a bootstrap confidence interval
   on the difference that excludes zero. Until then it is logged and inert.

If sentiment fails that test — and it probably will at these horizons — you will
have spent two weeks instead of two months finding out, and the plumbing stays
useful for the volatility-forecasting use, which is more likely to work.

---

## 8. Timeframes: 15m was going to kill you

The brief lists 15m / 1h / 4h / 1d and says the choice should be tested. Fine in
principle, but one arithmetic fact settles most of it before any test.

Round-trip cost on a liquid crypto perp is roughly 10–20 bps taker plus slippage.
A 15m-bar system targets moves of maybe 0.3–0.6%. Cost is then 20–50% of the
gross move — you need a very large edge just to break even, and cost estimation
error alone swamps the signal. A 4h-bar system targeting 1.5–3% moves puts cost
at 5–10% of the gross move. Same edge, dramatically better survival odds.

**Change (original crypto scope):** v1 uses **4h decision bars** with **1d
context features**, derived by resampling a single 1h base dataset.

**Superseded for the primary (equity) scope** by
[ADR-016](ADR/016-daily-decision-bars.md): US cash sessions are 6.5 hours and do
not divide into 4h bars. v1 equities use **daily session bars** and next-open
execution. The 4h analysis remains the correct one for the optional crypto
sleeve (M7).

---

## 9. Correlation: skip the covariance matrix

The brief asks for correlation, sector exposure, market beta, and factor
exposure. Estimating a 100×100 covariance matrix from noisy crypto returns
produces an object that is mostly estimation error, and it will confidently tell
you that two assets are uncorrelated shortly before they both go to zero
together.

Crypto's correlation structure is, to first order, one factor (BTC) plus a
handful of clusters. That structure is stable and can be written down by hand.

**Change:** v1 uses a **static cluster map in config** (`MAJOR`, `L1`, `L2`,
`DEFI`, `MEME`, `AI`, ...) with per-cluster position and risk caps, plus a
rolling **beta to BTC** as a single continuous exposure measure. Total open risk
("portfolio heat") is capped. No covariance matrix. Upgrade path documented, with
the trigger being "more than 15 concurrent positions", which v1 will never hit.

This is both simpler and more robust than the estimated version. Details in
[`09-PORTFOLIO_AND_RISK.md`](09-PORTFOLIO_AND_RISK.md).

---

## 10. Infrastructure: none of it, yet

Your existing VPS runs Docker, PostgreSQL, and n8n, and the previous project's
architecture put FastAPI and n8n in the diagram from day one. The temptation to
reuse that is strong and should be resisted for this phase.

Research needs: historical bars, a deterministic loop, and files. Millions of 4h
bars across 100 symbols is a few hundred megabytes of Parquet, queryable with
DuckDB faster than the same data in Postgres over a network. A control-plane API
has nothing to control, because there is no live loop.

**Change:** v1–v3 are **CLI plus Parquet plus DuckDB**. No database, no web
framework, no orchestrator, no Docker. Each deferred piece has an explicit
trigger:

| Component | Introduced when |
|---|---|
| PostgreSQL | Live order state needs durable, crash-safe storage (M5) |
| FastAPI | There is a running loop whose state someone needs to inspect (M5) |
| n8n | Scheduled reports and alerts are needed (M5) |
| Docker | Deploying to the VPS (M5) |

This removes roughly 60% of the moving parts from everything before live
trading, and none of it is wasted — the ports are defined so the storage and
execution adapters slot in without touching decision logic.

---

## 11. Full list of what was cut from v1, and why

| Cut | Reason | Returns in |
|---|---|---|
| Weighted score with tunable weights | Dimensionally incoherent, overfitting engine (§2) | Never |
| Regime detector as a pipeline stage | Fragile categorical as a hard dependency (§3) | Never; regime lives in features |
| 4 of 6 strategies | Trial-budget discipline (§4) | M4, one at a time, each must pass holdout |
| Multi-timeframe decisions | Cost arithmetic and trial budget (§8) | M4 at the earliest |
| All machine learning | No baseline to beat yet; ML on an unvalidated pipeline is noise-fitting | M6, gated on baseline edge |
| Real sentiment data | Highest cost, lowest expected value, highest leakage risk (§7) | M2 plumbing, M4 promotion test |
| Equities | Originally deferred. **Reversed:** now the primary market (ADR-015) | v1 / M0–M6 |
| Crypto | Originally v1. **Deferred** to optional M7 | M7, after an equity holdout |
| Covariance / factor model | Estimation error exceeds signal at this position count (§9) | When >15 concurrent positions |
| PostgreSQL, FastAPI, n8n, Docker | Nothing to serve or orchestrate yet (§10) | M5 |
| Live execution | Obviously | M5, after 60 days paper |
| Monte Carlo path simulation | Bootstrap CIs give the same decision-relevant answer for 5% of the work | Optional, M4 |
| Trailing stops, pyramiding, partial exits | Each multiplies the exit-rule search space | M4, individually tested |

**Kept in full, no reduction:** point-in-time correctness, cost modelling,
portfolio-level risk, the decision audit trail, walk-forward validation,
robustness and stability analysis, and the requirement that every rejected
opportunity is logged with its reason.

---

## 12. What the brief asked for that is missing here, deliberately

For completeness, so nothing looks like an oversight:

- **"Design a better domain model if appropriate."** Done — see
  [`02-DOMAIN_MODEL.md`](02-DOMAIN_MODEL.md). The brief's `Opportunity` carried
  both `probability`, `expected_return`, `expected_loss`, `expectancy`,
  `sentiment_score`, `final_score` *and* `confidence`. Six of those seven are
  either redundant or derived. The replacement carries the ranking statistic, its
  components as diagnostics, and full provenance for the audit trail.
- **"15 things to design in Phase 1."** All covered, but not as 15 separate
  abstractions. The system needs **5 Protocols**, not 15 interfaces. Everything
  else is a pure function taking a typed input and returning a typed output,
  which is more testable than an interface and cheaper to read. See
  [`03-INTERFACES.md`](03-INTERFACES.md).
- **"Component diagram."** In [`01-ARCHITECTURE.md`](01-ARCHITECTURE.md). It has
  9 boxes, not 20. The brief's diagram had 6 sequential stages between "signal"
  and "ranking" (score, probability, expected return, risk/reward, sentiment
  adjustment, cost adjustment); those collapse into one function that computes
  one number.

---

## 13. The honest expectation

You should hold this expectation explicitly, because it determines whether the
project is a success or a disappointment.

The original (crypto) version of this paragraph said the probability of a durable
live edge was **low**. That was correct for liquid crypto perps. The equity pivot
improves the *prior*, not the guarantee. See [§14.3](#143-probability-of-finding-an-edge).

What is *highly likely* to have value either way:

1. A research harness that cannot lie to you. Point-in-time data, counted trials,
   a locked holdout, and cost modelling that is pessimistic on purpose. Most
   people never build this and therefore never find out that they have no edge.
2. A definitive, documented answer to "does 12–1 cross-sectional momentum on a
   liquidity-ranked US equity/ETF universe survive realistic retail costs in
   2018–present?" That answer is worth having regardless of its sign.
3. Reusable infrastructure for the next idea, and the one after that.

The way this project fails is by spending six months building the system in the
brief and never reaching step 2. The way it succeeds is by reaching a valid
holdout answer — which is what [`15-ROADMAP.md`](15-ROADMAP.md) is scheduled for
— and then deciding what to do with that answer.

If the M3 result is that there is no edge after costs, the correct action is to
stop and change the hypothesis, not to add sentiment, ML, crypto, and two more
timeframes until the number turns positive. That last sentence is the most
valuable one in this document.

---

## 14. The equity pivot

The original brief asked for crypto first with equities later. That order was
reversed. Full reasoning: [ADR-015](ADR/015-equities-first.md). Short version:

| | Crypto first | Equities first |
|---|---|---|
| Round-trip `cost_r` | ~0.16 R | ~0.02–0.05 R |
| Gross edge needed to matter | ~0.20 R | ~0.05 R |
| Documented anomaly | weak / short sample | 12–1 momentum, 30 years of literature |
| History | ~6 years, one giant bull | 25+ years, many regimes |
| New complexity | funding, 24/7, liquidation tails | corporate actions, calendar, earnings gaps, borrow |

Spending six weeks proving a 0.20 R edge does not exist, before testing a market
that needs 0.05 R, was the wrong order. Crypto is **M7, optional**, and only after
an equity holdout result exists. It will not rescue a failed equity baseline.

### 14.1 What changed in the design

| Decision | Crypto-first | Equities-first |
|---|---|---|
| Decision clock | 4h bars | **Daily session bars** (ADR-016) |
| Context | 1d per-symbol ER | **SPY market regime** (RISK_ON / NEUTRAL / RISK_OFF) |
| Primary strategy | Donchian breakout | **`xsec_momentum_v1`** (ADR-019) |
| Secondary | Range fade | Donchian breakout (range fade dropped from v1) |
| Universe | 120 liquid perps | Liquidity-ranked top 1,000 US stocks+ETFs, delisted included (ADR-018) |
| Beta | BTC | **SPY** |
| Clusters | crypto themes | **GICS sectors + ETF types** |
| Costs | taker + funding | commission/share + borrow + dividends |
| Execution | next 4h open | **next session market-on-open** |
| New gates | — | **earnings window** (ADR-020), penny-price, hard-to-borrow |
| Data | Binance klines (free) | **Paid survivorship-free vendor** (~$60–70/mo) |

The load-bearing method is unchanged: one ranking statistic in R, empirical
walk-forward LCB, panel loop, holdout lockbox, no weights.

### 14.2 What you must pay for

There is no free US equity dataset that includes delisted names, unadjusted
prices, and a corporate-actions table. Yahoo produces a backtest that looks
better than reality by an unbounded amount. Budget **$60–70/month** (Norgate
Platinum or Sharadar). If that is unacceptable, the honest options are to stop
or to treat the output as an exercise. Details:
[`04-DATA_AND_UNIVERSE.md §1`](04-DATA_AND_UNIVERSE.md#1-data-source-this-decision-is-load-bearing).

### 14.3 Probability of finding an edge

Numbers, not vibes. These are calibrated guesses, not forecasts. They are here
so a later positive backtest can be compared against the prior rather than
celebrated.

**Cross-sectional 12–1 momentum on liquid US names, net of honest retail costs,
on the 2018–present holdout:**

| Outcome | Rough probability | Why |
|---|---|---|
| Holdout `mean_r` CI excludes zero (criterion 2) | **30–40%** | The anomaly is real in the literature. Post-2003 Sharpes of 0.2–0.4 after costs are the relevant prior, not 1990s paper Sharpes of 0.8. The holdout contains a momentum crash (Jan 2021) and a concentration rally (2023–25) that hurt classic momentum. |
| That result also clears deflated Sharpe > 0.5 and 1.5× cost (full M3 go) | **20–30%** | Deflation for trials, gap-through-stop losses, and borrow on the short leg eat a lot of the paper premium. |
| The edge is still there in live paper/live at retail scale | **15–25%** | Crowding (MTUM and cousins), capacity is not the issue — **implementation shortfall and crash risk** are. A retail account can trade this universe; it cannot trade it better than the ETFs that already do. |
| Donchian breakout on the same universe independently passes | **10–15%** | Time-series trend on large-cap US equities is weaker than cross-sectional momentum and more of a market-beta bet. Kept as a contrast, not as the hope. A 20–40 name multi-asset ETF trend sleeve (the actual CTA claim) would have a higher prior — that is an M6 experiment, not v1. |
| Crypto sleeve later produces a durable edge the equity sleeve lacked | **very low** | Higher costs, shorter sample, one regime. If equities fail, adding crypto is not a rescue. |

**The most likely M3 outcome is a small positive development result that shrinks
or vanishes on the holdout.** That is a successful project: you will have
measured the gap between the Jegadeesh–Titman paper and a retail implementation,
which is the actual question.

**What would make me more optimistic:** restricting to mid-caps (where momentum
is historically stronger) while keeping the $5M ADV floor; a long-only version
that skips the expensive, hard-to-borrow short leg; or, after M3, a small
multi-asset ETF trend sleeve that actually spans rates/FX/commodities rather
than more US stocks. All are legitimate later experiments if the baseline is
mixed, not if it is dead. Do not swap the v1 core to “diversified trend
following” before measuring the equity-momentum question this project exists
to answer.

**What would make me more pessimistic:** using Yahoo data; fitting the 12–1 skip
month; turning off the earnings gate after seeing a few gap losses; adding
crypto before the equity holdout.

The system is well-designed for this hunt. The market is not obligated to
cooperate. Treat a clean negative as the default, and a deployable live edge as
a bonus you have to earn twice: once on the holdout, once in paper.
