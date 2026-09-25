# Edge Estimation and Scoring

> **The core of the system.** If you read one document, read this one.
>
> Module: `src/scout/scoring/`

---

## 1. The design in one page

The system produces **one** number per candidate, with a unit:

```text
ev_net_r  =  ev_r_lcb  -  cost_r          [units: R]

  R          = one unit of trade risk = stop_atr * atr per unit, times quantity.
               A VOLATILITY SIZING UNIT. It does not require a stop order to
               exist; where one does exist at that distance, a stop-out loses
               exactly 1 R.

  ev_r_lcb   = one-sided lower confidence bound on the mean realised R of
               historically resolved setups in the same bin, using only
               setups that resolved strictly before the decision time.

  cost_r     = full round-trip execution cost for this symbol and size,
               expressed in the same R units.
```

**On the definition of R.** In the original crypto design R was defined as the
stop distance, which is equivalent whenever a stop is placed at `stop_atr * atr`.
It is stated as a *sizing unit* here because `xsec_momentum_v1` sizes on a 5 ATR
unit while placing its stop as a rarely-binding disaster stop
([ADR-019](ADR/019-cross-sectional-momentum-primary.md)). Every consumer in this
document is indifferent, because all of them only ever divide dollars by
`risk_capital_usd`. One consequence to keep in mind when reading `mae_r` and
`stop_rate` for that strategy: an `mae_r` of 0.9 is a near-miss for
`donchian_breakout_v1` and merely a 9% adverse move for `xsec_momentum_v1`.

**Equities note.** Overnight gaps mean a "1 R" loss is a *modal* outcome, not a
bound. A stock gapping 15% through a 5.4% stop loses 2.8 R. This is the weakest
joint in the R framework and is why [ADR-020](ADR/020-earnings-gate.md) exists;
the labeling pass models it honestly (§2.2) rather than clipping losses at 1 R.

Trade only if `ev_net_r >= min_ev_net_r`. Rank by `ev_net_r / expected_bars_held`.

That is the whole scoring system. There are **no weights**. Everything the brief
wanted to weight enters instead as one of exactly three mechanisms:

| Mechanism | Nature | Examples |
|---|---|---|
| **Gate** | binary, before scoring | liquidity, spread, data quality, regime, warm-up, minimum bin samples |
| **Conditioning** | selects which bin's history applies | strategy, direction, volatility bucket |
| **Size multiplier** | multiplicative in `(0, 1]`, after ranking | sentiment penalty, portfolio heat scaling, thin-liquidity discount |

Each is individually testable, and none can be turned into a knob that
manufactures a trade out of weak evidence. Rationale and the alternatives
rejected: [`ADR/002-single-ranking-statistic.md`](ADR/002-single-ranking-statistic.md)
and [`00-REVIEW.md §2`](00-REVIEW.md#2-the-critical-flaw-the-weighted-sum-opportunity-score).

---

## 2. Step 1 — Labeling historical setups

Before any edge can be estimated, every setup the strategies would have produced
across history must be detected and resolved. This is an **offline pass**, run by
`cli/label_setups.py`, producing `data/labels/setups_<strategy_id>.parquet`.

### 2.1 Detection pass

```python
for asset_id in candidate_assets:
    rows = feature_panel.slice_symbol(asset_id)
    for row in rows:                      # every session, ignoring eligibility
        for strategy in strategies:
            if not strategy.regime_allows(row):     # per-symbol AND market regime
                continue
            setup = strategy.detect(row)
            if setup is not None:
                setups.append(setup)
```

Detection **ignores universe eligibility and portfolio constraints on purpose**.
The bin statistics should describe "what happens when this pattern occurs",
estimated from the largest valid sample available. Eligibility is a separate
question, applied at decision time. Conditioning the sample on eligibility would
shrink it and introduce a subtle selection effect that varies over time as
liquidity thresholds bind differently.

Two exceptions:

- Setups on bars where `is_warm` is False are excluded, because their features are
  not computable.
- **`xsec_momentum_v1` setups are an unavoidable partial exception**, because
  `mom_252_xs_pct` is defined relative to the *eligible* population
  ([`05-FEATURES_AND_REGIME.md §8`](05-FEATURES_AND_REGIME.md#8-cross-sectional-features)).
  An ineligible symbol has a NaN rank and therefore produces no setup, so its
  labeled sample is conditioned on eligibility whether we like it or not. This is a
  documented limitation, not a bug: the alternative — ranking against all
  candidates — would let untradeable names determine the decile boundaries, which
  is worse. It does mean the momentum bins have a smaller sample than the breakout
  bins, and the cold-start drought (§5) lasts longer for that strategy.

Also note that the **earnings gate is not applied during labeling**, for the same
reason eligibility is not: the bin should describe the pattern, and the gate is a
decision-time filter. But this has a consequence worth stating, because it cuts the
other way from the usual conservatism: the labeled sample *includes* earnings-gap
outcomes that live trading will not experience, so `std_r` is inflated and
`ev_r_lcb` is therefore **pessimistic** for the gated system. The paired
gate-on/gate-off sensitivity in [ADR-020](ADR/020-earnings-gate.md) measures the
size of this effect.

### 2.2 Triple-barrier resolution

For each setup, walk forward from the entry bar. Entry is at the **open of the
bar after `setup.ts`** — this is the bar convention and it is what makes the label
match what a live trade would have experienced.

```python
def resolve_setup(setup: Setup, forward: pd.DataFrame, cfg: LabelConfig) -> ResolvedSetup:
    """`forward` = this symbol's sessions with ts > setup.ts, ascending."""

    if len(forward) == 0:
        return open_result(setup)

    entry = forward.iloc[0]["open"]
    sign = setup.direction.sign

    # Re-anchor stop and target on the ACTUAL entry, preserving the strategy's
    # intended distances. The setup's prices were computed from the decision
    # bar's close; the fill happens at the next open, which gaps.
    risk_per_unit = abs(setup.reference_price - setup.stop_price)
    stop = entry - sign * risk_per_unit
    if setup.target_price is None:
        target = None                       # exit on time or rank only
    else:
        reward_per_unit = abs(setup.target_price - setup.reference_price)
        target = entry + sign * reward_per_unit

    for i in range(len(forward)):
        bar = forward.iloc[i]

        # GAP-THROUGH RULE: a session OPENING beyond the stop fills at the OPEN,
        # not at the stop level. The loss then exceeds 1 R. This is the normal
        # case in equities, not the tail.
        gapped_through = (bar["open"] <= stop) if sign > 0 else (bar["open"] >= stop)
        if gapped_through:
            return resolved(outcome=STOP, exit_price=bar["open"], bars_held=i + 1, ...)

        hit_stop = (bar["low"] <= stop) if sign > 0 else (bar["high"] >= stop)
        hit_target = (target is not None and
                      ((bar["high"] >= target) if sign > 0 else (bar["low"] <= target)))

        # CONSERVATIVE TIE RULE: if a single bar touches both barriers, we
        # cannot know the intrabar order from OHLC, so assume the stop.
        if hit_stop:
            return resolved(outcome=STOP, exit_price=stop, bars_held=i + 1, ...)
        if hit_target:
            # Symmetric gap handling: a session OPENING beyond the target fills at
            # the OPEN, which is BETTER than the target. Correct, and it must not
            # be clipped to the target level — that would understate winners while
            # the stop rule overstates losers, biasing every bin downward.
            gap_fill = (bar["open"] >= target) if sign > 0 else (bar["open"] <= target)
            exit_px = bar["open"] if gap_fill else target
            return resolved(outcome=TARGET, exit_price=exit_px, bars_held=i + 1, ...)
        if i + 1 >= setup.max_hold_bars:
            return resolved(outcome=TIME, exit_price=bar["close"], bars_held=i + 1, ...)

    return open_result(setup)      # ran out of data: OPEN, excluded from stats
```

Five rules in that function carry most of the honesty of the whole system:

1. **Re-anchoring on the actual entry.** The setup's stop was computed from the
   decision bar's close, but the fill happens at the next open. If the market gaps
   1.5 ATR in your favour overnight, a naive implementation keeps the original stop
   and target, which makes the trade look like it had a 3.5-ATR reward for a
   0.5-ATR risk. Preserving *distances* rather than *levels* keeps the labeled R/R
   equal to the intended R/R.

   This mattered occasionally in crypto. **In equities it matters on every single
   trade**, because every entry is preceded by an overnight gap. Getting it wrong
   here is not an edge case; it is a systematic distortion of the entire label set.
   Note also that the gap is *not* charged as a cost — it is unbiased market
   movement, and charging it would be double-counting
   ([`08-COSTS.md §2.3`](08-COSTS.md#23-slippage)).
2. **The gap-through-stop rule.** A session opening below a long's stop fills at the
   open, and the loss exceeds 1 R. Clipping every loss at exactly 1 R is the
   standard shortcut and it systematically understates tail risk and overstates
   Sharpe. In crypto this was a rare liquidity event; in equities it is a routine
   consequence of overnight news. Expect a meaningful fraction of `STOP` outcomes to
   have `realised_r_gross` below −1.0, and expect the worst few to be near −4.0.
   If none are, the rule is not implemented.
3. **Gap-through-target is handled symmetrically.** A session opening above a long's
   target fills better than the target. It is tempting to clip this to the target
   "to be conservative", and that would be wrong: combined with rule 2 it would
   overstate every loser and understate every winner, biasing every bin downward by
   an amount that grows with gap frequency. Conservatism means being pessimistic
   about *unknowns*, not being pessimistic about arithmetic.
4. **The conservative tie rule.** OHLC data cannot tell you whether the low or the
   high came first within a session. Assuming the favourable order is the single
   largest source of fake backtest profit in stop-and-target systems, and its effect
   grows with volatility, so it inflates exactly the periods that look most
   exciting. Always assume the stop.
5. **`OPEN` setups are excluded, not counted as flat.** Counting an unresolved setup
   as zero R would bias `mean_r` toward zero and, worse, would systematically
   exclude the *slowest* winners at the end of every sample.

`target is None` (`xsec_momentum_v1`) reduces this to a two-barrier problem: stop
and time. The rank exit is **not** modelled in labeling, because it depends on the
cross-sectional state of the whole panel at each future session rather than on this
symbol's path. Consequence: labeled `bars_held` for momentum is an upper bound on
realised holding period, so `ev_per_bar_r` slightly understates that strategy. The
direction of the error is conservative, and the size is measurable by comparing
labeled `bars_held` against realised `ClosedTrade.bars_held` in the M3 report.

### 2.3 Also record

- `mae_r` — maximum adverse excursion in R over the holding period. If winners
  routinely have `mae_r` of 0.9, the stop is one tick from being hit and the
  strategy is fragile to a slightly different stop distance.
- `mfe_r` — maximum favourable excursion. If losers routinely reach `mfe_r` of
  1.8 before stopping out, the target is too far away, and that is diagnostic
  information the brief correctly asked for.
- `realised_r_gross` — before costs, from the labeling pass.
- `resolution_ts` — the exit bar's `ts`. **This is the field that makes the
  walk-forward causal**, and getting it wrong invalidates everything downstream.

---

## 3. The resolved setup table

`data/labels/setups_<strategy_id>.parquet`

| Column | Type | Notes |
|---|---|---|
| `asset_id` | string | permanent id. **All joins use this** |
| `symbol` | string | display label only |
| `setup_ts` | ts UTC | decision session close |
| `entry_ts` | ts UTC | next session's close time (fill was at that session's open) |
| `resolution_ts` | ts UTC | exit session close. **Null for OPEN.** |
| `strategy_id` | string | |
| `direction` | string | |
| `regime` | string | per-symbol, at detection |
| `market_regime` | string | at detection |
| `vol_bucket` | string | at detection |
| `reference_price`, `entry_price`, `stop_price`, `exit_price` | double | adjusted |
| `target_price` | double, **nullable** | null for `xsec_momentum_v1` |
| `risk_per_unit` | double | |
| `reward_risk_ratio` | double, **nullable** | null when `target_price` is null |
| `entry_gap_atr` | double | `(entry_price - reference_price) / atr` at detection, signed by direction. New for equities; the diagnostic for rule 1 above |
| `outcome` | string | `SetupOutcome` |
| `bars_held` | int32 | sessions |
| `realised_r_gross` | double | `sign * (exit - entry) / risk_per_unit`. **May be well below −1.0** |
| `mae_r`, `mfe_r` | double | |
| `atr_pct`, `efficiency_ratio_20`, `beta_bench_90`, `adv_usd_60` | double | features at detection, for later bin experiments |
| `mom_252_xs_pct`, `overnight_var_share_60`, `xs_population` | double | ditto; the leading candidates for a fourth bin dimension |
| `had_earnings_in_window` | bool | whether the earnings gate *would* have fired. Enables the ADR-020 paired sensitivity without a second labeling pass |

Expected volume, much larger than the crypto estimate: 1,000–1,500 symbols over 25
years of daily bars is roughly 6–9M symbol-sessions. At a 1–3% setup rate for
`donchian_breakout_v1` that is 60,000–270,000 rows, and `xsec_momentum_v1` fires on
roughly 20% of eligible symbol-sessions by construction (two deciles), so about 1.5M
rows. Partition that file by year.

The momentum sample being an order of magnitude larger than the breakout sample is
a genuine advantage: `ev_r_lcb`'s small-sample penalty is small for it from early in
the walk-forward, so its cold-start drought is short.

**Sanity checks before proceeding, all mandatory.** Pooled over everything:

| Check | `donchian_breakout_v1` | `xsec_momentum_v1` |
|---|---|---|
| No-edge `mean_r` | near 0 | near 0 |
| No-edge win rate | near `1/(1 + 2.0) ≈ 0.33` | near 0.50 (no target, so R/R is set by the path) |
| `stop_rate` | 0.4–0.6 | **under 0.10** — a 5 ATR stop should rarely bind |
| `time_rate` | 0.1–0.3 | **over 0.85** |
| `min(realised_r_gross)` | **below −1.0** | **below −1.0** |
| `mean(entry_gap_atr)` | near 0 | near 0 |

If pooled `mean_r` is 0.4, you have a bug, not an edge. The most likely culprits, in
order:

1. A missing split adjustment, producing a fake 90% one-day return that every
   momentum feature reads as the strongest signal in the universe
   ([`04-DATA_AND_UNIVERSE.md §6.7`](04-DATA_AND_UNIVERSE.md#67-data-quality-checks-and-is_suspect)).
2. A survivorship-biased candidate list.
3. A cross-sectional rank computed over the full panel rather than per timestamp.
4. The missing `.shift(1)` on the Donchian channel.
5. A tie rule resolving in your favour, or losses clipped at −1.0 R.
6. Stop and target levels not re-anchored on the actual entry.

`mean(entry_gap_atr)` materially different from zero is diagnostic of a
**one-session alignment error**: if entries are being taken at the *same* session's
open rather than the next one, the gap will correlate with the signal and the mean
will be positive. It is the cheapest available detector of the most damaging
possible bug, and it costs one column.

---

## 4. Step 2 — Bins

```python
@dataclass(frozen=True, slots=True)
class BinKey:
    strategy_id: str
    direction: Direction
    vol_bucket: VolBucket
```

Cardinality: 2 strategies × 2 directions × 3 volatility buckets = **12 bins**.

### Why these three dimensions and not more

The binning decision is a direct trade-off: more dimensions means more relevant
conditioning but fewer samples per bin, and the confidence-bound penalty grows as
`1/sqrt(n)`. With 60,000–270,000 resolved breakout setups and about 1.5M momentum
setups, 12 bins gives thousands to hundreds of thousands per bin, which yields a
tight bound even at the earliest walk-forward evaluation point.

- **`strategy_id`** — mandatory. Different rules, different outcome distributions,
  and radically different `stop_rate` and `time_rate` profiles. Pooling them would
  be meaningless.
- **`direction`** — the long/short asymmetry in equities is large and structural. A
  long book in a rising market and a short book in the same market are not two
  samples of one process. Pooling directions would make every long look better than
  it is during a bear market and vice versa. This also delivers the brief's
  requirement to evaluate long and short independently, and given that shorts cost
  2.2× longs ([`08-COSTS.md §3.3`](08-COSTS.md#33-example-c--liquid-large-cap-short))
  it is doing real work.
- **`vol_bucket`** — outcome distributions differ sharply by volatility regime, and
  `cost_r` is much larger in low-volatility regimes (the risk distance shrinks while
  spread does not). Without this dimension the system would systematically over-trade
  low-volatility setups.

**Per-symbol regime is deliberately not a bin dimension in v1.** For
`donchian_breakout_v1` it is already a hard gate, so including it would split the
bin into two near-mirror halves and halve the sample for no new information. For
`xsec_momentum_v1` the field is not read at all.

**Market regime is deliberately not a bin dimension either**, and this one is a
closer call than it looks. Both strategies allow `RISK_ON` and `NEUTRAL`, so the
dimension would have cardinality 2 rather than being constant, and there is a real
hypothesis that momentum behaves differently in the two. Excluded from v1 for two
reasons: it would take 12 bins to 24 with a strongly unbalanced split (`RISK_ON` is
most sessions), and market regime is *already* reported as a performance breakdown
in the M3 diagnostics, which answers the question descriptively without spending
sample. **This is the strongest candidate for a fourth dimension at M6**, ahead of
`overnight_var_share_60`.

**Assets are pooled.** This is the brief's own §17 principle ("one framework plus
asset-specific calibration") implemented correctly: pooling is the default, and
per-asset separation requires evidence. The test to justify separation is in §9 —
and note that with equity sample sizes it can now actually pass, which it
essentially never could with crypto data. §9 has been tightened accordingly.

**ETFs are pooled with stocks in v1**, which is a defensible simplification and a
real one. A sector ETF's outcome distribution is narrower than a single name's, its
gap risk is much smaller, and it has no earnings event. If the M3 report shows the
ETF subset's realised R materially diverging from the pooled prediction, `is_etf`
becomes the fourth bin dimension — it is cheap (cardinality 2), it has a clear
mechanism, and the population split is roughly 85/15 rather than 95/5.

### Adding a bin dimension

Requires all three of:

1. `min(n)` across the resulting bins, measured at the **earliest** walk-forward
   evaluation point, is at least `min_bin_samples`.
2. A pre-registered hypothesis for why outcomes differ along that dimension,
   written before looking at the split.
3. One trial recorded in the registry.

---

## 5. Step 3 — Bin statistics and the confidence bound

For bin `k` at evaluation time `as_of`:

```python
sample = resolved[
    (resolved.bin_key == k)
    & (resolved.resolution_ts.notna())
    & (resolved.resolution_ts < as_of)          # STRICTLY before. Causality.
]
```

Then:

```python
n        = len(sample)
mean_r   = sample.realised_r_gross.mean()
std_r    = sample.realised_r_gross.std(ddof=1)
stderr   = std_r / sqrt(n)
ev_r_lcb = mean_r - z * stderr                  # z from config, default 1.28
```

`z = 1.28` is the one-sided 90% normal quantile: we are 90% confident the true
mean is at least `ev_r_lcb`. Config-selectable; `z = 1.645` (95%) is the
conservative alternative, and the sensitivity of results to `z` must be reported
in M3 (§10).

### Gross in the bins, net at decision time

`ev_r_lcb` is computed from **`realised_r_gross`** — the labeling pass has no
account, no size, and no notional, so it cannot compute costs. Costs are
subtracted once, at decision time, at the size actually being considered:

```text
ev_net_r        = ev_r_lcb (gross)  -  cost_r (this symbol, this size, this horizon)
realised net R  = ClosedTrade.realised_r  = net_pnl_usd / risk_at_entry_usd
```

Both sides of the calibration check in §10.1 are therefore **net**, and the
comparison is valid. Do not apply costs in the labeling pass: cost depends on
notional, which depends on equity and volatility at decision time, so a
cost-adjusted label would bake in one specific account size and become wrong for
every other one.

### Why this single expression replaces five of the brief's score dimensions

```text
ev_r_lcb = mean_r  -  z * std_r / sqrt(n)
           ───┬──      ──┬──   ──┬──
              │           │       └── few samples  →  larger penalty
              │           └────────── noisy outcomes → larger penalty
              └────────────────────── the actual expected value
```

- The brief's **"confidence"** dimension is the `1/sqrt(n)` term. A setup type
  with 40 historical examples is automatically penalised relative to one with
  4,000, with no weight to choose.
- The brief's **"volatility quality"** dimension is the `std_r` term. A setup
  whose outcomes are wildly dispersed is automatically penalised. Note this is
  *outcome* dispersion, not price volatility, which is the right thing to
  penalise and is not what the brief was measuring.
- The brief's **"probability"** and **"expected return"** dimensions are both
  inside `mean_r` by construction. `mean_r = Σ p_i · r_i` over the empirical
  outcome distribution, which is the exact quantity the brief's
  `P(win)·AvgWin - P(loss)·AvgLoss` was trying to approximate — but computed
  from the realised distribution rather than a three-point approximation, so
  timeouts and partial moves are included automatically instead of being an
  awkward special case.

Five knobs became zero knobs, and the remaining `z` is a stated confidence level
rather than a fitted weight.

### The heavy-tail caveat

`std_r / sqrt(n)` assumes an approximately normal sampling distribution of the
mean. Trade returns are heavy-tailed and right-skewed for trend strategies. With
`n > 500` the CLT is adequate; below that the bound is optimistic. Therefore:

```yaml
edge:
  lcb_method: normal          # normal | bootstrap
  bootstrap_iterations: 2000
  bootstrap_seed: 20260827    # fixed: determinism is non-negotiable
```

`bootstrap` computes the 10th percentile of 2,000 bootstrap resample means.
Default `normal` for speed during development; **`bootstrap` is mandatory for the
holdout run**, and the difference between the two must be reported. If the two
methods disagree materially, trust the bootstrap.

### Diagnostics stored on every `BinStats`

`win_rate`, `avg_win_r`, `avg_loss_r`, `target_rate`, `stop_rate`, `time_rate`,
`median_r`, `p05_r`, `p95_r`, `mean_bars_held`. None of these enter the ranking.
They exist so that a surprising `ev_net_r` can be explained. In particular
`time_rate` above about 0.5 means the strategy is mostly producing timeouts and
neither barrier is meaningful.

### The as_of grid

Computed on a **monthly** grid (the first trading session of each month, at its
session close). Lookup rounds **backward**: a decision on 2023-04-17 uses the
2023-04-03 statistics.

The grid is on **sessions**, not calendar dates, so that "the first of the month"
never lands on a holiday and silently shifts the boundary by a day.

Backward rounding is the safe direction. Forward rounding, or exact-time
recomputation with a subtle boundary error, would let a setup that resolved on
2023-04-10 inform a decision on 2023-04-05. Monthly recomputation is also about
300× cheaper than per-bar recomputation, and the statistics move slowly enough
that the loss of precision is immaterial.

### Cold start

If `n < min_bin_samples` (default **100**), the bin is unusable and every candidate
mapping to it is rejected with `INSUFFICIENT_BIN_SAMPLES`.

Consequence: the early part of any backtest produces almost no trades. **This is
correct and must not be "fixed".** It is what an honest walk-forward looks like, and
it is why the warm-up split exists. The alternative — seeding bins with statistics
computed from the full history — is in-sample leakage of the most damaging kind,
because it leaks the answer directly into the ranking.

With equity sample sizes the drought is short and asymmetric between the strategies,
which is worth anticipating so it is not mistaken for a bug:

- `xsec_momentum_v1` fires on about 20% of eligible symbol-sessions by construction,
  so with a few hundred eligible symbols it accumulates 100 resolved setups per bin
  within a few months of the first eligible session. Its drought is roughly one
  quarter.
- `donchian_breakout_v1` fires on 1–3% of sessions, so its `HIGH`-volatility short
  bins in particular may take a year or more to fill.

Both are gated independently, so the momentum strategy trades while the breakout
strategy is still cold. Combined with the 273-session feature warm-up and the
252-session market-regime warm-up, the first genuinely tradeable session is roughly
18 months after the data start — hence a warm-up split running to the end of 2003
against a data start in 1998
([`04-DATA_AND_UNIVERSE.md §8`](04-DATA_AND_UNIVERSE.md#8-data-splits)).

---

## 6. Step 4 — Costs in R

Full model in [`08-COSTS.md`](08-COSTS.md). The conversion is what matters here:

```text
cost_r = total_cost_usd / risk_capital_usd

  where risk_capital_usd = equity_usd * risk_fraction_per_trade
```

This works because position size is chosen so that a stop-out loses exactly
`risk_capital_usd`. Therefore `qty * risk_per_unit = risk_capital_usd`, and a
cost of `X` USD is exactly `X / risk_capital_usd` in R. Same unit as `ev_r_lcb`,
so the subtraction is meaningful.

### The magnitude, which is the crux of the whole project

```text
notional     = qty * entry_price = risk_capital / risk_per_unit * entry_price
             = risk_capital / (stop_atr * atr_pct)

with stop_atr = 3.0 and atr_pct = 0.018 (a liquid large cap on daily bars):
notional     = risk_capital / 0.054 ≈ 18.5 * risk_capital

cost_bps     ≈ 12.5 bps round trip (see 08-COSTS.md SS3.1 for the itemisation)
cost_usd     = 0.00125 * 18.5 * risk_capital ≈ 0.023 * risk_capital

cost_r       ≈ 0.023 R
```

**Roughly 0.023 R per round trip for a long in a liquid name**, rising to about
0.054 R at the bottom of the universe and about 0.051 R for a short (borrow plus
dividends). Full derivations and the three worked examples:
[`08-COSTS.md §3`](08-COSTS.md#3-worked-examples).

The comparison that justifies the entire equity pivot:

| | Crypto perp | Equity large-cap long |
|---|---|---|
| `cost_r` | 0.159 R | **0.023 R** |
| Gross `mean_r` needed for a net Sharpe of 0.5 | 0.33 R | **0.15 R** |
| Cost as a share of the required gross edge | 48% | **15%** |

Note that leverage on risk capital *fell* from 41.7× to 18.5×, because the equity
stop distance (5.4% of price) is much wider than the crypto one (2.4%). Wider stops
mean less notional per unit of risk, and less notional means less cost. This is the
mechanism behind the 7× reduction, alongside the absence of funding and a 5×
smaller fee-and-spread component.

Writing this number down before any implementation is the most useful thing in this
document: it tells you what magnitude of edge you are hunting. A gross `mean_r` of
0.15 R over 21 sessions on a 5.4% risk unit is a 0.8% average gain, which is
squarely inside the range the published cross-sectional momentum literature reports.
The crypto equivalent required 0.33 R, which was not.

The arithmetic that connects `mean_r` to a Sharpe ratio is in
[`08-COSTS.md §3.4`](08-COSTS.md#34-what-this-means-the-required-edge-and-whether-it-is-plausible)
and it should be read alongside this section. Any strategy whose gross `mean_r` is
below about 0.05 R is not a strategy at any cost level.

### Provisional versus final cost

`cost_r` is nearly invariant to `risk_fraction` (both numerator and denominator
scale with it), except through the nonlinear impact term. So:

1. **Ranking** uses a provisional notional from `equity * risk_fraction`.
2. **Sizing** recomputes with the final notional and multipliers.
3. If `ev_net_r` drops below `min_ev_net_r` at step 2, the trade is dropped with
   `BELOW_EV_THRESHOLD`.

Step 3 binds only for large orders in thin markets, which is precisely where it
should.

---

## 7. Step 5 — Building the Opportunity

```python
def build_opportunity(setup, row, stats, cost, asset) -> Opportunity:
    ev_net_r = stats.ev_r_lcb - cost.cost_r
    expected_bars_held = max(stats.mean_bars_held, 1.0)
    return Opportunity(
        symbol=setup.symbol, ts=setup.ts, strategy_id=setup.strategy_id,
        direction=setup.direction, setup=setup,
        ev_net_r=ev_net_r,
        ev_per_bar_r=ev_net_r / expected_bars_held,
        ev_r_lcb=stats.ev_r_lcb, ev_r_point=stats.mean_r,
        cost_r=cost.cost_r, expected_bars_held=expected_bars_held,
        bin_key=stats.key, bin_n=stats.n, bin_std_r=stats.std_r,
        bin_win_rate=stats.win_rate,
        regime=row.regime, market_regime=row.market_regime,
        vol_bucket=row.vol_bucket,
        adv_usd_60=..., spread_bps_est=..., beta_bench_90=row.beta_bench_90,
        cluster=asset.cluster, is_etf=asset.is_etf,
    )
```

---

## 8. Step 6 — Threshold and ranking

```python
def rank(candidates, cfg) -> tuple[Opportunity, ...]:
    eligible = [c for c in candidates if c.ev_net_r >= cfg.min_ev_net_r]
    return tuple(sorted(
        eligible,
        key=lambda c: (-c.ev_per_bar_r, c.symbol, c.strategy_id),   # stable ties
    ))
```

### The threshold: `min_ev_net_r`, default 0.05

Not fitted. Derived — but the derivation changes with the equity cost structure,
and it is worth being explicit about why the number did not.

In the crypto design the threshold was set by **cost-estimation error**:
`cost_r ≈ 0.06–0.12 R` with easily ±50% uncertainty meant an `ev_net_r` of 0.02 R
sat inside the error bar of the cost model itself. With `cost_r ≈ 0.023 R`, that
uncertainty is now about ±0.012 R, so cost error no longer sets the floor.

The binding consideration is instead **the edge you need for the result to be worth
trading at all**. A net Sharpe of 0.5 requires net `mean_r ≈ 0.125 R`
([`08-COSTS.md §3.4`](08-COSTS.md#34-what-this-means-the-required-edge-and-whether-it-is-plausible)),
and `ev_r_lcb` sits below `mean_r` by the small-sample penalty. A 0.05 R floor on
the *lower bound* therefore admits candidates whose point estimate is roughly
0.08–0.15 R and rejects everything below — which is the right cut. The two
derivations happen to land on the same number for entirely different reasons.

**`min_ev_net_r` is not a parameter to sweep.** It is reported at three values
(0.02, 0.05, 0.10) as a sensitivity check, not optimised. If results are only good
at exactly 0.037, there is no edge.

### The ranking key: `ev_per_bar_r`

Ranking by `ev_net_r` alone would prefer a 0.20 R trade held 40 sessions over a
0.12 R trade held 15 sessions, but the second earns 0.008 R per session against the
first's 0.005 — more per unit of the genuinely scarce resource, which is a capital
slot under a portfolio heat cap.

This matters more with two strategies of different holding periods than it did with
two of similar ones. `xsec_momentum_v1` holds 21 sessions and
`donchian_breakout_v1` up to 40, so the per-bar normalisation is what stops the
longer-horizon strategy monopolising slots on the strength of a larger absolute
`ev_net_r`.

The threshold applies to `ev_net_r` (is it worth doing at all?) and the sort
applies to `ev_per_bar_r` (which first, given limited slots?). Those are two
different questions and conflating them is why holding-period blindness is a
common and expensive bug.

`expected_bars_held` comes from the bin's `mean_bars_held`, so it is measured, not
assumed.

### Tie-breaking

Explicit, alphabetical, secondary keys. Never rely on input order: input order
depends on dict iteration and symbol-list order, which makes the run
non-reproducible in exactly the situation where two candidates are equally good
and only one slot is available.

---

## 9. Should an asset get its own calibration?

The brief asked for "asset-specific calibration only where statistically
justified". Here is the test, so that "justified" is not decided by eye.

For candidate symbol `s` and bin `k`, with pooled sample `A` (all other symbols) and
symbol sample `B`:

1. Require `n_B >= 200`.
2. Welch's t-test on `realised_r_gross` between `A` and `B`.
3. Bonferroni-correct for the number of symbols tested: require
   `p < 0.05 / n_symbols_tested`.
4. Require the difference to be economically material: `|mean_B - mean_A| > 0.10 R`.
5. Require it to hold **out of sample**: split `B` in half by time; the sign and
   approximate magnitude must appear in both halves.
6. Record the test as a trial in the registry.

All six, or the asset is pooled.

**This test was near-unpassable with crypto sample sizes, and that was doing much of
the work. It is not any more, and the change is dangerous.** With 25 years of daily
data, a symbol like `AAPL` can accumulate several hundred `xsec_momentum_v1` setups,
so `n_B >= 200` is satisfiable for hundreds of symbols. Two consequences:

- Step 3's Bonferroni correction now has to be honoured strictly. Testing 1,000
  symbols means requiring `p < 5e-5`, and with a large `n_B` that is reachable by
  chance often enough to matter. The count is `n_symbols_tested`, not
  `n_symbols_that_looked_interesting`; screening first and testing second is the
  bias this correction exists to prevent, and it is invisible after the fact.
- Step 5 becomes the real gate rather than a formality. A 25-year sample splits into
  two 12-year halves, which is a genuine out-of-sample test.

**Additional requirement for equities, and it is a hard one:** per-symbol
calibration is **only permitted for ETFs**, not for individual stocks. The reason is
mechanistic. An ETF is a persistent, rule-defined object; `XLK` in 2005 and `XLK` in
2025 are the same instrument tracking the same rule, so a per-symbol statistic
describes something stable. A company is not: `AAPL` in 2005 was a consumer
electronics turnaround and in 2025 is the largest company in the world. A
per-symbol edge fitted on the former does not describe the latter, and the
statistical test cannot detect that because the sample is drawn from both.

Reject per-stock calibration by construction rather than by threshold. This is the
one place where the equity pivot makes a test *easier* to pass, and easier is not
better.

---

## 10. Required M3 diagnostics

Produced by `research/diagnostics.py`, in every holdout run's report. These are
what make the estimator falsifiable rather than merely plausible.

1. **Calibration plot.** Bucket closed trades by predicted `ev_net_r_at_entry`
   (deciles), plot mean realised net R per bucket against the bucket's predicted
   midpoint, with bootstrap error bars. A working estimator gives a monotone,
   roughly 45-degree line. A flat line means the ranking carries no information
   and the whole system is a coin flip with extra steps. **This is the single
   most important plot the project produces.**
2. **Bin table.** Per bin: `n`, `mean_r`, `std_r`, `ev_r_lcb`, `win_rate`,
   `mean_bars_held`, realised trade count, realised `mean_r`.
3. **Cold-start profile.** Trades per month over the run. Confirm the early
   drought and confirm it ends.
4. **`z` sensitivity.** Full metrics at `z ∈ {0.0, 1.28, 1.645}`. `z = 0` means
   ranking on the raw mean; if `z = 0` is materially better, the small-sample
   penalty is costing more than it saves and that is worth knowing.
5. **Cost sensitivity.** Full metrics at `cost_multiplier ∈ {1.0, 1.5, 2.0, 3.0}`
   (the set defined in
   [`12-RESEARCH_PROTOCOL.md §6.2`](12-RESEARCH_PROTOCOL.md#62-cost-sensitivity)).
   A wider range than crypto used, because `cost_r` is now small enough that a 3×
   error would still leave a viable strategy, and demonstrating that is worth more
   than demonstrating survival at 1.5×.
6. **Tie-rule sensitivity.** Re-run the labeling with the optimistic tie rule
   (target wins when a bar touches both). The gap between the two is a direct
   measure of how much of the result depends on intrabar luck. A large gap means the
   stop is too close to the entry relative to bar range. Expect this gap to be
   *smaller* than it would have been in crypto, because `xsec_momentum_v1` has no
   target and so no tie to resolve — which is worth noting as a structural advantage
   of that strategy rather than a fortunate accident.
7. **Bootstrap versus normal LCB.** Report both.
8. **Market-regime breakdown.** `ev_net_r` and realised R split by `market_regime`,
   with the two failure signatures listed in
   [`05-FEATURES_AND_REGIME.md §7`](05-FEATURES_AND_REGIME.md#7-market-regime).
9. **Gap decomposition.** Distribution of `realised_r_gross` for `STOP` outcomes,
   and the fraction below −1.5 R. This is the direct measurement of the risk the
   R framework does not bound
   ([ADR-020](ADR/020-earnings-gate.md)). If the fifth percentile of stopped trades
   is below −3 R, position sizing is mis-calibrated regardless of the aggregate
   metrics and the daily-loss circuit breaker needs re-deriving.
10. **Earnings-gate paired sensitivity.** Full metrics with the gate on and off,
    from the same label set via `had_earnings_in_window`. Required by
    [ADR-020](ADR/020-earnings-gate.md).
11. **Factor-loading diagnostic.** Regress the strategy's returns on market, size,
    value, and momentum factor returns. A strategy whose alpha is not
    distinguishable from zero has found nothing, however good its Sharpe. For a
    long-biased equity system this is the most important single number after the
    calibration plot, because a positive Sharpe from market beta alone is the default
    outcome, not a success.

---

## 11. When ML is allowed to replace the estimator

At M6, and only if all of the following hold:

- The rule-based baseline has produced a holdout result with at least 200 closed
  trades and a bootstrap 90% CI on `mean_r` that excludes zero.
- The calibration plot from §10.1 is monotone.
- The trial budget for M6 has been pre-registered.

Then, and only then, the estimator becomes a Protocol:

```python
class EdgeEstimator(Protocol):
    def estimate(self, setup: Setup, row: FeatureRow, as_of: datetime) -> BinStats: ...
```

with `BinnedEmpiricalEstimator` (the v1 implementation) and, for example,
`GradientBoostedEstimator`. The ML variant must return the *same* `BinStats`
shape, including an honest `ev_r_lcb` — which for a model means an out-of-fold
prediction interval, not a training-set residual. Nothing downstream changes,
and the comparison is a paired holdout run against the baseline.

If the baseline shows no edge, ML will not find one. It will find the same
absence of edge, plus an overfit, and it will take three months. That
conditionality is the point of putting ML at M6.
