# Research Protocol

> How to avoid fooling yourself. Module: `src/scout/research/`
>
> This document is a **process specification**, not a suggestion. The trial
> registry and the holdout lockbox are enforced in code.

---

## 1. Data splits

```text
1998-01-01 ──── 2004-01-01 ──────────── 2017-12-31 ─── present
   warm-up          DEVELOPMENT              HOLDOUT
   (no trades)      (unlimited use)          (3 evaluations, ever)
```

| Split | Period | Rules |
|---|---|---|
| Warm-up | 1998-01-01 → 2003-12-31 | Features and bins warm up. No trading. |
| Development | 2004-01-01 → 2017-12-31 | Free use. Every run is registered. 14 years, includes 2008. |
| Holdout | 2018-01-01 → present | Lockbox: 3 evaluations total for the project's lifetime. |
| Forward | after go-live | The only genuinely out-of-sample data. Paper trading, M5. |

The holdout is roughly 8 years and ~36% of the scored sample. It contains the
2018 Q4 selloff, COVID, the January 2021 momentum crash, the 2022 bear, and the
2023–25 concentration rally. A holdout that contains only one regime tests only
one thing; this one is hostile on purpose.

The holdout **range is a constant in `research/splits.py`**, not a YAML value.
You cannot widen it by editing a config file.

Canonical statement: [`04-DATA_AND_UNIVERSE.md §8`](04-DATA_AND_UNIVERSE.md#8-data-splits).

**The walk-forward is not a split; it is a property of the estimator.** Bin
statistics at time `t` use only setups resolved before `t`, at every point in
both development and holdout. So the development period is *already* an honest
walk-forward of the estimator. The holdout exists to catch the other thing you
can overfit: the thresholds, the strategy parameters, the gates, the bins, and
the hundred small choices in between.

---

## 2. The trial registry

Every backtest run appends one row to `experiments/registry.csv` **automatically**,
from `write_run_outputs`. There is no manual step and no way to opt out.

| Column | Notes |
|---|---|
| `trial_id` | monotonic integer |
| `run_id` | matches `results/<run-id>/` |
| `timestamp_utc` | wall-clock; this file is a lab notebook, not a decision input |
| `git_sha` | `--dirty` suffix when the tree is not clean |
| `config_hash` | sha256 of the resolved config |
| `split` | `DEVELOPMENT` \| `HOLDOUT` \| `FORWARD` |
| `period_start`, `period_end` | |
| `strategies` | comma-separated ids |
| `n_trades`, `mean_r`, `sharpe`, `max_dd_pct`, `total_return_pct` | headline results |
| `notes` | free text from `--notes`; required for holdout runs |

### Why this matters more than it looks like it does

The deflated Sharpe ratio requires the number of trials. Without a count you
cannot compute it, and without it you cannot distinguish "Sharpe 1.4" from
"the best of 200 attempts at Sharpe 1.4", which are entirely different claims.
The registry turns a number you would otherwise guess into a number you have.

`git_sha` with a `--dirty` marker is there because a result produced from an
uncommitted tree cannot be reproduced. Holdout runs from a dirty tree are
**refused**.

---

## 3. The holdout lockbox

`src/scout/research/lockbox.py`

```python
LOCKBOX_PATH = Path("experiments/holdout_lockbox.json")

def request_holdout_evaluation(reason: str, git_sha: str) -> None:
    """Raises ScoutConfigError if the budget is exhausted.

    State: {"budget": 3, "used": 1, "evaluations": [
              {"ts": ..., "git_sha": ..., "reason": ..., "run_id": ...}]}

    The file is COMMITTED to git. Every holdout evaluation is therefore a
    visible commit with a stated reason and a diff. Spending a peek silently
    is not possible without an obvious, auditable commit.
    """
```

`run_backtest.py` calls this whenever the configured period overlaps the holdout.
Bypassing it requires `--force-holdout`, which prints a large warning, records
`forced: true`, and consumes budget anyway.

### Why three

The holdout's value is destroyed by reuse. Each evaluation that informs a change
converts the holdout a little more into training data, and the conversion is
silent. Three evaluations is enough for the plan in
[`15-ROADMAP.md`](15-ROADMAP.md):

1. **M3** — the baseline result. The answer to "does this work at all".
2. **M4** — the best development candidate after the parameter and strategy work.
3. **Reserve** — the sentiment promotion test, or a final pre-go-live check.

If you find yourself wanting a fourth, the honest options are: accept the
development result as your estimate and discount it heavily, or wait for real
forward data. Adding budget is not one of them.

This mechanism is roughly 40 lines of code and it removes the most common way
solo quant projects deceive themselves. It is the highest return-on-effort item
in the entire repository.

---

## 4. Metrics

`research/metrics.py`. All computed from `trades.csv` and `equity.csv`.

### Return and risk

| Metric | Definition |
|---|---|
| `total_return_pct` | `equity[-1] / equity[0] - 1` |
| `cagr_pct` | annualised from the equity curve |
| `sharpe` | annualised, from **bar-level** equity returns, risk-free = 0 |
| `sortino` | same, downside deviation only |
| `max_drawdown_pct` | worst peak-to-trough |
| `max_drawdown_duration_days` | longest time under water |
| `calmar` | `cagr_pct / max_drawdown_pct` |
| `ulcer_index` | RMS of the drawdown series |

Sharpe from bar-level returns, annualised with `sqrt(bars_per_year)`. Do not
compute it from trade returns: trade-level Sharpe ignores idle capital and is not
comparable to any published figure.

### Trade statistics

`n_trades`, `win_rate`, `avg_win_r`, `avg_loss_r`, `mean_r`, `median_r`, `std_r`,
`profit_factor`, `expectancy_r`, `payoff_ratio`, `max_consecutive_losses`,
`avg_bars_held`, `mae_r_mean`, `mfe_r_mean`.

### Activity and cost

`trades_per_month`, `exposure_pct` (fraction of bars with any position open),
`turnover_annual`, `total_fees_usd`, `total_funding_usd`,
**`cost_drag_pct`** (total costs as a fraction of gross P&L).

`cost_drag_pct` is the honesty metric. If it exceeds 50%, the strategy is a fee
generator and no amount of parameter work fixes that — only a longer holding
period or a larger target does.

### Statistical significance

| Metric | Definition |
|---|---|
| `mean_r_ci_low`, `mean_r_ci_high` | bootstrap 90% CI on mean R, 2000 iterations, fixed seed |
| `sharpe_ci_low`, `sharpe_ci_high` | same, on Sharpe |
| `deflated_sharpe` | Bailey & López de Prado, using the registry trial count |
| `p_value_mean_r` | one-sided bootstrap p-value that mean R > 0 |

### Deflated Sharpe

```python
def deflated_sharpe(sharpe, n_trials, n_obs, skew, kurtosis) -> float:
    """Bailey & Lopez de Prado (2014). Adjusts the observed Sharpe for:
      - the number of trials (from the registry — not a guess),
      - non-normality of returns (skew and kurtosis),
      - sample length.

    Interpretation: the probability that the true Sharpe exceeds zero, given
    that this was the best of `n_trials` attempts.
    """
```

`n_trials` is read from the registry, counting development runs since the last
strategy-code change. **Report `sharpe` and `deflated_sharpe` together, always.**
A Sharpe of 1.3 from 200 trials is a weaker claim than a Sharpe of 0.9 from 5,
and only the deflated figure shows that.

---

## 5. Stability analysis

The brief was right that this matters more than the headline number. A single
Sharpe figure is one draw; stability is what tells you whether the edge is a
property of the market or of your sample.

Every dimension below produces a table and a plot in `research/stability.py`.
The **question to ask of each is the same: is the sign consistent, and is the
dispersion explicable?** Not "is every cell positive" — demanding that would
select for overfit configurations.

| Dimension | Grouping | Failure signal |
|---|---|---|
| Time | per calendar year, per quarter | All the profit in one year |
| Regime | per-symbol TREND_* / RANGE / CHOP, and market RISK_ON / NEUTRAL / RISK_OFF | Only works in one regime while the gate claims to handle several |
| Volatility | `VolBucket` | Only works in `HIGH`, where costs and gaps are worst |
| Direction | long vs short separately | Only long, in a sample with a large up-move — a beta bet |
| Asset | per symbol, and top-5 contribution share | Top 3 symbols are more than 60% of P&L |
| Cluster | per GICS / ETF cluster | One sector dominates |
| Strategy | per `strategy_id` | One strategy carries everything (then delete the other) |
| Market beta | regression of strategy returns on **SPY** returns | High beta with a low alpha t-stat: you built a leveraged index fund |
| Momentum crashes | the six worst known windows named in ADR-019 | Loss > 15% in any of them → do not proceed |

### The SPY-beta check

```python
alpha, beta, t_alpha = ols(strategy_session_returns ~ spy_session_returns)
```

Report `beta`, `alpha_annualised_pct`, and `t_alpha`. If `beta > 0.5` and
`t_alpha < 2.0`, the system is a leveraged SPY position with extra steps, and the
correct comparison is not to zero but to buy-and-hold — which is why
`equity.png` mandates the SPY (total-return) benchmark on the same axes.

This is the check most likely to deliver bad news, and the most valuable one for
that reason.

---

## 6. Robustness suite

`research/robustness.py`, run with `scout robustness --config <cfg>`. Every item
is a **sensitivity report, not an optimisation**. The output is one table
showing how the headline metrics move; no item's result is used to select a
configuration.

### 6.1 Parameter sensitivity

For each strategy parameter, evaluate at `[0.7×, 0.85×, 1.0×, 1.15×, 1.3×]` of
its default, one at a time (not a grid).

Read it as follows:

- **Flat surface** — the parameter does not matter. Consider deleting it.
- **Smooth, gently sloped** — healthy. The default is in a plateau.
- **Sharp peak at the default** — you have overfit, or you got lucky. Either way
  the live result will be the average of the neighbourhood, not the peak.

A one-at-a-time sweep costs `5 × n_params` runs instead of `5^n_params`. It
misses interactions, and that is an accepted trade: a full grid would burn the
trial budget for a marginal amount of information, and the plateau question is
answerable one dimension at a time.

### 6.2 Cost sensitivity

`cost_multiplier ∈ {1.0, 1.5, 2.0, 3.0}`. **A strategy that dies at 1.5× is not
deployable.** Live costs exceed modelled costs — that is the reliable direction
of the error. The 3.0× point is unique to equities: `cost_r` is small enough that
surviving a 3× error is informative.

### 6.3 Execution-assumption sensitivity

| Variant | Tests |
|---|---|
| Optimistic tie rule (target wins) | How much of the result is intrabar luck |
| Entry at `t+2` open instead of `t+1` | Sensitivity to execution delay |
| No gap penalty on stops | How much comes from tail-loss modelling |
| `slippage_vol_coef` × 2 | Volatility-scaled slippage |

The gap between the conservative and optimistic tie rules is the single most
informative number here. A large gap means the stop sits inside typical bar range
and the result is a coin flip on intrabar path.

### 6.4 Universe sensitivity

| Variant | Tests |
|---|---|
| `min_adv_usd` at 2× and 0.5× | Liquidity dependence |
| Top 200 by ADV only | Whether the tail is carrying the result |
| Exclude SPY, QQQ, and the sector ETFs | Whether it works beyond the index products |
| Long-only (shorts disabled) | How much of the result is the expensive short leg |
| Drop a random 30% of symbols, 20 seeds | Asset-selection luck |

The last one is a genuine bootstrap over the universe and it is the honest answer
to "would this have worked if I had picked a different 700 names".

### 6.5 Temporal robustness

| Variant | Tests |
|---|---|
| Block bootstrap of the equity curve, 30-bar blocks | Path dependence of drawdown |
| Start-date jitter, `±90` days in 30-day steps | Start-date luck |
| Rolling 12-month Sharpe | Time-varying edge |

### 6.6 Sentiment lag

`ingest_lag_minutes ∈ {60, 240}`. Required when sentiment is enabled
([`10-SENTIMENT.md §6`](10-SENTIMENT.md#6-the-promotion-gate)).

---

## 7. Required workflow

```text
1. Form a hypothesis. Write it down BEFORE coding, in the PR description or
   in docs/results/. If you cannot state why the edge should persist, stop.

2. Implement. Unit tests first for anything with a formula.

3. Run on DEVELOPMENT only. The registry records the trial.

4. Inspect, in this order — and stop at the first failure:
     a. funnel.csv       — is the pipeline even considering things?
     b. n_trades         — is there enough sample to say anything?
     c. cost_drag_pct    — is this a fee generator?
     d. calibration.png  — does ev_net_r predict realised R at all?
     e. stability tables — is the sign consistent across years and regimes?
     f. THEN look at Sharpe and return.

5. Iterate on DEVELOPMENT. Every run is registered.

6. When a candidate is genuinely finished — not "looks good", finished —
   commit everything, then spend ONE lockbox evaluation.

7. Write the holdout result to docs/results/<milestone>.md, whatever it says.

8. If the holdout is materially worse than development (it will be), the
   holdout is your estimate. Not the average. Not development.
```

Step 4's ordering is deliberate and it is the discipline that makes the rest
work. Looking at Sharpe first is how you end up explaining a number produced by
a broken funnel.

---

## 8. Anti-patterns, and what each one actually does

| Anti-pattern | What it does to you |
|---|---|
| Running the holdout to "check progress" | Converts the holdout into training data, silently and permanently |
| Sweeping a parameter and taking the best | Selects the noise peak; live result is the neighbourhood mean, which is lower |
| Adding a filter because a losing trade looked avoidable | Fits the sample's specific losses; ~always negative out of sample |
| Deleting an asset because it lost money | Survivorship bias, applied by hand |
| Extending the sample after seeing a bad result | Selection on the outcome; the extension is not out of sample |
| Reporting Sharpe without the trial count | Makes a claim you have no evidence for |
| "The regime filter needs tuning" | Almost always means the edge does not exist and the filter is being asked to hide it |
| Adding a strategy because the first one underperformed | Doubles the trial budget spend for one hypothesis |
| Trusting a result with fewer than 100 trades | A 90% CI on mean R at n=50 spans roughly ±0.3 R, which is wider than any plausible edge |

---

## 9. Go / no-go criteria for M3

Stated **now**, before any implementation, so the bar cannot move afterwards.
This is the most important section in this document.

### Proceed to M4 if all of these hold on the holdout

| # | Criterion | Threshold |
|---|---|---|
| 1 | Closed trades | ≥ 150 |
| 2 | Bootstrap 90% CI on `mean_r` | lower bound > 0 |
| 3 | `deflated_sharpe` | > 0.5 |
| 4 | Survives 1.5× cost | `mean_r` still positive |
| 5 | Calibration | monotone increasing across at least 3 of 5 buckets |
| 6 | Direction | long *and* short `mean_r` both positive, or one flat and one clearly positive |
| 7 | Concentration | top 3 symbols ≤ 60% of gross P&L |
| 8 | Beta | `beta_spy < 0.5`, or `t_alpha > 2.0` |
| 9 | Max drawdown | < 25% |
| 10 | Momentum-crash windows | no window named in ADR-019 loses more than 15% |

### If 2, 3, or 4 fail

**Stop. Do not proceed to M4.** The correct next action is to change the
*hypothesis*, not the parameters. Options, in order of expected value:

1. Drop the short leg and re-evaluate long-only on development (one trial). The
   short side is where borrow, hard-to-borrow, and crash convexity live.
2. Try a longer holding period (63 sessions). The cost arithmetic in
   [`08-COSTS.md §3`](08-COSTS.md#3-worked-examples) still favours it.
3. Accept that this hypothesis does not work on this universe and write it up.
   **Do not add crypto to rescue it.**

**Adding sentiment, ML, crypto, or a second timeframe to rescue a failed baseline is the
single worst available action.** It will produce a positive backtest, because
with enough additional degrees of freedom anything will, and that backtest will
be pure noise. The infrastructure you will have built is still worth having; a
fabricated edge is not.

### If 5 through 9 fail but 1 through 4 pass

Proceed to M4 with the specific weakness named in
`docs/results/m3_holdout.md`, and target it directly. A failure of 8 (beta) means
the next work is a market-neutral variant, not more parameter tuning.
