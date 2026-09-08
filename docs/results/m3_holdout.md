# M3.7 Holdout — lockbox evaluation 1 of 3

Inspected in protocol order
([`12-RESEARCH_PROTOCOL.md` §7](../12-RESEARCH_PROTOCOL.md#7-required-workflow),
then §9 go/no-go). **Do not tune. Do not spend another lockbox evaluation on
this hypothesis.**

| | |
|---|---|
| `run_id` | `20260908-085154-xsec-momentum-donchian` |
| Config | `config/holdout.yaml` (warmup_end 2018-01-01, end 2026-08-01, `edge.lcb_method: bootstrap`) |
| `config_hash` | `99ab80c62636b286788be0117c42ad18ce38234b9e20ecad4e7684f08700b1c8` |
| Snapshot | `20260830-sharadar` |
| Strategies | `xsec_momentum_v1`, `donchian_breakout_v1` |
| `--notes` | M3.7 lockbox evaluation 1 of 3 |
| Lockbox `git_sha` | `841b834e…` (clean at spend) |
| Registry `git_sha` | `841b834e…--dirty` (lockbox JSON is written before `write_run_outputs`) |
| Registry | trial 3 |
| Results | `results/20260908-085154-xsec-momentum-donchian/` |
| Decision | **STOP** |

Criterion 2 fails: bootstrap 90% CI on `mean_r` is (−0.0015, 0.170). The
lower bound is not greater than zero. Protocol: if 2, 3, or 4 fail, do not
start M4. Change the hypothesis.

Artifacts: `funnel.csv`, `trades.csv`, `equity.csv`, `metrics.json`,
`decisions.parquet`, `bins.csv`, `config.yaml`, `config_hash.txt`, `run.log`,
and all six plots including `plots/calibration.png`.

This is lockbox evaluation **1 of 3**. `experiments/holdout_lockbox.json`
`used` is 1. Criterion 4 (1.5× cost) is a post-hoc on these trades, not a
second holdout.

---

## Go / no-go (protocol §9, all ten rows)

Roadmap M3.7 says “nine” criteria; §9 has ten numbered rows. All ten are
evaluated.

| # | Criterion | Result | Pass? |
|---|---|---|---|
| 1 | Closed trades ≥ 150 | 411 closed (418 accepted; 7 open at 2026-07-31) | **PASS** |
| 2 | Bootstrap 90% CI on `mean_r`, lower bound > 0 | (−0.00147, 0.170) | **FAIL** |
| 3 | `deflated_sharpe` > 0.5 | 0.761 (`n_trials` = 2) | **PASS** |
| 4 | Survives 1.5× cost (`mean_r` still positive) | post-hoc `mean_r` 0.0828 | **PASS** |
| 5 | Calibration monotone increasing across ≥ 3 of 5 buckets | 1 of 4 adjacent steps up; top bucket worst | **FAIL** |
| 6 | Long *and* short `mean_r` both positive, or one flat and one clearly positive | LONG 0.084 on n=411; SHORT n=0 | **FAIL** |
| 7 | Top 3 symbols ≤ 60% of gross P&L | `top_n_pnl_share` = 35.4% of `sum_r` | **PASS** |
| 8 | `beta_spy < 0.5` or `t_alpha > 2.0` | β = 0.064 | **PASS** |
| 9 | Max drawdown < 25% | 4.85% | **PASS** |
| 10 | Momentum-crash windows: no named window loses more than 15% | 2021-01 = +1.52%; others empty | **PASS** |

**STOP.** Criterion 2 is the hard gate. The CI on development trial 2 just
cleared zero (0.00065). The holdout CI just misses it. That is not a near-miss
to round in our favour.

---

## 1. Funnel

`funnel.csv` counts every snapshot name every session (`n_decisions` =
94 027 473). Same coverage counter as development, not “names we could have
traded today.”

| stage | rejection_reason | n | % of funnel |
|---|---|---:|---:|
| REJECT | STALE_DATA | 37 123 860 | 39.48 |
| REJECT | INSUFFICIENT_HISTORY | 29 647 912 | 31.53 |
| REJECT | LOW_LIQUIDITY | 11 104 878 | 11.81 |
| REJECT | NOT_IN_UNIVERSE | 8 177 708 | 8.70 |
| REJECT | LOW_PRICE | 4 076 550 | 4.34 |
| REJECT | EARNINGS_IN_WINDOW | 1 405 463 | 1.49 |
| REJECT | DATA_SUSPECT | 841 254 | 0.89 |
| REJECT | NO_SETUP | 684 003 | 0.73 |
| REJECT | REGIME_BLOCKED | 427 110 | 0.45 |
| REJECT | MARKET_REGIME_BLOCKED | 233 720 | 0.25 |
| REJECT | WIDE_SPREAD | 139 492 | 0.15 |
| REJECT | BELOW_EV_THRESHOLD | 102 801 | 0.11 |
| REJECT | BELOW_TOP_N | 30 713 | 0.033 |
| REJECT | MAX_POSITIONS | 26 672 | 0.028 |
| REJECT | ALREADY_IN_POSITION | 2 345 | 0.002 |
| REJECT | BETA_CAP | 1 778 | 0.002 |
| REJECT | CLUSTER_CAP | 791 | <0.001 |
| ACCEPTED | | 418 | 0.00044 |
| REJECT | SIZE_BELOW_MIN_NOTIONAL | 4 | <0.001 |
| REJECT | DATA_GAP | 1 | <0.001 |

Holdout `STALE_DATA` leads `INSUFFICIENT_HISTORY` (development was the other
way around) because the eligible set is older names plus 2018–2026 listings;
delisted-but-still-in-the-candidate-file rows dominate the coverage counter.
No single gate rejects 99.9% of eligible candidates.

---

## 2. Trade count

`n_trades` = **411** closed. Above 150.

| | n |
|---|---:|
| `donchian_breakout_v1` | 255 |
| `xsec_momentum_v1` | 156 |
| LONG | 411 |
| SHORT | 0 |
| TIME / STOP / TARGET | 217 / 144 / 50 |
| Unique symbols | 294 |
| First entry | 2018-01-03 |
| Last entry | 2026-06-18 |
| qty max / p99 | 1 105 / 545 shares |

Zero shorts. Development had six, all in 2004, `mean_r` −0.20. The holdout
did not grow a short book; it deleted the remainder. Criterion 6 is not “the
short side is flat.” It is unmeasured.

`exposure_pct` = 92%. `avg_bars_held` ≈ 22. Same full-book, few-accepts
shape as development.

No `|R| > 5`. Worst: WFC Donchian 2018-01-23, −1.85 R (gap through the stop).
Best: MRNA momentum 2020-11-11, +3.89 R (TIME).

---

## 3. Cost drag

`cost_drag_pct` = **4.75%** (fees $360 / gross price P&L $7 574). Not a fee
generator.

Net P&L $10 779. Dividends $3 566. Borrow $0, funding $0 — there were no
shorts to borrow.

### Criterion 4 (1.5× cost), post-hoc

Not a second lockbox run. Extra cost on each closed trade is
`0.5 × (fees + borrow + funding)`. Converted through that trade’s
`risk_at_entry = net_pnl / realised_r`:

| | |
|---|---|
| `mean_r` at 1.0× | 0.0843 |
| post-hoc `mean_r` at 1.5× | 0.0828 |

Still positive. **PASS.** Do not spend evaluation 2 to re-simulate this.

Trades were already filtered on `ev_net_r >= 0.05` at 1.0× cost. A drop of
trades that would fail `min_ev_net_r` at 1.5× would need `cost_r` on the
trade row; it is not stored. The R-adjustment above is the quantity the
criterion names (`mean_r` still positive).

---

## 4. Calibration

`plots/calibration.png` is the actual plot. Five predicted-EV buckets,
low → high:

| bucket | predicted mid | mean realised R | 90% CI | n |
|---|---:|---:|---:|---:|
| 1 | 0.057 | 0.249 | 0.133, 0.371 | 83 |
| 2 | 0.072 | 0.029 | −0.115, 0.168 | 82 |
| 3 | 0.096 | 0.156 | −0.075, 0.373 | 82 |
| 4 | 0.120 | 0.119 | −0.105, 0.336 | 82 |
| 5 | 0.148 | −0.133 | −0.342, 0.078 | 82 |

Not monotone. The lowest predicted bucket is the best realised cell. The
highest predicted bucket is the only clearly negative cell. One of four
adjacent steps increases. Protocol: a flat (here, inverted) line means the
ranking carries no information. **Same answer as development.**

The predicted range across five buckets is ~0.09 R. Trade `std_r` is
1.06 R. qcut is slicing noise. Ten buckets: 8 is the only later cell that is
clearly positive (0.34 R); 7, 9, and 10 are negative or mixed. Do not retune
`min_ev_net_r` or the bins to make this plot slope up.

---

## 5. Stability

### Per-year `mean_r`

| year | n | mean_r | sum_r |
|---|---:|---:|---:|
| 2018 | 44 | −0.037 | −1.61 |
| 2019 | 52 | 0.107 | 5.56 |
| 2020 | 50 | 0.139 | 6.96 |
| 2021 | 52 | −0.003 | −0.18 |
| 2022 | 24 | 0.299 | 7.17 |
| 2023 | 58 | 0.016 | 0.94 |
| 2024 | 58 | 0.033 | 1.93 |
| 2025 | 45 | 0.181 | 8.16 |
| 2026 | 28 | 0.205 | 5.73 |

Sign is **not** consistent. 2020, 2022, 2025, and 2026 carry most of the R.
2018 is the hole. 2022 has the highest `mean_r` on the smallest n (24). That
is a thin book in a bear, not evidence of crash convexity.

Max drawdown **4.85%**, trough 2025-03-10, peak 2024-03-21. Duration 571
days. Ending equity $111 602 vs peak $111 029 — the 2024 peak is recovered
by 2026-07-31.

### Per-regime

| regime | n | mean_r |
|---|---:|---:|
| TREND_UP | 272 | 0.058 |
| CHOP | 78 | 0.143 |
| RANGE | 59 | 0.128 |
| TREND_DOWN | 2 | 0.058 |

TREND_DOWN is n=2. CHOP is the best cell here and was the worst in
development. That is not stability.

### Direction / strategy / concentration

| group | n | mean_r |
|---|---:|---:|
| LONG | 411 | 0.084 |
| SHORT | 0 | — |
| donchian_breakout_v1 | 255 | 0.044 |
| xsec_momentum_v1 | 156 | 0.151 |

`top_n_pnl_share` (top 3 symbols / total `sum_r`) = **35.4%**. Under 60%.
MRNA, COST, AGQ (two trades each).

### SPY beta

`beta` = 0.064, `alpha_annualised` ≈ 0.35%, `t_alpha` = 0.35. Not a
leveraged index. Also not significant alpha. Starting equity $99 999,
ending $111 602 (+11.6%).

### Momentum-crash windows (ADR-019)

| window | in sample? | return | fails 15%? |
|---|---|---:|---|
| 1932-07, 1932-08, 1939-09 | no | NaN | no |
| 2001-01, 2009-04 | development | NaN | no |
| 2021-01 | yes | **+1.52%** | no |

January 2021: the book did not lose 15%. It also did not have a short side
to be long the crash. Criterion 10 passes on the number the table names.

---

## 6. Sharpe — looked at last

| | |
|---|---|
| `mean_r` | 0.0843 R |
| bootstrap 90% CI | **−0.00147**, 0.170 |
| `p_value_mean_r` | 0.054 |
| Sharpe | 0.42 (CI −0.12, 1.03) |
| deflated Sharpe | 0.76 (`n_trials` = 2) |
| CAGR | 1.29% |
| total return | 11.6% |
| max DD | 4.85% |
| win rate | 46.7% |

Eight and a half years to make twelve percent, with an inverted calibration
plot and a CI on `mean_r` that includes zero. Development said this was a
small, fragile, long-only remainder. The holdout is the same remainder,
out of sample, and it does not clear the CI bar that was written down before
the run.

---

## Decision

**STOP. Do not proceed to M4.**

Criterion 2 failed. Protocol next actions, in order:

1. Drop the short leg and re-evaluate long-only on development (one trial).
   The short leg is already gone in the fills; this is “admit the book is
   long-only” rather than a new detector.
2. Try a longer holding period (63 sessions).
3. Accept that this hypothesis does not work on this universe and write it up.
   **Do not add crypto to rescue it.**

**Do not add sentiment, ML, crypto, or a second timeframe to rescue this
baseline.** That would spend lockbox evaluations 2 and 3 on a ranking
statistic the calibration plot already said is noise.

Lockbox remaining: **2 of 3**. They are not a budget for retrying this
config.

---

## What this is not

It is not a parameter problem. The inverted calibration is the same finding
as M3.6. Do not retune `min_ev_net_r`, bins, or the earnings gate.

It is not a cost-model problem. 1.5× still leaves `mean_r` positive; the CI
failure is about the mean, not the fee line.

It is not permission to peek at the holdout again “to check a bug.” Evaluation
1 is spent. A code bug that invalidated the run would be filed here; none
turned up. `SIZE_BELOW_MIN_NOTIONAL` (4) and zero shorts are sample findings.
The registry `--dirty` suffix is the lockbox write ordering, not a dirty
tree at spend time.
