# M3.6 Development run

Inspected in protocol order
([`12-RESEARCH_PROTOCOL.md` §7](../12-RESEARCH_PROTOCOL.md#7-required-workflow)).
**Do not tune. Do not spend a lockbox evaluation.** Trial 1 is an accounting
invalidation (appendix). Trial 2 is the measurement.

| | Trial 2 (valid) | Trial 1 (invalid) |
|---|---|---|
| `run_id` | `20260907-154935-xsec-momentum-donchian` | `20260906-203830-xsec-momentum-donchian` |
| Config | `config/development.yaml` (1998-01-01 → 2017-12-31, warmup_end 2004-01-01) | same |
| `config_hash` | `d36dea1f0fd93b5e0f0341de3f93affd3c54a500dbe799a06d6a8cda3fb40240` | same |
| Snapshot | `20260830-sharadar` | same |
| Strategies | `xsec_momentum_v1`, `donchian_breakout_v1` | same |
| `--notes` | M3.6 after close_raw qty and cash_raw dividends | M3.6 development baseline |
| `git_sha` | `ba62e113…--dirty` | `ba62e113…--dirty` |
| Registry | trial 2 | trial 1 |
| Results | `results/20260907-154935-xsec-momentum-donchian/` | `results/20260906-203830-xsec-momentum-donchian/` |

Trial 2 artifacts: `funnel.csv`, `trades.csv`, `equity.csv`, `metrics.json`,
`decisions.parquet`, `bins.csv`, `config.yaml`, `config_hash.txt`, `run.log`,
and all six plots including `plots/calibration.png`.

Panel rebuilt after ADR-017 qty/`cash_amount` units; labels and edge table
rebuilt from that panel. Same `config_hash` as trial 1.

---

## 1. Funnel

`funnel.csv` counts every snapshot name every session (`n_decisions` =
153 688 689). That is a coverage counter, not “names we could have traded
today.” `INSUFFICIENT_HISTORY` (55%) and `STALE_DATA` (21%) are mostly names
that were never eligible at `t`. They are not a 99.9% gate on the live
universe.

| stage | rejection_reason | n | % of funnel |
|---|---|---:|---:|
| REJECT | INSUFFICIENT_HISTORY | 84 739 014 | 55.14 |
| REJECT | STALE_DATA | 32 310 834 | 21.02 |
| REJECT | LOW_LIQUIDITY | 13 660 590 | 8.89 |
| REJECT | NOT_IN_UNIVERSE | 8 178 842 | 5.32 |
| REJECT | LOW_PRICE | 6 337 192 | 4.12 |
| REJECT | DATA_SUSPECT | 3 418 238 | 2.22 |
| REJECT | EARNINGS_IN_WINDOW | 2 377 482 | 1.55 |
| REJECT | NO_SETUP | 1 064 290 | 0.69 |
| REJECT | REGIME_BLOCKED | 649 802 | 0.42 |
| REJECT | WIDE_SPREAD | 396 266 | 0.26 |
| REJECT | MARKET_REGIME_BLOCKED | 296 552 | 0.19 |
| REJECT | BELOW_EV_THRESHOLD | 139 762 | 0.091 |
| REJECT | MAX_POSITIONS | 67 669 | 0.044 |
| REJECT | BELOW_TOP_N | 43 886 | 0.029 |
| REJECT | ALREADY_IN_POSITION | 4 387 | 0.003 |
| REJECT | BETA_CAP | 2 603 | 0.002 |
| ACCEPTED | | 694 | 0.00045 |
| REJECT | CLUSTER_CAP | 585 | <0.001 |
| REJECT | DATA_GAP | 1 | <0.001 |

The early-gate counts match trial 1. The book now fills: `MAX_POSITIONS`
(67 669 vs 12 253) and `ALREADY_IN_POSITION` (4 387 vs 2 824) rose because
trades continue through 2017 instead of dying in 2011. `BELOW_EV_THRESHOLD`
fell (139 762 vs 190 949) once evaluate-time `cost_r` stopped seeing
billion-dollar `risk_capital`.

No single gate rejects 99.9% of eligible candidates.

---

## 2. Trade count

`n_trades` = **687** closed (694 accepted; 7 still open at 2017-12-31;
1 `DATA_GAP`). Above the “fewer than 100, say nothing” line.

| | n |
|---|---:|
| `donchian_breakout_v1` | 431 |
| `xsec_momentum_v1` | 256 |
| LONG | 681 |
| SHORT | 6 |
| TIME / STOP / TARGET | 365 / 248 / 74 |
| Unique symbols | 431 |
| First entry | 2004-01-05 |
| Last entry | 2017-11-17 |
| qty max / p99 | 665 / 380 shares |

Trades in every development year 2004–2017 (13 in 2008; 42–64 otherwise).
The 2012–2017 silence from trial 1 is gone.

CHKAQ appears twice at 67 and 251 shares, `realised_r` 0.27 and 0.99. No
`|R| > 5`. The $315 B path is closed.

Shorts: six Donchian names, all in May–August 2004. Four stopped at about
−1 R, two won. Short `mean_r` is −0.20 on n=6. That is “the short side
barely exists,” not a missing detector (the parquet still has short
candidates; they do not clear EV / caps / size).

`exposure_pct` = 91%. “No trade is normal” is about new accepts vs the
candidate set, not about sitting in cash. With ~4 trades/month and
`avg_bars_held` ≈ 23, the book is usually full.

---

## 3. Cost drag

`cost_drag_pct` = **10.4%** (fees $584 / gross price P&L $5 672). Not a fee
generator.

Net P&L $11 274. Dividends $6 192 (55% of net). That is cash on a long-only
23-bar book, not restated Sharadar figures. The largest dividend is WCRX
2010-09-09: vendor `cash_amount` = $8.50, no later SPLIT, qty 86 → $731.
The stock gapped through the stop on the ex-date (29.29 → 20.70);
`realised_r` = −0.04. Price drop and cash cancel. That is the conservative
intrabar/gap rule doing its job, not another CHKAQ.

`total_funding_usd` = 0 (almost no shorts filled).

---

## 4. Calibration

`plots/calibration.png` is the actual plot. Five predicted-EV buckets,
low → high:

| bucket | predicted mid | mean realised R | 90% CI | n |
|---|---:|---:|---:|---:|
| 1 | 0.067 | 0.062 | −0.020, 0.145 | 138 |
| 2 | 0.086 | 0.134 | 0.010, 0.270 | 137 |
| 3 | 0.104 | 0.090 | −0.068, 0.262 | 137 |
| 4 | 0.126 | 0.026 | −0.142, 0.198 | 137 |
| 5 | 0.177 | 0.010 | −0.155, 0.186 | 138 |

Not monotone. The top two predicted buckets are the worst realised.
Protocol: a flat (here, inverted) line means the ranking carries no
information. **That is the answer, not a tuning problem.**

The predicted range across five buckets is only ~0.11 R. Trade `std_r` is
1.03 R. qcut is slicing noise. Ten buckets show the same: bucket 5 is the
only clearly positive cell (0.29 R); 6 and 9 are negative. Do not retune
`min_ev_net_r` or the bins to make this plot slope up.

---

## 5. Stability

### Per-year `mean_r`

| year | n | mean_r | sum_r |
|---|---:|---:|---:|
| 2004 | 61 | 0.042 | 2.54 |
| 2005 | 54 | 0.116 | 6.28 |
| 2006 | 52 | 0.122 | 6.33 |
| 2007 | 54 | 0.086 | 4.62 |
| 2008 | 13 | −0.020 | −0.26 |
| 2009 | 35 | −0.123 | −4.30 |
| 2010 | 48 | 0.353 | 16.95 |
| 2011 | 59 | −0.014 | −0.84 |
| 2012 | 64 | 0.040 | 2.57 |
| 2013 | 46 | 0.288 | 13.24 |
| 2014 | 56 | −0.097 | −5.45 |
| 2015 | 54 | −0.293 | −15.84 |
| 2016 | 49 | 0.014 | 0.70 |
| 2017 | 42 | 0.422 | 17.73 |

Sign is **not** consistent. 2010 and 2017 carry most of the R. 2015 is the
hole: Donchian 19 trades at −0.70 R, momentum 35 at −0.07 R, 22 stops / 1
target. Worst single trade is LMT Donchian 2015-08-17, −2.77 R (gap through
the stop). Dispersion is a thin long-only trend book in 2015, not one
journal entry.

Max drawdown **7.85%**, trough 2016-01-15, peak 2013-06-10. Duration 1 662
days. Equity does not retake the 2013 peak inside the sample
($114 518 → $105 523 → $111 974 at 2017-12-29).

### Per-regime

| regime | n | mean_r |
|---|---:|---:|
| TREND_UP | 457 | 0.088 |
| CHOP | 136 | −0.067 |
| RANGE | 84 | 0.169 |
| TREND_DOWN | 10 | −0.110 |

CHOP is negative. RANGE is the best cell and a small n. TREND_DOWN is n=10.

### Direction / strategy / concentration

| group | n | mean_r |
|---|---:|---:|
| LONG | 681 | 0.067 |
| SHORT | 6 | −0.199 |
| donchian_breakout_v1 | 431 | 0.090 |
| xsec_momentum_v1 | 256 | 0.022 |

`top_n_pnl_share` (top 3 symbols / total `sum_r`) = **41%**. Under the 60%
gross-P&L concentration bar on this metric. STRZA (6 trades, 8.09 R) is the
largest name.

### SPY beta

`beta` = 0.067, `alpha_annualised` ≈ 0.18%, `t_alpha` = 0.26. Not a
leveraged index. Also not significant alpha. Starting equity $99 998,
ending $111 974 (+12.0%).

### Momentum-crash windows (ADR-019)

| window | in sample? | return | fails 15%? |
|---|---|---:|---|
| 1932-07, 1932-08, 1939-09 | no | NaN | no |
| 2001-01 | warmup, no trades | NaN | no |
| 2009-04 | yes | **0.0** | no |
| 2021-01 | holdout | NaN | no |

April 2009: 21 sessions, one unique equity value, **zero** overlapping
trades. The strategy did not lose 15% in a crash it was not in. That is
not evidence it survives a crash.

---

## 6. Sharpe — looked at last

| | |
|---|---|
| `mean_r` | 0.0645 R |
| bootstrap 90% CI | 0.00065, 0.129 |
| `p_value_mean_r` | 0.0485 |
| Sharpe | 0.29 (CI −0.16, 0.74) |
| deflated Sharpe | 0.72 (`n_trials` = 2) |
| CAGR | 0.81% |
| total return | 12.0% |
| max DD | 7.85% |
| win rate | 49.3% |

The CI on `mean_r` just clears zero. Sharpe’s CI includes zero. Fourteen
years to make twelve percent, with an inverted calibration plot and a
2013–2016 underwater stretch that never fully recovers. This is a small,
fragile, long-only remainder after costs — not a candidate to “check on
the holdout.”

---

## Bugs filed (do not tune around them)

Trial 1 items 1–4 (qty vs `close_raw`, restated dividend cash, evaluate-time
`cost_r` after a fake credit, registry `max_dd_pct` key) are **fixed** in
this tree. Trial 2 is the check.

No new unit bug turned up in trial 2. The inverted calibration, the missing
short side, and 2015 are sample findings, not code defects.

---

## What this is not

It is not a holdout. It is not a parameter problem. Calibration is not
monotone; protocol says that is the answer. Do not retune `min_ev_net_r`,
bins, or the earnings gate to chase a slope.

Do not start M3.7 until the git tree is clean, historical borrow/dividend
inputs are ingested, and `edge.lcb_method` is `bootstrap` — and only then
if you still want the lockbox spent on this hypothesis. The development
picture does not argue for rushing that.

---

## Appendix — Trial 1 (invalid)

`run_id` `20260906-203830-xsec-momentum-donchian`. **Not a candidate.**
Headline metrics were one journal entry, not an edge.

CHKAQ (Chesapeake, Sharadar `asset_id` 197696): sizing used adjusted
`reference_price` ≈ 9e-8 so qty ≈ 1.8e10 shares; Sharadar `cash_amount`
$17.50 was a 2011 quarterly restated onto post-2020 reverse-split shares
(`SPLIT` 2020-04-15, `split_ratio` 0.005). Ledger: `qty × 17.50` =
$3.153e11 on 2011-06-29. After that, `risk_capital` exploded evaluate-time
`cost_r` and 2012–2017 accepted 0 trades.

Spec: ADR-017 wins. Share qty and dollar gates use `close_raw`. Vendor
dividend cash is converted to contemporaneous $ per then-share (later SPLIT
ratios only) before `_cash_factor`. Trial 2 is that spec on the same
config.
