# Lab knowledge base

Durable findings across hypotheses (per-hypothesis lessons are in lessons.md). Append-only: a correction
is a new entry that supersedes the old one. Written by the Librarian, the Chair and the owner;
rendered by the framework.

## Markets and mechanisms

### K-0012 US stock intraday-return reversal is fragile

A weekly long/short on intraday-return reversal in LIQ-500 beat T-bills in dev on the point estimate and against random entries, but not in the S&P 500 universe, with most of the return in the earliest era and a peaked parameter choice, the same pattern as research 5.

*confidence medium · 2026-10-06 · human · evidence: H-0007; research 5*

## Data

### K-0001 Sharadar SFP adjusted-close spikes

closeadj has one-day spikes that reverse the next day (191 cases in 67 funds 1998-2026, e.g. SSO 2014-06-24 x3.95), split adjustments applied to a single day. The SFP loader drops those days (data.drop_spikes, |log move| > 0.4 reversed within 0.1).

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05; lab/framework/data.py drop_spikes*

### K-0002 DefiLlama stablecoin history is recomputed

The DefiLlama stablecoin endpoint recomputes the whole history on every call and adds coins and chains retroactively, so past values are not what was visible at the time. Any signal on it carries look-ahead in levels and growth rates, worst in 2018-2020.

*confidence high · 2026-10-06 · human · evidence: catalog stablecoin_supply_1d; H-0005*

### K-0003 Single stocks fit in RAM only up to LIQ-500

The LIQ-1000 panel has 4 581 stocks; a card on LIQ-500 or the S&P 500 loads ~2 400 columns (float32) and peaks at 1.7-2.4 GB in G2 on this 3.9 GB machine. LIQ-1000 does not fit.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05 (loaders); H-0007 G2 run*

### K-0004 No earnings dates in the lab

Sharadar SF1 filing dates exist on disk but there is no loader, and announcement dates are not in any loadable dataset. Earnings-event ideas can only use proxies (volume seasonality), which H-0008 showed to be weak.

*confidence medium · 2026-10-06 · human · evidence: H-0008; catalog*

### K-0014 Only crypto has forward data

Forward daily bars come from the Binance public API (spot and USD-M perps with funding), reachable from this VPS. ETFs and US stocks have no free forward source yet (Sharadar not renewed), so hypotheses on them are parked after G4 until the Archivist validates a source.

*confidence high · 2026-10-06 · human · evidence: lab/framework/forward.py; decision Q3*

## Methodology

### K-0005 Benchmark-relative binary switches start handicapped

A strategy that moves the whole book between two assets on a state variable carries a different average exposure than the constant-mix benchmark; it must beat the benchmark's risk-adjusted return with less diversification. Test the state variable as a tilt around the benchmark weights, or against a control with the same average exposure.

*confidence high · 2026-10-06 · human · evidence: H-0001; H-0003; research 2, 9*

### K-0006 Short dev windows cannot confirm crypto ideas

Crypto dev data ends 2022-12-31 and starts 2017-2020 depending on the dataset, so weekly strategies have few independent decisions; point estimates can beat BTC while the bootstrap bound stays below zero. Prefer daily decisions or many instruments, and treat a near miss as 'cannot tell', not as evidence.

*confidence high · 2026-10-06 · human · evidence: H-0004; H-0005; H-0006*

### K-0007 Run the mechanism diagnostic before the strategy

Three cards died at G1 before their own mechanism checks could run. An event study or conditional-return diagnostic of the mechanism (with a placebo) is cheaper and more informative than a full strategy when the mechanism itself is uncertain. Diagnostics on dev data are trials too and must be counted.

*confidence medium · 2026-10-06 · human · evidence: lessons H-0006..H-0008*

## The judge (framework behaviour)

### K-0008 Gate runner fixes found by real hypotheses

Real runs exposed judge bugs that the canaries did not: a per-period bootstrap bound reported next to annualized Sharpe (H-0001), two requirements on one dataset overwriting each other (H-0003), generic series filtered by the card's guessed field names (H-0005), spot+perp cards getting the cross-asset benchmark (H-0006), a random-entry null drawn from illiquid stocks (H-0007). Every fix re-ran the canaries; affected verdicts were voided only when the bug changed them.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05*

### K-0009 G0 look-ahead tests are weak on sparse decisions

Truncation and perturbation at random rows rarely hit the decision rows of weekly or monthly strategies, and rank-based weights often do not change under perturbation. G0 now also cuts on rows where the weights change; the Skeptic must still read the code of sparse strategies.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05; lab/tests/test_dryrun.py*

## How the lab works

### K-0010 Fail loudly on missing inputs

A strategy that silently holds nothing when a series or instrument is missing passes G0 and dies at G1 for the wrong reason (H-0005, first run). Builders index inputs directly so a missing one raises.

*confidence high · 2026-10-06 · human · evidence: H-0005*

### K-0011 Scouts converge without a brief

Without steering, three of the first four cards used SPY/IEF and monthly switching. Rotating briefs (least-used horizon x asset group, multi-timeframe while under half of the cards combine timeframes) produced three different markets and horizons in the next three runs.

*confidence medium · 2026-10-06 · human · evidence: H-0001..H-0004; H-0006..H-0008*

### K-0013 Paper trading is model execution on forward data

The lab places no orders. Paper = the strategy run by the lab engine on daily bars that did not exist when it was judged, appended once per day and never recomputed. It is clean out-of-sample evidence for the signal and the cost model's assumptions, not for real fills or slippage; G5 therefore cannot test tracking or realized costs.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-06 (step 4c); lab/framework/paper.py*
