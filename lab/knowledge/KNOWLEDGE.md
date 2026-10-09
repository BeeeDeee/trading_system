# Lab knowledge base

Durable findings across hypotheses (per-hypothesis lessons are in lessons.md). Append-only: a correction
is a new entry that supersedes the old one. Written by the Librarian, the Chair and the owner;
rendered by the framework.

## Markets and mechanisms

### K-0012 US stock intraday-return reversal is fragile

A weekly long/short on intraday-return reversal in LIQ-500 beat T-bills in dev on the point estimate and against random entries, but not in the S&P 500 universe, with most of the return in the earliest era and a peaked parameter choice, the same pattern as research 5.

*confidence medium · 2026-10-06 · human · evidence: H-0007; research 5*

### K-0024 Large-cap short-horizon event anomalies vanish; mandated-flow events survive

Five-session event studies on documented US single-stock anomalies (ex-dividend run-up, stale 52-week-high breakouts, insider purchase filings, correlated-peer catch-up, abnormal volume, predicted earnings dates) found nothing distinguishable from time-shifted placebos among the most liquid stocks, and gross effects were far below spread cost. The only two events with an effect that cleared or nearly cleared the placebo bar were driven by mandated flows rather than information or sentiment: the month-turn beta spread and index inclusion. Together with K-0012 this suggests the large-cap short-horizon cross-section is efficient at this cost level, and the remaining candidates are flow events.

*confidence medium · 2026-10-09 · librarian · evidence: H-0010; H-0011; H-0012; H-0013; H-0014; H-0015; H-0016; H-0017*

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

### K-0020 Spot catalog range was longer than the loadable data

The catalog said Binance spot daily data runs to 2026-10-03 but the research 9 panel the loader reads ends 2026-08-31; the gate runner takes the end of a card's data from the catalog range, so perp-plus-spot cards would have run on a month of missing spot rows. The range now says 2026-08-31.

*confidence high · 2026-10-06 · human · evidence: lab/data/catalog.yaml*

### K-0023 Attached stock datasets fit in RAM with at most six fields

SF1, insider and 13F fields are daily float32 matrices on the SEP stocks, 70 MB each. A LIQ-500 card with six fields peaks at 2.3 GB through G0-G2 on this machine (limit 2.8 GB for the cycle), three fields at 2.1 GB, so the per-card cap is six. Loading them takes seconds to a minute; fundamentals change only on filings, so signals built from them are slow.

*confidence high · 2026-10-07 · human · evidence: calibration quality_stocks; PLAN decision log 2026-10-07*

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

### K-0019 The judge was calibrated against five known strategies

Low-vol stocks, funding carry, 12-1 momentum, 5-day reversal and a BTC trend filter were run through the real gates in a throwaway lab (lab calibrate). Verdicts agree with the owner's earlier studies: nothing passes unexplained, low-vol reaches G2 and dies on publication decay, carry dies on sub-period concentration, the rest die at G1. This shows the judge is not lenient; it cannot show that it passes a genuinely good real strategy because none is known.

*confidence medium · 2026-10-06 · human · evidence: PLAN decision log 2026-10-06; docs/research-lab/calibration*

### K-0021 Mechanism tests before strategies

Cards can declare an event and a horizon; the judge runs an event study against time-shifted placebos on dev rows before simulating the strategy. A strategy whose claimed mechanism is absent dies at G1 with a mechanism reason, which separates 'the idea is wrong' from 'this strategy was too weak'. It counts as one trial of the family. Placebo percentile is the gate, not a t-statistic.

*confidence medium · 2026-10-06 · human · evidence: lab/framework/diagnostic.py; canaries mechanism_**

### K-0025 Read the placebo average, not the raw abnormal return

The placebo keeps the same stocks and shifts the dates, so it absorbs any drift that belongs to the stock selection (names with a clean earnings spike, stocks at 52-week highs). In H-0011 and H-0015 the raw abnormal return was positive and still no better than the placebo average. Cards and lessons should judge an event set against the placebo average and spread, and a positive gross number alone is not evidence of a mechanism. Event-matched controls (same prior return, size or distance to the high) would separate selection from timing.

*confidence medium · 2026-10-09 · librarian · evidence: H-0011; H-0015*

### K-0026 The placebo gate has little power for sparse event sets

With a few hundred events on a few hundred dates the placebo distribution is wide, and a real effect that exceeds costs several times can still land just under the percentile bar (H-0017, which also lost a few placebo runs). The gate is a fixed threshold, not a statistic, so the verdict stands, but a Scout who wants to test a sparse event should widen the event set (more occurrences, both directions) instead of tweaking hold or filters on the same events, which would be counted as new trials of the same data. Reading near misses as a coin flip is more honest than reading them as promising.

*confidence medium · 2026-10-09 · librarian · evidence: H-0017; H-0012; decision K-0021*

### K-0027 A passing mechanism test does not make a tradable stand-alone window book

H-0016 is the first card whose event study cleared the placebo bar and was then simulated as a strategy: the month-turn beta spread exists but the book is invested about a fifth of the days, so its Sharpe is capped by the window's own Sharpe and it missed the absolute bar and the bootstrap bound against T-bills. Calendar windows should be tested as a tilt on an always-invested benchmark with the same average exposure (see K-0005), or combined into one overlay, not as flat-otherwise books against cash. The mechanism gate and the strategy gate answer different questions and should be read separately.

*confidence medium · 2026-10-09 · librarian · evidence: H-0016; H-0001; K-0005*

## The judge (framework behaviour)

### K-0008 Gate runner fixes found by real hypotheses

Real runs exposed judge bugs that the canaries did not: a per-period bootstrap bound reported next to annualized Sharpe (H-0001), two requirements on one dataset overwriting each other (H-0003), generic series filtered by the card's guessed field names (H-0005), spot+perp cards getting the cross-asset benchmark (H-0006), a random-entry null drawn from illiquid stocks (H-0007). Every fix re-ran the canaries; affected verdicts were voided only when the bug changed them.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05*

### K-0009 G0 look-ahead tests are weak on sparse decisions

Truncation and perturbation at random rows rarely hit the decision rows of weekly or monthly strategies, and rank-based weights often do not change under perturbation. G0 now also cuts on rows where the weights change; the Skeptic must still read the code of sparse strategies.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-05; lab/tests/test_dryrun.py*

### K-0017 The random-entry null was survivorship-biased

G2's random-entry null drew random stocks from those listed in more than half of the dev window, a pool chosen with knowledge of who lasted. Its median Sharpe (0.62) beat both the market (0.36) and a real low-volatility book, so no long-only stock strategy could pass. Found by calibrating on real data. The null is now built date by date from the instruments eligible on that date, with the real book's retention so that turnover matches.

*confidence high · 2026-10-06 · human · evidence: docs/research-lab/calibration/lowvol_stocks.json; lab/framework/nulls.py*

### K-0018 Position-entry minimum rejected always-invested books

min_position_entries counted only flat-to-position entries: a delta-neutral funding carry has 4 entries but 156 weekly rebalances and was rejected at G1 despite a Sharpe of 8.4. For books invested at least half the time the rebalancing dates now count as bets (gates.yaml continuous_min_exposure).

*confidence high · 2026-10-06 · human · evidence: docs/research-lab/calibration/funding_carry.json*

### K-0028 G0 checks are vacuous when the synthetic market lacks the card's data, and card diagnostics cannot be reported

Three of eight cards in this batch had a G0 look-ahead check that exercised nothing on the synthetic market: no dividends (H-0010), no insider events (H-0012), and only 80 names so the book never traded (H-0013). The Builders substituted hand-made tests, which no gate reads. Separately, two cards promised sanity diagnostics (regular-payer share in H-0010, volume hit rate of the predicted earnings days in H-0015) that the Builders could not output because no channel to the gate result exists, so a failed proxy cannot be told from an absent effect. Extend the synthetic market or the diagnostic output, and the Skeptic should read the code of data-specific cards.

*confidence high · 2026-10-09 · librarian · evidence: H-0010; H-0012; H-0013; H-0015*

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

### K-0015 systemd hardening can silently break the agent user switch

ProtectKernelTunables, ProtectKernelModules, LockPersonality (and other options) imply NoNewPrivileges, which makes sudo refuse to switch to labagent. The first production cycles failed this way; the unit now omits them, and three agent runs failing before any model turn pause the lab with a critical alert instead of retrying every 30 minutes.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-06 (first production cycles); deploy/research-lab/research-lab-cycle.service*

### K-0016 A deny rule meant for the lab state blinded the agents

The agents' settings denied Read(//srv/**) to protect the lab state; in production the workspaces live under /srv/research-lab/workspaces, so every file tool was denied and two Scout runs produced nothing. Deny rules must name the protected directories exactly; a test now checks that none covers a workspace. The Scouts reported the problem as owner questions instead of improvising, which is the intended behaviour.

*confidence high · 2026-10-06 · human · evidence: PLAN decision log 2026-10-06; messages 70, 71*

### K-0022 First production mechanism tests killed two ideas without a strategy run

Two of the first cards with a mechanism test (ex-dividend price pressure, breakouts through stale 52-week highs, both LIQ-500 market-neutral, 5-day horizon) died at G1 on the event study: placebo percentiles 0.64 and 0.46, gross abnormal return per event covering only 4 % and 28 % of the round-trip cost. The strategy was never simulated, so each death cost a Scout run and a Builder run but no gate time, and the reason is mechanism-level, not 'the strategy lost to the benchmark'.

*confidence high · 2026-10-07 · human · evidence: H-0010; H-0011; gate_results G1 mechanism*
