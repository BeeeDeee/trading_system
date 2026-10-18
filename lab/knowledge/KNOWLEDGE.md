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

### K-0038 The month-turn high-beta advantage passes the placebo test but fades in the later years

Two designs on the same month-turn window in liquid US stocks (a high-minus-low beta spread, and a long-only rotation between a high-beta and a low-beta quintile) both cleared the mechanism test against time-shifted placebos, and in both the effect was strongest in the earliest third of the sample and weakest in the latest, in the rotation reversing sign. The two cards share events and data, so this is one piece of evidence seen twice, not two confirmations. It fits a published flow effect that is being arbitraged away, or a beta payoff that depends on the bull-market era. The rotation also gave no drawdown protection relative to the market. A month-turn beta claim should be expected to fail sub-period and publication-decay checks unless a new design shows the latest years separately.

*confidence medium · 2026-10-12 · librarian · evidence: H-0016; H-0025; K-0027; K-0030*

### K-0039 Large-cap earnings-date events fail the placebo test: premium, surprise, delay

Real release dates now support three claims in liquid US stocks over a five-session window: a long premium around the predicted date (H-0023), surprise drift at concurrent filings (H-0024) and a short on late releases (H-0026). None cleared the placebo bar. Surprise drift is a near miss whose bar lies above cost, so it is a 'cannot tell'; the delay signal has a bar below cost and reversed sign in the latest third, so it is not tradable in this form. Large-cap releases are pre-announced and heavily covered, so timing and surprise are likely priced quickly. With K-0024, further date-based earnings cards in this universe have a low prior unless they change the horizon to the release session, the stocks to less-covered names, or the data to announced dates.

*confidence medium · 2026-10-12 · librarian · supersedes K-0037 · evidence: H-0023; H-0024; H-0026; K-0024; K-0034*

### K-0041 The trading-window net-flow spread does not survive a move from the filing date to the real release date

The net-repurchaser-minus-stock-compensation-supplier spread in liquid US stocks passed the event-study gate clearly when anchored to the quarterly filing (H-0019) but fell short of the placebo bar when anchored to the real earnings release (H-0027), where it also built slowly with the horizon, had nothing in the first sessions and reversed sign in the latest third. A flow concentrated just after the trading window opens would show up at the release anchor and early horizons; it did not. The filing-anchored result is better read as filing-linked news or earnings-season drift, or as a slow style difference between repurchasers and issuers that the placebo (same stocks) cannot net out. Treat the trading-window flow family as closed for five-session books in LIQ-500; any remaining claim is a slow tilt, to be argued as a style premium.

*confidence medium · 2026-10-13 · librarian · evidence: H-0019; H-0027; K-0025; K-0030*

### K-0044 Weekly return-sort effects in the most liquid US stocks are smaller than cost, in either sign

Past-month-return effects in liquid US stocks have now been tested with a reversal sign (H-0007, K-0012), with a conditional continuation sign (H-0029, highest-turnover stocks) and with a reversal sign after removing earnings sessions and adjusting for correlated peers in the S&P 500 (H-0030). None gave a tradable five-session effect: H-0029 was inside the placebo cloud with a bar below cost and a negative latest third, H-0030 cleared the placebo bar but earned well under round-trip cost, and H-0007 was early-era and small-name. Conditioning the sort (news, turnover, intraday split, peers) changes the sign or the significance, not the order of magnitude relative to cost. A further conditioner on the same one-month sort in the large-cap universe has a low prior; a cost-light implementation or a less-covered universe would be a different claim.

*confidence medium · 2026-10-13 · librarian · evidence: H-0007; H-0029; H-0030; K-0012; K-0024*

### K-0046 The release-session reaction drift reproduces on the same events in three book designs, and the long side alone shows it

Three cards built on the same real-release-date reaction events (a long/short book, a fixed-slot long/short book, and a long-only tilt on positive reactions) each cleared the placebo bar with gross above cost and a positive sign in every third of the sample. They share events, so this is one piece of evidence seen three times, not three confirmations. What is new: the positive side alone carries drift, so the effect is not only a short-side artifact, and within the positive-only set it weakens from the earliest to the latest third while staying positive. The effect is real in an equal-weight event study but small next to cost: the market-neutral books lost to T-bills, and the long-only tilt on an always-invested core came nearest to a pass, beating the market on the point estimate with a reliable bound but not by the required margin.

*confidence low · 2026-10-14 · librarian · supersedes K-0043 · evidence: H-0028; H-0031; H-0033; K-0043*

### K-0049 Crypto forced-flow rebounds exist before stress but a taker cannot keep them

The conditional rebound after fast, high-volume, taker-dominated moves in liquid crypto perps (H-0035, both directions) left a positive average trade after modeled taker costs and beat random-timed trades, so a liquidity-provision effect appears to exist. It did not survive costs at twice the model, and its profit sat in the first two thirds of the 2020-2022 dev period while the last third lost. Together with the hourly reversal finding of study mrel-1h (harvestable only as a maker) and the daily dip-buy H-0006 (could not tell), the pattern is that crypto reversal is paid to whoever provides liquidity at the stress moment and is small or negative for a taker who enters a candle later with optimistic slippage. The effect looked faster than the hold: the result halved with a one-candle later fill. Whether it is absent in sustained crashes is untested.

*confidence low · 2026-10-14 · librarian · evidence: H-0035; H-0006; prior_studies mrel-1h*

### K-0056 Stress rebounds: positive on average, near generic drift, weakest lately

A rebound after forced selling appears as a positive pooled average in three settings: daily crypto dip-buying (H-0006, could not tell), fifteen-minute crypto cascades (H-0035) and stocks against Treasuries after sessions where stocks fell and long Treasuries rose (H-0041). In each the effect is near, not clearly beyond, what time-shifted placebo windows give, and the latest third of dev is the weakest or negative, possibly because stress episodes ran on instead of reversing (not verified by an era or episode split). For the ETF case random windows already favour equities over Treasuries, so the raw gap overstates the event. Studies 6 and 7 found the same for VIX-conditioned reversal: a market-level bounce, not a selection effect, and no gain over holding the market. New stress-rebound cards in other assets are largely the same evidence again, not independent confirmations. A Scout should state how the design separates a one-off shock from a persistent crash.

*confidence low · 2026-10-15 · librarian · evidence: H-0041; H-0035; H-0006; K-0049; K-0052; prior_studies 6-7*

### K-0058 BTC-led altcoin catch-up is parameter-fragile in the lab

The rule 'long lagging altcoins after a BTC hourly jump beyond k sigma, hold four hours' looks strong on the dev period at the primary setting but its edge needs a high threshold: lower thresholds lose most of it and the result is dominated by one sub-period. It failed G2 on neighbor stability and block concentration, while survival under delay and doubled costs was fine. A narrowed grid after the fact would be a new trial, not a rescue.

*confidence medium · 2026-10-15 · human · evidence: H-0040 G2 run 84*

### K-0060 Short-horizon effects that clear the pooled placebo test are usually weakest or reversed in the latest third of dev

Across the lab's event studies that cleared or came near the placebo bar, the effect is typically largest in the earliest third of the dev period and smallest in the latest, in several cases with the opposite sign: month-turn beta, index inclusion, trading-window flow, short-horizon reversal, weekday mood in speculative stocks, twin-ETF gaps and perp funding crowding. Only earnings-reaction drift stayed positive in every third, and it also weakened. New: the pattern also holds for a full strategy that passed its absolute and bootstrap bars, the BTC-led altcoin catch-up (H-0040), where the middle third carried most of the return and the latest third lost, so G2 killed it on concentration. The three crypto perp cards (H-0035, H-0039, H-0040) share one short dev window, a single boom-and-bust cycle, so their thirds are regimes rather than independent samples, and the latest third was the weakest in each. A pooled result therefore overstates what a card can expect going forward. A Scout should expect decay by default and state the latest-third sign as a kill condition. Three equal blocks of one history cannot separate publication decay from a change of regime or cost level.

*confidence medium · 2026-10-15 · librarian · supersedes K-0052 · evidence: H-0016; H-0017; H-0025; H-0027; H-0029; H-0032; H-0035; H-0036; H-0037; H-0039; H-0040; K-0038; K-0046*

### K-0063 Fed-calendar equity premia against bonds: positive on average, near the placebo bar, not tradable, weakest lately

Two designs on the Fed's pre-announced calendar, the scheduled announcement sessions (H-0042) and the even-week cycle of FOMC time (H-0045), both gave a positive equity-over-bond or high-beta-over-low-beta average that was inside or just under the time-shifted placebo cloud. The two live at different horizons: the announcement design shows its gain on the announcement session only, while the cycle design shows nothing at one session and builds over several, so the cycle result is not the announcement-day effect again. Both are weakest in the latest third of dev, which includes publication, and gross is thin relative to cost. A design that marks both sides (even long, odd short) gives a placebo mean near zero, which is a cleaner test than the one-sided H-0042, where part of the gain was generic equity drift. A further Fed-calendar card on equities and Treasuries has a low prior unless it changes the instrument (index futures, equity-only comparison) or the horizon; this extends K-0060 without superseding it.

*confidence low · 2026-10-16 · librarian · evidence: H-0042; H-0045; K-0060; K-0034*

### K-0066 Perp funding does not tell forced crash selling from informational selling on a daily horizon

Two daily designs have used the perpetual-futures funding market as a state variable in liquid Binance perps over the 2020-2022 dev window. A cross-section on the funding level (H-0039) gave a small multi-day price effect that barely cleared the placebo bar and sat in one episode. A crash-day split on the change in funding (H-0049: flushed versus unflushed coins among same-day relative losers, both sides marked, same-date unmarked peers as the base, so the generic market bounce is netted out and the placebo mean sits near zero) gave nothing at the first day and the wrong sign at the multi-day horizons, worst in the latest third, the bear market. So once the generic stress bounce of K-0056 is removed, a funding-based flush label did not add a rebound and may have marked persistent selling. Funding looks like a weak and regime-dependent separator, not a reliable detector of forced liquidations. Whether it works at intraday frequency or for makers is untested, and the dev window holds one boom-and-bust cycle.

*confidence low · 2026-10-17 · librarian · evidence: H-0049; H-0039; H-0006; K-0056; K-0060; K-0049*

### K-0068 Mechanical stock-bond flow: the study effect appeared only when the trigger was a large stock-bond move

Three daily-ETF cards tried to trade mechanical stock-bond reallocation pressure. The month-end rule on month-to-date drift (H-0002) carried almost no edge. The threshold-drift trigger (H-0046) passed the event study clearly, but its events were large relative stock-bond move days, so generic spread reversal could not be separated from rebalancer flow, and the book lost to the plain mix. The volatility-control trigger (H-0050) removed that confound: the event day is quiet and the cause lies many sessions earlier. There the average was slightly wrong-signed, inside the placebo cloud, and negative in the latest third. The simplest reading is that the H-0046 pass was mostly generic reversal after large moves (K-0056), not rebalancer flow. The H-0050 sample was thin and clustered in high-volatility regimes, so this is a lean, not a refutation. A further mechanical-flow card on daily ETFs has a low prior unless the flow is observed where it trades (futures, closing auction) or a control separates it from stress reversal.

*confidence low · 2026-10-17 · librarian · evidence: H-0050; H-0046; H-0002; K-0056; K-0064*

### K-0070 Macro-calendar cards fail in four designs; moving to a volatility ETP did not help

Four designs on the scheduled US macro calendar have ended at the event-study gate: announcement-session beta premium (H-0042), even-week FOMC-cycle tilt (H-0045), cross-asset reaction drift (H-0047) and short volatility over release windows through a VIX-futures ETP (H-0052). The last was the route K-0063 and K-0065 left open. It gave a positive average, but placebo windows of the same instrument already earned over half of it through roll-down, the effect grew with the horizon like generic carry, and the first third of dev was flat while the middle third carried most (not the early-heavy decay of K-0060). The placebo bar lay above cost although gross was several times cost, because a volatility ETP is noisy. A release-time volatility crush is not visible in daily ETP bars. A further macro-calendar card on daily ETFs has a low prior; implied volatility or index futures measured directly are untested and not loadable.

*confidence low · 2026-10-17 · librarian · supersedes K-0065 · evidence: H-0052; H-0047; H-0045; H-0042; K-0063; K-0065; K-0034; K-0025; K-0060*

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

### K-0014 Only crypto has forward data

Forward daily bars come from the Binance public API (spot and USD-M perps with funding), reachable from this VPS. ETFs and US stocks have no free forward source yet (Sharadar not renewed), so hypotheses on them are parked after G4 until the Archivist validates a source.

*confidence high · 2026-10-06 · human · evidence: lab/framework/forward.py; decision Q3*

### K-0020 Spot catalog range was longer than the loadable data

The catalog said Binance spot daily data runs to 2026-10-03 but the research 9 panel the loader reads ends 2026-08-31; the gate runner takes the end of a card's data from the catalog range, so perp-plus-spot cards would have run on a month of missing spot rows. The range now says 2026-08-31.

*confidence high · 2026-10-06 · human · evidence: lab/data/catalog.yaml*

### K-0023 Attached stock datasets fit in RAM with at most six fields

SF1, insider and 13F fields are daily float32 matrices on the SEP stocks, 70 MB each. A LIQ-500 card with six fields peaks at 2.3 GB through G0-G2 on this machine (limit 2.8 GB for the cycle), three fields at 2.1 GB, so the per-card cap is six. Loading them takes seconds to a minute; fundamentals change only on filings, so signals built from them are slow.

*confidence high · 2026-10-07 · human · evidence: calibration quality_stocks; PLAN decision log 2026-10-07*

### K-0035 Real earnings-release dates now exist (sec_8k_item202), from late 2004 and with coverage gaps

The EDGAR 8-K Item 2.02 flag (ea_reaction_day) replaces the volume proxies of H-0008 and H-0015 and was used first by H-0023. It starts with the August 2004 reform, so predicted dates (last year plus 52 weeks) exist from 2005; the 1998-2004 part of dev has no earnings events. The catalog quality line shows filings for only 10 218 of 17 958 CIKs, and the known biases say that CIKs EDGAR no longer serves and releases filed under other items are missed. Firms that delisted are the likely gaps, so event sets built on it may lean toward survivors; no one has measured how far. The dataset has no hit-rate output either: whether a rule such as 'last year plus 52 weeks' finds the actual release day was a declared diagnostic of H-0023 that never reached the gate result (see K-0033). Treat the dataset as usable but unvalidated for coverage in point-in-time liquid universes.

*confidence medium · 2026-10-12 · librarian · supersedes K-0004 · evidence: H-0023; H-0008; H-0015; catalog sec_8k_item202*

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

### K-0034 Compare the placebo bar with the round-trip cost: many events can still leave a bar above cost

A mechanism failure at G1 is a 'cannot tell' when the placebo bar (its upper percentile) lies above the round-trip cost, because a tradable effect could still miss it. H-0017 (sparse events) and now H-0024 (thousands of events on over a thousand dates) are such cases: right sign, gross above cost, percentile just under the bar, bar above cost. Event count does not cure it, because events within a date share one cross-section and earnings seasons cluster dates. When the bar lies below cost (H-0020) the failure means the effect is not tradable in that form. Widening the event set helps only if it narrows the cloud below the cost; tuning filters or horizon on the same events creates new trials. Read near misses as a coin flip, not as promise.

*confidence medium · 2026-10-11 · librarian · supersedes K-0031 · evidence: H-0017; H-0020; H-0024; K-0026; K-0031*

### K-0053 The cost check now binds more often than the placebo test, and near-cost results are sensitive to the cost tier

H-0030, H-0036 and H-0039 passed the placebo percentile (H-0039 by a hair) and failed only on gross against round-trip cost; H-0036 and H-0039 ended within a fifth of the cost, H-0030 well under. H-0037 failed both. The cost ratio prices the event legs only: hedge baskets, the weekend or overnight holds, and tier choices (taker spread, perp tier) enter only when a strategy runs, so a ratio close to one is neither a safe bound nor a refutation. Before filing a daily or weekly rotation of a taker book, a Scout should write down the expected gross per event against the cost tier of the instrument, and treat a card whose gross is a fraction of cost as untradable regardless of the placebo result. This extends K-0034 (placebo bar against cost) to the cost check itself.

*confidence medium · 2026-10-15 · librarian · evidence: H-0030; H-0036; H-0037; H-0039; K-0034; K-0044*

### K-0064 A strong mechanism pass did not carry over to a long-only always-invested tilt

Several cards cleared the event-study gate and then failed the strategy gate (H-0016, H-0019, H-0028, H-0031, H-0046). K-0048 suspected hedge basket, shorts, costs and cash, because the long-only tilt H-0033 came close. H-0046 removes most of that: a long-only tilt on an always-invested book, no shorts, hedge or cash, with the strongest placebo score so far and gross several times cost, still lost to the constant mix on Sharpe. Remaining suspects: the study pools four equity ETFs against four bond peers while the book trades SPY and IEF only; the effect built over several sessions while the book held one; the earliest third carries most of the mean; and the benchmark or book construction is unverified (return, drawdown and Sharpe disagreed oddly; owner asked). A rough estimate suggests the Sharpe margin was within reach had the study effect carried over. Scouts should state the expected annual gain for the traded instruments only next to the required margin.

*confidence low · 2026-10-16 · librarian · supersedes K-0048 · evidence: H-0046; H-0033; H-0031; H-0028; H-0025; H-0019; H-0016; K-0048; K-0053*

### K-0071 Slow-horizon event studies: the placebo bar sits several times above cost, so only a large effect can pass G1

H-0054 (about three-month residual momentum) and H-0055 (about two-month variance-risk tilt) are, as far as the lessons show, the first cards with a primary horizon beyond a month that reached a mechanism verdict. Both had gross several times round-trip cost, the right sign in every third and a placebo mean near or below zero, and both ended under the percentile bar (mid-eighties percentile) with a bar roughly seven times cost. At the five-session horizon of earlier cards the bar sat near cost (K-0034). The cloud widens with the horizon while cost does not, and monthly or weekly signals give overlapping windows with few independent episodes. The brief's one-to-six-month horizon therefore meets the least powerful form of the G1 mechanism test: an effect must be several times cost to pass, and near misses are 'cannot tell' (K-0026). A Scout should state the expected placebo bar next to expected gross and cost; widening the event set helps only if it adds independent dates. Whether G1 should use a horizon-aware bar for slow claims is a judge-design question for the owner.

*confidence low · 2026-10-18 · librarian · evidence: H-0054; H-0055; K-0026; K-0034; H-0052*

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

### K-0032 Events on young or departing stocks, or needing index membership, do not fit the point-in-time universes

Three cards in two families hit the same wall in different forms. H-0018 (deleted index members) failed at G0 because weights outside the S&P 500 universe are refused. H-0021 (IPO lockup expiry in LIQ-500) found zero events across the whole dev period, so G1 rejected it on the event count with no mechanism result; recent IPOs apparently never sit in the liquid universe at the event date (cause not confirmed). H-0022 (S&P changes read inside LIQ-500) was parked by the owner because the PIT S&P 500 mask is not available to liq_n cards, although the catalog lists alt_universe among the sharadar_sep extras. None produced evidence about its mechanism. Before a Builder run, a Scout should check that the event subject can be a member of the declared universe at the event date and that every input is delivered for that universe kind; a cheap event-count feasibility check would save runs.

*confidence medium · 2026-10-11 · librarian · supersedes K-0029 · evidence: H-0018; H-0021; H-0022; K-0029*

### K-0051 The point-in-time universe rule also forces early exits in crypto perp cards

K-0032 recorded that the engine refuses weights outside the declared point-in-time universe for stock events. H-0035 shows the same rule on crypto_top_n: a position held across a universe update was refused at G0, and the Builder had to close such trades at the first decision row outside the universe, contradicting the card's 'closed on schedule' clause. The deviation touched few trades, but the clause cannot be honored and a Scout should not promise it. Event cards on a rolling top-n universe should state an exit rule that tolerates leaving the universe. Separately, the intraday data has no forward source, so such cards can pass at most G4.

*confidence medium · 2026-10-14 · librarian · evidence: H-0035; H-0018; K-0032; catalog binance_perp_15m*

### K-0054 H-0038 had events at G0 and none at G1; H-0039 got no universe at G0

K-0047 said G0 events_in_dev matches the G1 event count. That still holds for H-0036 and H-0039 (counts agree to within a fraction of a percent) and for H-0037 (1040 against 1006). It fails for H-0038: G0 counted 241 events and passed its look-ahead cuts, G1 counted zero events with a null abnormal return, so the mechanism was never measured. Possible causes, unconfirmed: a two-instrument universe (TLT, SHY) leaves no peer set for the abnormal return; a tradable or universe mask is missing for the generic auction dataset (the catalog gives it no cost model and calls it signal only); an event definition used by G0 but not G1. Separately, H-0039 first crashed at G0 because the gate passed no crypto_top_n universe on the daily perp data; the Builder computed its own top twenty by 30-day dollar volume, so the event set may differ from the judge's universe definition. Skeptic messages are again absent for all four cards. K-0047 stands for its other points.

*confidence low · 2026-10-15 · librarian · supersedes K-0047 · evidence: H-0036; H-0037; H-0038; H-0039; K-0047; K-0051*

### K-0055 The G1 event study needs unmarked peers on each date: marking every universe instrument gives zero events

H-0041 marked all five universe instruments on event dates, as its card said. The Builder tried it in the lab: marking five gave zero events, marking three gave zero, marking two gave events, and dates with only two unmarked instruments with data were dropped. The abnormal return is measured against the unmarked peers, so the judge seems to need at least three per date (inferred from behaviour, code not read). Same symptom as H-0038 (two-instrument universe, events at G0, none at G1, K-0054), which this makes the leading explanation. Consequences: the card as written could not run; the Builder narrowed the measured claim (SPY against IEF, not against TLT) and G1 and G0 event counts then agreed (482 both); nothing in the card schema warns of the peer rule. Also events() returns a single mask, so a declared complementary event set (equity drops on days when Treasuries fell) cannot be reported, the old diagnostic-channel gap again. Skeptic messages are absent for this card too.

*confidence medium · 2026-10-15 · librarian · evidence: H-0041; H-0038; K-0054*

### K-0057 lab try knows instruments named by the strategy (intraday crypto_top_n)

Intraday crypto_top_n cards may name a leader or exclusion (e.g. BTCUSDT, ETHUSDT) in strategy.py. Since commit 405b0961 the Builder's lab try adds instrument-like string literals (…USDT) to the synthetic market; before, data.col('BTCUSDT') crashed in lab try (synthetic names were AAAUSDT..FFFUSDT) although real G0 on real data was unaffected. A Builder rewrite after such a block may not reuse earlier code (old workspace files are gone), so re-verify parity with the card's reference.

*confidence high · 2026-10-15 · human · evidence: H-0040 message 288*

### K-0061 G2 random-timing null is far below zero in three cards, so passing it may only measure costs

In H-0007 (weekly US stocks), H-0035 and H-0040 (fifteen-minute crypto perps) the random-entry or random-timing check passed with a median Sharpe far below zero. In H-0035 the null was several times worse than the strategy's own cost drag would imply. In H-0040 it is less extreme: the strategy's modeled cost was a bit under two fifths of gross, which puts its cost drag in Sharpe terms at well under one, and the null's median was roughly one and three quarters times that. Either the null trades more or costs more than the real book (the strategy skips signals while a position is held and is invested a few percent of the time), or something other than timing dominates. The null's turnover, exposure and cost per trade are still not reported next to the strategy's, so a pass on this check should not be read as evidence of timing skill, and the percentile reached by an all-negative null is uninformative. The question to the owner from H-0035 is still open.

*confidence low · 2026-10-15 · librarian · supersedes K-0050 · evidence: H-0007; H-0035; H-0040; K-0017*

### K-0062 Zero events can pass G0 and then read as a mechanism failure

Three cards ended on the event count with no mechanism result: H-0021 (empty by universe), H-0038 (events at G0, none at G1) and H-0043 (zero at G0 and at G1). H-0043 is new: G0 passed with zero events, and the signal was a two-sigma residual gap that fires regularly in any healthy series, so the cause is almost surely the inputs (ICE BofA spread coverage in the lab's FRED copy, the valid-day rule, or NaN propagation), not the market. The registry shows g1_mechanism_events, which a Scout can read as a failed mechanism. Suggested handling: G0 should flag a card whose events_in_dev is zero; the Builder should report events by year before submitting; the Archivist should confirm first-valid dates of long-history series. The card's fail-loudly coverage check cannot be point-in-time at G0.

*confidence low · 2026-10-16 · librarian · evidence: H-0043; H-0038; H-0021; K-0054; K-0032*

### K-0069 G1 event-count code can hide a measured mechanism failure; date tripwires cannot fire at G0

H-0049 (193 events against a minimum of 200) and H-0050 (172 marked events on 43 event dates) are both registered as g1_mechanism_events, yet G1 measured the mechanism in both: wrong-signed average, placebo percentile far under the bar, negative cost ratio, weak latest third. A Scout reading the registry takes them for sparsity problems more events would cure; the sign says otherwise (the opposite of K-0062). New in H-0050: the card wrote its tripwire in event dates (flag if under 50 dates), but G0 reports marked events (four ETFs per date), so it showed 172 and the tripwire could not fire; the dates appeared only in G1 metrics. Both Scouts forecast event rates far above the real count. Skeptic messages were absent again. Suggested: list all failing checks in the registry reason; report event dates next to marked events at G0 and flag counts under the G1 minimum; Scouts state expected dates and marked events separately.

*confidence medium · 2026-10-17 · librarian · supersedes K-0067 · evidence: H-0049; H-0050; K-0062; K-0054; K-0032*

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

### K-0059 Cards imported from the owner's earlier research carry a contaminated holdout and an undercounted trial family

H-0040 is the first card that is an earlier owner rule re-run through the lab (research 16 and 17, selected on data that includes the lab's holdout). The card says so and states that its family trial count misses at least dozens of earlier configurations. Consequences: the judge's trial counter and multiple-testing penalty see one trial where there were many; a G4 holdout pass would not be out-of-sample; only forward data after the owner's registration date is clean. The card also changed the leader, universe ranking, costs and fill timing, so it is a re-implementation, not a replication; the lab's result and the owner's can differ for that reason as well. Suggested handling: the Chair adds the earlier configurations to the family count, and a Scout or owner states the provenance on the card before the run, as here.

*confidence medium · 2026-10-15 · librarian · evidence: H-0040; prior_studies.md; card notes of H-0040*

## Superseded

- K-0004 No earnings dates in the lab
- K-0028 G0 checks are vacuous when the synthetic market lacks the card's data, and card diagnostics cannot be reported
- K-0029 The engine cannot hold stocks that have left the declared universe, so exit-from-universe events are untestable
- K-0030 A passing mechanism test is necessary, not sufficient: pooled events hide era fade, clustering and re-weighting costs
- K-0031 Compare the placebo bar with the round-trip cost to tell a power problem from an untradable effect
- K-0033 G0 look-ahead checks are vacuous for event cards, and diagnostics cannot be reported (now five cards)
- K-0036 G0 look-ahead checks are vacuous for event cards, and diagnostics cannot be reported (now seven cards)
- K-0037 Earnings-announcement premium: absent as a tradable effect in large liquid US stocks, with real dates too
- K-0040 G0 look-ahead checks are vacuous for event cards; diagnostics cannot be reported (eight cards)
- K-0042 A passing mechanism test can still end in a book below cash, even without fade
- K-0043 The market's own release-session reaction predicts same-direction returns in liquid stocks
- K-0045 G0 look-ahead checks stay vacuous for cards with minimum-size rules (now eleven cards)
- K-0047 G0 now reports real event counts and look-ahead cuts for event cards; Skeptic messages still absent
- K-0048 Fixed event slots did not close the gap between a passing event study and a failing market-neutral book
- K-0050 G2 random-timing null is deeply negative in two cards, so passing it may only measure costs
- K-0052 Short-horizon effects that clear the pooled placebo test are usually weakest or reversed in the latest third of dev
- K-0065 Macro-calendar cards on asset-class ETFs fail in three designs; reaction drift is single-stock only
- K-0067 The G1 reason code can name the event count while the mechanism was measured and failed; the count was visible at G0
