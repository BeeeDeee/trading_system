# Lessons from the lab's hypotheses

Written by the Librarian, rendered by the framework. Qualitative on purpose: no metrics.

## H-0001 – Turn-of-month equity timing - SPY during the month-end/month-start flow window, IEF otherwise

- family: `turn-of-month-flows` · mechanism: calendar · instruments: SPY around the month turn, IEF the rest of the month, one side at a time, daily ETFs 2002 to 2020
- outcome: rejected at G1: Sharpe far below the stock-bond benchmark on the point estimate; bootstrap bound looks inconsistent (flagged)
- lesson: The design handicapped itself: equities are held only in a short window, so average equity weight is far below the benchmark's and the rest of the month is spent in intermediate Treasuries, which earned little in this period. For the rule to win, the window would have to carry nearly all of the equity premium; it did not. This does not show the window effect is absent: the mechanism checks (window vs outside, random windows) never ran. The calendar was known in advance, so no look-ahead; the Builder's calendar approximations are minor.
- avoid: Do not re-propose full switching between SPY and IEF on a calendar window benchmarked against a constant stock-bond mix, nor re-tune the window or swap in IVV/TLT: same design, same handicap. Lower drawdown is no reason to retry.
- open questions: To test the calendar premium itself, keep average equity exposure equal to the benchmark (constant mix plus a tilt inside the window) or compare window vs outside equity returns against random-window placebos. Other calendar effects (option expiry, index rebalancing, pre-holiday) are untested.
- related: H-0002, H-0003

## H-0002 – Front-running month-end 60/40 rebalancing - SPY vs IEF spread in the last days of the month, direction set by month-to-date drift

- family: `rebalancing-flow-pressure` · mechanism: flow_structural · instruments: half-weight SPY vs IEF spread through the last days of each month, flat otherwise, daily ETFs 2002 to 2020
- outcome: rejected at G1: barely above T-bills, below the card's absolute Sharpe bar; bootstrap bound below zero
- lesson: Not a coverage problem: it traded every month and was invested about a quarter of the time, yet the month-to-date-drift sign rule carried almost no edge in liquid ETFs. The source paper measures a next-day price impact on futures; holding a whole window dilutes a one-day effect, and a half-weight ETF spread earns little per unit of gross. Implementation deviated: the engine cannot exit at the close, so the position was exited at the next month's first open, carrying it into the first-day inflow period the card wanted to avoid. That cannot by itself explain a missing edge, but the card was not tested exactly as written.
- avoid: Do not re-run the same SPY/IEF month-end window with another direction rule or window length: it is already covered by H-0001 and this card. Do not rely on close-of-day execution without checking the engine supports it.
- open questions: Does the pressure exist at all? A diagnostic on next-day returns after rebalancing-sign days, on futures-like or higher-impact instruments, would separate a one-day effect from a weekly hold. The paper is from 2025, so the holdout is the only out-of-publication check.
- related: H-0001

## H-0003 – Volatility-timed equity exposure - SPY when short-term SPY volatility is below its long-run level, IEF otherwise, weekly

- family: `volatility-managed-exposure` · mechanism: volatility · instruments: SPY when short-run volatility is at or below its long-run level, IEF otherwise, weekly, daily ETFs 2003 to 2020
- outcome: rejected at G1: Sharpe below the stock-bond benchmark, bound well below zero, deeper drawdown, entries just under the minimum
- lesson: The rule is always fully invested in one asset, so in the calm state it carries more equity risk than a constant mix and gets caught by fast crashes and V-shaped rebounds; return was about level with the benchmark but the risk-adjusted return was not beaten. The vol state flips rarely, so weekly decisions on one asset pair cannot reach the entry minimum: a structural limit of slow binary switches. The volatility-timing premium itself was not isolated, because the mechanism checks never ran. An earlier G0 crash was a Builder bug, since fixed.
- avoid: Another whole-portfolio binary switch between SPY and IEF on a single state variable (trend, volatility, calendar): H-0001 and H-0003 both lost to the constant mix this way, and studies 2 and 9 point the same direction. Do not tune windows or the threshold to reach the entry count.
- open questions: Continuous scaling of risk exposure (the Moreira-Muir form) that keeps average exposure near the benchmark, applied across several risk assets so decisions and entries are plentiful, and judged against a static mix with the same average exposure.
- related: H-0001

## H-0004 – Crypto new-listing supply overhang - short recently listed Binance pairs, long seasoned pairs, dollar-neutral, weekly

- family: `new-listing-overhang` · mechanism: flow_structural · instruments: Binance USDT spot: short recent listings, long pairs listed over a year, dollar-neutral, weekly, 2017 to 2022
- outcome: rejected at G1: positive point estimate over T-bills but below the absolute Sharpe bar; bootstrap bound below zero
- lesson: A test that could not tell, more than a refutation. The dev window is only a few years of weekly data, so the Sharpe interval is wide. The card's own seasoned-leg rule forces cash for roughly the first year, and the strategy was invested only part of the time, which dilutes the measured Sharpe. The short leg was a spot-short simplification with no borrow or funding cost, so even a pass would have been flattered. The spread may also mix listing age with size, liquidity and beta rather than isolate overhang.
- avoid: Do not re-submit with a different age window or skip period: neighbors on the same few years add no information. Do not read the result as support for shorting alts; study 10 showed crypto short legs flipping sign between dev and holdout.
- open questions: An implementable version (perpetuals with funding costs) that controls for size and liquidity, on a longer history. It needs perp loaders that do not exist yet; without them later gates cannot say more.
- related: H-0005

## H-0005 – Stablecoin net issuance as crypto inflow gauge - hold liquid Binance coins only while aggregate USD stablecoin supply is growing, weekly

- family: `stablecoin-flow-pressure` · mechanism: flow_structural · instruments: inverse-volatility basket of ten liquid Binance coins, held only while stablecoin supply grew, weekly, 2018 to 2022
- outcome: rejected at G1: beat BTC on the point estimate, bootstrap bound below zero; first run voided (evaluator bug)
- lesson: Not a clean refutation: the point estimate was good but not reliable. Bitcoin finished the dev window near where it began, so almost any rule that sat out drawdown phases would beat it on excess return; only the noise-aware bound is informative. The strategy was invested most of the time because supply grew through most of the sample, leaving few independent risk-off episodes, and drawdown stayed close to the benchmark's. The comparison also confounds the signal with holding an altcoin basket instead of BTC; no always-on basket control was run, so the signal's contribution is unknown. The first G1 run (no exposure) was an evaluator bug and is not evidence.
- avoid: Do not pair a new flow signal with a different asset basket than the benchmark without an unconditional basket as control. Do not retune the lookback to move the bound.
- open questions: Stablecoin issuance as a conditioner on BTC and ETH only, with an always-on control and a price-trend regime control (study 9 found trend filters add little). Needs a longer sample with more than one supply contraction.
- related: H-0004

## H-0006 – Crypto liquidation-flush rebound - buy liquid coins after an extreme daily drop, only while BTC is in a weekly uptrend, hold 3 days

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: Binance USDT spot liquid coins, long only after extreme one-day drops, only in a BTC weekly uptrend, 2019-2022
- outcome: rejected at G1: beat the benchmark on the point estimate, but the bootstrap bound of the Sharpe difference was below zero; exposure below floor
- lesson: More a test that could not tell than a refutation. The book was in cash most of the time, so a few clustered episodes drove the result and the dev window (about three years, apparently cut to the start of the perp diagnostic data) left the Sharpe interval wide. Drawdown still exceeded the benchmark's despite low exposure: losses probably came from episodes where many coins triggered together while the lagging weekly BTC filter was still ON, the falling-knife risk the card named. The pre-registered diagnostics (regime, funding flush, cascade shape) were never reached, so the leverage-flush mechanism is untested.
- avoid: Do not retune the drop threshold, hold length or regime length, or widen the universe, to lift exposure or the bound: the grid reuses the same few episodes. Do not file another long-only crypto dip-buy gated by a BTC trend filter (studies 9 and 13 point the same way).
- open questions: Test the mechanism as an event study before any strategy: after extreme drops, do forward returns differ by falling funding versus not, or by jump-like versus diffuse drops, over the full spot history from 2017 and perps from 2019? A rule that does not depend on a diagnostic dataset would also get a longer dev window.
- related: H-0007, H-0005

## H-0007 – Intraday-component reversal confirmed by intraday-component momentum - weekly long/short in LIQ-500, buy last week's intraday losers among 12-month intraday winners, short the mirror

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: US liquid stocks, weekly dollar-neutral long/short on intraday-return reversal confirmed by intraday momentum, 1998-2020
- outcome: passed G1 narrowly; rejected at G2: primary was the grid peak, early-era concentration, no edge among S&P members
- lesson: A thin edge that is early-era and small-name. Return decayed across the three sub-periods, the earliest dominating, the same pattern as study 5. It vanished among S&P members, and it was fragile to costs: doubled costs left almost nothing, tripled costs made it negative. The random-entry check passed with random books deeply negative, so it is probably only measuring costs (flagged to the owner). The decomposition, confirmation and overnight-placebo checks never ran, so whether the intraday split adds anything over close-to-close reversal is unknown.
- avoid: Another weekly cross-sectional US stock reversal with a new conditioning variable (trend, VIX, intraday split, momentum confirmation): this is the third try after studies 5 and 7, and each time the profit sits in the early years and the less liquid names. Do not add grid points or lookbacks.
- open questions: Run the pre-registered mechanism checks as cheap diagnostics restricted to after 2005 and to large caps. Only a low-turnover or maker-style implementation could survive costs; a taker book at weekly turnover cannot (compare mrel-1h).
- related: H-0006, H-0008

## H-0008 – Earnings-announcement premium via volume seasonality - weekly LIQ-500 long/short, long stocks whose volume 52 and 13 weeks ago predicts an announcement next week

- family: `earnings-announcement-premium` · mechanism: calendar · instruments: US liquid stocks, weekly dollar-neutral: long predicted announcers (volume seasonality), short the rest, 1998-2020
- outcome: rejected at G1: lost money versus T-bills, bootstrap bound below zero; coverage was fine, so not a power problem
- lesson: A real negative on a long sample, not a power problem. Three explanations remain and none was separated, because the predictor check (are predicted weeks really high-volume announcement weeks?) and the six-week-shifted placebo never ran: (1) volume seasonality does not identify announcers (it also catches index rebalances, option expiries and other recurring spikes); (2) the premium is absent in large liquid names, or lives in a few days around the event and is diluted by a full-week hold; (3) a few equal-weight long names with jump risk against a broad equal-weight short is not matched on size or beta, so it measures more than the premium.
- avoid: Do not retune the volume threshold, the lags or the tolerance: a negative result over this many years has no peak to refine. Do not use a volume peak as a proxy for announcement dates again without validating it.
- open questions: The missing input is real earnings dates (filing dates in sf1 lag the press release, so they would not do); an Archivist task. With dates, run an event study of returns on announcement days against size- and beta-matched controls before any strategy. Check first whether the volume rule hits announcement weeks at all.
- related: H-0007, H-0001

## H-0009 – Disposition-gated news drift - after a stock-specific volume shock, follow the move only when it has the same sign as holders' 12-month capital gain, LIQ-500 long/short, hold 5 days

- family: `disposition-news-drift` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), daily long/short following volume-shock moves that agree with holders' capital gain or loss, short hold, 1998 to 2020
- outcome: rejected at G1: lost money outright, far below T-bills, deep drawdown, bootstrap bound well below zero; ample coverage, so not a power problem; mechanism checks never ran
- lesson: A clear negative on a long, dense sample: the book was mostly invested and traded constantly, yet lost steadily. Not separated: (1) wrong sign, since volume-driven moves in liquid stocks may partly reverse (K-0012, H-0007); (2) spread costs of a high-turnover taker book, plus short-side squeeze risk with no borrow cost modelled. Whether any drift exists before costs is unknown. No sign of a framework bug.
- avoid: Do not retune the volume threshold, reference window or hold, or swap the overhang for another gate: a long dense negative has no peak to refine. No other daily US large-cap event-follow book at this turnover.
- open questions: An event study first: forward returns after volume-shock days by news direction and overhang sign, against delayed-entry and misaligned placebos, gross of costs, by era and in large caps. Real announcement dates (K-0004) would isolate true news days.
- related: H-0007, H-0008, H-0006

## H-0010 – Ex-dividend price pressure - LIQ-500 neutral, long payers before predicted ex-day, short just after, 5 days

- family: `dividend-month-premium` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500), market-neutral: long regular quarterly payers in the sessions before a predicted ex-dividend day, short just after the ex-day, five-session windows, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: abnormal return was inside the placebo cloud and a tiny fraction of the round-trip cost; no strategy run
- lesson: No ex-dividend price pressure visible in large liquid names. The event sample was large, so this is not a power problem. The pooled abnormal return faded across the three sub-periods (positive early, negative in the latest) and was far below the cost of trading every name twice per dividend cycle. Pre-ex and post-ex events were pooled in one test and the by-side run-up the card promised was not reported, so which leg is dead is unknown. The synthetic market has no dividends, so G0 look-ahead was vacuous and the regular-payer share was never reported.
- avoid: Do not re-run the ex-day clock with another window length, the pre-ex leg alone, or another payer-regularity rule: the gap to costs is too wide for tuning to close. Do not file the same ex-day events under a new name (dividend capture, high-yield pressure).
- open questions: The original dividend-month premium is monthly (predicted-dividend months against other months, month-long hold) and was not tested; it has a different cost profile but is likely weak in large caps. File it under the same family so the trials stay counted. A by-side diagnostic would tell whether the run-up exists at all.
- related: H-0011, H-0015

## H-0011 – 52-week-high anchoring - LIQ-500 market-neutral, long stocks that just broke through a stale 52-week high (not on a top-decile move day), short the rest, hold 5 days

- family: `52-week-high-anchoring` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), market-neutral: long stocks closing above a stale 52-week high on a non-extreme day, short the rest, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: placebo sets of the same stocks earned more than the breakout events, and cost was not covered; no strategy run
- lesson: The positive point estimate is not evidence of a breakout effect. The placebo average (same stocks, shifted dates) was higher than the event average, so stocks that make 52-week highs simply tend to do well in any week, probably winner drift rather than anything special about the crossing day. The sub-periods disagreed in sign (early era negative, middle strongly positive), and events cluster on few bull-market dates, so the effective sample is smaller than the event count. Longer horizons look larger, but the placebo drifts too and they are slower than the brief allows.
- avoid: Do not retune gap_days, hold_days or the top-decile exclusion. Do not refile stale-high breakouts as momentum or near-high variants on the same universe: the card itself accepted that this is the breakout family of study 1.
- open questions: A fair test of the anchoring story must net out stock-identity drift, for instance comparing breakout stocks with stocks matched on prior-year return and distance to the high, and probably needs a monthly horizon where the literature places the effect.
- related: H-0009, H-0007

## H-0012 – Non-routine insider purchase drift - LIQ-500 market-neutral, long stocks with a first open-market insider purchase filing after a half-year without any, hold 5 days

- family: `insider-purchase-information` · mechanism: other · instruments: US liquid stocks (LIQ-500), market-neutral: long stocks with a first insider open-market purchase filing after a half-year quiet window, five-day hold, 2009 to 2020
- outcome: rejected at G1 on the mechanism test: abnormal return after the filing was slightly negative and below the placebo middle; no strategy run
- lesson: No post-filing drift in large liquid names. Entry is two sessions after the filing, the Form 4 is public at once and large caps are heavily followed, so any reaction is likely already in the price. Events are sparse and cluster in sell-offs, so the placebo distribution is wide: only an effect larger than costs could have shown, and none did. The middle sub-period was clearly negative, consistent with buying into declines that continued (the reversal confound the card named but never checked). The data holds only counts of purchase transactions, so the non-routine proxy is crude. G0 look-ahead was vacuous: the synthetic market produced no events.
- avoid: Do not retune the quiet window (91 or 365 days) or the hold. Do not read this as proof that insider buying is uninformative; it shows only that the filing-day drift is not harvestable in LIQ-500.
- open questions: Insider information may live in less-covered names, at longer horizons, or in features the loader lacks (purchase size, officer role, several insiders buying together). None is loadable today; size and role would be an Archivist task.
- related: H-0009, H-0013

## H-0013 – Peer lead-lag catch-up - LIQ-500 market-neutral, long stocks whose return-correlated peers rose last week while they themselves did not move much, short the mirror, hold 5 days

- family: `peer-lead-lag-diffusion` · mechanism: cross_asset_information · instruments: US liquid stocks (LIQ-500), weekly dollar-neutral: long stocks whose most return-correlated peers rose while they did not move, short the mirror, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: abnormal return slightly negative, below the placebo middle, positive in at most one sub-period; no strategy run
- lesson: A clear negative on the largest event sample of the batch, so not a power problem. Statistically chosen correlation peers carry no usable catch-up over a week in LIQ-500; this agrees with the earlier cross-market lead-lag scan and with the idea that diffusion in well-covered names is over within about a day. Only the very shortest horizon leaned positive, and by far less than costs; this is not an invitation to a one-day version. The book would rotate nearly all names weekly, so even a weak real effect could not be harvested. G0 look-ahead was vacuous (the synthetic market has too few names for the book to trade); the Builder repeated the checks by hand on a larger market.
- avoid: Do not vary peers_k, formation_days or corr_window, and do not swap correlation peers for another clustering: it is the same test. No weekly full-rotation statistical-peer book.
- open questions: Diffusion along explicit economic links (suppliers, customers, parents) or in less-covered names might differ, but neither link data nor small caps are loadable.
- related: H-0007, H-0009

## H-0014 – High-volume visibility premium - LIQ-500 market-neutral, weekly, long abnormal-volume stocks and short abnormally quiet stocks matched on past-week return, hold 5 days

- family: `high-volume-visibility-premium` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), weekly market-neutral: long abnormal-volume stocks, short abnormally quiet ones, matched within past-week-return quintiles, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: high-volume against quiet stocks showed essentially zero abnormal return, mid-placebo; no strategy run
- lesson: Flat result on the largest event sample in the lab, so not a power problem. The visibility premium of the literature builds over several weeks and sits in less visible firms; LIQ-500 stocks are always on screens, so a volume burst adds little new attention. The descriptive longer horizons did not turn positive either, so waiting longer would not help. The latest sub-period was negative. Matching on past-week return removed reversal by construction, and what remained was nothing, which suggests that earlier volume-based books owed their moves to returns, not volume. The long leg alone was not reported separately.
- avoid: Do not retune reference_days, leg_quantile or the hold. Do not file another abnormal-volume cross-section on LIQ-500 under a new attention story (visibility, most-active lists, volume shock): with H-0009 this is the second dead volume-signal book.
- open questions: Any volume effect probably lives in smaller or less covered names, or needs volume tied to an identified event; neither is loadable at present.
- related: H-0009, H-0008

## H-0015 – Filing-anchored earnings-announcement premium - LIQ-500 dollar-neutral, long stocks whose predicted announcement (last year's SF1-validated reaction day) is 3 days ahead, hold 5 days

- family: `earnings-announcement-premium` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), dollar-neutral: long before a predicted earnings reaction day, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: positive abnormal return, but no larger than placebo sets of the same stocks; no strategy run
- lesson: Second trial of the family, and it did not settle the question left by H-0008. The placebo average was almost as high as the event average, so the positive number belongs to which stocks are picked (names with a clean earnings-day volume spike year after year), not to the dates. The card's own proxy check (do predicted days really show a volume spike) was not computed because the Builder had no output channel, so we still cannot say whether the date prediction worked or the premium is absent in large caps. Treat as: no evidence of an exploitable premium, proxy quality unknown.
- avoid: Do not file a third volume-based date proxy, and do not tune lead_days or min_spike before the hit rate of the proxy is measured. Do not read a positive gross abnormal return without comparing it with the placebo average.
- open questions: Real earnings dates (Archivist task, see K-0004), or a framework change so that proxy hit-rate diagnostics reach the gate output. With true dates, test announcement days against size- and beta-matched controls.
- related: H-0008, H-0014

## H-0016 – Turn-of-month beta spread - LIQ-500 dollar-neutral, long the top beta quintile and short the bottom quintile only over the 5-day month turn, flat otherwise

- family: `turn-of-month-flows` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), dollar-neutral: long the top beta quintile, short the bottom quintile, only over a five-session window at the month turn, flat otherwise, 1998 to 2020
- outcome: mechanism test passed; rejected at G1 on the strategy: Sharpe far below the bar and the bootstrap bound against T-bills below zero
- lesson: The first card to pass the mechanism check and then be simulated, so it separates the two questions: the month-turn window does reward high beta over low beta in LIQ-500 more than at other dates, but the premium is too small to trade alone. The book is invested about a fifth of the time, a beta spread at gross one has a net beta well under one, and the rest of the year earns cash, so the window's own Sharpe is the ceiling, as the card warned. The effect was weakest in the latest third of the sample, which fits publication decay. A passing mechanism with a failing strategy is not a refutation of the calendar flow.
- avoid: Do not retune post_days, beta_lookback or the quintile cut to lift the Sharpe, and do not build another stand-alone window book against cash. Do not return to SPY/IEF switching (H-0001).
- open questions: Use the window as a tilt on an always-invested benchmark (overweight equity or beta inside the window, same average exposure otherwise), judged against the constant mix as K-0005 asks. Several calendar windows (pre-holiday, option expiry) in one low-turnover overlay would raise exposure; each is a separate card with its own trial count.
- related: H-0001, H-0002

## H-0017 – Index-inclusion price-pressure reversal - S&P 500 universe, short stocks in their first days as index members, long the other members, hold 5 days

- family: `index-inclusion-price-pressure` · mechanism: flow_structural · instruments: S&P 500 members: short stocks in their first sessions after joining the index, long the other members as hedge, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test by a hair: right sign in every sub-period and cost covered, but the placebo percentile was just under the bar
- lesson: The closest call of the batch and more a 'cannot tell' than a refutation. Only a few hundred events on fewer dates make the placebo distribution wide, so a gate that works for tens of thousands of events is close to a coin flip for a real effect of this size. In its favour: the right sign in every sub-period, a gross effect far above cost, and larger descriptive moves at longer horizons, which suggests the reversal runs slower than the five-day hold. Against: it faded from the early to the late sub-period, as expected once the effect is published, and a short book of a few single names is idiosyncratic. The gate stays the gate.
- avoid: Do not resubmit the same events with hold 3 or 10, a changed seasoning filter or any tweak aimed at the percentile: that is a retry on the same data and counts as new trials.
- open questions: A different card with many more events and a slower horizon where the reversal seems to live, or both additions and deletions in one test. Other index-driven events with more occurrences (annual reconstitutions) would raise power, but no such membership data is loadable today.
- related: H-0002, H-0016

## H-0018 – Index-change price-pressure reversal, both directions - S&P 500, short new members and long just-deleted stocks, hedged with other members, hold 5 days

- family: `index-inclusion-price-pressure` · mechanism: flow_structural · instruments: S&P 500 members: short new members and long just-deleted stocks, hedged with the other members, five-day hold (never run)
- outcome: parked before any gate: the deletion leg needs weights outside the S&P 500 universe, which the engine refuses (g0_outside_universe). Nothing was simulated.
- lesson: A feasibility failure, not evidence about index-change pressure. The card named the risk and asked for a loud G0 failure instead of a silent drop of the deletion leg, and the Builder did that. The deletion side, where the literature finds the fuller reversal, cannot be tested under the sp500 universe kind, because a deleted stock stops being a member the day it becomes an event. The question H-0017 left open (more power through both directions) is still open.
- avoid: Do not file a card whose key leg needs weights outside the declared universe (deleted index members, names dropped after the event): it stalls at G0 again. Do not drop the deletion leg quietly and resubmit addition-only events under a new id.
- open questions: Can deletions be read inside a liq_n universe that uses S&P membership as an input? That loses additions outside the liquid set and needs a Chair or owner decision first. Other index-driven events with more occurrences are untested.
- related: H-0017

## H-0019 – Trading-window net flow - LIQ-500 dollar-neutral, after a new quarterly filing long net repurchasers and short net stock-compensation issuers, hold 5 days

- family: `trading-window-flows` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500), dollar-neutral: after each new quarterly filing long net repurchasers and short heavy stock-compensation issuers, five-day hold, 1998 to 2020
- outcome: mechanism test passed clearly (the strongest in the lab so far); rejected at G1 on the strategy: net Sharpe about level with T-bills, far below the absolute bar, bootstrap bound below zero.
- lesson: Best mechanism result so far, still no strategy. The event average sat far above the placebo cloud and average, and gross covered the round-trip cost about twice. But the effect faded across all three sub-periods, the latest near the placebo level and below cost; longer horizons were as large, so it reads as a slow filing-linked drift (news or earnings season) more than a temporary flow concession; and clustered filings mean a book invested part of the time with small weights, plus daily re-equalising turnover the event cost ratio omits. Net return was about cash. No sign of a bug.
- avoid: Do not retune quantile or hold_days, and do not refile it as buyback yield or net issuance. Do not read the mechanism pass as a tradable edge (K-0027).
- open questions: Is the spread flow or news? A control matched on return around the filing would tell; true earnings dates (K-0004) would separate filing from announcement. If real, it needs a slow low-turnover tilt on an always-invested book against a same-exposure benchmark.
- related: H-0016, H-0012, H-0015, H-0010

## H-0020 – Option-expiration hedge-unwind cycle - LIQ-500 beta-neutral, long the most option-active (most liquid) names and short the least liquid through the monthly expiration week, reversed the week after

- family: `option-expiration-hedging-flows` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500), beta-neutral: long the most liquid names and short the least liquid through the monthly option-expiration week, reversed the week after, 1999 to 2020
- outcome: rejected at G1 on the mechanism test: pooled abnormal return positive but tiny, inside the placebo cloud (percentile below the bar), gross a small fraction of the round-trip cost; strategy not run.
- lesson: No visible dealer-hedging cycle between the most and least liquid names, and not a power problem: the event count was huge, the placebo cloud narrow and its upper bar far below trading cost, so any tradable effect would have shown. The earliest sub-period leaned positive, the middle was about zero, the latest barely positive, which fits fading after publication and weekly options. Dollar volume only proxies option open interest, so no hedge pressure and a poor proxy are not separated. The expiration-week-only check the card promised was not reported.
- avoid: Do not retune leg_n or lookback_days, drop the reversed week, or swap in another liquidity or size sort on the same dates. Many events do not mean power: they share a few dates and one cross-section.
- open questions: A fair test needs option open interest (not loadable). Other pre-announced calendar flows (index reconstitution, pre-holiday) are separate cards and better tested as a tilt on an always-invested book (K-0027).
- related: H-0016, H-0001, H-0017

## H-0021 – IPO lockup-expiry supply overhang - LIQ-500 beta-neutral, short stocks around the 180th calendar day after their first trading day, hedged with the other members, hold 5 days

- family: `new-listing-overhang` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500), beta-neutral: short each stock around the end of its standard six-month IPO lockup, hedged with the other members, five-day hold, 1998 to 2020 (no event ever fired)
- outcome: rejected at G1 on the event count: no lockup events at all in dev, so no mechanism result and no strategy run. Untestable, not refuted.
- lesson: The test could not tell. A point-in-time liquid universe apparently never holds a stock when its lockup ends: recent IPOs are not among the most liquid names after six months, or the liquidity pool needs more history. The Builder's literal rule (fire only if the stock is a member on the first eligible day, otherwise never) makes a miss permanent. The card named this risk and the gate reported the empty set as designed. Nothing is learned about lockup pressure. G0 look-ahead was vacuous.
- avoid: Do not resubmit lockup events in LIQ-500 or the S&P 500 with another lead, hold or lockup length: an empty event set stays empty. Do not infer that lockups carry no price effect.
- open questions: A cheap feasibility count of listing age at membership, before any card, would show whether the lab sees young stocks at all. Lockup events likely need a wider or listing-age-aware universe, which RAM limits (K-0003) may forbid: an owner or Chair decision. Borrow cost of recent IPOs is not modelled either.
- related: H-0004, H-0018, H-0022

## H-0022 – S&P 500 change price-pressure reversal, both directions, read inside LIQ-500 - long liquid stocks just deleted from the index, short just-added ones, dollar-neutral, hold 5 days

- family: `index-inclusion-price-pressure` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500) with S&P 500 membership as an input: long liquid stocks just deleted from the index, short just-added ones, dollar-neutral, five-day hold (never built or run)
- outcome: parked by the owner before any gate: the PIT S&P 500 mask is not available to LIQ-500 cards. Nothing simulated.
- lesson: A feasibility failure, the second in this family. The card was the workaround H-0018 proposed (read S&P membership as an input inside LIQ-500), but the input is not delivered for that universe kind, although the catalog lists the extra. Both routes to the deletion leg are now closed: inside the S&P universe a deleted stock cannot be held (H-0018); inside LIQ-500 the membership signal is missing. H-0017's open question (more power from both directions) cannot be asked with today's loaders.
- avoid: Do not file a third both-direction index-change card, or rebuild this one on a proxy for membership, until the owner confirms the loader supplies the PIT S&P mask for the declared universe. Do not resubmit addition-only events with new holds: that is H-0017 again.
- open questions: The membership history exists on disk but has no loader; adding one, or exposing the mask to LIQ-500 cards, is an owner or Archivist task. Then additions plus deletions in one liquid universe remain the natural way to give the placebo test more power than H-0017 had.
- related: H-0017, H-0018

## H-0023 – Earnings-announcement premium on real 8-K dates - LIQ-500 dollar-neutral, long a 5-session window around last year's Item 2.02 date + 52 weeks

- family: `earnings-announcement-premium` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), dollar-neutral, long a five-session window around the predicted release date
- outcome: rejected at G1 on the mechanism test: right sign, many events, but inside the placebo cloud and gross below round-trip cost; strategy not run
- lesson: First test with real release dates, so it settles what H-0008 and H-0015 could not. Many events, right sign, no fade, yet indistinguishable from placebo windows of the same stocks, and the placebo upper bar sits about at the round-trip cost while the event gross is below it. Any effect in LIQ-500 is smaller than the cost of trading it this way; not a power problem. The one-session horizon was negative, so no pre-announcement run-up shape. The card's date hit-rate was never reported, and the dataset misses some firms.
- avoid: Do not retune window_days, the lead or the 45-day rule, nor refile the same dates as a run-up or attention premium: a fourth trial of the same test. Do not read the positive gross as support (K-0025).
- open questions: The premium may sit in less covered names or in the release session itself, which a five-session LIQ-500 window cannot see. Surprise or date-shift conditioning is a different claim and must name its mechanism (Chair decision on H-0024/H-0019).
- related: H-0008, H-0015, H-0024, H-0019

## H-0024 – Concurrent-filing earnings-surprise drift - LIQ-500 dollar-neutral, long/short top/bottom quintile earnings change at filings that coincide with a volume spike, hold 5 days

- family: `post-earnings-announcement-drift` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), dollar-neutral: long the best and short the worst earnings-change quintile at quarterly filings that coincide with a volume spike, five-day hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: abnormal return positive, cost covered, many events, but the placebo percentile stayed under the bar. The strategy was not run.
- lesson: Nearer 'cannot tell' than refuted. Unlike H-0011 and H-0015 the placebo average was near zero, so this is not stock selection. But the placebo upper bar lay above the round-trip cost and the gross was only modestly above cost, so the effect could miss the bar. Dense events share a few earnings seasons, so the cloud stays wide (K-0031). Sub-periods were unstable, not fading: first and latest positive, middle negative. Longer horizons were not smaller, which fits drift. Whether the volume filter finds the release day is unknown (K-0028).
- avoid: Do not retune quantile or vol_mult, or move to the longer horizon the descriptive numbers seem to favour: same events, new trials aimed at the percentile. A part-time clustered book would earn little even if real (K-0030).
- open questions: True announcement dates (H-0023 waits on them) would replace the volume proxy. Is this the same filing-season news drift that H-0019 saw as a net-flow spread? A control conditioning on filing window and surprise together would separate them. Follow-ups should be slow tilts on an always-invested book.
- related: H-0019, H-0009, H-0015, H-0008, H-0023, H-0017

## H-0025 – Turn-of-month beta rotation, always invested - LIQ-500 long-only, high-beta quintile over the month turn (day -1 to +3), low-beta quintile the rest of the month

- family: `turn-of-month-flows` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), long-only: high-beta quintile over the month turn, low-beta quintile otherwise
- outcome: mechanism test passed; rejected at G1 on the strategy: Sharpe margin over SPY a hair under the fixed bar, bootstrap bound above zero. G2 not reached.
- lesson: The always-invested tilt asked for by K-0027 did far better than the flat-otherwise book of H-0016, but do not read the miss as promising. The pooled spread decayed across eras and was negative in the latest third, so G2's sub-period and publication checks would likely have killed it. Drawdown was as deep as the benchmark's. The comparison is confounded: an equal-weight liquid-stock book against cap-weighted SPY, with no static low-beta control and no switch A/B split (no channel). Full rotation twice a month leaves costs close to the gross effect.
- avoid: Do not retune post_days, beta_lookback or leg_quantile, or change the benchmark, to close the margin: new trials on the same data. No fourth month-turn beta book without a new reason.
- open questions: Is it the month-turn flow or low beta plus an equal-weight tilt? Only a same-universe control separates them. A small month-turn tilt on an always-invested cap-weighted book would cost far less; other calendar windows are separate cards.
- related: H-0016, H-0001, H-0019

## H-0026 – Overdue earnings release - LIQ-500 market-neutral, short stocks whose Item 2.02 release is several sessions past last year's same-quarter date, hedged with the other members, hold 5 days

- family: `earnings-date-delay-signal` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), market-neutral: short stocks whose earnings release is late versus last year's date, hedged with other members
- outcome: rejected at G1: right sign on average, but under the placebo bar and gross below round-trip cost; strategy not run
- lesson: Not a power problem: the placebo upper bar lies below the round-trip cost, so a tradable effect would have cleared it (K-0034). Overdue stocks underperformed slightly on average, but the latest third reversed sign (one of the card's own kill conditions) and the one-session horizon was wrong-signed. The flag fires on a sizeable share of firm-quarters, so benign delays (calendar shifts, other filing items, dataset gaps) probably dilute it. Large-cap release dates are pre-announced, so lateness is likely priced. The rule's hit rate was never reported (K-0036).
- avoid: Do not retune delay_sessions or hold_days, or move to the longer horizon that looks larger descriptively: new trials on the same events. Do not refile the same anniversary rule as late reporter or disclosure-timing; it shares data and date prediction with earnings-announcement-premium, so those trials count together.
- open questions: A cleaner delay event (explicit late-filing notice, or announced versus expected dates) might behave differently, but neither is loadable today. Less-covered names are untested. The release-session reaction after an overdue flag was not isolated.
- related: H-0023, H-0024, H-0015, H-0008, H-0019

## H-0027 – Release-anchored trading-window net flow - LIQ-500 dollar-neutral, at each real 8-K earnings release long net repurchasers and short net stock-compensation suppliers, hold 5 days

- family: `trading-window-flows` · mechanism: flow_structural · instruments: US liquid stocks (LIQ-500), dollar-neutral: at each real earnings release long net repurchasers and short net stock-compensation suppliers, five-session hold, 2004 to 2020
- outcome: rejected at G1 on the mechanism test: placebo percentile under the bar, gross just under round-trip cost, latest third wrong sign; strategy not run
- lesson: The release-anchored version did not reproduce H-0019's pass, so that pass was probably filing-linked drift, not a flow concentrated just after the trading window opens. The effect built slowly with the horizon (nothing in the first sessions, larger at the longest horizons), the slow-drift shape the card named as a kill condition; the latest third reversed sign, another declared kill condition. Gross sat just under round-trip cost and the placebo bar about at cost, so the percentile miss is a near miss, but the two declared failures make it a refutation of the window-flow story in this form.
- avoid: Do not retune leg_quantile, hold_days or the net-flow score, move to the longer horizon that looks larger, or refile the same events as buyback yield, net issuance or insider selling after the release: a third trial of the same idea. Do not read the positive gross as support (K-0025).
- open questions: If net-flow information exists it is slow and style-like (repurchasers versus issuers), so it would need a low-turnover tilt on an always-invested book against a same-exposure benchmark (K-0027, K-0030), and it should be argued as a style premium, not a calendar flow. Event-matched controls were not run.
- related: H-0019, H-0028, H-0023

## H-0028 – Earnings-reaction drift on real 8-K release dates - LIQ-500 dollar-neutral, follow the sign of a large standardized release-session reaction, hold 5 days

- family: `post-earnings-announcement-drift` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), dollar-neutral: follow the sign of a large release-session reaction on real earnings dates, five-session hold, 2004 to 2020
- outcome: mechanism test passed clearly; rejected at G1 on the strategy: net Sharpe below T-bills and the bootstrap bound of the difference below zero, with a deep drawdown for a market-neutral book
- lesson: Strongest mechanism result in the lab, and the book still lost to T-bills. Following a large standardized release-session reaction earned same-direction abnormal returns, far above the placebo cloud, above cost, positive in every third including the latest, mostly in the first sessions after entry. Yet the dollar-neutral book, invested about half the time, earned less than cash. This is not the fade or thin exposure of H-0016/H-0019, so the gap is unexplained: event weights shrink when many names report together, hedge-basket turnover, and a generic cost tier for names that just moved several deviations. The gate output has no gross/net split.
- avoid: Do not retune z_threshold or hold_days, and do not refile the same reaction-following events as earnings momentum or announcement-return continuation: they share events and count together. Do not read the mechanism pass as a tradable edge (K-0030).
- open questions: Reconcile event study and book first: gross by sleeve, event-weighted versus equal-weighted, cost of the hedge basket, and cost on reaction-day opens. The market's own reaction looks like a better surprise measure than the accounting change of H-0024. A lower-turnover implementation, or a tilt on an always-invested book, is the natural next form.
- related: H-0024, H-0023, H-0026, H-0019

## H-0029 – Turnover-conditioned short-term momentum - LIQ-500 dollar-neutral, weekly: among the highest-turnover stocks of the past month, long the past-month winners and short the past-month losers, hold 5 days

- family: `short-term-momentum` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), weekly dollar-neutral: among the highest-turnover stocks of the past month, long past-month winners and short losers, five-session hold, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: small positive gross, inside the placebo cloud, gross a small fraction of round-trip cost, latest third negative; strategy not run
- lesson: The turnover-split momentum claim is not visible in LIQ-500 at a weekly horizon. Not a power problem: the event sample was very large, the placebo bar sat below round-trip cost and gross covered only a fraction of cost, so a tradable effect would have shown. The first two thirds leaned positive and the latest reversed, the early-era pattern of the other one-month-return books (study 5, H-0007, K-0012), and the descriptive horizons were not monotone. The paper forms and holds monthly; here selection rolls weekly with heavy overlap, so even a real effect would face high turnover.
- avoid: Do not add turnover, volume or size conditioners to the one-month return sort in LIQ-500 (high-turnover momentum, low-turnover reversal, size splits): another look at the same cross-section. Do not retune formation_days or quantile.
- open questions: A monthly-rebalanced version has a different cost profile but the latest-third reversal and the wide gap to cost give it a low prior. The conditioning may live in less-covered names, which are not loadable.
- related: H-0007, H-0009, H-0014, H-0030

## H-0030 – No-news shock reversal in the S&P 500 - dollar-neutral, buy peer-adjusted daily losers and sell winners on sessions without an 8-K earnings release, staggered 5-day hold, 2005-2020

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: S&P 500 members, daily dollar-neutral: long peer-adjusted one-day losers, short winners, on sessions without an earnings release, staggered five-session hold, 2005 to 2020
- outcome: rejected at G1: the no-news reversal cleared the placebo test, but gross per event covered only a fraction of round-trip cost; strategy not run
- lesson: The reversal exists but is untradable at this horizon. Removing earnings sessions and netting out correlated peers left a real effect, positive in all three blocks, with a huge event set, so not a power problem; but it is well under the cost of trading each cohort in and out. It was strongest early, nearly absent in the middle and partly back late, and it grew with the horizon, unlike a fast liquidity concession: a five-session hold may be too short. Other news (M&A, guidance) stays in the sample. Strategy gates were not reached. G0 was vacuous until the Builder altered the flag test (K-0045).
- avoid: Do not refile this sort with another conditioner (more news filters, peer count, leg size, hold of three or seven sessions): K-0044 says conditioning changes significance, not size relative to cost. No daily-rebalanced taker book on it.
- open questions: Does the reversal live at a monthly or slower horizon with much lower turnover? That would be one new trial in the same family, argued from the slow build seen here, not tuned. Whether the peer adjustment or the news exclusion carried the significance was not separated.
- related: H-0007, H-0029, H-0006, H-0028

## H-0031 – Earnings-reaction drift, fixed risk-scaled event slots - LIQ-500, follow large 8-K release-session reactions, no gross normalization, liquid net hedge, hold 5 days

- family: `post-earnings-announcement-drift` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), market-neutral: follow large standardized release-session reactions with fixed risk-scaled slots, net hedge in the 100 most liquid members, five-session hold, 2004 to 2020
- outcome: mechanism test passed clearly again; rejected at G1 on the strategy: net Sharpe below T-bills, bootstrap bound of the difference below zero, exposure above the floor
- lesson: The reaction drift passed again, far above the placebo cloud, above cost and positive in every third, yet the fixed-slot book, invested a bit under half the time, earned less than cash. Un-normalized slots were the main fix proposed after H-0028, so weight normalization is not what separates study from book. Gross covered cost by only a modest multiple, so the hedge, higher costs on names that just moved several deviations and the T-bill hurdle left nothing. The gap is still not decomposed: the gate gives no gross/net or long/short split.
- avoid: The card's own stop rule has fired: no further market-neutral reaction-following book with a new weighting, hedge basket, z_threshold, hold_days or event_weight. Do not read the mechanism pass as a tradable edge.
- open questions: Gate output with gross versus net and long versus short sleeves would show where the return went. The long-only tilt on an always-invested core (H-0033) came much closer, which points at the short side, hedge or cash handling as the main suspects.
- related: H-0028, H-0033, H-0024

## H-0032 – Fundamental anomalies resolved at earnings news - LIQ-500 dollar-neutral, long good-composite / short bad-composite stocks only in a 5-session window around their predicted 8-K release, hold 5 days

- family: `anomalies-at-earnings-news` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), dollar-neutral: long good / short bad accruals-profitability-asset-growth composite, held only in a five-session window around the predicted 8-K release, 2005 to 2020
- outcome: rejected at G1 on the mechanism test: event average above the placebo average but under the placebo 95th-percentile bar, gross below round-trip cost, latest third negative; strategy not run
- lesson: No sign that fundamental anomalies concentrate at earnings news in LIQ-500. The window earned more than the always-on placebo average but stayed below its upper bar, and the bar sat about at cost, so this is not tradable in this form and not just a power problem. It faded era by era and turned negative in the latest third, a declared kill condition that fits decay after publication. The first session was flat and the effect grew with the horizon: the slow shape of an always-on anomaly, not an announcement-day jump. The hit rate of the predicted dates was never reported, so some dilution is possible.
- avoid: Do not retune half_window or leg_quantile, add anomalies, or refile the composite as quality or accruals at earnings: the predicted-date rule is shared with H-0023 and H-0026, so those trials count together. No short-window books on slow fundamentals in LIQ-500.
- open questions: If the composite earns anything it is a slow style tilt at low turnover on an always-invested book, not a window effect. Measuring how often the predicted window contains the real release would settle the dilution question.
- related: H-0023, H-0026, H-0028, H-0019

## H-0033 – Positive earnings-reaction tilt on a cap-weighted core - LIQ-500 long-only, overweight stocks after a large positive 8-K release-session reaction for 5 days, no shorts, no hedge basket

- family: `post-earnings-announcement-drift` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), long-only: cap-weighted core plus an overweight on large positive release-session reactions, five-session hold, vs SPY
- outcome: rejected at G1: mechanism passed, the book beat SPY on the point estimate and the bootstrap bound of the Sharpe difference was above zero, but the margin stayed under the fixed bar
- lesson: Closest the family has come. The positive side alone shows the drift, above the placebo bar, well above cost, positive in all thirds, so H-0028's pass did not live only in the short side. But the effect weakened from each third to the next, and the tilt is a limited share of a core that mostly tracks the market, so the extra risk-adjusted return was too small for the margin. No core-only or random-tilt control is reported, so how much of the difference to SPY is tilt and how much is the liquid core is unknown. A G2 sub-period check would probably have been hard.
- avoid: Do not enlarge the tilt budget, max_name_weight, z_threshold or hold_days to close the margin: new trials on the same events, with more tracking error and cost. Do not bring shorts or a hedge basket back (H-0028, H-0031).
- open questions: A control that separates the tilt from the core, and a cheaper or slower way to take the same long-side drift, deserve a card. The era decay needs to be looked at on its own before more PEAD trials are spent.
- related: H-0028, H-0031, H-0024

## H-0035 – Intraday liquidation-cascade rebound on Binance perps - fade 1-hour moves of 4+ sigma on 3x volume dominated by aggressive takers, both sides, hold 2 hours

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: Binance USD-M perps, top liquid names, 15-minute candles, both sides, two-hour hold
- outcome: passed G1; rejected at G2: net Sharpe at doubled costs was not positive and the last third of dev lost, so one block carried more than all the profit
- lesson: Passed G1 but died at G2 on cost and era. Modeled costs took about half of gross, so doubling them left nothing. The first two thirds of dev earned and the last lost, so one block carried more than the whole profit. Neighbors paid more with larger moves and shorter holds, and a one-candle delay roughly halved the result: the rebound is real but front-loaded, and a late taker pays for the first part. Random-timed books lost heavily, mostly cost drag, so that pass says little about timing. A position leaving the universe mid-hold broke G0; the Builder closed such trades early, a small deviation from the card.
- avoid: Do not refile the rule with the best-looking neighbor (higher threshold, shorter hold): dev-selected, a new trial on the same events. Do not trust modeled slippage at cascade moments.
- open questions: Why did the last third of dev lose: cascades running on in a persistent crash? An era split of the same events would tell; the holdout is untouched. A maker-style entry cannot be simulated here.
- related: H-0006, H-0007, H-0030
