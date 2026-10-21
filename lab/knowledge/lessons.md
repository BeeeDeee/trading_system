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

## H-0036 – Day-of-week mood in speculative stocks - LIQ-500 beta-neutral, long high-idiosyncratic-volatility stocks over the last session of the week, short them over the first session, flat otherwise

- family: `day-of-week-mood` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), beta-neutral: long the high idiosyncratic-volatility fifth over the last session of the week, short it over the first session of the next week, flat otherwise, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: the event study cleared the placebo bar by a wide margin with the right sign, but gross per event stayed just under the round-trip cost; strategy not run
- lesson: The weekday effect on speculative stocks exists in LIQ-500 at the session level: far above the placebo cloud, concentrated in the first session and smaller at longer horizons, as Birru places it. But gross landed just under the round-trip cost, and the first two thirds of dev carried it: the latest third was flat to slightly negative, the card's own decay condition, consistent with a 2018 publication. The cost ratio prices only the speculative leg; the real book also trades a hedge basket, so net would be worse. A near miss on cost plus decay, not an absent mechanism and not a power problem.
- avoid: Do not retune ivol_lookback_days or leg_quantile, add other weekdays, or refile as a Friday or Monday effect on the same sort to close the small cost gap: the latest third says the effect has already faded, and a taker book that turns over twice a week cannot get it back.
- open questions: Only a much cheaper way to take the same side (a small weekday tilt on an always-invested book, or a maker-style entry) could matter, and the decay makes even that unattractive. The weekday pattern of the index itself is a different and weaker claim. The post-publication years were seen only through the latest third.
- related: H-0016, H-0030, H-0029, H-0007

## H-0037 – Twin-ETF premium reversion - market-neutral, fade a large price gap between two US ETFs holding the same basket (HY credit, aggregate bonds, TIPS, EM and developed equity), hold 5 days

- family: `etf-premium-arbitrage` · mechanism: liquidity_provision · instruments: five pairs of US ETFs on the same basket (credit, bonds, TIPS, EM and developed equity), market-neutral twin-gap fade, five-session hold, 2007 to 2020
- outcome: rejected at G1 on the mechanism test: right sign in every third, but under the placebo bar and gross about half of round-trip cost; strategy not run
- lesson: Wrapper gaps close slowly in the right direction, but the gain is small and inside the placebo cloud. The placebo bar sits just under the cost and the gross is well under it, so this is mostly an effect too small to trade, with some power loss from few events on few dates. The spread built up to five sessions and was smaller at ten, so it is not an index-difference drift. It was largest in the earliest third, which includes the 2008 credit stress, and weaker afterwards. Borrow fees and index differences between twins were not modelled; both would lower a pass.
- avoid: Do not retune lookback_days or z_entry, move the cost floor, or add pairs after seeing results. Do not add exact twins on liquid underlyings (the card excluded them: arbitrage there is intraday). Do not call another daily fade of a recent deviation a new mechanism without wrapper evidence.
- open questions: The AP-band story is about price against NAV or intraday premium, which the catalog does not hold; daily closes of two wrappers mostly measure noise common to both. A stress-only version has too few events for the placebo gate (K-0026).
- related: H-0006, H-0030, H-0035

## H-0038 – Treasury auction supply cycle - short TLT in the 3 sessions up to each 5y-30y coupon auction cluster, long TLT in the 3 sessions after it, flat otherwise (SHY as duration control)

- family: `treasury-auction-supply-concession` · mechanism: flow_structural · instruments: TLT short before and long after 5y to 30y Treasury coupon auction clusters, flat otherwise, SHY as duration control, 2002 to 2020
- outcome: rejected at G1 on the event count: zero events in the mechanism test although G0 counted events; untestable, not refuted
- lesson: The test could not tell, and the cause is unknown. G0 passed with a real event count while G1 saw none, which looks like a framework or design mismatch rather than a market result. Candidate causes, unconfirmed: the abnormal-return step needs a peer set and the universe holds only TLT and SHY; the event mask is cut by a tradable or universe field the generic auction dataset lacks; or the first-session-of-short-state event is lost between diagnostic and gate. The Builder had warned its harness showed no events. Nothing is learned about a supply concession.
- avoid: Do not refile with another window_days, extra instruments or a calendar proxy for auction dates (the H-0008 mistake) before the zero-event cause is known. A plain re-run after a fix is a Chair decision on trial counting, not a new idea.
- open questions: Once events register: does TLT lag SHY before the auctions and recover after? Split mid-month (10y, 30y) from end-of-month (5y, 7y) clusters, as the card asked and the Builder did not implement. The dataset shows an auction only about a week ahead, so a longer pre-window cannot be tested.
- related: H-0037, H-0002, H-0008

## H-0039 – Perp leverage-crowding unwind - top-20 Binance perps, dollar-neutral, short the highest and long the lowest trailing funding within past-week-return halves, hold 5 days

- family: `perp-leverage-crowding` · mechanism: flow_structural · instruments: Binance USD-M perps, PIT top twenty, dollar-neutral: short highest and long lowest trailing funding within past-week-return halves, five-day hold, 2020 to 2022 dev
- outcome: rejected at G1 on the mechanism test: price-only event study just cleared the placebo bar, but gross stayed under the round-trip cost; strategy not run
- lesson: The crowding signal has a positive price effect that barely clears the placebo bar, but it is not tradable in this form and it is one episode: the middle third of dev earned many times the first and the latest third lost. The placebo bar lay below cost, so it is not just a power problem. The effect grew with the horizon and was smallest at one session, so it reads as a multi-day unwind, not a one-day reversal. Cautions: many perps sit at the default funding rate, so ties broken by liquidity can make the score a liquidity sort; no volatility-matched control; funding received by the short leg would flatter any strategy.
- avoid: Do not retune funding_days or hold_days, or add the volatility control or a longer hold as a rescue on the same events: new trials on the same data. Do not read a placebo pass as support when the latest block flips sign. Do not rank on funding values tied at the default rate.
- open questions: Does the unwind exist outside the 2021 leverage boom? A per-era report and a tie-free score (funding relative to its own history or to basis) would say; the holdout is untouched. The Builder built its own top-twenty universe because the gate supplied none.
- related: H-0035, H-0006, H-0004

## H-0040 – BTC-led catch-up in lagging altcoins - after a 1-hour BTC jump of 3+ sigma, long each top-volume altcoin that has moved less than half as much, hold 4 hours

- family: `btc-lead-altcoin-lag` · mechanism: cross_asset_information · instruments: Binance USD-M perps, 15-minute candles: long lagging top-volume altcoins after a large one-hour BTC upward jump, four-hour hold, dev 2020 to 2022
- outcome: passed G1; rejected at G2: neighbours too far below the primary, and one dev third carried most of the return
- lesson: Strong at the primary setting and robust where short-horizon crypto rules usually die: it survived doubled costs and a one-candle later fill, and every neighbour stayed positive. It failed on stability. The edge rose with the jump threshold, so the lowest threshold in the grid lost most of it, and one third of dev carried most of the return while the latest third lost. Trades cluster on BTC jump days, so independent episodes are far fewer than the trade count. Earlier owner research saw the same weakness in later years, so the 2023 to 2026 holdout is contaminated and cannot arbitrate.
- avoid: Do not narrow the grid to the higher thresholds, switch the leader to spot BTC or ETH, add the short side, or change the lag ratio or hold after seeing this: dev-selected variants of a rule already tried in dozens of earlier configurations (research 15 to 17) are new trials, not rescues (K-0058). Do not refile it as altcoin beta catch-up or leader-follower under another name.
- open questions: The owner's forward paper test (from 2026-10-01) is the only clean out-of-sample evidence; wait for it. A new card could ask whether the lag lives in particular follower liquidity tiers, or needs a news-driven BTC jump rather than a liquidation cascade (overlaps H-0035). An episode split of the losing block is untested.
- related: H-0035, H-0039, H-0013, H-0006

## H-0041 – Flight-to-safety reversal - after a session where US stocks fall sharply while long Treasuries rally, tilt a 60/40 SPY/IEF book to 100 % SPY for 5 sessions

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: SPY, IEF and TLT daily ETFs 2002-2020: tilt of a stock-bond mix to all SPY for a week after stocks fall and TLT rises
- outcome: rejected at G1 (mechanism): gross above cost but placebo percentile under the bar, bar above cost; latest third negative
- lesson: Near miss, closer to 'cannot tell' than refuted: after sessions where stocks fell and long Treasuries rose, stocks beat Treasuries on average with gross above cost, but under the placebo bar, and the bar itself sat above cost. Random windows also favour equities over Treasuries, so part of the raw number is generic drift. The first two thirds of dev were positive and the latest third negative, a declared kill condition, possibly selloffs that kept going in crash episodes (unverified). The effect also grew with the horizon, so it looks more like slow drift than a quick unwind. The card's key control could not be reported.
- avoid: Do not retune the multiple, hold or TLT condition, or add a VIX or credit conditioner: same events, new trials. Do not refile stress rebounds of equities against bonds under a new name; studies 6 and 7 share the mechanism. Do not mark every universe instrument as an event.
- open questions: Is this only the market bounce after stress of studies 6 and 7, and does the Treasuries-up condition add anything? The control (equity drops with Treasuries falling) could not be reported and needs its own card. A per-episode view (2008, 2020) would show if the latest-third loss is one crash.
- related: H-0003, H-0006, H-0030, H-0035, H-0038

## H-0042 – Macro-announcement beta premium - shift IEF into the 3 highest-beta ETFs over scheduled CPI, payrolls and FOMC days, 2 sessions

- family: `macro-announcement-premium` · mechanism: risk_premium · instruments: ten US ETFs: high-beta against low-beta over two sessions around CPI, payrolls and FOMC days, 2002 to 2020
- outcome: rejected at G1 on the mechanism test: right sign and positive in every third of dev, but under the placebo upper bar, and gross per event was below the round-trip cost; strategy not run
- lesson: The announcement-day beta spread across ETFs is positive in every third of dev but sits inside the cloud of time-shifted placebo windows of the same ETFs, and gross is below round-trip cost. The placebo upper bar is itself below cost, so this is an effect too small to trade in this form, not a power problem. The placebo average was positive too (equities beat bonds and gold in any window), so part of the raw gain is generic drift. The gain shows at the two-session horizon only and is gone at three and five sessions: it sits on the announcement session and is not kept. Weakest in the latest third. The tilt never ran.
- avoid: Do not retune tilt, beta_lookback or the window, and do not split the same events into CPI, payrolls and FOMC subsets after seeing this: new trials on the same dates. Do not refile as pre-FOMC drift or announcement-day beta on these ten ETFs.
- open questions: A beta sort inside equities on announcement days removes the equity-versus-bond drift, but the month-turn beta spread faded (K-0038): state the latest-third sign as a kill condition. A split by release type could not be reported by the gate.
- related: H-0016, H-0025, H-0023, H-0041

## H-0043 – Credit-ETF gap to its bond index - tilt the IEF sleeve of a 60/40 book into HYG/JNK/LQD for 5 days after the ETF falls well below what its ICE BofA OAS and Treasury yields imply

- family: `etf-premium-arbitrage` · mechanism: liquidity_provision · instruments: HYG, JNK and LQD tilted out of the IEF sleeve of a SPY/IEF mix after the ETF falls below what ICE BofA spreads and Treasury yields imply; SPY, IEF, TLT and SHY as unmarked peers; 2003 to 2020
- outcome: rejected at G1 on the event count: zero events in the mechanism test and already zero at G0, so no mechanism result and no strategy run. Untestable, not refuted.
- lesson: The test could not tell, and a zero here looks like a data or pipeline fault, not a market result. A two-sigma residual gap should fire regularly in any healthy series over many years, so an empty set points at the inputs: the lab's copy of the ICE BofA spread series may start late or be thin, the valid-day rule (all four FRED series present and changed) may drop most days, or gaps may blank the 252-session regression window. The Builder could not enforce the card's coverage check at G0 and asked for verification. Nothing is learned about credit-ETF gaps to bond value.
- avoid: Do not refile with other z_entry or gap_days values: the empty set does not depend on them, and a lower threshold cannot fix missing data. Do not resubmit before the cause is known; a plain rerun after a data fix is a Chair decision on trial counting. Do not count this as evidence against H-0037's open question.
- open questions: Once the spread series are confirmed to cover the period, the card's question stands: does the wrapper's excess move against bond value revert, or does the ETF lead the evaluated index (the death the card predicted)? A cheap event-count and coverage check by year before filing would avoid another empty run.
- related: H-0037, H-0038, H-0021, H-0041

## H-0045 – FOMC-cycle equity tilt - 60/40 SPY/IEF book overweight SPY in even weeks and underweight in odd weeks of FOMC cycle time, weekly switches

- family: `macro-announcement-premium` · mechanism: risk_premium · instruments: SPY against IEF in a 60/40 book, overweight SPY in even weeks and underweight in odd weeks of FOMC cycle time; TLT, SHY, LQD as unmarked peers; daily ETFs 2002 to 2020
- outcome: rejected at G1: gross covered cost, but the placebo percentile was just under the bar, which itself lies above cost; latest third negative; strategy not run
- lesson: No reliable FOMC-cycle premium of SPY over bonds in 2002-2020. Marking both even weeks (long) and odd weeks (short) left the placebo mean near zero, so this is not generic equity drift. Gross cleared cost but stayed just under a placebo bar that itself lies above cost: closer to 'cannot tell' than refuted. Against it: nothing at one session (so it is not the H-0042 announcement day), gone at ten, the first two thirds carried it and the latest third, after publication, reversed, and doubled costs would very likely have erased it.
- avoid: Do not retune tilt, offset or the week definition, split the same dates by cycle week or by FOMC meeting, or refile as pre-FOMC drift, Fed-information premium or even-week effect: same dates, new trials that count under macro-announcement-premium. Do not read the positive gross as support without the placebo bar and the latest-third sign.
- open questions: The paper's effect may sit in the equity premium itself, not against Treasuries: an equity-only even-versus-odd test is a different card. An era split of the latest third is untested; ETFs have no forward data, so G4 is the ceiling.
- related: H-0042, H-0041, H-0025

## H-0046 – Threshold-rebalancing pressure - when a simulated 60/40 SPY/IEF book drifts past its band, tilt a 60/40 book against the rebalancers' next-session trade for one session

- family: `rebalancing-flow-pressure` · mechanism: flow_structural · instruments: a 60/40 SPY/IEF book tilted for one session against a simulated threshold rebalancer; event study on SPY, VTI, DIA and IWM against bond ETF peers; daily ETFs 2003 to 2020
- outcome: rejected at G1: mechanism test passed clearly (above placebo, gross several times cost), but the tilt book's Sharpe was below the 60/40 benchmark and the bootstrap bound was below zero
- lesson: Strongest placebo result in the lab and gross well above cost, yet an always-invested SPY/IEF tilt did not beat the plain mix, so thin exposure, shorts, hedges or cash are not the whole gap between study and book. Differences: four equity ETFs against four bond peers versus SPY and IEF only; the effect built over about five sessions and was gone by ten, so the fixed one-session hold caught part; it shrank each third. The generic reversal of the stock-minus-bond spread after large relative moves is unseparated. Benchmark-relative return, drawdown and Sharpe disagreed oddly (asked owner).
- avoid: Do not refile with a longer hold, another threshold_pp or tilt, other ETFs, or the same breach days under a new name (band breach, pension rebalancing): same events, new trials. Do not read the mechanism pass as a tradable edge or as proof that rebalancers cause it.
- open questions: A control card: large relative stock-bond moves that do not breach the band, and bond-driven versus equity-driven breaches, to separate rebalancing from generic spread reversal. An equity-only or futures test may fit the paper better.
- related: H-0002, H-0041, H-0044, H-0033, H-0025, H-0045

## H-0047 – Post-macro-release drift across asset classes - on CPI, payrolls and FOMC days, long the ETFs with the largest vol-standardized reaction and short the smallest, hold 5 sessions, dollar-neutral

- family: `post-macro-announcement-drift` · mechanism: cross_asset_information · instruments: 19 US asset-class ETFs, long/short by release-day reaction on CPI, payrolls and FOMC days, five-session hold, 2002 to 2020
- outcome: rejected at G1 on the mechanism test: abnormal return after the release-day reaction had the wrong sign, below the placebo cloud, negative in every third of dev; strategy not run
- lesson: The continuation claim failed in the wrong direction. Ranked by release-day reaction, the extreme ETFs did slightly worse than their unmarked peers, with the same sign in every third (no decay story), nothing at the first session and growing out to ten. Not a power problem: both sides were marked, so the placebo mean was near zero, and the placebo upper bar was at about round-trip cost. Slow-capital drift after macro news is not visible across asset-class ETFs, and the stock-level reaction drift (K-0046) does not carry over. The reversal is of the order of cost, so a mirror trade would roughly break even before any real friction.
- avoid: Do not flip the sign and refile as macro-release reversal or overreaction: a dev-selected variant on the same events, a new trial of this family. Do not retune top_k or hold_days, split by CPI, payrolls or FOMC, or swap ETFs.
- open questions: Release days are not separated from any large day: a reversal may be generic ETF behaviour (H-0037, H-0041, K-0056) and the calendar may add nothing. A non-release-day control would tell.
- related: H-0042, H-0045, H-0041, H-0037, H-0028, H-0033

## H-0048 – Month-end Treasury index-extension tilt - 60/40 SPY/IEF book that holds its bond sleeve in TLT instead of IEF over the last 2 sessions of each month

- family: `bond-index-extension-flow` · mechanism: flow_structural · instruments: 60/40 SPY/IEF book with the bond sleeve held in TLT over the last two sessions of each month; SHY, LQD, AGG, TIP as unmarked peers; daily ETFs 2002 to 2020
- outcome: rejected at G1: placebo bar cleared narrowly and positive in every third, but gross per event was about half of round-trip cost; strategy not run
- lesson: The month-end duration concession shows up in TLT minus IEF but is too small to trade. It cleared the placebo bar narrowly, with a placebo mean near zero (not generic duration drift), stayed positive in every third while shrinking toward the latest, and sat on the last session with a give-back at five and ten sessions, as the paper predicts. Gross was about half the round-trip cost and the placebo bar lies below cost, so this is not a power problem; gross also fell well short of the card's own forecast. A long-minus-belly pair probably cancels the common part of the concession, which IEF already earns.
- avoid: Do not retune window_sessions (the gain is bunched at the last session, the round trip is paid either way), swap in other long ETFs, or refile as index extension or month-end duration demand: same dates, new trials. No post-window short leg after seeing the give-back.
- open questions: Is the concession mainly in holding duration at all over the window (duration against bills), not long against belly? That changes average duration against the benchmark (K-0005) and needs its own exposure argument. ETFs have no forward data; G4 is the ceiling.
- related: H-0001, H-0002, H-0038, H-0046, H-0016

## H-0049 – Flushed vs unflushed crashes in crypto perps - after a large bottom-tercile daily drop, long coins whose perp funding collapsed, short coins whose funding held, BTC/ETH hedged, hold 3 days

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: Binance USD-M perps, top thirty by volume ex BTC/ETH: after big daily drops, long coins with collapsing funding, short coins with steady funding, BTC/ETH hedge, 2020 to 2022
- outcome: rejected at G1: code says event count (just under the minimum), but the mechanism was measured and failed: wrong-signed, inside the placebo cloud, cost not covered, latest third clearly negative
- lesson: Funding falling on a crash day did not separate rebounds from continuation. Pooled, flushed coins did slightly worse than unflushed ones: nothing at one day, wrong-signed at three and five. The first two thirds of dev leaned right but inside noise; the latest third (2022 bear market) held about half the events and was clearly wrong-signed, the card's own kill condition; negative funding there may mark persistent selling, not a one-off flush (unverified). The code says too few events, but the mechanism was measured and absent: more events would not fix the sign.
- avoid: Do not lower k_sigma to reach the event minimum, retune hold_days or the funding comparisons, or flip the sides (long unflushed, short flushed): a mirror trade chosen from a wrong-signed bear-market block is a dev-selected new trial. Do not refile as funding-conditioned dip-buy, liquidation-flush rebound or leverage-flush reversal: with H-0006 it is the same family and idea.
- open questions: The by-side result (flushed longs and unflushed shorts, each against unmarked peers) was not reported, so which leg failed is unknown. Whether any crash reversal exists in perps outside a bear regime cannot be told from one three-year dev window with a single boom-and-bust cycle. K-0049 suggests any payoff goes to those providing liquidity at the stress moment, not to a daily taker.
- related: H-0006, H-0039, H-0035, H-0040

## H-0050 – Volatility-control re-leveraging flow - tilt a 60/40 SPY/IEF book to all SPY for 2 sessions when an old large S&P day is about to leave the 20/60-session realized-vol windows

- family: `rebalancing-flow-pressure` · mechanism: flow_structural · instruments: SPY, IVV, VTI and DIA marked against IEF, TLT, SHY and AGG as unmarked peers; the unrun book was a 60/40 SPY/IEF mix tilted to all SPY for two sessions; daily ETFs 2003 to 2020
- outcome: rejected at G1: coded as event count, but the mechanism was also measured and failed (wrong-signed, inside the placebo cloud, latest third negative)
- lesson: No sign of a vol-control roll-off flow in daily ETFs. After an old large S&P day was due to leave the realized-vol windows, SPY did not beat the bond ETFs over two sessions: the average was slightly wrong-signed and well inside the placebo cloud, only the middle third was positive, and the latest third (most vol-control AUM) was negative, the card's own kill condition. The trigger fired on far fewer dates than forecast, so the test was thin and its placebo bar sat above the effect the book needed; but sign, horizon shape and era give no reason to expect a hidden effect.
- avoid: Do not lower the threshold, add EWMA or other vol-window variants, or change hold_days to raise the event count: same trigger, new trials. Do not flip the sign. Do not refile as vol-targeting or risk-parity deleveraging flow with the same SPY/IEF tilt (H-0041, H-0046 share it). State expected event dates and marked events separately.
- open questions: Rule-mandated flows may exist where they trade (index futures, closing auction), which daily ETF bars cannot show. A deleveraging trigger, when a large day enters the window, is a separate claim, confounded with stress rebounds (K-0056), low prior.
- related: H-0046, H-0041, H-0002, H-0003, H-0049

## H-0052 – Macro-release volatility crush - short the VIX-futures ETP (VIXY) over the open-to-open step containing each scheduled CPI, payrolls or FOMC release plus one step, bond and gold ETFs as unmarked peers

- family: `macro-announcement-premium` · mechanism: volatility · instruments: VIXY (short-term VIX-futures ETP) short over each scheduled CPI, payrolls and FOMC release step; SHY, IEF, TLT, AGG and GLD as unmarked peers; daily ETFs 2011 to 2020
- outcome: rejected at G1 (mechanism): positive abnormal return, gross several times cost, but placebo percentile under the bar; strategy not run
- lesson: Release timing added little to ordinary short-VIXY carry. The event average was positive, but time-shifted placebo windows, which carry the same roll-down, already earned over half of it, so the event sat inside the placebo cloud. Gross was well above cost, yet the placebo bar was above cost too, so a coin-flip near miss (K-0034), not a refutation: VIXY's daily noise makes a few hundred events a weak test. Against the story: the first third of dev was about flat, the middle third carried most, and the effect kept growing with the horizon, the shape of roll-down drift, not of a release-time crush. The latest third stayed positive but smaller than the middle.
- avoid: Do not retune post_steps, split the dates by release type, swap in another VIX ETP or add a volatility conditioner: same dates, new trials. Do not read a positive gross without the placebo mean (K-0025). Jump risk and borrow cost of a short-volatility book are untested.
- open questions: A VIX ETP bundles the release effect with a structural roll cost that a placebo on the same instrument cannot fully net out. Implied volatility measured directly (index options, futures curve) would isolate the claim but is not loadable.
- related: H-0042, H-0045, H-0047

## H-0053 – Return seasonality net of seasonal reversal - LIQ-500 dollar-neutral, monthly: long same-month-minus-other-month winners, short losers, hold 1 month

- family: `return-seasonality` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), monthly dollar-neutral: long strong same-calendar-month history, short weak, one-month hold
- outcome: rejected at G1 on the mechanism test: abnormal return after entry was wrong-signed in every third, far under the placebo bar; strategy not run
- lesson: The seasonal claim is absent in the most liquid US stocks, not just too small: stocks with the strongest same-month history did slightly worse than unmarked peers over the month, with the wrong sign in every third, the pre-publication years included, so this is not decay. The event count was very large and the placebo mean sat near zero as designed, so stock identity and power are not the story. Only the shortest horizons leaned faintly positive, far under round-trip cost. Builder deviations (holiday month-end handling, 'has a return' read as listed and tradable) are too small to flip a sign.
- avoid: Do not retune years_back or leg_quantile, restrict to January or single months, or flip the sign: a mirror trade chosen from a wrong-signed result is a dev-selected new trial. Do not refile quarterly or dividend-month variants as seasonality; they share mechanisms with the earnings and dividend families and count together.
- open questions: If seasonality exists it probably lives where the literature finds it strongest: smaller, less covered stocks (not loadable now, K-0003), or in aggregates such as asset-class ETFs (untested). The earnings-month component could be tested with real release dates, but K-0039 gives that a low prior in LIQ-500.
- related: H-0010, H-0023, H-0015, H-0054

## H-0054 – Residual momentum, beta-neutral - LIQ-500 monthly, long top / short bottom quintile of vol-scaled market-adjusted 12-2 month residual return

- family: `cross-sectional-momentum` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), monthly market-neutral: entries into top and bottom quintile of residual momentum, three-month horizon
- outcome: rejected at G1: right sign, gross well above cost, positive in all thirds, but placebo percentile under the bar; strategy not run
- lesson: Near miss, closer to 'cannot tell' than refuted. Entries into the residual-momentum legs earned a positive abnormal return over the long horizon, gross several times round-trip cost, positive in every third, placebo mean near zero, growing with the horizon (slow drift, not a quick reversal). The percentile stayed under the bar, which sits far above cost: monthly dates with overlapping long windows make the placebo cloud wide (K-0034). The earliest third carried most and the later two were small (K-0060). The event study uses no beta-neutral sizing, so the card's distinctive claim, beta removal, was never tested.
- avoid: Do not retune formation_months or leg_quantile, add a volatility or size conditioner, or move to the horizon that looks larger: same entries, new trials. Do not refile as total-return, twelve-month or industry momentum under a new name; the family carries the momentum trials of study 1 and the calibration run.
- open questions: If the drift is real it is a slow style premium, better argued as a low-turnover tilt on an always-invested book against a same-exposure benchmark (K-0027). A by-side report and an era split were not reported. The card itself expected the absolute Sharpe bar to be the likely strategy death.
- related: H-0029, H-0011, H-0033, H-0007

## H-0055 – Variance-risk-premium equity tilt - 60/40 SPY/IEF book, more SPY when VIX-implied variance is high relative to SPY realized variance, weekly

- family: `variance-risk-premium-timing` · mechanism: risk_premium · instruments: SPY against IEF, TLT, SHY, AGG peers; unrun book: weekly SPY/IEF tilt on VIX-implied minus realized variance, 2003 to 2020
- outcome: rejected at G1: right sign, gross well above cost, positive in all thirds, but placebo percentile under the bar; strategy not run
- lesson: Right direction, closer to 'cannot tell' than refuted. Marking both high and low premium weeks, the stock-over-bond spread was positive, gross several times cost, positive in every third including the latest, and it grew with the horizon, so it is not the quick stress bounce of K-0056. It stayed under a bar far above cost: weekly signals persist and long windows overlap, so independent episodes are few and the cloud is wide (K-0026). The middle third carried most. The placebo mean sat below zero, so generic drift did not fully cancel, and no by-side split was reported. The card's own estimate put the Sharpe gain under the required margin unless the effect was at the top of its range.
- avoid: Do not retune tilt, rank_days or the quintile cuts, or swap in realized-variance-only or VIX-level rules (H-0003 and studies 6 and 7 cover them). Do not add a trend or credit conditioner. Do not file a card whose own estimate of the gain on the traded pair is under the required margin without saying why the real effect should be larger.
- open questions: Which side earns it, the overweight after high premium or the underweight after low premium, and does it survive outside the post-crisis recovery? That needs a by-side and an era report. More equity-bond pairs add events but not independent episodes. Implied variance measured directly (options, futures) is not loadable. ETFs have no forward data, so G4 is the ceiling.
- related: H-0003, H-0041, H-0045, H-0046

## H-0056 – Oil-specific shock underreaction - weekly tilt of a 60/40 SPY/IEF book against the past month's WTI move not explained by the stock market, hold ~1 month

- family: `oil-shock-equity-underreaction` · mechanism: cross_asset_information · instruments: SPY against IEF, TLT, AGG, SHY peers; unrun book: weekly 60/40 SPY/IEF tilt on WTI shock net of stocks
- outcome: rejected at G1: mechanism wrong-signed, lower tail of the placebo cloud, same sign in all thirds; strategy not run
- lesson: The underreaction direction failed outright: after a large oil-specific rise SPY did not lag Treasuries, it led them, with the same sign in every third and a gap that widened with the horizon. The placebo mean was near zero (both signs marked), so this is neither generic drift nor a power problem, and the card's own latest-third condition failed too. The oil residual left after removing the stock move may mark reflation or risk-on news that equities keep pricing, not neglected news. Events cluster in few oil episodes, so a reliable reverse effect is not shown either.
- avoid: Do not flip the sign and refile as oil-shock momentum or a reflation tilt: a mirror chosen from a wrong-signed dev result is a new trial of this family. Do not retune lookback_days or tilt_per_z, change the residualization window, or swap WTI for Brent or oil futures. Do not refile other commodity-to-equity lead-lag (gold, copper, gas) on SPY/IEF without saying why the channel differs.
- open questions: Does the sign depend on why oil moved (inflation news versus demand), or on the side (rises versus falls)? The gate reports no by-side or by-cause split, so this is unknown. Sector-level responses (energy producers versus oil consumers) are untested. The published effect is reported to have weakened after 2008, so it may simply be gone in the post-publication years.
- related: H-0013, H-0040, H-0047, H-0053, H-0055, H-0041

## H-0057 – Hedging-pressure premium - CFTC positioning into commodity and currency ETFs, monthly dollar-neutral, long unusual speculator net long, short the mirror

- family: `hedging-pressure-premium` · mechanism: risk_premium · instruments: commodity, metal and G10 currency ETFs, monthly dollar-neutral on CFTC speculator positioning against its own history, 2007 to 2020
- outcome: rejected at G1 on the mechanism test: average about zero, mid placebo cloud, below cost; only the earliest third positive; strategy not run
- lesson: No monthly hedging-pressure premium shows when the positioning z-score is traded through ETFs. The event set was large and the placebo upper bar far above round-trip cost, so the effect the card forecast would have shown; the average was about zero with no consistent shape across horizons. Only the earliest third (few ETFs, commodity boom and bust) was clearly positive, the latest negative: the decay of K-0060. Likely dilution: futures-holding funds carry heavy roll and tracking noise, and physical metal and currency-deposit trusts do not earn the futures roll that is part of the premium. This does not refute the premium in futures themselves.
- avoid: Do not retune z_window_weeks, avg_reports or the return-half split, split commodities from currencies after seeing this, or flip the sign: a mirror trade chosen from a weak or wrong-signed block is a new trial. Do not refile the same CFTC level under another name (COT extremes, speculator positioning, insurance premium).
- open questions: The CFTC data now loads, so the weekly-change card H-0051 (opposite sign, short horizon, same data and almost the same ETFs) can be judged on its own; if filed, say why its trials count apart from this one. Testing positioning against futures returns rather than ETF returns is not possible with today's loaders.
- related: H-0051, H-0060

## H-0058 – Token vesting overhang on Binance perps - beta-neutral, short coins aged 30-365 days since listing, long coins listed over 2 years, weekly

- family: `new-listing-overhang` · mechanism: flow_structural · instruments: Binance USD-M perps, point-in-time top thirty: young listings as the short-side event set, other members as peers; weekly, 2020 to 2022
- outcome: rejected at G1 on the mechanism test: wrong sign, low in the placebo cloud; young coins led in the first two thirds and lagged only in the latest
- lesson: The overhang story was not visible: in the liquid top thirty, young coins did not lag seasoned ones; they led through the boom years and lagged only in the 2022 bear market. That fits a higher beta to the market direction better than vesting supply, and the G1 event study is raw, so the card's beta-neutral sizing, its distinguishing control, was never tested. The liquidity filter also admits only young coins that already drew volume, which favours winners. The placebo cloud was very wide for a quarter-long altcoin horizon. The marked count per date looked far above the forecast: possible misread of listing age (question to the owner).
- avoid: Do not resubmit with other min_age_days, young_max_days or leg sizes: a further trial of this family on the same few years. Do not flip to long young coins; the sign comes from one boom. Do not treat the top-N liquidity filter as a control for young-coin selection.
- open questions: A beta- and size-matched comparison of young against seasoned coins, plus a per-era report, would separate overhang from beta. Listing age from token launch or unlock data is not loadable. The dev window holds a single boom and bust (K-0006), so forward or holdout data is the next real evidence.
- related: H-0004, H-0021, H-0039

## H-0059 – R&D-increase underreaction - LIQ-500 long-only, hold for 12 months stocks whose trailing R&D spending and R&D intensity just rose sharply, monthly rebalance

- family: `rd-investment-underreaction` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), long-only: stocks whose R&D and R&D intensity just rose sharply, held for months, monthly, 1998 to 2020
- outcome: rejected at G1 on the mechanism test: positive and several times cost in every third, but equal to same-stock placebo windows; strategy not run
- lesson: The timing of the R&D increase adds nothing visible. The positive number belongs to which stocks are picked, not when: R&D-intensive stocks beat the other liquid names in almost any window of 1999 to 2020, so placebo windows of the same stocks earned nearly the same (K-0025). The placebo cloud was comparatively narrow, so this is not a power failure for an effect of the claimed size. The sign held in every third and grew with the horizon, so what exists looks like a slow style premium of R&D-heavy growth names, which a same-stock placebo removes by design. The test sees only the first quarter of a twelve-month drift claim.
- avoid: Do not retune growth_min, hold_months or the intensity floor, or refile as intangible investment, innovation or capitalized-R&D underreaction: same stocks, same events. Do not read the positive raw gross as support for the mechanism.
- open questions: If the object is the style premium of R&D-intensive stocks, it is a different card: a low-turnover tilt on an always-invested cap-weighted core judged against SPY with a same-exposure control (K-0005, K-0027), stating the growth-era and high-beta confound. A matched control of R&D-intensive stocks without an increase would separate timing from level.
- related: H-0032, H-0054, H-0033

## H-0060 – Cross-country long-term reversal - 21 country equity ETFs, long the 5 worst and short the 5 best 48-month returns skipping the last year, monthly

- family: `long-term-reversal` · mechanism: behavioral · instruments: SPY and iShares single-country equity ETFs, monthly long the worst and short the best multi-year performers with the last year skipped, 2003 to 2020
- outcome: rejected at G1 on the mechanism test: the average was slightly wrong-signed, low in the placebo cloud and below cost; wrong sign in the first two thirds, about zero in the latest; strategy not run
- lesson: No multi-year country reversal in this ETF set: long-term losers did not beat long-term winners over the following quarter, and both sides were marked, so generic drift is netted out. The placebo bar lay well above cost and its mean was below zero, so a modest effect would have been tested. Power for the real claim is limited, though: a four-year formation window leaves few independent cycles in eighteen years, and the test sees only one quarter of a drift claimed over years. Persistent US outperformance and the emerging-market cycle may have kept the legs in place (unverified). USD country ETFs also carry currency, so equity overshooting is not isolated.
- avoid: Do not retune formation_months, leg_size or the skip, add countries after the fact, or flip to country momentum because the sign was wrong: a mirror chosen from dev is a new trial, and momentum belongs to the cross-sectional-momentum family.
- open questions: Currency alone behaved differently (H-0062); whether the equity or the currency part drives country ETF returns is not separated, and no by-leg report was produced. G1 cannot test horizons beyond one quarter.
- related: H-0062, H-0054, H-0044

## H-0061 – Default-spread business-conditions tilt - 60/40 SPY/IEF book, more SPY when the Moody's Baa-Aaa spread is high against its own 5-year history, less when low, monthly, tilt persists for months

- family: `default-spread-risk-premium-timing` · mechanism: risk_premium · instruments: SPY marked against Treasury and aggregate-bond ETF peers on the Baa-Aaa spread versus its own history; unrun book: 60/40 SPY/IEF tilt, 2003 to 2020
- outcome: rejected at G1 on the mechanism test: slightly wrong-signed, low in the placebo cloud; the first third wrong-signed, later two small; strategy not run
- lesson: No default-spread premium appeared. When the spread was wide against its own history, SPY lagged Treasuries in the first third of dev, which holds the financial crisis: spreads widened and equities kept falling, the persistent-crash risk the card named. The later thirds were right-signed but small, well inside the placebo cloud, and nothing grew with the horizon. A wrong sign is not a near miss. Within one history, high compensation in bad states cannot be separated from persistent crashes. The event count fell short of the forecast, so signal history may not have warmed up the first years (question to the owner).
- avoid: Do not retune window_days or tilt, add a trend or volatility conditioner to avoid crashes (H-0003, H-0055, studies 2 and 9), or flip the sign. Do not refile term-spread, credit-spread or yield-curve timing of SPY against IEF under a new family: same events, same pair.
- open questions: More independent bets (several equity markets or sectors against bonds) would add cross-sections but not crises. The spread could be tested as a crash marker (drawdown reduction) rather than as a premium; that is a different claim.
- related: H-0055, H-0003, H-0041, H-0056

## H-0062 – G10 currency value (PPP reversion) - FRED spot vs its 5-year average, long 2 cheapest / short 2 richest currency ETFs, monthly

- family: `long-term-reversal` · mechanism: risk_premium · instruments: six G10 currency trust ETFs, monthly dollar-neutral: long two cheapest, short two richest against own five-year average spot, 2007 to 2020
- outcome: mechanism test passed; rejected at G1 on the absolute net Sharpe floor, though it beat T-bills with a reliable bootstrap bound
- lesson: First slow-horizon card to clear the placebo bar clearly: cheap currencies beat rich ones, building with the horizon, positive in the populated thirds, gross many times cost, placebo mean near zero. The low-volatility book beat T-bills reliably but missed the absolute Sharpe floor, as the card predicted. Caveats: the warm-up seems not fed by pre-ETF history, so the effective sample is about nine years, the first third is nearly empty and the publication check has little earlier data; carry is mixed in (rich currencies were low-yield); the score is nominal. Real in this sample, not established.
- avoid: Do not tune lookback_years or n_per_leg, add currencies, or lever or vol-target the book to clear the Sharpe floor: dev-selected rescues are new trials. Do not file a value score on the same six trusts under another name.
- open questions: Separate value from carry (value within carry-matched currencies, or inflation-adjusted anchors); more currencies or futures would add breadth but need data. Check what history the signal really received (question to the owner). ETFs have no forward data, so G4 is the ceiling.
- related: H-0060, H-0044, H-0051

## H-0063 – Stop-cascade breakout from compressed ranges on Binance perps - follow the first taker-dominated, high-volume 15m close outside an unusually narrow 24h range, both sides, hold 6 hours

- family: `stop-cascade-breakout` · mechanism: flow_structural · instruments: Binance perps top fifteen, 15-minute bars, long/short breakouts from compressed ranges, six-hour hold
- outcome: passed G1 narrowly; rejected at G2: neighbors unstable, one-candle delay erased it, doubled costs made it negative
- lesson: Gross per trade was only just above the card's own bar and modeled costs took most of it, so G1 was a thin pass. G2 shows the continuation is front-loaded: one candle later flipped the sign, which is the card's own condition for 'the card is wrong' (the H-0035 shape, now on the continuation side). The short-range neighbors lost money. All three dev thirds were positive, but the latest was clearly the weakest. A small fast edge that a taker reaches too late.
- avoid: Do not refile the better-looking wider-range or longer-hold neighbors, or tighter flow thresholds: dev-selected, a new trial on the same events. Do not relabel it as volatility squeeze, range expansion, Donchian or liquidation-follow breakout on the same 15-minute perps. Do not rely on a modeled next-open fill for a move defined by taker aggression.
- open questions: Does the continuation exist at a slower horizon (many hours to days) where a late fill matters less, or only for fast or maker fills that this engine cannot simulate? Untested: an episode split of the weak latest block, and whether market-wide state (all coins breaking at once) separates real runs from single-coin noise.
- related: H-0035, H-0040, H-0039, H-0006

## H-0064 – Pre-announcement liquidity-provision reversal - LIQ-500, long 3-session losers / short winners just before their predicted 8-K earnings release, hold 5 sessions

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: US liquid stocks (LIQ-500), unhedged long/short: three-session losers long, winners short, just before predicted 8-K release, five-session hold
- outcome: mechanism test passed clearly; rejected at G1 on the strategy: Sharpe beat T-bills reliably but stayed under the absolute floor, deep drawdown; latest third wrong-signed
- lesson: First conditioner on a return sort in LIQ-500 that lifted event gross clearly above cost (contrast H-0030, K-0044). But it decayed: strongest in the earliest third (financial-crisis rebounds), weaker in the middle, wrong-signed in the latest, a declared kill condition. It kept building from one to ten sessions, not a quick concession. The unhedged book earned far less than the event study implies, with a deep drawdown for a neutral book; the gap is undecomposed (no gross/net or sleeve split).
- avoid: Do not retune formation_days, the lead, leg_quantile or hold, nor add a hedge basket or long-only leg to reach the Sharpe floor: same events, new trials. Do not read the pooled pass as live while the latest third is negative (K-0060).
- open questions: Is the decay an episode (crisis rebounds in the first third) or publication? An episode split of the same events would tell. A control with the same sort at dates away from the predicted release would show whether the window enlarges reversal. The hit rate of the predicted date is unreported.
- related: H-0030, H-0007, H-0023, H-0026, H-0028, H-0029

## H-0065 – Quarter-end window-dressing cycle - LIQ-500 dollar-neutral, follow residual QTD winners minus losers in the last 10 sessions of a quarter, fade them in the first 5 of the next

- family: `quarter-end-window-dressing` · mechanism: calendar · instruments: US liquid stocks (LIQ-500), dollar-neutral: follow residual quarter-to-date winners before quarter end, fade them after
- outcome: rejected at G1 on the mechanism test: average about zero, placebo percentile far under the bar, gross a small fraction of cost; strategy not run
- lesson: No quarter-end window-dressing cycle visible in LIQ-500. Not a power problem: many events, placebo bar below round-trip cost, gross a small part of cost. Signs flipped between thirds (wrong, right, wrong) and longer horizons were no better. Caveat: the follow and fade entries sort on nearly the same residual with opposite sides, so generic momentum or reversal cancels, but so would two real effects of opposite sign; the by-phase report asked for never reached the gate. Only about two event dates per quarter: few independent episodes.
- avoid: Do not retune pre_days, post_days or leg_quantile, or keep only the fourth quarter after seeing the thirds: new trials on the same few dates. Do not refile as quarter-end momentum or turn-of-quarter reversal on the same sort.
- open questions: A phase-by-phase event study (pressure and release separately, placebo mean checked) would show whether one phase hides in the pool. Fourth-quarter tax-loss selling alone is narrower and has very few dates. Holdings evidence (13F) is not loadable for enough quarters.
- related: H-0016, H-0020, H-0054, H-0030, H-0029

## H-0066 – Earnings-news transfer to not-yet-reporting peers - LIQ-500 long/short, follow a large 8-K release reaction in the announcer's correlated peers, hold 5 days

- family: `peer-lead-lag-diffusion` · mechanism: cross_asset_information · instruments: US liquid stocks (LIQ-500), long/short: follow a large 8-K release reaction in correlated peers that have not reported, five-day hold
- outcome: rejected at G1 on the mechanism test: small positive average, placebo percentile well under the bar, gross a fraction of cost; strategy not run
- lesson: A dated, verified news event did not rescue peer diffusion. Peers moved slightly with the announcer at short horizons, but the average sat inside the placebo cloud, the bar was below round-trip cost (a tradable effect would have shown) and longer horizons turned negative: noise or overreaction, not slow transfer. Thirds ran from wrong to right, the reverse of decay; early 8-K coverage gaps may let already-reported peers in (K-0035), untested. The peer map is only a return correlation. With H-0013, undated and dated versions are both flat.
- avoid: Do not retune z_threshold, peers_k or hold_days, nor trade the mirror (overreaction) because longer horizons are negative: dev-selected sign (K-0072). No further correlation-peer cards on LIQ-500.
- open questions: Explicit economic links (suppliers, customers, industry codes) are the untested channel but not loadable. Less-covered names are untested. 8-K coverage by year would show whether early dilution hid anything.
- related: H-0013, H-0028, H-0047, H-0056, H-0067

## H-0067 – Earnings-release volume visibility premium - LIQ-500 dollar-neutral, long announcers with abnormally high release-session volume, short abnormally low, matched on reaction sign and size, hold 5 days

- family: `high-volume-visibility-premium` · mechanism: behavioral · instruments: US liquid stocks (LIQ-500), dollar-neutral: among announcers, long high and short low release-session volume, matched on reaction
- outcome: rejected at G1 on the mechanism test: slightly negative average, placebo percentile far under the bar, no third positive; strategy not run
- lesson: Answers the question H-0014 left open: abnormal volume tied to a real release session, inside matched reaction bins, gives no visibility premium in LIQ-500. The average was wrong-signed in every third, worst in the latest, and below the same-stock placebo mean. The placebo bar was below round-trip cost, so not a power problem. Very short horizons were about zero. Large caps are always visible; release volume probably measures news size and disagreement that matching only partly removes. The negative sign is dev-selected.
- avoid: Do not trade the mirror (low release volume over high) or retune leg_quantile, hold_days or the bins. No more volume-based attention stories on LIQ-500: with H-0009 and H-0014 this is the third dead one.
- open questions: A volume effect, if any, lives in less-covered names or at slower horizons, neither tested. Volume as a conditioner of reaction drift is the same data as the PEAD family and would count there.
- related: H-0014, H-0009, H-0028, H-0031, H-0024

## H-0068 – Post-release attention unwind in past winners - LIQ-500 market-neutral, short 12-month winners for 5 days after a real 8-K release with a non-extreme reaction

- family: `earnings-announcement-premium` · mechanism: behavioral · instruments: LIQ-500 stocks, market-neutral: short year-long winners after a real 8-K release, 2004 to 2020
- outcome: rejected at G1 on the strategy: the mechanism test cleared the placebo bar only at the margin, but the book earned less than T-bills and the bootstrap bound of the Sharpe difference was below zero
- lesson: Mechanism test passed only at the margin: the event average just cleared the placebo bar, gross was a modest multiple of cost and below the card's own central expectation. The give-back built with the horizon instead of appearing in the first sessions, and almost all of the pooled mean sat in the earliest third (middle about zero, latest barely positive). The book, invested about a third of the time, earned less than T-bills: a thin per-event gap on few small slots a year cannot reach the Sharpe bar. Same profile as H-0064.
- avoid: Do not retune formation window, winner quantile, reaction cut or hold: a longer hold is a new trial on the same events. Do not refile as salience give-back or attention unwind; trials count here.
- open questions: Release-specific give-back versus generic fading of winners is not separated: try a control sort at dates without a release. If slow drift, argue it as a low-turnover tilt, not a five-session market-neutral book.
- related: H-0064, H-0032, H-0023, H-0028, H-0067

## H-0069 – Retail opening-auction attention premium - LIQ-500 market-neutral, short both extreme up- and down-movers at the next open, hedged with the most liquid members, hold 1 session

- family: `retail-attention-opening-pressure` · mechanism: liquidity_provision · instruments: US liquid stocks (LIQ-500), market-neutral: short extreme up- and down-movers at the next open, one-session hold, 2001 to 2020
- outcome: rejected at G1 on costs: the mechanism beat the placebos in every third, but gross per event covered only a fraction of round-trip cost; strategy not run
- lesson: A small sign-independent effect exists in LIQ-500: after extreme stock-specific moves the next open-to-close return was below the other members, well outside the placebo cloud (whose mean leaned the other way), positive in all thirds with no decay, and it lasted a few sessions instead of giving back overnight. But it is far smaller than the literature figures the card quoted from all-stock samples, and the placebo bar lies well below round-trip cost, so this is not a power problem: the effect is real but untradable at taker cost. The card forecast this death. The hedge basket would have added cost on top.
- avoid: Do not retune k_sigma, lookback_days or the hold, and do not refile the same extreme-move events as an overnight-versus-intraday, opening-auction or retail-attention variant: same events, new trials. Do not narrow to the most liquid ranks just to cut cost unless the card argues the effect survives there; it probably shrinks with liquidity. No daily taker book on a gross this far under cost.
- open questions: The claim of the same sign after up- and down-moves was never separated: G1 pools both, so the gain may come from one side. A one-sided card is a new trial and needs its own argument. Maker-style fills or less liquid names would change the cost side but cannot be simulated or loaded.
- related: H-0030, H-0007, H-0036, H-0064, H-0014

## H-0070 – Equity-stress carry-unwind reversal in currencies - after an SPY down-shock day, long the CurrencyShares trust that fell most and short the one that rose most (relative to the six), hold 5 sessions

- family: `short-term-reversal` · mechanism: liquidity_provision · instruments: six G10 currency trust ETFs after SPY down-shock days: long the biggest relative loser, short the biggest relative winner, 5 sessions
- outcome: rejected at G1 (mechanism): average positive but inside the placebo cloud, upper placebo bar below cost, gross far below cost; no strategy run
- lesson: Not a power problem: the event set was large and the upper placebo bar sat below round-trip cost, so a tradable effect would have cleared it. The average was positive but barely above the placebo mean. It came from the crisis-heavy first third, faded in the middle and was wrong-signed in the latest third, a declared kill condition. The horizon profile was nearly flat from the first to the fifth session and much larger only at ten: slow drift, not a concession that reverses within the week. The long side is the higher-beta currency, so what remained may be the equity rebound after the shock rather than a currency liquidity concession (K-0074).
- avoid: Do not retune k or hold_days, move to the ten-session horizon that looks larger, or flip to following the shock. Do not refile as carry-unwind, safe-haven or funding-currency reversal on the same six trusts with another trigger (VIX, FX volatility, dollar move): same events, new trials. Do not read this as evidence on H-0062's currency value: different horizon and mechanism.
- open questions: Forced FX unwinds need data where they trade (intraday FX, currency futures); daily trust opens miss a reversal in the Asian session. Matching the legs on equity beta would separate concession from rebound, but the prior is low. The slow ten-session build is a carry or value question for H-0062's family.
- related: H-0062, H-0041, H-0049, H-0006, H-0035

## H-0071 – Leverage-led vs spot-led big moves in Binance alts - after a 2-sigma day, fade coins whose perp-vs-spot volume ratio spiked, follow coins whose spot volume led, BTC-hedged, hold 3 days

- family: `perp-leverage-crowding` · mechanism: flow_structural · instruments: Binance USD-M perps outside BTC and ETH (point-in-time top thirty), big-move days classified by the perp-versus-spot volume ratio, signal read in spot volume, BTC-hedged, 2020 to 2022
- outcome: rejected at G1: wrong-signed average inside the placebo cloud, gross below cost, only the earliest third positive
- lesson: Splitting big-move days by where the volume traded (perp versus spot) did not separate forced from informational moves. The leverage-led minus spot-led gap was slightly wrong-signed and inside the placebo cloud, more negative at five and ten days, and positive only in the earliest third (the 2020 boom). Thousands of events and a placebo bar well under cost: no separation in this form, not a power problem. A perp volume surge may mark where informed traders act, so the ratio is a weak proxy for liquidation. Second failed forced-vs-informational separator in perps after funding (H-0049).
- avoid: Do not refile the split with another venue or volume ratio, k_sigma, hold or baseline window: same question on the same events (H-0049, K-0066). Do not trade the mirror (follow leverage-led, fade spot-led): the sign is dev-selected and sits inside the placebo cloud. Do not read the boom-era third as a regime worth a card.
- open questions: Forced flow is observed directly only in open interest and liquidation data, which are not loadable; a volume ratio never saw it. An intraday version at the cascade moment is untested, and the dev window holds a single boom-and-bust cycle, so the thirds are regimes, not independent samples.
- related: H-0039, H-0049, H-0035, H-0006

## H-0072 – Mortgage-convexity hedging continuation - follow a large IEF move for 5 sessions while the 10y yield is near its 1-year low, long/short

- family: `mortgage-convexity-hedging` · mechanism: flow_structural · instruments: IEF (seven-to-ten-year Treasury ETF) marked on large days while the ten-year yield was near its one-year low, SHY, SHV and BIL as unmarked near-cash peers, long/short, 2002 to 2020
- outcome: rejected at G1: average inside the placebo cloud, gross a small fraction of cost, latest third negative
- lesson: Large IEF days while the ten-year yield sat near its one-year low did not continue beyond what time-shifted placebos show. The average was barely positive, mid-cloud, a small fraction of round-trip cost; the first sessions were flat to slightly negative and the gain appeared only at longer horizons, the shape of generic bond drift, not hedging flow (two declared kill conditions fired: horizon shape, latest third negative). The placebo bar was under cost, so a tradable effect would have shown: not a power problem. The event study covered only dates with enough unmarked peers (about 2007 onward).
- avoid: Do not refile with another k, band, hold or a different convexity-state proxy (mortgage rate, other yield-distance rule) or another duration ETF on the same days: same events, new trials. Do not file large-day continuation or reversal in Treasury ETFs under a new flow story (auction, hedging, index extension) without observing the flow itself.
- open questions: The control the card asked for (same events with the state filter off, and by side) was not reported, so whether the convexity state changed anything is unknown. Direct MBS duration or swap-market hedging flow is not loadable; a daily Treasury ETF may be too coarse to show a few basis points of yield amplification.
- related: H-0038, H-0047, H-0041, H-0048, H-0070
