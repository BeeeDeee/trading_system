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
