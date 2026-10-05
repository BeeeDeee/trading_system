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
