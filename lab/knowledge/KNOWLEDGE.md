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

### K-0046 The release-session reaction drift reproduces on the same events in three book designs, and the long side alone shows it

Three cards built on the same real-release-date reaction events (a long/short book, a fixed-slot long/short book, and a long-only tilt on positive reactions) each cleared the placebo bar with gross above cost and a positive sign in every third of the sample. They share events, so this is one piece of evidence seen three times, not three confirmations. What is new: the positive side alone carries drift, so the effect is not only a short-side artifact, and within the positive-only set it weakens from the earliest to the latest third while staying positive. The effect is real in an equal-weight event study but small next to cost: the market-neutral books lost to T-bills, and the long-only tilt on an always-invested core came nearest to a pass, beating the market on the point estimate with a reliable bound but not by the required margin.

*confidence low · 2026-10-14 · librarian · supersedes K-0043 · evidence: H-0028; H-0031; H-0033; K-0043*

### K-0049 Crypto forced-flow rebounds exist before stress but a taker cannot keep them

The conditional rebound after fast, high-volume, taker-dominated moves in liquid crypto perps (H-0035, both directions) left a positive average trade after modeled taker costs and beat random-timed trades, so a liquidity-provision effect appears to exist. It did not survive costs at twice the model, and its profit sat in the first two thirds of the 2020-2022 dev period while the last third lost. Together with the hourly reversal finding of study mrel-1h (harvestable only as a maker) and the daily dip-buy H-0006 (could not tell), the pattern is that crypto reversal is paid to whoever provides liquidity at the stress moment and is small or negative for a taker who enters a candle later with optimistic slippage. The effect looked faster than the hold: the result halved with a one-candle later fill. Whether it is absent in sustained crashes is untested.

*confidence low · 2026-10-14 · librarian · evidence: H-0035; H-0006; prior_studies mrel-1h*

### K-0058 BTC-led altcoin catch-up is parameter-fragile in the lab

The rule 'long lagging altcoins after a BTC hourly jump beyond k sigma, hold four hours' looks strong on the dev period at the primary setting but its edge needs a high threshold: lower thresholds lose most of it and the result is dominated by one sub-period. It failed G2 on neighbor stability and block concentration, while survival under delay and doubled costs was fine. A narrowed grid after the fact would be a new trial, not a rescue.

*confidence medium · 2026-10-15 · human · evidence: H-0040 G2 run 84*

### K-0060 Short-horizon effects that clear the pooled placebo test are usually weakest or reversed in the latest third of dev

Across the lab's event studies that cleared or came near the placebo bar, the effect is typically largest in the earliest third of the dev period and smallest in the latest, in several cases with the opposite sign: month-turn beta, index inclusion, trading-window flow, short-horizon reversal, weekday mood in speculative stocks, twin-ETF gaps and perp funding crowding. Only earnings-reaction drift stayed positive in every third, and it also weakened. New: the pattern also holds for a full strategy that passed its absolute and bootstrap bars, the BTC-led altcoin catch-up (H-0040), where the middle third carried most of the return and the latest third lost, so G2 killed it on concentration. The three crypto perp cards (H-0035, H-0039, H-0040) share one short dev window, a single boom-and-bust cycle, so their thirds are regimes rather than independent samples, and the latest third was the weakest in each. A pooled result therefore overstates what a card can expect going forward. A Scout should expect decay by default and state the latest-third sign as a kill condition. Three equal blocks of one history cannot separate publication decay from a change of regime or cost level.

*confidence medium · 2026-10-15 · librarian · supersedes K-0052 · evidence: H-0016; H-0017; H-0025; H-0027; H-0029; H-0032; H-0035; H-0036; H-0037; H-0039; H-0040; K-0038; K-0046*

### K-0063 Fed-calendar equity premia against bonds: positive on average, near the placebo bar, not tradable, weakest lately

Two designs on the Fed's pre-announced calendar, the scheduled announcement sessions (H-0042) and the even-week cycle of FOMC time (H-0045), both gave a positive equity-over-bond or high-beta-over-low-beta average that was inside or just under the time-shifted placebo cloud. The two live at different horizons: the announcement design shows its gain on the announcement session only, while the cycle design shows nothing at one session and builds over several, so the cycle result is not the announcement-day effect again. Both are weakest in the latest third of dev, which includes publication, and gross is thin relative to cost. A design that marks both sides (even long, odd short) gives a placebo mean near zero, which is a cleaner test than the one-sided H-0042, where part of the gain was generic equity drift. A further Fed-calendar card on equities and Treasuries has a low prior unless it changes the instrument (index futures, equity-only comparison) or the horizon; this extends K-0060 without superseding it.

*confidence low · 2026-10-16 · librarian · evidence: H-0042; H-0045; K-0060; K-0034*

### K-0070 Macro-calendar cards fail in four designs; moving to a volatility ETP did not help

Four designs on the scheduled US macro calendar have ended at the event-study gate: announcement-session beta premium (H-0042), even-week FOMC-cycle tilt (H-0045), cross-asset reaction drift (H-0047) and short volatility over release windows through a VIX-futures ETP (H-0052). The last was the route K-0063 and K-0065 left open. It gave a positive average, but placebo windows of the same instrument already earned over half of it through roll-down, the effect grew with the horizon like generic carry, and the first third of dev was flat while the middle third carried most (not the early-heavy decay of K-0060). The placebo bar lay above cost although gross was several times cost, because a volatility ETP is noisy. A release-time volatility crush is not visible in daily ETP bars. A further macro-calendar card on daily ETFs has a low prior; implied volatility or index futures measured directly are untested and not loadable.

*confidence low · 2026-10-17 · librarian · supersedes K-0065 · evidence: H-0052; H-0047; H-0045; H-0042; K-0063; K-0065; K-0034; K-0025; K-0060*

### K-0083 On real earnings dates only own-price-move events cleared the placebo test; no such book survived

On real 8-K earnings dates the mechanism tests that cleared the placebo bar condition on the stock's own price move at or around the release: reaction drift (H-0028, H-0031, H-0033), the pre-release reversal (H-0064) and, new, the post-release give-back of year-long winners (H-0068). Sorts on a second variable or another stock did not (H-0024, H-0026, H-0027, H-0032, H-0067, H-0066), nor did date-only premia (H-0023). New: H-0068 shows a slow own-price sort can pass, but only at the margin, with the mean in the earliest third and the effect still building beyond the hold: slow drift, not an announcement effect. Its market-neutral book lost to T-bills like those of H-0028, H-0031 and H-0064; only the long-only H-0033 came near. Hindsight pattern, not a tested rule. State the expected annual gain of the book, not the per-event gap.

*confidence low · 2026-10-20 · librarian · supersedes K-0081 · evidence: H-0028; H-0031; H-0033; H-0064; H-0068; H-0023; H-0024; H-0026; H-0027; H-0032; H-0066; H-0067; K-0064*

### K-0084 Stress rebounds: positive on average, near generic drift, weakest lately, also in currencies with both sides marked

A rebound after forced selling shows as a positive pooled average in four settings: daily crypto dip-buying (H-0006), fifteen-minute crypto cascades (H-0035), stocks against Treasuries (H-0041) and, new, relative reversal among G10 currency trusts after equity down-shock days (H-0070). In each it is near, not clearly beyond, the time-shifted placebo, and the latest third of dev is weakest or negative. H-0070 marked both sides inside one asset class and standardized the trigger by trailing volatility, so generic drift was netted out; gross still covered little of cost and the placebo bar lay below cost, so it is not a power problem. Its horizon profile was flat early and larger only at ten sessions, the shape of slow drift, and the long leg was the higher-beta currency (K-0074), so a beta confound is the leading reading. New stress-rebound cards are largely the same evidence again.

*confidence low · 2026-10-20 · librarian · supersedes K-0056 · evidence: H-0070; H-0041; H-0035; H-0006; H-0049; K-0049; K-0060; K-0074; prior_studies 6-7*

### K-0088 Short VIXY timing: the unconditional roll-down is the effect

Two cards short the VIX-futures ETP VIXY on a timing rule: over scheduled macro-release steps (H-0052) and in steep contango of the implied-volatility curve measured directly from index options (H-0073). Both had positive averages with gross many times cost, and in both time-shifted placebos of the same instrument earned most of the average, so the rule added little beyond holding the short. Both effects grew roughly evenly with the horizon, the shape of carry, not of an event. K-0070 left implied volatility measured directly as an untested control; the slope is now that control for the carry reading and it does not rescue the idea. In H-0073 the effect was clearly weaker in the later thirds of dev (K-0060), so the roll-down itself may have shrunk. Untested: tail days and drawdown of a timed versus an untimed short, borrow cost, and the actual futures curve. A further short-VIXY timing card has a low prior unless it argues a tail-risk gain, not an average one.

*confidence low · 2026-10-21 · librarian · evidence: H-0052; H-0073; K-0070; K-0060; K-0025*

### K-0090 CFTC speculator positioning through commodity and currency ETFs: levels and weekly changes both come out empty

Two designs on the same CFTC non-commercial positioning and almost the same commodity, metal and currency ETFs found no tradable effect: monthly positioning levels against their own history (H-0057) and the weekly change in speculator net position traded for five sessions (H-0051). Averages were about zero or wrong-signed and inside the placebo cloud, with no consistent sign across the thirds of dev: the levels design was positive only in the earliest third, the changes design only in the latest. In H-0051 the short horizons were flat and only the longest leaned the predicted way. Likely reasons are the reporting lag (entry days after the as-of date), futures roll and tracking noise in the ETFs, and physical trusts that do not hold the futures. The published premiums concern futures, so this does not refute them. A third ETF-based CFTC card has a low prior unless it changes the instrument to the futures themselves or controls for the market's own past return.

*confidence low · 2026-10-25 · librarian · evidence: H-0051; H-0057*

### K-0093 S&P 500 membership-change reversal: two designs end just under the placebo bar, with slow build and a faded latest third

Two designs on the forced trading around S&P 500 membership changes in liquid US stocks, additions only (H-0017) and additions plus deletions read inside LIQ-500 (H-0022), both showed the predicted sign, a gross effect well above cost and a placebo mean near zero, and both landed just under the placebo percentile bar, with a bar above cost. Both effects were small in the first sessions and larger at the longest horizons, so the reversal is slower than a five-session hold, and in both the latest third of dev was weakest (about zero in the second). Adding the deletion leg gave more events but did not remove the near miss. They share most of their events, so this is one piece of evidence seen twice. Mandated-flow events remain the nearest thing to a live effect in the large-cap short-horizon cross-section (K-0024), but the gate cannot confirm them with a few events per month. Further tests on the same membership events are retries; a card needs a slower horizon argued from the build-up, or many more events.

*confidence low · 2026-10-25 · librarian · evidence: H-0017; H-0022; K-0024; K-0026; K-0034; K-0060*

### K-0094 Reaction drift is specific to earnings releases: large moves on non-earnings 8-K sessions show none

On real earnings-release sessions, following a large standardized own-price reaction drifts in the same direction (H-0028, H-0031, H-0033). The same rule applied to sessions with a filed non-earnings 8-K (H-0034) gave a slightly wrong-signed average, inside the placebo cloud, with signs mixed across thirds and horizons beyond the first sessions leaning negative. Moves on sessions with no earnings release at all reverse slightly (H-0030). So a filing-identified news event did not turn reversal into drift: what separates the earnings case may be the scheduled date, the consensus benchmark and the attention an earnings release draws, not the presence of news. Caveats: the 8-K item types are pooled and the filing session can come after the press release, so the dated reaction may be a tail or a routine filing. Not tested: single item types, press-release timestamps, a news-versus-no-news comparison within the same stocks. Dated news outside earnings now has one flat verdict; a further filing-based drift card needs a reason that the item type is a surprise.

*confidence low · 2026-10-25 · librarian · evidence: H-0034; H-0030; H-0028; H-0031; H-0033; K-0083; K-0024*

### K-0103 Closed-end fund gaps revert in a pooled seven-class book; only the net Sharpe floor fails

A pooled book of closed-end funds from seven classes, long after a large shortfall against a beta-adjusted ETF anchor and short the anchor, again cleared the placebo test at the top of the cloud, with gross several times cost and a positive sign in every third. The middle third, which likely holds 2008, was strongest and the latest weakest but positive. Muni and country funds reuse the event type of H-0077 and H-0078, so this is partly the same evidence, not an independent confirmation. Pooling cured exposure and entry counts and passed the bootstrap check, but net Sharpe stayed under the absolute floor while gross cleared it: cost of thinly traded funds took the margin. The anchor short lost money alone again, so the long leg may include asset-class beta (K-0074). The effect was small at one session and built over about two weeks, so it is not a one-day bounce.

*confidence low · 2026-10-28 · librarian · supersedes K-0100 · evidence: H-0079; H-0078; H-0077; K-0100; K-0074; K-0060*

### K-0110 Net repurchasers versus issuers: filing-anchored pass, no release flow, no slow drift

The same repurchaser-versus-issuer sort was tested at three anchors in LIQ-500. Anchored to the quarterly filing, a five-session spread cleared the placebo bar clearly (H-0019). Anchored to the real release it did not, built slowly and reversed lately (H-0027). As a slow style premium, the route K-0041 left open, new entries to the extreme quintiles over a quarter (H-0086) averaged below same-stock placebos, were slightly negative in the first weeks, and faded from clearly positive in the earliest third to clearly negative in the latest. Reading: the filing-anchored pass was filing-linked news or an early-era effect, not a trading-window flow and not a drift visible in large caps. Caveats: the placebo bar lay well above cost, repurchasers have lower beta than issuers so late bull years bias the raw comparison (K-0074), no split by side was reported, and the year-long hold was never simulated. Treat the family as closed for event-study cards in LIQ-500 unless beta-neutral, side-split or in less covered stocks.

*confidence low · 2026-10-29 · librarian · supersedes K-0041 · evidence: H-0019; H-0027; H-0086; K-0041; K-0060; K-0074*

### K-0112 Currency value predicts country returns over a quarter in two vehicles, but neither dollar-neutral book is a strategy

Country equity ETFs sorted on their own past return showed no reversal (H-0060). A five-year-average value score in G10 currency trusts passed the mechanism test (H-0062). The same score, measured only in FX and applied to unhedged country equity ETFs, also passed it (H-0088): above the placebo cloud, gross several times cost, positive in every third, so the local equity leg did not offset the currency recovery over a quarter. This answers the question K-0073 left open. The two currency cards share the signal, so this is one piece of evidence seen twice, on a short sample with few value cycles and a nominal score mixed with carry. Neither book became a strategy: the trust book missed the absolute Sharpe floor, and the equity book earned about T-bill returns, gross near zero, the long leg's gain mirrored by the short leg's loss. Study and book disagree for reasons not separated: G1 sees only the first quarter of a year-long hold, leg betas were not reported, and the book is in cash until the first signal. A further currency-value card has a low prior unless it reconciles study and book, hedges equity beta, or separates value from carry.

*confidence low · 2026-10-29 · librarian · supersedes K-0073 · evidence: H-0088; H-0062; H-0060; K-0073; K-0074; K-0076; K-0097*

### K-0119 Daily reversal in liquid crypto perps: four conditioners found none

Daily designs on the top perps (2020-2022 dev) conditioned the big moves on funding level (H-0039), funding change (H-0049), perp-versus-spot volume (H-0071) and, new, a US implied-volatility stress gate with both sides marked (H-0081). None gave an average clearly above the placebo cloud with the predicted sign; H-0039 barely cleared, in one episode. In H-0049, H-0071 and H-0081 longer horizons turned negative, the shape of continuation. The placebo bar lay under cost in H-0071 and H-0081, so not a power problem. A positive third was a different third each time: regimes of one boom-and-bust cycle (K-0060). The unconditional daily reversal among these perps was never reported and is the missing control. Open interest and liquidations are not loadable; maker-side reversal (K-0049) and smaller pairs are untested.

*confidence low · 2026-10-31 · librarian · supersedes K-0085 · evidence: H-0081; H-0071; H-0049; H-0039; H-0006; K-0085; K-0084; K-0049; K-0060*

### K-0121 Holders' capital-gain overhang does not predict a short-horizon edge in liquid US stocks: two designs, both wrong-signed

Two designs gated the news response of liquid US stocks by holders' unrealized gain. H-0009 followed volume-shock moves that agreed with the 12-month gain and lost money. H-0091 used real earnings-release reactions and long gain-overhang, short loss-overhang stocks, a contrast built to cancel generic reaction drift; its average was wrong-signed, below every time-shifted placebo, negative in every third of dev and at every horizon, and absent in the first session. The shape is slow drift, not an announcement effect, and fits the winner give-back of H-0068 (K-0083) because overhang is mostly a past-winner marker. A beta gap between legs (K-0074) was not neutralized; no split by news sign or no-release control sort was run. So a disposition channel is not supported in large caps over days, and the opposite sign is not shown to be an effect either. A new overhang card should separate good-news and bad-news cells and match legs on beta and past return.

*confidence low · 2026-11-01 · librarian · evidence: H-0091; H-0009; H-0068; K-0083; K-0074; K-0060*

### K-0127 A predicted-release fundamentals sort cleared the placebo bar but not cost, and grew rather than decayed

K-0083 found that on real 8-K dates only sorts on the stock's own price move cleared the placebo test. Earnings seasonality (H-0093) sorts on a second variable and cleared the percentile narrowly, with a placebo mean near zero and a bar below cost; the composite sort H-0032 did not. Its shape fit the claim (small early, larger by the end of the week, flat beyond) but gross stayed below round-trip cost. The latest block was the strongest, the reverse of K-0060, yet the first block had almost no events, so only two blocks are usable. This weakens K-0083 without refuting it. The date hit rate was never measured. A further predicted-window fundamentals sort in LIQ-500 has a low prior unless gross is argued clearly above cost.

*confidence low · 2026-11-01 · librarian · evidence: H-0093; H-0032; H-0092; K-0083; K-0126; K-0060; K-0064*

### K-0128 Supply-side crypto sorts did not find the expected laggard in four designs; persistent coin identity carries the average

Four cards tried to short coins on a supply-side characteristic: young spot listings (H-0004, could not tell), perps aged between one month and one year (H-0058, wrong-signed), the first weeks after a perp launch (H-0082, wrong-signed) and, new, the fastest token supply growth within beta halves (H-0085). In H-0085 the time-shifted placebo of the same coins was about as negative as the marked average, so the sort picks persistent coin identity (old slow-issuance coins against emission-heavy ones) and says nothing about timing. In the dev window the slow-issuance side lagged, and the sign swung with the cycle across thirds, as in H-0058, so beta-matching legs by halves did not cure the confound that K-0074 names. Only protocol issuance was visible, not vesting unlocks. All of it sits in one boom-and-bust window (K-0006). A further overhang or dilution card has a low prior unless it uses within-coin timing (changes in issuance), matches on prior return, or sees unlock schedules.

*confidence low · 2026-11-03 · librarian · supersedes K-0107 · evidence: H-0085; H-0058; H-0082; H-0004; K-0107; K-0074; K-0006; K-0025*

### K-0139 Filing-anchored sorts pass twice in LIQ-500; a learned conditional form did not

Two sorts with different content, net repurchasers against issuers (H-0019) and low against high quarterly accruals (H-0092), were both timed to the quarterly filing and held for a week in LIQ-500. Both cleared the same-stock placebo bar with gross above round-trip cost, and both books ended near T-bills (K-0064). They share filing dates, universe and seasons, so this is one piece of evidence seen twice. H-0019 faded era by era, H-0092 was positive in every third, and the effect in H-0092 was present in the first session and kept building. New: H-0097 added the release reaction and event time in a learned long-only form. Fresh-filer entries stayed inside the placebo cloud, gross was under cost, and the latest third was wrong-signed, so the claim that filing information is conditional on the headline reaction adds nothing visible over the plain accrual sort. The release-anchored sort (H-0027) and the slow style (H-0086) also failed, so the anchor matters and the pass may be filing-linked or early-era. A further filing card in LIQ-500 has a low prior; untested are the long leg alone, beta-matched legs and less covered stocks.

*confidence low · 2026-11-05 · librarian · supersedes K-0126 · evidence: H-0092; H-0097; H-0019; H-0027; H-0086; K-0110; K-0064; K-0074; K-0060*

### K-0143 Learned earnings-reaction cards: ownership, event time and non-filed gaps added nothing

K-0141 asked for a new information source or for the interaction to be reported directly. Both came in and failed. Institutional ownership with arbitrage risk (H-0102) gave entries below same-stock placebos, gross a fraction of cost and a slow-drift horizon shape. Event-time features of the latest reaction scored over all members (H-0104), tested as picks only the gbm makes against a ridge, were also below placebos and grew with horizon instead of fading. Overnight gaps with no filing and earnings windows excluded (H-0105) went wrong-signed against the ridge. With H-0095, H-0100 and H-0101, five learned cards in the reaction family have died on the mechanism test, and none beat the plain reaction rule of H-0033; the only near miss (H-0100) had a bar under cost. Second variables did not add at the release (K-0083), and a learned combination of them does not either. A further learned card on release reactions or overnight news in LIQ-500 has a low prior. Untested: other stocks (not loadable) and holds beyond a week argued from the slow build.

*confidence low · 2026-11-06 · librarian · supersedes K-0141 · evidence: H-0102; H-0104; H-0105; H-0101; H-0100; H-0095; H-0033; K-0141; K-0083*

### K-0148 Sector-ETF flow days: the constituent reversal did not reproduce once market-wide stress was removed

H-0106 found a reversal after sector-ETF flow days that beat placebos and covered cost (K-0146), but its sector moves were mostly market-wide. H-0107 removed that by construction: the push was the wrapper's move residual to SPY, stocks with an own filing that day were excluded, industry ETFs added flow days outside market stress. Entries were no better than same-stock placebos, gross was about half of cost, the first sessions were negative, ten sessions flat, and the thirds disagreed in sign. The placebo bar lay above cost, so a small effect was unresolved, but nothing suggests a hidden tradable one. The H-0106 pass most likely belonged to market-wide stress days (K-0084), not to wrapper-specific basket flow. The cards also differ in model, entry rule and side, so this is a lean, not a refutation. Untested: ordinary-day control, up-push side, ridge rule. A further ETF-flow card has a low prior without a stress control or ownership data.

*confidence low · 2026-11-06 · librarian · supersedes K-0146 · evidence: H-0107; H-0106; K-0146; K-0084; K-0024*

### K-0160 The linear reversal score in the volatile half of liquid stocks did not reproduce when re-specified

K-0155 read H-0115 as a clean liquidity concession: a ridge score in the high-volatility half of a liquid universe beat same-stock placebos, covered cost, kept its sign in every third and eased after the first week. H-0117 re-used the claim with the score rebuilt from the lesson text, new leg entries as the event, and a buffered, beta-neutral book as the change. Its entries stayed under the placebo percentile bar, gross per entry was under round-trip cost, a large share of the average was earned by the same stocks on shifted dates, the latest third was much weaker, and the effect grew toward the longest horizon instead of easing. The two cards differ in inputs, event definition and trial luck, so this is not a refutation of H-0115, and it is not a confirmation either. Reading: the volatile-half concession is, so far, one pass that did not repeat, with a market-exposure reading (K-0074) still open. Treat it as unconfirmed; the nonlinear failures of K-0155 stand. A further card on this score needs the exact H-0115 inputs, a beta-matched or volatile-versus-calm control, or a different information source.

*confidence low · 2026-11-08 · librarian · supersedes K-0155 · evidence: H-0117; H-0115; H-0094; H-0103; H-0112; K-0155; K-0074; K-0060*

### K-0162 Learned date-level state and calendar interactions in liquid US stocks: two nulls and one near miss at a legal deadline

Two learned cards gave a gradient-boosted score date-level inputs that are the same for every stock on a date: month-turn and macro-release clocks (H-0098) and volatility and drawdown state (H-0099). Both ended with entries no better than same-stock placebos and gross far under cost. A third, the October and December tax-deadline clocks interacting with holders' loss state (H-0118), came closer: right-signed, placebo mean near zero, gross just above round-trip cost, percentile just under the bar, and a bar above cost, so a small real effect is not excluded. Unlike the usual decay (K-0060) its earliest third was wrong-signed and the later two positive, and the horizon profile was slow rather than a quick release. So the only date-level conditioner that did not read as null is a clock tied to a dated legal deadline, and that is one near miss on a part-time book. In none of the three could the model's use of the date inputs be seen. The reading of K-0137 stands for month-turn, macro-release and volatility states: switching a static tilt with them adds nothing. A further card of this type should report how the state was used, or be a tilt on an always-invested core.

*confidence low · 2026-11-09 · librarian · supersedes K-0137 · evidence: H-0118; H-0098; H-0099; H-0110; K-0137; K-0152; K-0027; K-0034; K-0060*

### K-0169 Learned large-cap sorts: the dollar price grid and macro-cleaned residuals add nothing either

K-0166 listed fundamentals, ownership, attention, calendar, tax clock, return-distribution shape and information-asymmetry markers as learned conditioners that did not lift a short-horizon large-cap sort above cost. Two more sources failed. H-0122 used the price level on the dollar grid (next-session hold, picks only a gbm makes against a ridge): entries beat peers slightly, but a sizeable share was earned by the same stocks on shifted dates, gross was a small fraction of cost, and the effect was largest early and grew with the horizon. H-0123 split the weekly move into a cross-asset part and a residual for a ridge: entries matched placebos, gross was near zero, and the thirds disagreed. In both the placebo bar lay below cost, so not a power problem. In the most liquid stocks neither a behavioral microstructure input nor a cross-asset input brings gross near a taker's round trip. Any grid effect probably sits in low-priced names or needs resting orders. A further learned conditioner has a low prior.

*confidence low · 2026-11-10 · librarian · supersedes K-0166 · evidence: H-0122; H-0123; H-0121; H-0117; K-0166; K-0152; K-0142; K-0160; K-0060*

### K-0176 Macro-news diffusion into stocks: a plain credit rule and an inflation sort failed

K-0173 read the credit pass of H-0126 as support for an explicit balance-sheet link plus a clean shock series. Two cards built exactly that and failed. H-0128 replaced the learned model with a plain leverage sort on the same credit-specific shock: right-signed but inside the placebo cloud, gross far under cost, wrong-signed in the earliest third, growing toward the longest horizon like slow style drift. H-0129 used a clean inflation-expectations shock and the nominal-versus-real asset split: inside the cloud, peaking in the first sessions and negative at longer horizons, not a build over the week. In both the placebo bar lay under cost. So the H-0126 pass is one result its linear control did not repeat: it needed the model's coverage and volatility inputs or was luck; the ridge control never ran. Oil (H-0114) is the only other pair that passed, once, with a wrong-signed latest third. Rates, dollar and non-oil commodities stay unsupported. A link and a clean shock are not sufficient; the prior for another card is low.

*confidence low · 2026-11-11 · librarian · supersedes K-0173 · evidence: H-0128; H-0129; H-0126; H-0114; H-0116; H-0119; K-0173; K-0172; K-0074*

### K-0179 US risk-state inputs do not visibly re-order the crypto cross-section over days

Three cards conditioned a crypto coin sort on a TradFi state. A VIX-gated reversal in perps (H-0081) was inside the placebo cloud and below cost. A learned sort on beta, idiosyncratic risk and liquidity crossed with VIX and real-yield shifts (H-0130), tested as the picks only the gbm makes against a ridge on the same features, was positive on the point estimate but inside the cloud, below cost, flat or negative at one and ten days, with almost all of the gain in one third of dev and about nothing in the latest. The third, a VIX-switched momentum card (H-0124), never produced a score. So the flight-to-quality ladder story (levered altcoin holders de-risking with a lag after a US shock) found no usable ordering at the daily to weekly scale, and nonlinear state-by-characteristic interactions added nothing over the linear control, as in liquid US stocks (K-0162). Caveats: one boom-and-bust window, events on few dates, the sign flip between risk-off and risk-on was never reported, and the beta gap between picks is not neutralised (K-0074). Not tested: slower horizons, BTC alone as the traded asset, funding or stablecoin data as the transmission channel.

*confidence low · 2026-11-11 · librarian · evidence: H-0130; H-0081; H-0124; K-0119; K-0162; K-0142; K-0074; K-0006*

### K-0180 Stock-bond flow: a rally-side control did not reproduce the H-0046 pass

H-0046 passed the event study on band-breach days of both directions, so stress rebound could not be separated from rebalancer flow (K-0068). H-0050 (quiet trigger) was slightly wrong-signed. New: H-0131 kept only the rally side, where no stress rebound exists. Equities lagged bonds, above a negative placebo mean (equity premium) but inside the placebo cloud, gross about equal to cost, latest third reversed, and the effect grew with horizon out to weeks: slow drift, not a week-long concession. Two different controls now fail to reproduce the pass, so the reading that H-0046 caught generic reversal after large spread moves and stress rebound (K-0084) is stronger. Still a lean: triggers differ, the H-0131 placebo bar lay above cost, H-0050 was thin. Four equity ETFs share dates, so independent evidence is the event dates. A further daily-ETF mechanical-flow card has a low prior unless the flow is seen where it trades.

*confidence low · 2026-11-11 · librarian · supersedes K-0068 · evidence: H-0131; H-0050; H-0046; H-0002; K-0068; K-0084; K-0060; K-0034*

### K-0184 Nearness to the 52-week high has no timing content in three settings: it picks persistent winners

Three anchoring cards died at the mechanism gate: stale-high breakouts in liquid US stocks (H-0011), nearness to the high within return-by-volatility cells (H-0083), and liquid crypto spot coins nearest their one-year high (H-0135). In H-0011 and H-0135 the raw gap was positive and above cost, but same-name placebos on shifted dates earned the same, so the sort selects names that do well in any week. In H-0135 the effect was largest early and nearly gone in the latest block, and grew with the horizon like the placebo. Reading: nearness to the high marks past winners; any premium is momentum or coin identity. Untested: crypto matched on prior return, and a dated own-price event conditioned on distance to the high. A further plain near-high card has a low prior.

*confidence low · 2026-11-12 · librarian · evidence: H-0135; H-0011; H-0083; K-0025; K-0060; K-0074; K-0006*

### K-0185 Earnings-reaction drift in liquid US stocks lives in the first weeks; held a quarter it matches same-stock placebos

On real 8-K dates a large positive reaction cleared the same-stock placebo over the first week in three designs (H-0028, H-0031, H-0033). H-0136 held the same kind of events to the next release. The abnormal return stayed positive, above cost, positive in every third with the latest among the strongest, and grew with horizon. But placebos of the same stocks on shifted dates earned more on average, so beyond the first weeks the set marks stocks that do well in any quarter (repeat positive reactors, persistent winners; K-0025, K-0184). This answers the open item of K-0143 (holds beyond a week) with a negative, and it is not the later-year large-cap decay of the literature, since the latest third did not weaken. Caveats: raw peers (K-0074); confirming releases of held stocks overlap; the book was never run. A longer-hold card on the plain rule has a low prior; only a control matched on reaction history could separate timing from identity.

*confidence low · 2026-11-12 · librarian · evidence: H-0136; H-0028; H-0031; H-0033; H-0104; K-0025; K-0046; K-0083; K-0143; K-0184*

### K-0194 15m crypto perp taker rules at the US open: coin-specific volume selection in alts did not lift the breakout above cost

K-0192 found that of four breakout conditioners on BTC and ETH only a narrow-range one cleared round-trip cost, and that fragilely. H-0142 changed the cross-section instead of the state: only liquid alts with a coin-specific opening-volume shock, both sides, exit at the US close. It was worse than the unselected BTC/ETH rule. Derived gross per trade was near zero against a round trip that is larger for alts, trades were plentiful, the bootstrap bound of the mean net trade was below zero and drawdown was deep with little exposure. So the idea that a breakout carries information only when the coin has its own news did not transfer from stocks to crypto at taker cost in 2020-2022; the three variants that used a market-wide state or selection (volatility, alignment, coin volume) all failed and only contraction came close. A selection that picks high-volume alts probably picks high-beta flow in a factor-driven market. Unmeasured: the no-filter and lowest-volume controls, side split and thirds, because the judge does not run them (K-0193); and the maker route and holds longer than a session.

*confidence low · 2026-11-13 · librarian · supersedes K-0192 · evidence: H-0142; H-0137; H-0138; H-0139; H-0140; K-0192; K-0193; K-0006; prior_studies 16*

### K-0195 Cross-sectional sorts of ETFs on past returns: three designs, none clearly above same-ETF placebos

Monthly-formation sorts of country and asset-class ETFs have now ended at the mechanism test three times without a clear gap over the same ETFs on shifted dates: country long-term reversal (H-0060, wrong-signed), bond-to-equity cross-asset momentum (H-0087, small and right-signed but inside the cloud, latest third negative) and, new, a learned frog-in-the-pan momentum score ranking ETFs inside fixed risk and defensive sleeves (H-0143). In H-0143 the average was tiny, over half of it was earned by the same ETFs on shifted dates, the thirds were unstable, and the effect peaked near one month and turned negative at a quarter. That is not the slow build and early-era weight of stock residual momentum (H-0054, a near miss). Reading, inferred: a cross-section of a few dozen ETFs gives few independent picks per date and a wide placebo cloud, and any in-sleeve momentum is small next to ETF round-trip cost. Plain in-sleeve momentum was never run as a control, so a path-shape conditioner was tested without knowing whether its base effect exists. A further ETF momentum card has a low prior unless it argues a horizon of about a month or less, or reports the plain sort first.

*confidence low · 2026-11-13 · librarian · evidence: H-0143; H-0060; H-0087; H-0054; K-0074; K-0060; K-0112*

### K-0196 Crypto spot trend sorts: near-high and a learned life-cycle score both fail the same-coin placebo

Two cards sorted liquid Binance spot coins on past trend and died at the mechanism test alike. Nearness to the one-year high (H-0135) was positive but no better than the same coins on shifted dates. A learned two-month score on six-month trend, run stage and volume trend (H-0144) was below its placebo, flat at short horizons, and wrong-signed in both later thirds, boom era included. The placebo mean was positive in both, so the sort picks coin identity, not timing. With K-0128 and study 9, no coin-characteristic sort of the crypto spot cross-section has beaten its same-coin control. The ridge control and a BTC-direction split were not reported. The long-horizon momentum claim is outside the sixty-day schema cap. One boom-and-bust window (K-0006). Low prior for more unless dated own-price events or BTC alone.

*confidence low · 2026-11-13 · librarian · evidence: H-0144; H-0135; H-0134; K-0184; K-0128; K-0006; prior_studies 9*

### K-0197 Perp crowding markers: no reproducible edge in six cards

Six cards used leveraged-demand markers of liquid crypto perps: funding level (H-0039), funding change after crashes (H-0049), perp-versus-spot volume (H-0071), a US stress gate (H-0081), a learned momentum-by-basis score (H-0134) and, new, a slow own-history basis sort in half-year cohorts (H-0145). None cleared the same-coin placebo test with a bar under cost. H-0145 is a cannot-tell: right-signed with a negative placebo mean, so not coin identity, but the cloud was wide, the early third wrong-signed and the middle third carried the average. Open interest and liquidations remain unmeasured; the plain linear basis sort was never reported. Low prior for another crowding card on these perps without forced-flow data or a multi-cycle history.

*confidence low · 2026-11-13 · librarian · evidence: H-0145; H-0039; H-0049; H-0071; H-0081; H-0134; K-0119; K-0006*

### K-0198 A cut arbitrage link is not enough: single-company receipts did not revert during home closures, basket wrappers did

K-0183 read wrapper deviations as reverting where creation and redemption cannot arbitrage them: country ETFs on home holidays (H-0133) and closed-end funds against an ETF anchor (K-0103). H-0157 applied the same cut link to single-company receipts of Asian companies during multi-day home closures. The unanchored excess move did not revert but continued, in every third and more at ten sessions than at three, on only a few dozen closure episodes (about one event date a year), so the sample is small and the sign is weak evidence. Reading, inferred and not tested: a basket wrapper has an anchor in its holdings that returns when the home market reopens, while a single-company receipt's move during the closure may carry company news that no home price contradicts. So what reverts is the flow part of a gap, not the information part, and a cut link is at most necessary. Which stocks the correlation rule assigned to home exchanges was never reported. A further wrapper card needs basket wrappers or many more closure dates, not single-company receipts.

*confidence low · 2026-11-14 · librarian · supersedes K-0183 · evidence: H-0157; H-0133; H-0079; H-0037; H-0080; K-0103; K-0183*

### K-0199 Volatility risk premium: the cross-section of stocks by loading on volatility innovations is empty too

K-0129 read the failed implied-over-realized rankings as inferred evidence that any premium is global and cancels in a relative sort, and left a time-series design as the only open route. H-0155 tested the cross-sectional form of the global premium directly: liquid stocks most hurt by VIX innovations, matched on beta and idiosyncratic volatility, held for months. Entries were right-signed in every third, but the same stocks on shifted dates earned far more, short horizons were negative and there was no build to a quarter, so the sort picks stocks that do well anyway and timing adds nothing. Beta was held fixed by cell, so the usual beta confound (K-0074) is an unlikely excuse; the placebo bar lay above cost but the event mean was below the placebo mean, so power was not the limit. The family now has five cards across stocks, ETFs and a volatility ETP. The loading was estimated from a noisy collinear regression; loadings from option prices are not loadable. A further card here has a low prior unless it is a time-series design across many markets with an always-invested control.

*confidence low · 2026-11-14 · librarian · supersedes K-0129 · evidence: H-0155; H-0084; H-0073; H-0052; H-0055; K-0129; K-0074*

### K-0200 Session-VWAP setups on BTC/ETH five-minute perps: four directions, none pays at taker cost

Four cards from one leaflet traded the UTC-session VWAP and its bands on BTC and ETH five-minute perps: continuation after acceptance outside the bands (H-0147), a fade of a rejected band-edge test (H-0148), the re-entry after a failed excursion (H-0149) and a held VWAP test continuing to the far band (H-0150). All died at G1 with negative net Sharpe, a bootstrap bound below zero and drawdown over the limit. Trade counts were large, several times the forecast for the two later setups: the faster clock buys events, not gross per event. Derived gross per trade (from mean net trade and cost share; not reported by the judge) was closest to cost for H-0147 and a small fraction of cost for the others. H-0149 and H-0150 take opposite sides of the same re-entries and both lost, so no direction is left to flip. With H-0137 to H-0142 and study 16 these are nine price-level taker rules on BTC and ETH with one fragile near pass (K-0194). Owner controls did not run (K-0193). Untested: maker execution, holds beyond a session.

*confidence low · 2026-11-14 · librarian · evidence: H-0147; H-0148; H-0149; H-0150; H-0137; H-0142; K-0193; K-0194; prior_studies 16*

### K-0201 Quarter-horizon stock sorts: nine designs, none beat the same-stock placebo, which takes most of the average

Since K-0111, five more designs held liquid US stocks on a quarter-long label and ended at the mechanism test: asset growth by cash-flow discretion (H-0146), healthy cheap stocks (H-0151), large share-count jumps (H-0152), bond-proxy dividend stocks by rate state (H-0154) and aggregate-volatility loading (H-0155). With H-0054, H-0059, H-0083 and H-0086 that is nine, none passed: wrong-signed (H-0151); below the placebo mean (H-0154, H-0155, H-0086); mostly placebo identity (H-0146, H-0059); tiny in a wide cloud (H-0152); near misses with bars above cost (H-0054, H-0083). In the new cards the placebo mean was positive and about as large as the event mean: the sort picks stocks that earn on any date. Entry-only events remove that trap but widen the cloud (K-0111). Four of the five were long-only against raw peers (K-0074, K-0153). The three learned cards reported no ridge control and no split by state or side (K-0135). A further quarter-horizon stock sort has a low prior unless it reports the linear control and a beta-matched peer set.

*confidence low · 2026-11-14 · librarian · supersedes K-0111 · evidence: H-0146; H-0151; H-0152; H-0154; H-0155; H-0086; H-0083; H-0059; H-0054; K-0111; K-0074; K-0153; K-0135*

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

### K-0096 ICE BofA spread series in the lab's FRED copy are empty in dev

The G0 check of H-0075 reported fred_macro.BAMLC0A0CM and fred_macro.BAMLH0A0HYM2 as series without a single value in the dev period, with first valid value 2023-10-04 (FRED serves only a short ICE BofA history). This confirms the cause K-0062 suspected for H-0043, whose zero events came from the spread inputs, not from the market. The catalog entry for fred_macro lists both fields and a range from 1998 but does not say they start in 2023, so a Scout cannot see it. Separately, that G0 failure appeared on H-0075 although the card requested only DGS10; a re-run then passed. Cause not known (inferred: a whole-dataset check at that moment). Scouts should not use ICE BofA OAS series for any dev period before 2023, and the Archivist should add the start dates of these two series to the catalog known_biases. H-0043's question about credit ETF gaps stays untested.

*confidence medium · 2026-10-26 · librarian · evidence: H-0075; H-0043; K-0062*

### K-0102 Sharadar SFP lacks some delisted closed-end funds (CH, MF); the card's data check passes at DATA_READY and fails at G0

H-0078 listed 24 fund-ETF pairs on purpose to include delisted funds and reduce survivorship. G0 failed the first version with g0_no_data_adapter because CH and MF are not in Sharadar SFP; the owner removed both pairs and re-filed (version 2), about 14 hours of cycle time lost. The catalog check (DATA_READY) resolves datasets, not tickers, so a Scout cannot see ticker coverage before G0. What stays unchecked: ticker reuse after delisting (GCH, IF, TTF, SGF, ISL, TKF, BZF, SNF, and others were flagged by the card and the Builder), because the Builder has no data access and no gate reports first and last price dates per ticker. Missing funds are a data limit, not a selection, but they lean the fund list toward survivors, in the same direction as H-0077's list. Suggested: the Archivist reports first and last price date per instrument for named-ticker cards, and the DATA_READY check flags tickers absent from the dataset.

*confidence medium · 2026-10-27 · librarian · evidence: H-0078; H-0077; K-0099*

### K-0108 Signal-only history before the view start is not delivered (H-0082 G0 crash)

H-0082 requested Binance spot bars from 2017 as a signal-only dataset to verify that each coin's spot pair was seasoned before its perp launch. Its first G0 run crashed with 'view starts too late for the spot seasoning rule'. The Builder reports that the view starts at the perp panel's first date, so the spot history before it is not delivered although the catalog holds it. This is the direct confirmation of the cause K-0075 inferred from event counts (H-0061, H-0062). Consequences: any rule on a first-bar date, listing age or long look-back taken from a signal-only dataset is left-censored at the view start. The Builder had to count seasoning only when provable, so perp launches in roughly the first half year of dev were excluded for a data reason (longer for a stricter age rule), and G2 neighbors with a longer age lose early events too. The G1 event count came out well under the card's forecast, which fits but does not prove this cause. Scouts should compute first valid dates from the view start, not from the dataset range; the owner should either deliver signal-only history before the dev start or state in the catalog that it is not delivered.

*confidence medium · 2026-10-28 · librarian · supersedes K-0075 · evidence: H-0082; H-0061; H-0062; K-0075; K-0020*

### K-0124 lab fetch takes seconds for single-endpoint sources, minutes for paginated ones

Existing fetchers run in 0.4-7 s (FRED, Treasury, DefiLlama, CFTC: 1-4 requests). Upbit answers in ~1.4 s per request, so a paginated multi-coin source (~300 requests, estimated, not run) needs ~7 min; the framework caps a fetch at 900 s and 500 requests. Archivist timeout_s was raised 1500 -> 3000 on 2026-11-01 because 3 Archivist runs timed out at 1500 s after the Bash backgrounding fix (K-0116); whether 3000 s is enough is not yet known.

*confidence low · 2026-11-01 · human · evidence: lab/framework/ingest.py FETCH_LIMIT_S, MAX_REQUESTS; commit 51dd2563*

### K-0168 sharesbas is split-adjusted to today, close_u is not: shares x close_u overweights later splitters (H-0113)

H-0113 built its S&P 500 core from sf1_arq_sharesbas x close_u. In 2010 that gives a total cap of 24.9 tn USD (real ~6 tn) and a top-10 weight of 60 % (real ~20 %): Apple 2010 = 25.5 bn shares (after the later 7:1 and 4:1 splits) x 251 USD (before them). Names that split later, the later winners, are overweighted, so the weights carry future information. A split-consistent core (shares x split-adjusted price anchored at the last close) gives 8.2 %/yr vs SPY_TR 7.95 % on dev 1998-2020 (diff -0.3 %/yr, t -0.3); the raw core gives 15.5 % (diff +7.4 %/yr, t 4.2). The whole G1-G3 margin of H-0113 was this mismatch; the model window added +0.24 %/yr (t 0.35). The framework look-ahead test cannot see it: the share counts are in the dataset as stored. Other cards use sf1_*_marketcap (point-in-time at filing), and H-0033 anchors marketcap and grows it by total return; none multiplies sharesbas by close_u. Rule for Builder: never build market cap as sharesbas x close_u; use sf1_*_marketcap anchored at its last change and grown by total return, or an equivalent split-consistent price.

*confidence high · 2026-11-09 · human · supersedes K-0167 · evidence: H-0113 g3_dsr 2026-11-08; scratchpad core/core_diag.py (dev rows only, holdout unread)*

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

### K-0074 The G1 abnormal return is raw against unmarked peers, so a bucket with a different beta tests partly market direction

The G1 event study compares marked instruments with unmarked peers in raw returns. A card whose distinctive claim is beta neutrality (H-0054, sizing enters only in the strategy), or whose marked set differs in beta from its peers (young coins in H-0058, R&D-intensive growth stocks in H-0059), is therefore tested on a bucket that also carries market-direction exposure. In H-0058 the sign followed the market cycle across the thirds (young coins led in the boom, lagged in the bear), as a beta difference would produce; in H-0059 the same-stock placebo removes the level of the style but not a time-varying beta. Inferred from the metrics, not tested with a beta-matched control. A Scout should state the expected beta gap between marked and peers and choose peers or events that neutralize it. Whether G1 should offer a beta-adjusted abnormal return is a judge-design question for the owner.

*confidence low · 2026-10-19 · librarian · evidence: H-0058; H-0059; H-0054; K-0025; K-0060*

### K-0076 Slow-horizon event studies: one clear pass, most failures wrong-signed or equal to the placebo mean

Since K-0071, five more cards with a primary horizon of a quarter reached a mechanism verdict: H-0057 (about zero, decaying), H-0058 (wrong-signed), H-0059 (positive but equal to the same-stock placebo mean), H-0060 (wrong-signed) and H-0061 (wrong-signed, one crisis). H-0062 (currency value) passed with a placebo percentile near the top: gross about eight times cost against a bar about seven times cost, growing with the horizon. With the near misses H-0054 and H-0055 that makes eight slow verdicts: one pass, two near misses, five that more power would not cure. K-0071 stands for modest effects, but an effect of that size can pass, and most slow failures were signs or placebo-equal averages. Read a slow failure by its sign and placebo mean first, the bar second. The 60-session cap still shows only the first quarter of year-long claims.

*confidence low · 2026-10-19 · librarian · supersedes K-0071 · evidence: H-0057; H-0058; H-0059; H-0060; H-0061; H-0062; H-0054; H-0055; K-0026*

### K-0078 G2 random-timing null is far below zero in four cards; in H-0063 it matches cost drag

In H-0007 (weekly US stocks), H-0035, H-0040 and now H-0063 (fifteen-minute crypto perps) the random-entry or random-timing check passed with a median Sharpe far below zero. In H-0035 the null was several times worse than the strategy's cost drag; in H-0040 about one and three quarters times. H-0063 allows a sharper reading: doubling costs lowered its Sharpe by about one and a half, which is its cost drag at the model, and the null median (about minus one and eight tenths, 95th percentile still below zero) sits close to that. So the null is roughly 'same trades, zero gross, normal costs', and a percentile of one means only that gross is positive, not that the timing pays for itself: the same book failed doubled costs, delay and neighbors. The null's turnover, exposure and cost per trade are still not reported next to the strategy's, so for H-0007 and H-0035 the gap to cost drag is unexplained. Read a pass as 'gross above zero', never as evidence of tradable timing; an all-negative null makes the percentile uninformative. The question to the owner from H-0035 stays open.

*confidence low · 2026-10-19 · librarian · supersedes K-0061 · evidence: H-0007; H-0035; H-0040; H-0063; K-0017*

### K-0120 Scout forecasts of events, regime days and cost ran too high in six cards

H-0049, H-0050, H-0077, H-0089 and H-0090 forecast far more events than counted (K-0069, K-0098, K-0113); H-0090 forecast 600-1200, counted 66 on 30 dates. H-0081 adds: the VIX gate was expected on 35-45 percent of dev days, 400-500 dates, 4000-6000 events; G0 and G1 counted 1802 events on 207 dates, about a fifth of days (inferred). The card put the liquid-perp round trip at 25-35 bps; G1 used about 91 bps (cost model not read). Gross per event was about 3 bps, so it died on either cost. Scouts should count regime days from the actual series inside the usable period, state events per year per condition and intersection, and take the round trip from the catalog cost tier of the weakest instrument. Suggested: G0 flags events below the G1 minimum and reports events by year.

*confidence medium · 2026-10-31 · librarian · supersedes K-0114 · evidence: H-0081; H-0090; H-0089; H-0077; H-0049; H-0050; K-0069; K-0098; K-0113; K-0114*

### K-0125 Mean exposure of a filing-season event book came out well below entries x hold x weight / sessions

K-0104 found that mean exposure of slot books fits entries x hold / sessions x slot weight (H-0078, H-0079). H-0092 does not fit: 16204 entries, hold 5, slot weight 0.04, 5437 decision dates (5787 observations) give an arithmetic gross of 0.56-0.60, while G1 reported a mean exposure of 0.36, and the card forecast 0.4-0.55. Two readings, not separated: the card's gross cap of 1.0 scaled down all entries of peak filing-season sessions (events cluster in a few weeks per quarter), or the reported exposure nets long against short positions. Either way, a clustered event book is invested less than the arithmetic says, and the scaled slots are smaller than the card's weight. Cards with clustered events should forecast exposure from the busiest weeks, not from the annual average, and state which exposure definition they use. The same card forecast events (about 14000-16000, counted 16330) and the round trip (12-15 bps forecast against about 12 bps modelled) correctly, which fits the advice of K-0098 and K-0120. The exposure definition question of K-0101 stays open.

*confidence low · 2026-11-01 · librarian · evidence: H-0092; H-0078; H-0079; K-0104; K-0101; K-0098; K-0120*

### K-0134 Learned-model cards: a column set defined by the whole-view universe makes walk-forward scores depend on the future

H-0095 failed G0 look-ahead on its first run (truncation at 4 of 11 cuts, perturbation at 8). Cause per the Builder: features, labels and walk_forward were restricted to columns that are ever a universe member over the whole view. That set depends on future membership, and gbm fits depend on the column layout, so scores on earlier rows changed when a stock joined later. Fix: run on all columns and mask non-members per row; a regression test with late entrants and late-starting event data now covers it. Rule for learned cards: no column, row or training-sample subset may be chosen with information from the end of the view (ever-member lists, full-period filters, survivors). The synthetic lab try market could not show this, only G0 on real data did. H-0096 passed G0 on its first run.

*confidence medium · 2026-11-04 · librarian · evidence: H-0095; H-0096; run 275; run 276; K-0009*

### K-0147 A rare-state mechanism that passes its event study cannot carry a daily learned book (H-0106)

H-0106 is the first learned daily book whose mechanism test passed clearly: 5563 flow-day entries on 325 dates, placebo percentile 0.99, gross per entry about 28 bps against a round trip of about 13.5 bps. The book traded 124656 entries, so flow-day entries were about 4.5 percent of them. G1: gross Sharpe about 0.01, net about -0.76, long leg +0.22, short leg -0.61, cost drag about 4.7 percent a year. The other entries earned nothing and the short leg lost, as the card's own book arithmetic (K-0064) predicted. Rule: a rare-state mechanism needs an always-invested tilt or an event-slot book (K-0005, K-0027) with the expected annual gain written down. Forecast check (K-0144): event count and flow days were in range, gross per entry was above the central forecast, the first learned card not below its pessimistic case. Suggested for G1: report book gross for mechanism entries and for all others.

*confidence low · 2026-11-06 · librarian · evidence: H-0106; H-0094; H-0099; K-0027; K-0064; K-0097; K-0135; K-0144*

### K-0161 Book-change cards are gated by an entry event study; base score may be a rebuild

H-0117 was filed to test two book changes (a rank buffer and beta-neutral leg sizes) on a score that had passed in H-0115. Two things made that unanswerable. (1) G1 runs the mechanism test first and stops when it fails (strategy_not_run), but the mechanism test sees only new leg entries; continuing holdings, the buffer's purpose, and the leg sizing are visible only in the strategy run. A card whose claim is a book change therefore can die on a test that does not measure the change, and the card's own forecast named the absolute Sharpe floor as the likely death, not the event test. (2) The Scout states it did not see the exact H-0115 feature windows or alpha and rebuilt them from the lesson text, so 'same score, so the mechanism test reproduces a known pass' could not be checked and was not borne out. Suggested: when a card builds on an earlier pass, the Scout gets the earlier spec (features, windows, parameters), and a book-change card may state whether it wants the strategy run even if the entry test misses. Also: the registry code shows only the percentile failure while the cost check failed too (K-0069, K-0098).

*confidence low · 2026-11-08 · librarian · evidence: H-0117; H-0115; K-0007; K-0021; K-0069; K-0098*

### K-0163 An empty first G1 block also comes from model warm-up on short windows, and the registry hides it

K-0145 explained an empty first block by a dataset starting after the default dev start (8-K flag from 2004). H-0120 shows a second cause. It was a crypto learned card on 2020-2022 with a refit every 63 rows: G1 counted 0, 165 and 336 events in the three blocks, mechanism_blocks was 1 against a minimum of 2, with only the latest block positive. The first fit needs a minimum training sample, so the first third of a three-year window holds no scored entries. This fits the reading of K-0145 that an empty block cannot count, so a short-window learned card needs both later blocks positive; the Scout foresaw it in the card. The registry shows only the percentile code, although the block and cost checks also failed (K-0069, K-0098). Scouts of learned cards on windows of about three years should state which block will be empty and expect the block check to bind; suggested: report events per block at G0 and list every failing check in the registry reason.

*confidence low · 2026-11-09 · librarian · supersedes K-0145 · evidence: H-0120; H-0105; H-0102; H-0104; K-0145; K-0069; K-0098; K-0006*

### K-0172 Learned dollar-neutral books: the long leg earns and the short leg loses in four cards, also with beta-balanced legs

In four learned dollar-neutral or beta-neutral weekly books whose event test passed, the G1 decomposition shows a positive long leg and a negative short leg: H-0106 (K-0147), H-0114, H-0115 and H-0126. The earlier reading for the first three was market exposure or a sector tilt (K-0074). H-0126 weakens that reading: its legs were built inside beta terciles and balanced by construction, and the short leg still lost. So the pooled event average (side-adjusted, against raw unmarked peers) probably leans on the long side, and the book pays for a short leg that adds cost and no return. The event study reports no side split, and the card of H-0126 named none, so this is inferred from leg Sharpes, not tested. A Scout writing a learned neutral card should state what it expects from each leg, and a Builder or G1 should report the event average by side. A long-only tilt on an always-invested core avoids the short leg but keeps the limits of K-0064 and K-0153.

*confidence low · 2026-11-10 · librarian · evidence: H-0126; H-0114; H-0115; H-0106; K-0147; K-0074; K-0064; K-0153*

### K-0178 A learned crypto card with a VIX input did score: warm-up and VIX leading NaN no longer explain H-0124's zero events

K-0170 read H-0124 (learned crypto perps card, VIX and BTC-trend state) as probably never scored: G0 passed with zero events in 0.19 s and G1 counted none. H-0130 (learned crypto spot card, top twenty pairs, 2018-2022, six features each needing at most sixty days, VIX and 10-year real yield as raw date-level inputs, walk-forward refit every 63 rows) passed G0 with 554 events in about 12 s, and G1 counted the same 554 events on 118 dates, all with controls. Its first third of dev held no events (model warm-up, K-0163), the later two had 88 and 466. So a leading NaN in the VIX series and the minimum training sample are not sufficient causes for H-0124. The Builder's safeguards worked here: date-level features fed raw rather than through the cross-sectional z-score helper, coin features needing at least 45 of 60 pairs rather than all sixty, and non-members masked per row (K-0134). Candidates left for H-0124 (none verified): the perp universe the gate hands over, the all-valid sixty-day return rule, the BTC-trend feature. Also new: the card asked for event averages split by risk-off, risk-on and calm dates, and G1 reported none of them (K-0069).

*confidence low · 2026-11-11 · librarian · supersedes K-0170 · evidence: H-0130; H-0124; H-0120; K-0170; K-0163; K-0134; K-0069; run 347; run 348*

### K-0182 Disagreement events against a ridge control: six cards, no nonlinear gain, now also in crypto perps

The design of K-0138 and K-0142 (event = picks only the model makes against the linear control) has now run in six cards: H-0100, H-0103, H-0104, H-0105, H-0130 and H-0134. H-0134 is the first on a monthly horizon in perps. Its gbm-only picks did worse than the ridge-only picks at every horizon, below the placebo mean, with the sign set by the cycle (clearly negative in the 2021 block, slightly positive in 2022) and an empty first block from warm-up (K-0163). Three of the six are wrong-signed (H-0103, H-0105, H-0134), the rest positive but inside the cloud or far under cost; none beat its linear control. G0 counted a few percent more events than G1 (793 and 767), as in K-0095, on only 77 weekly dates, so the event count can pass while the independent sample is small. The ridge side was never run as a book, so a wrong-signed result does not say the linear score pays, and its sign is dev-selected. Crypto momentum itself remains unscored in the lab (H-0124 never scored). A new learned card should state what it concludes if the ridge side wins and ask for the linear form as the primary claim.

*confidence low · 2026-11-12 · librarian · supersedes K-0142 · evidence: H-0134; H-0130; H-0105; H-0104; H-0103; H-0100; K-0142; K-0138; K-0163; K-0179; K-0095*

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

### K-0051 The point-in-time universe rule also forces early exits in crypto perp cards

K-0032 recorded that the engine refuses weights outside the declared point-in-time universe for stock events. H-0035 shows the same rule on crypto_top_n: a position held across a universe update was refused at G0, and the Builder had to close such trades at the first decision row outside the universe, contradicting the card's 'closed on schedule' clause. The deviation touched few trades, but the clause cannot be honored and a Scout should not promise it. Event cards on a rolling top-n universe should state an exit rule that tolerates leaving the universe. Separately, the intraday data has no forward source, so such cards can pass at most G4.

*confidence medium · 2026-10-14 · librarian · evidence: H-0035; H-0018; K-0032; catalog binance_perp_15m*

### K-0054 H-0038 had events at G0 and none at G1; H-0039 got no universe at G0

K-0047 said G0 events_in_dev matches the G1 event count. That still holds for H-0036 and H-0039 (counts agree to within a fraction of a percent) and for H-0037 (1040 against 1006). It fails for H-0038: G0 counted 241 events and passed its look-ahead cuts, G1 counted zero events with a null abnormal return, so the mechanism was never measured. Possible causes, unconfirmed: a two-instrument universe (TLT, SHY) leaves no peer set for the abnormal return; a tradable or universe mask is missing for the generic auction dataset (the catalog gives it no cost model and calls it signal only); an event definition used by G0 but not G1. Separately, H-0039 first crashed at G0 because the gate passed no crypto_top_n universe on the daily perp data; the Builder computed its own top twenty by 30-day dollar volume, so the event set may differ from the judge's universe definition. Skeptic messages are again absent for all four cards. K-0047 stands for its other points.

*confidence low · 2026-10-15 · librarian · supersedes K-0047 · evidence: H-0036; H-0037; H-0038; H-0039; K-0047; K-0051*

### K-0057 lab try knows instruments named by the strategy (intraday crypto_top_n)

Intraday crypto_top_n cards may name a leader or exclusion (e.g. BTCUSDT, ETHUSDT) in strategy.py. Since commit 405b0961 the Builder's lab try adds instrument-like string literals (…USDT) to the synthetic market; before, data.col('BTCUSDT') crashed in lab try (synthetic names were AAAUSDT..FFFUSDT) although real G0 on real data was unaffected. A Builder rewrite after such a block may not reuse earlier code (old workspace files are gone), so re-verify parity with the card's reference.

*confidence high · 2026-10-15 · human · evidence: H-0040 message 288*

### K-0069 G1 event-count code can hide a measured mechanism failure; date tripwires cannot fire at G0

H-0049 (193 events against a minimum of 200) and H-0050 (172 marked events on 43 event dates) are both registered as g1_mechanism_events, yet G1 measured the mechanism in both: wrong-signed average, placebo percentile far under the bar, negative cost ratio, weak latest third. A Scout reading the registry takes them for sparsity problems more events would cure; the sign says otherwise (the opposite of K-0062). New in H-0050: the card wrote its tripwire in event dates (flag if under 50 dates), but G0 reports marked events (four ETFs per date), so it showed 172 and the tripwire could not fire; the dates appeared only in G1 metrics. Both Scouts forecast event rates far above the real count. Skeptic messages were absent again. Suggested: list all failing checks in the registry reason; report event dates next to marked events at G0 and flag counts under the G1 minimum; Scouts state expected dates and marked events separately.

*confidence medium · 2026-10-17 · librarian · supersedes K-0067 · evidence: H-0049; H-0050; K-0062; K-0054; K-0032*

### K-0082 lab try cannot exercise cards that need the 8-K flag: its synthetic ea_reaction_day never equals one

The Builders of H-0064, H-0066 and H-0067 each report that the synthetic ea_reaction_day in lab try is uniform noise between 0 and 10, never exactly 1, so lab try shows zero events and its checks are vacuous for these cards; they tested events, book and point-in-time behaviour on hand-made data (one temporarily rounded the flag). On real data G0 did report events (counts within a handful of G1's) and no look-ahead failures in 25 cuts, so G0 is the first meaningful check, and an event-logic bug would surface only there. Suggested: make the synthetic flag a sparse 0/1 event series so lab try produces events.

*confidence medium · 2026-10-20 · librarian · evidence: H-0064; H-0066; H-0067; K-0054; K-0057*

### K-0091 A card whose data period starts before the dataset's first row is blocked and cannot be revised in BLOCKED_DATA

H-0051 asked for the CFTC dataset from 2004-01-02, three days before its first report (2004-01-05). The card went to BLOCKED_DATA on the missing dataset, and the later period mismatch could not be fixed there because BLOCKED_DATA cards cannot be revised; the owner had to park it, reopen it and file a one-line version 2, about eight days after the card was created. Nothing else changed in the card. A Scout should read the first date of a new dataset in the catalog or from the Archivist before writing the period, and the framework could allow period-only revisions in BLOCKED_DATA or flag a period that starts before the dataset when the card is specified.

*confidence medium · 2026-10-25 · librarian · evidence: H-0051 transitions 2026-10-17 to 2026-10-25*

### K-0092 lab try cannot exercise weekly-report cards on the CFTC dataset: its synthetic COT series are daily random walks

The Builder of H-0051 reports that the synthetic cftc_cot_legacy_1d series in lab try are daily random walks, so no pair of reports exactly seven days apart exists and the book has zero trades and zero events there; it tested legs, weights, the seven-day rule, fewer than six eligible ETFs, late listing and truncation on hand-built weekly reports. On real data G0 reported events (about 2100) and no look-ahead failures in 25 cuts, and G1 counted nearly the same events, so G0 was the first meaningful check. Same kind of gap as K-0082 (synthetic 8-K flag never equal to one). Suggested: make synthetic report series weekly, stamped with a Friday release, so lab try produces events. The Builder also used the loader's usable-session date as the release date; whether the loader stamps holiday and shutdown delays correctly was not measured by any gate.

*confidence medium · 2026-10-25 · librarian · evidence: H-0051; K-0082*

### K-0095 PIT S&P 500 mask now reaches liq_n cards; G0 counted more events than G1

K-0032 recorded that H-0022 could not run because the PIT S&P 500 mask was not delivered to LIQ-500 cards. The owner reopened it after extras['sp500_member'] was deployed (prod ba7c4c2), the Builder read it directly, G0 passed with no look-ahead failures in 25 cuts, and G1 measured the mechanism, so the both-direction index-change test is feasible and a deleted stock that stays liquid can be held. The other parts of K-0032 stand: events on young or departing stocks that never sit in the declared universe give an empty set (H-0021), and weights outside the declared universe are refused (H-0018, K-0051). New: G0 counted 453 events in dev and G1 measured 395, about an eighth fewer, with zero events lacking controls. Cause unconfirmed: the G1 window start, the forward window at the end of dev, or a peer rule like K-0086. Scouts should expect G1 to cover somewhat fewer events than G0 reports.

*confidence medium · 2026-10-25 · librarian · supersedes K-0032 · evidence: H-0022; H-0018; H-0021; K-0032; K-0051; K-0086*

### K-0097 G1 now gives a gross/net split; H-0044 shows a positive event study beside a negative gross book

Earlier lessons (H-0028, H-0031) said the gate gave no gross versus net or long versus short split. H-0044's G1 result has one: gross Sharpe, net Sharpe, each leg's Sharpe and the yearly cost drag. It shows what K-0064 suspected: costs were a minor part of the loss. The book was negative before costs and in both legs, while the event study of nearly the same window was positive and just above the placebo bar. So the gap between study and book is construction (weights set by signal size versus equal weights by sign, exit at the next open versus the fix-day close, one shared dollar bet across six currencies) or a bug, not cost. Also: when a strategy is run, the G1 checks list holds only strategy checks, so a mechanism pass or near miss is visible only in the metrics block. Builders should report the event-study mean next to the book's gross mean per window on the same dates. A reconciliation question went to the owner.

*confidence low · 2026-10-26 · librarian · evidence: H-0044; H-0028; H-0031; H-0046; K-0064; K-0027*

### K-0098 The registry shows one failed G1 check; sparse event-slot books also need exposure arithmetic in the card

H-0077 is registered as g1_mean_exposure (mean exposure about 0.11 against a floor of 0.2), but its G1 result also failed the absolute Sharpe floor (net about 0.39, gross about 0.58, floor 0.5), while the bootstrap bound and excess checks passed. A Scout reading only the registry code would take it for a sizing problem that bigger slots cure; since Sharpe is roughly scale-free, they would not. Same pattern as K-0069 (the code hides what was measured). Cause of the thin exposure: G1 counted 658 events on 278 dates against a forecast of 1300 to 2700, and the book capped at five slots of seven percent each, so the average book held a fraction of the forecast. Rule for cards: write expected events per year, hold, slot weight and slot cap, and compute mean exposure (events x hold / 252 x weight) next to the absolute Sharpe floor, and state that gross Sharpe must clear the floor with margin before costs and hedge drag. Suggested: list every failing check in the registry reason.

*confidence medium · 2026-10-27 · librarian · evidence: H-0077; K-0069; K-0027; H-0016*

### K-0104 Mean exposure of paired slot books fits entries x hold / sessions x leg weight

H-0079 adds a second data point to K-0101. The card forecast mean exposure 0.25-0.30 on a one-leg count and 0.5-0.6 on gross; G1 reported 0.339 with 1192 entries over 4783 sessions, hold 5, weight 0.25 per leg. The formula entries x 5 / 4783 x 0.25 gives 0.31. H-0078 reported 0.199 with 813 entries over 5283 sessions; the formula gives 0.19. Both readings left open in K-0101 (entries count slots and exposure one leg, or entries count legs and exposure gross) give this same formula, so it can be used for forecasting while the definition stays unconfirmed. Until the owner answers, cards should compute exposure this way. The question in K-0101 stays open.

*confidence low · 2026-10-28 · librarian · supersedes K-0101 · evidence: H-0079; H-0078; H-0077; K-0101; K-0098*

### K-0106 lab try cannot exercise cards that pair perps with spot: the synthetic market has no spot columns and no '.P' ids

The Builder of H-0082 reports that the synthetic market in lab try has no spot instruments and no '.P' perp ids, so the spot-seasoning rule finds no pairs and lab try shows zero events and zero exposure; the logic was tested on hand-made data. This is the third data family with the same gap after the 8-K flag (K-0082) and the weekly CFTC reports (K-0092). On real data G0 then crashed once on the missing spot history (K-0075) and, after the fix, reported events within a handful of the G1 count (366 against 363) and no look-ahead failures in 25 cuts. So the first meaningful check of such cards is G0 on real data, and a data problem costs a full cycle. Suggested: give the synthetic market spot twins for its perps with a '.P' naming, listed at staggered dates, so lab try can produce events.

*confidence medium · 2026-10-28 · librarian · evidence: H-0082; K-0082; K-0092; K-0075*

### K-0115 Subscription 'session limit' errors now pause agent runs

The Claude subscription window error ('You've hit your session limit, resets 7am UTC') was not in RATE_LIMIT_WORDS, so no pause followed and every 30-minute cycle burnt another agent slot (4 Scout runs on 2026-10-30, 2 Builder runs on 2026-10-27). Since commit dac6ed66 the phrase triggers the same rate_limit_pause_hours pause as a 429.

*confidence high · 2026-10-31 · human · evidence: invocations 2026-10-27, 2026-10-30; commit dac6ed66*

### K-0116 Long Bash commands are waited for, not backgrounded (Archivist lab fetch)

Claude Code moves a Bash command that outlives its timeout (2 min default) to the background; the Archivist then polled it with Monitor, sleep and Read of /tmp, each a policy violation, and four runs a night were discarded from 2026-10-27 on, which left every BLOCKED_DATA card stuck. Since commit d980f31c agents run with BASH_DEFAULT_TIMEOUT_MS and BASH_MAX_TIMEOUT_MS of 15 minutes. The cause is inferred from the violation pattern, not from a transcript; if Archivist runs still end in policy violations after 2026-10-31, it is wrong.

*confidence medium · 2026-10-31 · human · evidence: alerts 597-600, 623-626, 661-663, 694-696; commit d980f31c*

### K-0117 lab try adds instruments named in the strategy to the daily crypto_top_n market

A daily crypto_top_n card that reads a named instrument (the BTC factor BTCUSDT.P, a leader) crashed lab try G0 because the synthetic market held only SYNxxUSDT names, although real G0 was unaffected. Since commit 51536516 the daily lab try adds the instrument-like literals of strategy.py and diagnostic.py, as the intraday one already did (K-0057); the .P suffix is accepted.

*confidence high · 2026-10-31 · human · evidence: H-0081 message 641; commit 51536516*

### K-0122 Scout wall-clock timeouts: 15 % of runs hit 900 s; a normal run takes ~430 s

Between 2026-10-05 and 2026-11-01, 16 of 107 completed Scout runs ended in a wall-clock timeout at the default 900 s (Builder, Librarian: none), clustered by day (10-27: 5, 10-30/31: 2 each), and on 10-30/31 timeouts were followed by session-limit failures. The first Scout run after the fix took 431 s and 22 turns, so 900 s left only ~2x headroom. Scout now has timeout_s 1500 and is told to do at most 3 WebSearch calls; the cause of the clusters (slow API vs a stuck search) is not verified. lab invocations now shows wall time and turns.

*confidence medium · 2026-11-01 · human · evidence: lab-prod invocations 2026-11-01; commit e6c24ced*

### K-0123 Rate-limit pause ends at the reset time named in the error, not after a fixed 5 h

A session-limit failure ('resets 7am (UTC)') now pauses agent runs until that reset time (orchestrator.pause_until), capped at rate_limit_pause_hours; a message without a reset time still pauses for the full 5 h. Before, a limit at 06:35 blocked runs until 11:35 although the quota returned at 07:00. Untested in production until the next session limit.

*confidence medium · 2026-11-01 · human · evidence: commit e6c24ced; K-0115*

### K-0131 G0 still passes no crypto_top_n universe on daily perp data (four cards)

H-0076 and H-0085 both crashed their first G0 with data.universe None on binance_perp_1d, as H-0039 and H-0081 did earlier, so the gap has recurred in four cards over about three weeks. Each Builder wrote the same fallback (top n by trailing 30-day dollar volume), which may differ from the judge's definition, so the event sets may not be the judge's. In H-0076 the fallback then failed G0 seven more times on truncated look-ahead views of only 18 rows: a loop over the volume window assumed at least as many rows as the window, and the cohort shift had the same pattern. The Builder cites run numbers 188 to 263 for the crash and fixed it with min(window, rows). The card was DATA_READY on 2026-11-02 and IMPLEMENTED on 2026-11-03, so about a day was lost on top of the data wait. Rolling-window code must accept views shorter than its window, because G0 cuts can be that short. Suggested: the gate passes the card's declared crypto_top_n universe on daily perp data, and G0 reports which universe the strategy used. Question sent to the owner.

*confidence medium · 2026-11-03 · librarian · supersedes K-0118 · evidence: H-0076; H-0085; H-0039; H-0081; K-0054; K-0118; run 188; run 263; run 264*

### K-0135 Learned-model cards: lab try cannot run them and the ridge control never runs

Two gaps, now seen in three learned cards (H-0094, H-0095, H-0096). (1) lab try: besides the universe of fewer than 50 members (K-0132), H-0095 shows that the synthetic 8-K flag is too sparse to reach walk_forward's minimum training sample (zero events) and that synthetic dollar volume has no cross-sectional dispersion, so a volume z-score is NaN. The Builder exercised the model path only by temporarily densifying events and by a regression test. On real data G0 was again the first meaningful check (event counts at G0 and G1 agreed within a handful: 8587 vs 8587 for H-0095, 15582 vs 15577 for H-0096). (2) Each of the three cards names a linear ridge baseline as a falsification criterion, and each ended at the G1 mechanism test before it could run. H-0096 asked the Skeptic or Librarian to have it run; neither has a command for it. So whether nonlinearity pays is untested in all three, and a Scout cannot tell a wrong learner from wrong features. Suggested: G1 reports the ridge event gross and percentile next to the model's whenever the card names a ridge baseline; the synthetic market gets a dense 0/1 8-K flag and volume dispersion.

*confidence medium · 2026-11-04 · librarian · supersedes K-0132 · evidence: H-0095; H-0096; H-0094; K-0132; K-0082*

### K-0153 Core-plus-tilt books have no core-only control, so a margin over SPY cannot be attributed to the tilt

H-0033 (positive-reaction tilt) and H-0113 (quarterly share-update tilt) are long-only books on a cap-weighted core with a tilt held a small part of the time, benchmarked against SPY total return. G1 reports no core-only run, no return of the tilt-window sleeve and no tracking of the core against SPY. In H-0113 G1 and G2 passed with a Sharpe margin over SPY far larger than its event study could supply: the event average was concentrated in the earliest third and about zero or negative in the middle third, yet the book beat the benchmark in every third; neighbors, an alternative universe and doubled costs all passed with the same Sharpe, and the random-entry null had a median Sharpe near zero for an always-invested book, which is odd (K-0078). The core is built from share counts times unadjusted close, with names leaving the index sent to cash, not from SPY, so the margin may come from core construction, a bug, or a window effect much larger than the events show. None of these is separated by the outputs. Suggested: G1 reports the core-only Sharpe and the window sleeve's return; the Builder reports the core's correlation with SPY. A question went to the owner.

*confidence low · 2026-11-08 · librarian · evidence: H-0113; H-0033; H-0107; K-0097; K-0078; K-0064*

### K-0154 First G3 verdict: DSR failed at the hurdle, and the family trial count seems to include grid points

H-0113 is the first card to reach G3 and the first to fail it. The deflated Sharpe came out just above one half, with the per-period Sharpe about equal to the hurdle (hurdle above the benchmark's own Sharpe) and a probability of backtest overfitting a little above one half. The result lists 12 family trials where the family index-inclusion-price-pressure holds three gated hypotheses (H-0017, H-0022, H-0113) and one parked (H-0018); 12 fits three gated hypotheses plus the nine grid configurations of this card (3 x 3), but this is inferred, the code was not read. If grid points count as trials, a card with a full grid starts with nine and each earlier family member adds to the hurdle. A book whose Sharpe merely clears the hurdle gets a DSR near one half by construction, so G3 is decided by the margin over the hurdle, not by passing G1 and G2. Reusing a family on purpose (as the Scout did) is the honest choice and has this price. Open: how n_family_trials is counted, and whether the benchmark's Sharpe enters the hurdle.

*confidence low · 2026-11-08 · librarian · evidence: H-0113; H-0017; H-0022; H-0018; K-0059; K-0087*

### K-0174 H-0127 supports the three-peer rule as the cause of H-0038's empty G1 event set

K-0086 inferred that G1 needs at least three unmarked peers per date (H-0041, H-0072) and K-0054 listed it as one possible cause of H-0038, whose universe held only TLT and SHY: G0 counted 241 events, G1 counted none. H-0127 filed the same Treasury auction cycle with five unmarked short-Treasury ETFs and a start date after the third peer lists. G0 counted 404 events and G1 measured 404, all with controls, so the mechanism was measured for the first time and the verdict is a measured negative. This supports the peer rule as the cause for H-0038 (inferred; gate code not read) and shows that G0 and G1 counts agree when peers exist from the start of the window. Still true from H-0072: when peers list late, G0 counts more events than G1. A card that marks every instrument it names, or leaves fewer than three unmarked, will read as zero events at G1 although G0 passed. Scouts should list the peers and the first date with three unmarked.

*confidence medium · 2026-11-11 · librarian · supersedes K-0086 · evidence: H-0127; H-0038; H-0072; H-0041; K-0086; K-0054; K-0170*

### K-0175 lab try cannot exercise two more data types: the Treasury auction calendar keys and the deposits field

Same gap as K-0082 (8-K flag), K-0092 (CFTC weekly reports) and K-0106 (perp and spot pairs). The Builder of H-0127 reports that the synthetic treasury_auctions_1d series never hit the exact integer session counts the card tests (next_coupon5_sessions equal to four, last_coupon5_sessions equal to zero), so lab try showed zero events and zero trades; the logic was tested on hand-made calendars (lead, cluster rules, overlap, missing price, each hold). The Builder of H-0129 reports that every synthetic stock has deposits above zero, so the card's bank exclusion removed all of them and lab try showed zero trades; it ran once with the filter off and passed G0 with 25 cuts and 3099 entries. On real data G0 then reported events (404 for H-0127, 42242 for H-0129) and no look-ahead failures, and G1 counted nearly the same, so G0 on real data was again the first meaningful check. Suggested: give the synthetic auction calendar integer countdowns and make deposits missing for most synthetic stocks.

*confidence medium · 2026-11-11 · librarian · evidence: H-0127; H-0129; K-0082; K-0092; K-0106; K-0135*

### K-0193 Intraday cards: a control built into strategy.py as a parameter still does not run; four cards, no mechanism evidence

K-0191: owner diagnostics written as separate modules are never run, and the intraday judge reports no side split or gross per trade. H-0140 took the other route: the pre-registered control was a side_mode argument (with, against, all) in the strategy function, same code path, to be run by the Builder or Chair. G1 judged only the primary run and rejected the card, so the against and all runs, side split and thirds were again not produced: a control exists only if the judge executes it. Gross per trade was again derivable only from mean net trade and cost share. The Builder had no timezone database and hand-coded the US daylight-saving rule (tested on March and November dates), and its 'all' mode still required a valid pre-session scale, so it is not exactly H-0137's trade set. Four cards (H-0137 to H-0140) have left no judge-produced evidence on side, anchor or control. Suggested: a judge-run controls list for intraday cards (named argument values run on dev, reporting trades, gross and net per trade, side split, thirds), counted as part of the card's trial, and gross per trade in the G1 metrics.

*confidence low · 2026-11-13 · librarian · supersedes K-0191 · evidence: H-0140; H-0139; H-0137; H-0138; H-0089; K-0186; K-0188; K-0191*

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

### K-0087 Family perp-leverage-crowding merged into short-term-reversal; H-0039 moves with it but tests a different mechanism

On the Librarian's question the Chair staged a merge of perp-leverage-crowding into short-term-reversal, because H-0071 shares the forced-versus-informational-flow question with H-0049, H-0035 and H-0006. Families cannot be split, so H-0039 moves too, which only makes the multiple-testing penalty stricter. H-0039 itself is a different mechanism: a cross-sectional sort on the trailing funding level (crowded longs against crowded shorts, held for days), not a rebound or fade after a big move. It was decided at G1 on cost, so nothing else changes. A reader of the registry should not take H-0039 as another stress-rebound test; for the mechanism evidence see K-0085. Read lessons filed under the old family name as part of short-term-reversal for the trial count.

*confidence high · 2026-10-21 · librarian · evidence: H-0039; H-0071; H-0049; H-0035; H-0006; K-0085; chair reply to question 531*

### K-0089 Family macro-announcement-premium merged into variance-risk-premium-timing

On the Librarian's question the Chair merged macro-announcement-premium into variance-risk-premium-timing, because the two short-VIXY cards (H-0052, H-0073) both found that same-instrument placebos earned most of the average (K-0088). The merge brings H-0042 and H-0045 into the shared count, so the family now holds six trials (3 + 3), which makes the multiple-testing penalty stricter and changes no past verdict. H-0055 stays where it was. Families cannot be split, so H-0042 and H-0045 are not short-VIXY tests: they are Fed-calendar equity-bond tilts (K-0063), and the mechanism evidence for them stays in K-0063 and K-0070. A reader should treat lessons filed under macro-announcement-premium as part of variance-risk-premium-timing for the trial count only. A new macro-calendar or VIXY card on this family starts with a heavier penalty than its name suggests.

*confidence high · 2026-10-21 · librarian · evidence: H-0052; H-0073; H-0042; H-0045; K-0088; K-0070; K-0087; chair reply to question 541*

### K-0165 Events defined as a subset of a model's own calls ran an order of magnitude below the Scout's count; cost matched

H-0121 marked only the gbm's continuation calls (new entries to a leg whose recent move agrees with the leg side). The card forecast about 4000 to 8000 events; G0 and G1 both counted 547 on 280 dates (about 34 a year), so roughly a tenth of the lower bound. The Scout had written that a model that never calls continuation would fail the event count, but still built the forecast from condition arithmetic: a quarter to a half of leg names. The round trip matched (about 11 bps forecast, 11.4 modelled); gross per event came in near 1.6 bps against a central forecast of 15 bps. K-0142 saw model-defined event counts run above forecast, so the direction is not systematic; the point is that such counts cannot be derived from conditions alone. A card with model-defined events should give a count range from a stated call rate, say what the card concludes at a call rate a tenth of that, and note that the G1 minimum of 200 events and 2 positive blocks may bind. The registry again shows only the percentile code while the cost check failed too (K-0069, K-0098).

*confidence low · 2026-11-09 · librarian · evidence: H-0121; K-0120; K-0142; K-0144; K-0069; K-0098*

### K-0181 H-0131 forecasts: events under the low end, ETF round trip about twice the card's

H-0131 forecast 220-350 event dates, 900-1400 marked events and a 2-6 bps ETF round trip. G1 counted 135 dates, 540 events (about 29 per year, 4 marked ETFs per date) and a 10 bps round trip, the same ETF tier gap as H-0127 (K-0177) and the perp gap of H-0081 (K-0120). The rally state fired on about 3 percent of sessions against a forecast of 5-8 (inferred, not in a G0 output). Gross landed between the pessimistic and central cases, below central, as in K-0171 and K-0177. The card did name its likeliest deaths (placebo cloud, negative latest third) and used the negative placebo mean as a harder bar. Scouts should take the daily-ETF round trip from the gate's cost tier (about 10 bps), count state days from the real series after warm-up, and lead with the pessimistic case when the premise is an unconfirmed pass.

*confidence medium · 2026-11-11 · librarian · evidence: H-0131; H-0127; H-0081; K-0120; K-0177; K-0171*

### K-0188 'Exactly the sibling's rule' cannot be verified: the Builder has no sibling code and G1 does not report gross per trade

H-0138 was specified as H-0137's rule plus one filter, with the instruction that H-0137's implementation wins if it differs from the card text and that deviations be reported. The Builder reported that H-0137's code was not in its workspace and built from the card text, so the claim of an identical rule rests on the text alone (one ambiguity was also noted: the volatility rank window counted the last sessions with valid data, which equals the card text only with complete data). The card's owner criterion compares the mean gross trade of the filtered rule with the unfiltered sibling, but the G1 result carries mean net trade and cost share of gross, not gross per trade or per side; the Librarian had to derive it, and H-0137's own gross is known only from its lesson. Suggested: give the Builder the sibling's code for cards that declare 'same rule plus one condition', and add gross per trade and the long/short split to the G1 metrics for intraday cards, where there is no mechanism channel (K-0186).

*confidence low · 2026-11-12 · librarian · evidence: H-0138; H-0137; G1 run 363; K-0186*

### K-0202 Scout forecasts H-0146 to H-0157: calendar-key dates and confirmation-close counts wrong, gross below central

Twelve cards. Event counts for slow stock sorts were in range or above (H-0146 forecast 3500-6500, counted about 14300; H-0154 forecast 3000-7000, counted 9296) and stock round trips matched (about 12 bps). Two kinds of count ran wrong. H-0157 forecast 250-700 events on 80-130 dates and G1 counted 173 on 18 dates: closures of different exchanges coincide and only the first unanchored session of each closure counts (K-0177). Confirmation-close intraday rules fired far more often than forecast: H-0149 forecast 250-700 trades, counted 2872; H-0150 forecast 1000-3000, counted 5980; H-0147 forecast 400-900, counted 2888; H-0148 (600-1500, counted 1699) was in range. Gross per entry or trade was below the central case in most cards (wrong-signed in H-0151 and H-0157, near zero in H-0156, H-0149, H-0150), and slow cards forecast raw gross without the same-stock placebo that absorbed it (K-0025). Scouts who led with a pessimistic case (H-0147 to H-0150) were closest. Registry codes again show one failed check (K-0069, K-0098).

*confidence low · 2026-11-14 · librarian · supersedes K-0177 · evidence: H-0146; H-0147; H-0148; H-0149; H-0150; H-0151; H-0152; H-0154; H-0155; H-0156; H-0157; K-0177; K-0144; K-0069; K-0098*

## Superseded

- K-0004 No earnings dates in the lab
- K-0028 G0 checks are vacuous when the synthetic market lacks the card's data, and card diagnostics cannot be reported
- K-0029 The engine cannot hold stocks that have left the declared universe, so exit-from-universe events are untestable
- K-0030 A passing mechanism test is necessary, not sufficient: pooled events hide era fade, clustering and re-weighting costs
- K-0031 Compare the placebo bar with the round-trip cost to tell a power problem from an untradable effect
- K-0032 Events on young or departing stocks, or needing index membership, do not fit the point-in-time universes
- K-0033 G0 look-ahead checks are vacuous for event cards, and diagnostics cannot be reported (now five cards)
- K-0036 G0 look-ahead checks are vacuous for event cards, and diagnostics cannot be reported (now seven cards)
- K-0037 Earnings-announcement premium: absent as a tradable effect in large liquid US stocks, with real dates too
- K-0040 G0 look-ahead checks are vacuous for event cards; diagnostics cannot be reported (eight cards)
- K-0041 The trading-window net-flow spread does not survive a move from the filing date to the real release date
- K-0042 A passing mechanism test can still end in a book below cash, even without fade
- K-0043 The market's own release-session reaction predicts same-direction returns in liquid stocks
- K-0044 Weekly return-sort effects in the most liquid US stocks are smaller than cost, in either sign
- K-0045 G0 look-ahead checks stay vacuous for cards with minimum-size rules (now eleven cards)
- K-0047 G0 now reports real event counts and look-ahead cuts for event cards; Skeptic messages still absent
- K-0048 Fixed event slots did not close the gap between a passing event study and a failing market-neutral book
- K-0050 G2 random-timing null is deeply negative in two cards, so passing it may only measure costs
- K-0052 Short-horizon effects that clear the pooled placebo test are usually weakest or reversed in the latest third of dev
- K-0055 The G1 event study needs unmarked peers on each date: marking every universe instrument gives zero events
- K-0056 Stress rebounds: positive on average, near generic drift, weakest lately
- K-0061 G2 random-timing null is far below zero in three cards, so passing it may only measure costs
- K-0062 Zero events can pass G0 and then read as a mechanism failure
- K-0065 Macro-calendar cards on asset-class ETFs fail in three designs; reaction drift is single-stock only
- K-0066 Perp funding does not tell forced crash selling from informational selling on a daily horizon
- K-0067 The G1 reason code can name the event count while the mechanism was measured and failed; the count was visible at G0
- K-0068 Mechanical stock-bond flow: the study effect appeared only when the trigger was a large stock-bond move
- K-0071 Slow-horizon event studies: the placebo bar sits several times above cost, so only a large effect can pass G1
- K-0072 Cross-market information-diffusion claims in liquid assets come out flat or wrong-signed from daily to monthly horizons
- K-0073 Multi-year reversal: absent in country equity ETFs, present for currency value in a short sample
- K-0075 Signal history before the dev start or the ETFs' first date seems not to warm up the first signals
- K-0077 Fifteen-minute crypto perp taker moves: a small edge in both directions, consumed within about a candle
- K-0079 Return-sort effects in liquid US stocks stay below cost, except a pre-release reversal that faded
- K-0080 Information-diffusion claims in liquid assets come out flat or wrong-signed, also with a dated news event
- K-0081 On real earnings dates the events that cleared the placebo test sort on the stock's own price move
- K-0085 Perp funding and the perp-versus-spot volume split both fail to separate forced from informational big moves
- K-0086 G1 needs at least three unmarked peers per date, so early event dates can be dropped at G1 but counted at G0
- K-0099 Wrapper gaps revert where no arbitrage link exists: closed-end funds cleared the placebo test, twin ETFs did not
- K-0100 Closed-end fund gaps revert against their ETF anchor in two asset classes; both books still fail at G1
- K-0101 Reported mean exposure of a paired fund-ETF book is about half the card's gross arithmetic; the cause is unknown
- K-0105 Wrapper-gap reversion shows in closed-end funds but not in ETFs
- K-0107 Newly available crypto coins did not underperform liquid peers in the dev window, in three designs
- K-0109 Monthly-formation, quarter-horizon stock sorts: the placebo bar is about the size of the literature's whole spread
- K-0111 Slow monthly stock sorts: four designs, none passed; entry-only events widen the placebo cloud
- K-0113 Scout event-count forecasts too high in four cards; joint tails of correlated variables are worst
- K-0114 Scout event-count forecasts too high in five cards; rare-instrument definitions worst
- K-0118 The gate again passed no crypto_top_n universe at G0 on daily perp data (H-0081, after H-0039)
- K-0126 Filing-anchored fundamentals sorts clear the placebo test in LIQ-500 twice, and both books stay near cash
- K-0129 Implied-over-realized volatility gives no usable relative signal across assets
- K-0130 Scout effect-size forecasts for slow cross-sectional cards ran far above what G1 measured in three cards
- K-0132 lab try cannot exercise a LIQ-500 card with a minimum-names floor: the synthetic universe has fewer than 50 members
- K-0133 Return-sort effects in liquid US stocks stay below cost, also as a learned composite of three sub-cost effects
- K-0136 Learned models on earnings-release events: near miss, wrong sign, no nonlinearity shown
- K-0137 Learned date-level state and calendar interactions in liquid US stocks: no timing value in two designs
- K-0138 A gbm-versus-ridge disagreement event runs the ridge control inside G1; first result is a small gain inside the cloud
- K-0140 Scout effect-size forecasts ran above G1 in seven cards; counts and cost were close
- K-0141 Learned release-event cards: a new information source, market attention load, did not help
- K-0142 Disagreement events against a ridge control: four cards, no nonlinear gain; G0 and G1 counts agree
- K-0144 Scout gross-per-entry forecasts ran above G1 in eleven cards; round trips matched
- K-0145 G1 blocks are thirds of the whole dev period; an empty block cannot count, a thin one can
- K-0146 Sector-ETF flow days: constituent reversal cleared the placebo and cost bars once, episodic and not isolated
- K-0149 Scout gross-per-entry forecasts ran above G1 in fourteen cards; event counts built from the condition arithmetic matched
- K-0150 Learned sign-split conditioners from new sources added no timing value in liquid US stocks
- K-0151 Scout gross-per-entry forecasts ran above G1 in fifteen cards; this time the event count also came in below the forecast
- K-0152 Return-distribution shape joins the learned conditioners that added nothing in liquid US stocks
- K-0155 A linear reversal score in the volatile half of liquid stocks cleared cost; nonlinear ones did not
- K-0156 An estimated stock-level oil link carried a delayed response, but it faded in the latest third
- K-0157 Scout gross forecasts split in the latest four cards: below, central and above the range
- K-0158 Estimated-exposure diffusion in liquid US stocks: oil passed once, rate, dollar and credit news did not
- K-0159 Scout gross forecasts: the estimated-exposure card H-0116 came in below its pessimistic case
- K-0164 Estimated-exposure diffusion: oil passed once; rates, dollar, credit and non-oil commodities did not
- K-0166 Information-asymmetry markers add nothing for a learned large-cap sort; continuation calls were rare
- K-0167 H-0113 core-only control: the reconstructed S&P 500 core beats SPY_TR by 7.9 %/yr on dev; the model window adds nothing
- K-0170 Zero events pass G0 and read as a mechanism failure: a learned crypto card that probably never scored
- K-0171 Scout forecasts: round trip and counts held; gross per entry fell below the central case in two more cards
- K-0173 Estimated-exposure diffusion: oil and credit passed the event test; rates, dollar and non-oil commodities did not
- K-0177 Scout forecasts: a calendar-key event count ran to a third of the forecast; gross short again
- K-0183 Wrapper-gap reversion needs a cut arbitrage link
- K-0186 Intraday cards: owner controls and mechanism claims outside the judge never run when G1 fails
- K-0187 15m crypto perp taker rules held for hours: gross at or below cost
- K-0189 15m crypto perp taker rules held for hours: gross at or below cost; a volatility filter lowered it
- K-0190 15m crypto perp taker rules: only the US-open breakout after a narrow-range day cleared cost, fragile
- K-0191 Intraday cards: owner diagnostics never run, after a G1 failure or a G1 pass
- K-0192 15m crypto perp taker rules: of four US-open breakout conditioners only the narrow-range one cleared cost, fragile
