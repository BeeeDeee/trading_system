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

### K-0068 Mechanical stock-bond flow: the study effect appeared only when the trigger was a large stock-bond move

Three daily-ETF cards tried to trade mechanical stock-bond reallocation pressure. The month-end rule on month-to-date drift (H-0002) carried almost no edge. The threshold-drift trigger (H-0046) passed the event study clearly, but its events were large relative stock-bond move days, so generic spread reversal could not be separated from rebalancer flow, and the book lost to the plain mix. The volatility-control trigger (H-0050) removed that confound: the event day is quiet and the cause lies many sessions earlier. There the average was slightly wrong-signed, inside the placebo cloud, and negative in the latest third. The simplest reading is that the H-0046 pass was mostly generic reversal after large moves (K-0056), not rebalancer flow. The H-0050 sample was thin and clustered in high-volatility regimes, so this is a lean, not a refutation. A further mechanical-flow card on daily ETFs has a low prior unless the flow is observed where it trades (futures, closing auction) or a control separates it from stress reversal.

*confidence low · 2026-10-17 · librarian · evidence: H-0050; H-0046; H-0002; K-0056; K-0064*

### K-0070 Macro-calendar cards fail in four designs; moving to a volatility ETP did not help

Four designs on the scheduled US macro calendar have ended at the event-study gate: announcement-session beta premium (H-0042), even-week FOMC-cycle tilt (H-0045), cross-asset reaction drift (H-0047) and short volatility over release windows through a VIX-futures ETP (H-0052). The last was the route K-0063 and K-0065 left open. It gave a positive average, but placebo windows of the same instrument already earned over half of it through roll-down, the effect grew with the horizon like generic carry, and the first third of dev was flat while the middle third carried most (not the early-heavy decay of K-0060). The placebo bar lay above cost although gross was several times cost, because a volatility ETP is noisy. A release-time volatility crush is not visible in daily ETP bars. A further macro-calendar card on daily ETFs has a low prior; implied volatility or index futures measured directly are untested and not loadable.

*confidence low · 2026-10-17 · librarian · supersedes K-0065 · evidence: H-0052; H-0047; H-0045; H-0042; K-0063; K-0065; K-0034; K-0025; K-0060*

### K-0077 Fifteen-minute crypto perp taker moves: a small edge in both directions, consumed within about a candle

Two cards on taker-dominated, high-volume fast moves in liquid Binance perps took opposite sides and found a small positive average trade after modeled costs on dev: a rebound after extended moves (H-0035) and a continuation after breaks out of a compressed range (H-0063). Neither survived a one-candle later fill or doubled costs, and in both the latest third of dev was the weakest. Both selected the moment the aggressive flow is already visible in the candle, so what is left after the signal candle closes is small. Together with K-0049 and the hourly study mrel-1h, the pattern is that the payoff on this time scale goes to fast fills and makers, and a taker entering at the next open pays for the front of the move. The BTC-led catch-up over hours (H-0040) survived delay and doubled costs and failed on stability instead, so slower follower effects are not ruled out. Cards on this time scale should state the expected gross per trade after a one-candle delay, not only at the next open.

*confidence low · 2026-10-19 · librarian · evidence: H-0035; H-0063; H-0040; K-0049; K-0060; prior_studies mrel-1h*

### K-0080 Information-diffusion claims in liquid assets come out flat or wrong-signed, also with a dated news event

Claims that news in one liquid market is priced late in another have not shown a delayed positive response from daily to monthly horizons. Statistical peer catch-up in US stocks (H-0013) was slightly negative, macro-release continuation across ETFs (H-0047) and the oil shock against Treasuries (H-0056) were wrong-signed in every third. New: H-0066 replaced the undated peer move by a verified event (a large real earnings-release reaction) and looked only at peers that had not reported; the average was still inside the placebo cloud with a bar below cost, and longer horizons turned negative. So undated noise in the trigger was not the only reason for H-0013's failure. These are not power failures. Do not trade the mirror: the sign is dev-selected and episodes are few. A new card should say why information would travel slowly through this particular channel; explicit economic links are untested and not loadable.

*confidence low · 2026-10-20 · librarian · supersedes K-0072 · evidence: H-0066; H-0056; H-0047; H-0013; H-0040; K-0076; prior_studies mrel*

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

### K-0105 Wrapper-gap reversion shows in closed-end funds but not in ETFs

In the etf-premium-arbitrage family the gaps that reverted were closed-end fund gaps against an ETF anchor (H-0077, H-0078, H-0079, K-0103), where no creation or redemption link exists. ETF gaps did not: twin ETFs (H-0037) fell short of the placebo bar, and a credit, EM, muni or loan ETF far below what SPY and IEF implied (H-0080) did not rebound, with a wrong-signed average at every horizon even though the test leaned its way (K-0074). Daily closes against liquid factor proxies are probably not a mispricing signal. H-0043 stays untested; H-0080 is its nearest replacement. A fair test needs premium to NAV, not loadable.

*confidence low · 2026-10-28 · librarian · evidence: H-0080; H-0037; H-0043; H-0077; H-0078; H-0079; K-0103; K-0084; K-0074*

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

### K-0129 Implied-over-realized volatility gives no usable relative signal across assets

Three designs on the options-implied fear of assets have ended at the mechanism gate. A time-series equity tilt on the implied-over-realized ratio was right-signed but a near miss with too few episodes (H-0055). Short volatility timed by the implied curve was mostly the unconditional roll-down (H-0073, K-0088). New: ranking twelve ETFs across asset classes on the ratio against each asset's own year (H-0084) gave a wrong-signed average at every horizon from a week to a quarter, widening with horizon, positive only in the earliest third and worst in the latest. The reading is inferred, not tested: raw returns of the rich-ratio leg carry stress continuation and beta (K-0074), and any premium that exists is probably global and common to assets rather than asset-specific, which a relative ranking cancels. A further volatility-premium card on this family has a heavy trial count and a low prior unless it is a time-series design with many markets and an always-invested control.

*confidence low · 2026-11-03 · librarian · evidence: H-0084; H-0055; H-0073; H-0052; K-0088; K-0070; K-0074; K-0060*

### K-0133 Return-sort effects in liquid US stocks stay below cost, also as a learned composite of three sub-cost effects

Past-return sorts in liquid US stocks, with reversal and continuation signs and many conditioners, almost all gave effects far below round-trip cost or nothing (H-0007, H-0029, H-0030, H-0065, H-0069). The exception, the reversal before a predicted earnings release (H-0064), cleared cost but was wrong-signed in the latest third and its book was far weaker than its event study. New: H-0094 merged no-news reversal, attention give-back and earnings-reaction drift into one learned score. Entries beat same-stock placebos in every third, but gross stayed under cost and faded across thirds. Combining sub-cost effects came closer than single sorts without clearing cost; whether nonlinearity matters is untested because the linear baseline never ran. A new return-sort card should argue gross above cost for its own entries, name the latest-third sign as a kill condition and include the linear control.

*confidence low · 2026-11-03 · librarian · supersedes K-0079 · evidence: H-0094; H-0064; H-0065; H-0007; H-0029; H-0030; H-0069; K-0060; K-0079*

### K-0137 Learned date-level state and calendar interactions in liquid US stocks: no timing value in two designs

Two learned cards gave a gradient-boosted score date-level inputs that are the same for every stock on a date, so that only an interaction could use them: month-turn and macro-release clocks with beta, volatility and liquidity (H-0098), and VIX and market-drawdown state with residual reversal, volatility and beta (H-0099). Both ended at the mechanism test with entries no better than same-stock placebos on shifted dates, in H-0098 clearly worse, and with gross far under round-trip cost. The placebo bars were under cost, so these are not power problems. Neither card could show whether the model split on the date inputs at all, because feature importance and the by-state split did not reach the gate. The reading is that liquid-stock selection does not become profitable by switching a static tilt with the calendar or the volatility state, and the month-turn fade of K-0038 and the weak reversal of K-0133 reappear in the learned form. A further card of this type has a low prior unless it reports how the model used the state.

*confidence low · 2026-11-05 · librarian · evidence: H-0098; H-0099; K-0038; K-0133; K-0084*

### K-0139 Filing-anchored sorts pass twice in LIQ-500; a learned conditional form did not

Two sorts with different content, net repurchasers against issuers (H-0019) and low against high quarterly accruals (H-0092), were both timed to the quarterly filing and held for a week in LIQ-500. Both cleared the same-stock placebo bar with gross above round-trip cost, and both books ended near T-bills (K-0064). They share filing dates, universe and seasons, so this is one piece of evidence seen twice. H-0019 faded era by era, H-0092 was positive in every third, and the effect in H-0092 was present in the first session and kept building. New: H-0097 added the release reaction and event time in a learned long-only form. Fresh-filer entries stayed inside the placebo cloud, gross was under cost, and the latest third was wrong-signed, so the claim that filing information is conditional on the headline reaction adds nothing visible over the plain accrual sort. The release-anchored sort (H-0027) and the slow style (H-0086) also failed, so the anchor matters and the pass may be filing-linked or early-era. A further filing card in LIQ-500 has a low prior; untested are the long leg alone, beta-matched legs and less covered stocks.

*confidence low · 2026-11-05 · librarian · supersedes K-0126 · evidence: H-0092; H-0097; H-0019; H-0027; H-0086; K-0110; K-0064; K-0074; K-0060*

### K-0143 Learned earnings-reaction cards: ownership, event time and non-filed gaps added nothing

K-0141 asked for a new information source or for the interaction to be reported directly. Both came in and failed. Institutional ownership with arbitrage risk (H-0102) gave entries below same-stock placebos, gross a fraction of cost and a slow-drift horizon shape. Event-time features of the latest reaction scored over all members (H-0104), tested as picks only the gbm makes against a ridge, were also below placebos and grew with horizon instead of fading. Overnight gaps with no filing and earnings windows excluded (H-0105) went wrong-signed against the ridge. With H-0095, H-0100 and H-0101, five learned cards in the reaction family have died on the mechanism test, and none beat the plain reaction rule of H-0033; the only near miss (H-0100) had a bar under cost. Second variables did not add at the release (K-0083), and a learned combination of them does not either. A further learned card on release reactions or overnight news in LIQ-500 has a low prior. Untested: other stocks (not loadable) and holds beyond a week argued from the slow build.

*confidence low · 2026-11-06 · librarian · supersedes K-0141 · evidence: H-0102; H-0104; H-0105; H-0101; H-0100; H-0095; H-0033; K-0141; K-0083*

### K-0148 Sector-ETF flow days: the constituent reversal did not reproduce once market-wide stress was removed

H-0106 found a reversal after sector-ETF flow days that beat placebos and covered cost (K-0146), but its sector moves were mostly market-wide. H-0107 removed that by construction: the push was the wrapper's move residual to SPY, stocks with an own filing that day were excluded, industry ETFs added flow days outside market stress. Entries were no better than same-stock placebos, gross was about half of cost, the first sessions were negative, ten sessions flat, and the thirds disagreed in sign. The placebo bar lay above cost, so a small effect was unresolved, but nothing suggests a hidden tradable one. The H-0106 pass most likely belonged to market-wide stress days (K-0084), not to wrapper-specific basket flow. The cards also differ in model, entry rule and side, so this is a lean, not a refutation. Untested: ordinary-day control, up-push side, ridge rule. A further ETF-flow card has a low prior without a stress control or ownership data.

*confidence low · 2026-11-06 · librarian · supersedes K-0146 · evidence: H-0107; H-0106; K-0146; K-0084; K-0024*

### K-0152 Return-distribution shape joins the learned conditioners that added nothing in liquid US stocks

A fourth learned card on a literature mechanism in liquid US stocks over a week (after H-0108, H-0109, H-0110) used a new source: the shape of the recent return distribution (H-0111, skewness from overnight and intraday half-day returns). Tail entries were no better than the same stocks on shifted dates, gross was a small fraction of cost, the first third of dev was wrong-signed and the effect showed only at the longest horizon. The placebo bar lay below cost, so not a power problem. With K-0133, K-0137 and K-0143, no conditioner tried (fundamentals, ownership, attention, calendar, tax clock, distribution shape) lifted a one-week large-cap sort above cost. The ridge control never ran (K-0135). The half-day split is coarse, so the published intraday-skew effect is not refuted.

*confidence low · 2026-11-07 · librarian · supersedes K-0150 · evidence: H-0111; H-0108; H-0109; H-0110; K-0133; K-0137; K-0143; K-0135*

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

### K-0111 Slow monthly stock sorts: four designs, none passed; entry-only events widen the placebo cloud

H-0086 is the fourth slow monthly-formation sort of liquid US stocks to end at the mechanism test (after H-0054, H-0059, H-0083): near misses with a bar above cost (H-0054, H-0083), right-signed but equal to the same-stock placebo (H-0059), and now below the placebo mean (H-0086). K-0109 said a card whose forecast is at or below the bar can only end in 'cannot tell'; H-0086 cited it, forecast an effect near the bar, and observed far less. New: to avoid the persistent-state trap of H-0059 it marked only new entries to the extreme quintiles. That left a few thousand events on a few hundred dates, and the placebo bar came out clearly wider than in H-0083 (inferred cause: fewer events per date, more dispersed stocks; not tested). So the cure for placebo-equals-level costs power, and a Scout should forecast the bar for the entry design, not copy it from a state design. Independent dates and dispersion matter, not event counts (K-0026, K-0034).

*confidence low · 2026-10-29 · librarian · supersedes K-0109 · evidence: H-0086; H-0083; H-0059; H-0054; K-0109; K-0026; K-0034*

### K-0120 Scout forecasts of events, regime days and cost ran too high in six cards

H-0049, H-0050, H-0077, H-0089 and H-0090 forecast far more events than counted (K-0069, K-0098, K-0113); H-0090 forecast 600-1200, counted 66 on 30 dates. H-0081 adds: the VIX gate was expected on 35-45 percent of dev days, 400-500 dates, 4000-6000 events; G0 and G1 counted 1802 events on 207 dates, about a fifth of days (inferred). The card put the liquid-perp round trip at 25-35 bps; G1 used about 91 bps (cost model not read). Gross per event was about 3 bps, so it died on either cost. Scouts should count regime days from the actual series inside the usable period, state events per year per condition and intersection, and take the round trip from the catalog cost tier of the weakest instrument. Suggested: G0 flags events below the G1 minimum and reports events by year.

*confidence medium · 2026-10-31 · librarian · supersedes K-0114 · evidence: H-0081; H-0090; H-0089; H-0077; H-0049; H-0050; K-0069; K-0098; K-0113; K-0114*

### K-0125 Mean exposure of a filing-season event book came out well below entries x hold x weight / sessions

K-0104 found that mean exposure of slot books fits entries x hold / sessions x slot weight (H-0078, H-0079). H-0092 does not fit: 16204 entries, hold 5, slot weight 0.04, 5437 decision dates (5787 observations) give an arithmetic gross of 0.56-0.60, while G1 reported a mean exposure of 0.36, and the card forecast 0.4-0.55. Two readings, not separated: the card's gross cap of 1.0 scaled down all entries of peak filing-season sessions (events cluster in a few weeks per quarter), or the reported exposure nets long against short positions. Either way, a clustered event book is invested less than the arithmetic says, and the scaled slots are smaller than the card's weight. Cards with clustered events should forecast exposure from the busiest weeks, not from the annual average, and state which exposure definition they use. The same card forecast events (about 14000-16000, counted 16330) and the round trip (12-15 bps forecast against about 12 bps modelled) correctly, which fits the advice of K-0098 and K-0120. The exposure definition question of K-0101 stays open.

*confidence low · 2026-11-01 · librarian · evidence: H-0092; H-0078; H-0079; K-0104; K-0101; K-0098; K-0120*

### K-0134 Learned-model cards: a column set defined by the whole-view universe makes walk-forward scores depend on the future

H-0095 failed G0 look-ahead on its first run (truncation at 4 of 11 cuts, perturbation at 8). Cause per the Builder: features, labels and walk_forward were restricted to columns that are ever a universe member over the whole view. That set depends on future membership, and gbm fits depend on the column layout, so scores on earlier rows changed when a stock joined later. Fix: run on all columns and mask non-members per row; a regression test with late entrants and late-starting event data now covers it. Rule for learned cards: no column, row or training-sample subset may be chosen with information from the end of the view (ever-member lists, full-period filters, survivors). The synthetic lab try market could not show this, only G0 on real data did. H-0096 passed G0 on its first run.

*confidence medium · 2026-11-04 · librarian · evidence: H-0095; H-0096; run 275; run 276; K-0009*

### K-0142 Disagreement events against a ridge control: four cards, no nonlinear gain; G0 and G1 counts agree

The design of K-0138 (mechanism event = picks or rows one model makes and the linear control does not, same features, rows, label and schedule) has now run in H-0100, H-0103 (a TCN against a ridge, both sides marked), H-0104 and H-0105. G0 and G1 event counts agreed within a handful (30603 and 30559, 15162 and 15158, 12349 and 12347), so the design works in the framework. Results: near the percentile bar with a bar under cost (H-0100), bottom of the cloud (H-0103), far under the bar (H-0104, H-0105). Two were wrong-signed (H-0103, H-0105), so there the disagreement set favoured the ridge side; the other two were positive but a small fraction of cost. The nonlinear form has not beaten its linear control in any of them, across three feature sets. Limits: the average mixes 'learner wrong' with 'control right' and the control's picks are never run as a book; the sign of a wrong-signed result is dev-selected, so the mirror is not a lead; event counts depend on model overlap and ran above forecast. A card that claims an interaction should still use this design and state what it concludes if the ridge side wins.

*confidence low · 2026-11-06 · librarian · supersedes K-0138 · evidence: H-0100; H-0103; H-0104; H-0105; K-0138; K-0135*

### K-0147 A rare-state mechanism that passes its event study cannot carry a daily learned book (H-0106)

H-0106 is the first learned daily book whose mechanism test passed clearly: 5563 flow-day entries on 325 dates, placebo percentile 0.99, gross per entry about 28 bps against a round trip of about 13.5 bps. The book traded 124656 entries, so flow-day entries were about 4.5 percent of them. G1: gross Sharpe about 0.01, net about -0.76, long leg +0.22, short leg -0.61, cost drag about 4.7 percent a year. The other entries earned nothing and the short leg lost, as the card's own book arithmetic (K-0064) predicted. Rule: a rare-state mechanism needs an always-invested tilt or an event-slot book (K-0005, K-0027) with the expected annual gain written down. Forecast check (K-0144): event count and flow days were in range, gross per entry was above the central forecast, the first learned card not below its pessimistic case. Suggested for G1: report book gross for mechanism entries and for all others.

*confidence low · 2026-11-06 · librarian · evidence: H-0106; H-0094; H-0099; K-0027; K-0064; K-0097; K-0135; K-0144*

### K-0151 Scout gross-per-entry forecasts ran above G1 in fifteen cards; this time the event count also came in below the forecast

H-0111 forecast a central gross of 12-18 bps per entry, about 4 at a quarter of that and 1-2 at a tenth, and a round trip of about 11 bps in LIQ-500. G1 measured about 2.4 bps gross (cost ratio about a fifth of an 11.2 bps round trip), between the quarter and tenth cases, so the pessimistic case was beaten again (fifteen cards with K-0149). The round trip matched. Unlike the three cards in K-0149, the event count missed: forecast 20000-27000 (25-35 entries on about 780 weekly dates), counted 15113 on 703 dates, about 21 per date. The tail-fraction arithmetic overstated entries per date, probably because the middle-range past-return filter and the previous-week entry rule overlap more than the independence arithmetic assumed (inferred, not tested), and fewer scored dates existed than forecast. Rule for Scouts: when an entry rule combines a tail entry with a second filter, count entries per date from a rough simulation of the actual overlap, not from the product of fractions, and keep the K-0149 advice to treat a literature gross as a ceiling and state what the card concludes at a tenth of it.

*confidence low · 2026-11-07 · librarian · supersedes K-0149 · evidence: H-0111; K-0149; K-0144; K-0140; K-0120*

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

### K-0062 Zero events can pass G0 and then read as a mechanism failure

Three cards ended on the event count with no mechanism result: H-0021 (empty by universe), H-0038 (events at G0, none at G1) and H-0043 (zero at G0 and at G1). H-0043 is new: G0 passed with zero events, and the signal was a two-sigma residual gap that fires regularly in any healthy series, so the cause is almost surely the inputs (ICE BofA spread coverage in the lab's FRED copy, the valid-day rule, or NaN propagation), not the market. The registry shows g1_mechanism_events, which a Scout can read as a failed mechanism. Suggested handling: G0 should flag a card whose events_in_dev is zero; the Builder should report events by year before submitting; the Archivist should confirm first-valid dates of long-history series. The card's fail-loudly coverage check cannot be point-in-time at G0.

*confidence low · 2026-10-16 · librarian · evidence: H-0043; H-0038; H-0021; K-0054; K-0032*

### K-0069 G1 event-count code can hide a measured mechanism failure; date tripwires cannot fire at G0

H-0049 (193 events against a minimum of 200) and H-0050 (172 marked events on 43 event dates) are both registered as g1_mechanism_events, yet G1 measured the mechanism in both: wrong-signed average, placebo percentile far under the bar, negative cost ratio, weak latest third. A Scout reading the registry takes them for sparsity problems more events would cure; the sign says otherwise (the opposite of K-0062). New in H-0050: the card wrote its tripwire in event dates (flag if under 50 dates), but G0 reports marked events (four ETFs per date), so it showed 172 and the tripwire could not fire; the dates appeared only in G1 metrics. Both Scouts forecast event rates far above the real count. Skeptic messages were absent again. Suggested: list all failing checks in the registry reason; report event dates next to marked events at G0 and flag counts under the G1 minimum; Scouts state expected dates and marked events separately.

*confidence medium · 2026-10-17 · librarian · supersedes K-0067 · evidence: H-0049; H-0050; K-0062; K-0054; K-0032*

### K-0082 lab try cannot exercise cards that need the 8-K flag: its synthetic ea_reaction_day never equals one

The Builders of H-0064, H-0066 and H-0067 each report that the synthetic ea_reaction_day in lab try is uniform noise between 0 and 10, never exactly 1, so lab try shows zero events and its checks are vacuous for these cards; they tested events, book and point-in-time behaviour on hand-made data (one temporarily rounded the flag). On real data G0 did report events (counts within a handful of G1's) and no look-ahead failures in 25 cuts, so G0 is the first meaningful check, and an event-logic bug would surface only there. Suggested: make the synthetic flag a sparse 0/1 event series so lab try produces events.

*confidence medium · 2026-10-20 · librarian · evidence: H-0064; H-0066; H-0067; K-0054; K-0057*

### K-0086 G1 needs at least three unmarked peers per date, so early event dates can be dropped at G1 but counted at G0

H-0041 showed that marking every universe instrument gives zero events at G1 and that dates with only two unmarked instruments with data were dropped (inferred: at least three unmarked peers per date; code not read). H-0072 adds a partial form of the same rule: G0 counted 647 events in dev, G1 measured 514, a gap of about a fifth. Its peers list from 2002 (SHY), 2007-01 (SHV) and 2007-05 (BIL), so events before the third peer exists were most likely dropped at G1 while G0 and the strategy still use them. The mechanism verdict then covers only about 2007 onward, not the full period of the card, and K-0054's statement that G0 and G1 counts agree does not hold when peers start late. Scouts with late-listing peers should state the first date with three unmarked peers and the expected events from that date; a G1 reading should say which years the event study covered.

*confidence low · 2026-10-21 · librarian · supersedes K-0055 · evidence: H-0072; H-0041; H-0038; K-0054; K-0055*

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

### K-0145 G1 blocks are thirds of the whole dev period; an empty block cannot count, a thin one can

H-0105 stated no data period. G1 reported 12347 events at about 540 per year, which implies about 23 years of denominator, i.e. the default dev start in 1998, while the 8-K flag starts in 2004, so its first block had zero events. mechanism_blocks appears to count blocks with a positive mean (H-0102 and H-0104 passed with two positive blocks of three, H-0103 and H-0105 scored zero), so an empty block can never count and the card needed both other blocks positive. Conversely H-0102 had 87 events in its first block, and that block counted as one of its two passing blocks, so a thin block can supply the pass. Both readings are inferred from the metrics, not from the code. Scouts should set the period start at the dataset's first usable date (8-K from 2004-09, 13F from mid-2013) so that the blocks are real. Suggested: report events by block at G0 and require a minimum event count per block before it counts.

*confidence low · 2026-11-06 · librarian · evidence: H-0105; H-0102; H-0104; H-0103; K-0035; K-0141; K-0120*

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
- K-0065 Macro-calendar cards on asset-class ETFs fail in three designs; reaction drift is single-stock only
- K-0066 Perp funding does not tell forced crash selling from informational selling on a daily horizon
- K-0067 The G1 reason code can name the event count while the mechanism was measured and failed; the count was visible at G0
- K-0071 Slow-horizon event studies: the placebo bar sits several times above cost, so only a large effect can pass G1
- K-0072 Cross-market information-diffusion claims in liquid assets come out flat or wrong-signed from daily to monthly horizons
- K-0073 Multi-year reversal: absent in country equity ETFs, present for currency value in a short sample
- K-0075 Signal history before the dev start or the ETFs' first date seems not to warm up the first signals
- K-0079 Return-sort effects in liquid US stocks stay below cost, except a pre-release reversal that faded
- K-0081 On real earnings dates the events that cleared the placebo test sort on the stock's own price move
- K-0085 Perp funding and the perp-versus-spot volume split both fail to separate forced from informational big moves
- K-0099 Wrapper gaps revert where no arbitrage link exists: closed-end funds cleared the placebo test, twin ETFs did not
- K-0100 Closed-end fund gaps revert against their ETF anchor in two asset classes; both books still fail at G1
- K-0101 Reported mean exposure of a paired fund-ETF book is about half the card's gross arithmetic; the cause is unknown
- K-0107 Newly available crypto coins did not underperform liquid peers in the dev window, in three designs
- K-0109 Monthly-formation, quarter-horizon stock sorts: the placebo bar is about the size of the literature's whole spread
- K-0113 Scout event-count forecasts too high in four cards; joint tails of correlated variables are worst
- K-0114 Scout event-count forecasts too high in five cards; rare-instrument definitions worst
- K-0118 The gate again passed no crypto_top_n universe at G0 on daily perp data (H-0081, after H-0039)
- K-0126 Filing-anchored fundamentals sorts clear the placebo test in LIQ-500 twice, and both books stay near cash
- K-0130 Scout effect-size forecasts for slow cross-sectional cards ran far above what G1 measured in three cards
- K-0132 lab try cannot exercise a LIQ-500 card with a minimum-names floor: the synthetic universe has fewer than 50 members
- K-0136 Learned models on earnings-release events: near miss, wrong sign, no nonlinearity shown
- K-0138 A gbm-versus-ridge disagreement event runs the ridge control inside G1; first result is a small gain inside the cloud
- K-0140 Scout effect-size forecasts ran above G1 in seven cards; counts and cost were close
- K-0141 Learned release-event cards: a new information source, market attention load, did not help
- K-0144 Scout gross-per-entry forecasts ran above G1 in eleven cards; round trips matched
- K-0146 Sector-ETF flow days: constituent reversal cleared the placebo and cost bars once, episodic and not isolated
- K-0149 Scout gross-per-entry forecasts ran above G1 in fourteen cards; event counts built from the condition arithmetic matched
- K-0150 Learned sign-split conditioners from new sources added no timing value in liquid US stocks
