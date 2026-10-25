# What the owner has already tested (and how it ended)

Eighteen earlier studies on the same data, all pre-registered, almost all negative. They count against you twice:
the families below are already "used" on this history, and their results are known. Re-proposing one of them
without a genuinely new mechanism is a wasted run. Building on a *specific lesson* from them is welcome.

| # | Question | Result |
|---|---|---|
| 1 | Ensemble of momentum, trend, breakout, reversal, low-vol, regime strategies on liquid US stocks (3 027 candidates, walk-forward selection) vs passive | **No.** Dev 2005-2019 low-vol ensemble Sharpe 0.93 vs SPY 0.49, DSR 0.999; locked holdout 2020-2026 CAGR 2.4 % vs SPY 15.5 %. A famous family passes in-sample and still fails. |
| 2 | 9-ETF allocation (inverse vol + per-class trend filter) vs 60/40 | **No.** 2010-2026 Sharpe 0.52 vs 0.85, robust to costs, ETF swaps, trend length. |
| 3 | Passive rules: rebalance frequency, equity share, foreign stocks, bond type, gold | No significant differences, except 5-10 % gold: Sharpe +0.04-0.07 (special period for gold). |
| 4 | Meta-layer (regimes, strategy momentum, LightGBM, ridge) choosing among 87 strategies from 25 families | **No.** Sharpe 0.52 vs SPY 0.56; all of the edge in 2020-2026. Rank IC 0.034; ~0.05 would be needed. |
| 5 | Short-term reversal with a trend filter on US stocks | **No.** Dev Sharpe 1.02 but DSR 0.52 (N = 993), an isolated peak; validation 2015-2019 Sharpe 0.32, 53 % of profit from 1999-2002. |
| 6 | Reversal only when VIX > 25 (Nagel 2012) | Not a strategy (invested 9 % of the time, CAGR 3 %). The effect looked real per trade, but study 7 showed it was the market bounce after high VIX, not stock selection. |
| 7 | SPY + reversal sleeve when VIX > 25 | **No.** Sharpe 0.686 vs 0.685; negative at costs x2. |
| 8 | Crypto funding carry (long spot + short perp) vs T-bill, 2023-2026 holdout | **No.** Dev 2020-2022 +12 %/y over T-bill; holdout +0.3-0.7 %/y, CI includes 0, negative since 2024-07. Funding compressed after Ethena. |
| 9 | Crypto trend filter on BTC/ETH; weekly altcoin momentum | **No.** SMA filter Sharpe 0.36 vs 0.34 (CI includes 0); filters did cut drawdowns (30-48 % vs 68 %). Altcoins 2022-2026: EW top-20 -45 %/y, no momentum variant had a positive CAGR. |
| 10 | Complementary strategy pairs (incl. shorts): stock sleeves, SPY + ETF trend L/S, crypto trend with perp shorts | **No.** Long-only stock pairs correlate >= 0.53 in crises; only stocks + bonds complement (= 60/40). ETF trend L/S earned in 2008 and lost 2010-2019. Crypto short leg flips sign between dev and holdout. |
| 11 | LightGBM on fundamentals, insiders, 13F changes selecting 50 of LIQ1000 monthly | **No (0 of 6).** Sharpe 0.57 vs SPY 0.82, below EW and below the price-only model. Turnover 15x/y. |
| 12 | Same model with a low-turnover hysteresis rule, forward test | Pre-registered, waiting for data (no Sharadar renewal). |
| 13 | "Smart Zones" (buy liquid crypto in the discount zone of the swing range) | **No (0 of 6).** Holdout Sharpe -0.92, worse than all 500 random-entry runs; failed already in dev. |
| 14 | Crypto multi-timeframe pullback: buy BTC/ETH after a strong 1 h drop (z <= -1.5 over 6 h) when the weekly and daily trends point up, hold 48 h (16 configurations) | **No (2 of 6).** Holdout 2022-2026 Sharpe 0.18, below 58 % of random entries in the same trend regime; the trend filter alone (long while W+D trend is up) had Sharpe 0.74. The edge sat in the higher-timeframe trend, not in the entry timing. |
| 15 | Crypto multi-timeframe 4 h / 1 h / 15 m on BTC/ETH: 4 h trend direction, 1 h break or liquidity-sweep setup, 15 m confirmation, stop behind the structure (16 configurations) | **No (4 of 6).** Best BRK_N48_R3 long: holdout Sharpe 0.43, beat 99 % of random entries and both ablations, but the CI of the mean trade includes 0 and Sharpe is -0.01 at 2x costs (0.22 with a one-candle delay). Shorts and liquidity-sweep setups lose in every variant. Trigger: a price-structure break, not taker flow. |
| 16 | Crypto intraday batch 1, 15 m BTC/ETH: eight simple families (opening-range breakout, funding extreme, BTC-led lead-lag, CME gap, squeeze, pullback, shock, range) with small grids (30 configurations; cumulative 46 with study 15) | **No.** Five families died in development; holdout rejected ORB (0 of 6, a bull-market artefact) and left FUND and LEAD at 5 of 6 (both failed only the CI of the mean trade; LEAD = long ETH for 4 h after a BTC 1 h move >= 3 sigma while ETH lags, 81 trades, all 4 LEAD variants and all holdout years positive). Holds under about two days lose everywhere on 15 bps round trip. |
| 17 | LEAD_K3_LONG unchanged on the top-10 altcoins (perps, BTC as leader, 2020-05..2026-09), one configuration | **Yes, 6 of 6**, thin margins: Sharpe 0.80, 2260 trades, event-bootstrap CI of the mean trade [+0.06, +0.36] %; Sharpe 0.06 at 20 bps per side; 2022, 2025 and 2026 negative; shorts and K=2 lose. Forward paper test from 2026-10-01 (to 2027-03-31). The rule was chosen on the study-16 holdout, so the 2022-2026 history is contaminated for it. |
| 18 | Two pre-registered forward hypotheses on LEAD after a post hoc backtest of 30 variants (hold, stop, delay, cost, size and filter variants; 25 of 30 have Sharpe > 0, every variant negative in 2022 and 2025): H1 = altcoin flat or down while BTC jumps (V05_lag0), H2 = LEAD_K3_LONG only while BTC spot is above its 200-day mean | Forward only, T0 2026-10-20, first decision 2027-10-19 (power about 10-30 %). Nothing evaluated yet. |
| mrel | Cross-market daily lead-lag scan (706k tests: stocks, ETFs, FX, macro, crypto) | **No.** 3 candidates survived 2022-23 validation (MUB autocorrelation, LQD <- EWG), all died on the 2024-2026 holdout (t ~ 0). |
| mrel-1h | Hourly crypto lead-lag | Short-term hourly reversal is real but only harvestable as a maker: taker turnover ~1 300x/y costs far more than it earns. |

Trial counting for the lab: configurations tried in these studies that belong to the same rule as a lab card are added
to its family's trial counter by the owner (`lab prior-trials`): btc-lead-altcoin-lag carries the four LEAD variants of
study 16, the 30 variants of the post hoc backtest and the extra filter of study 18 H2, cross-sectional-momentum the 864
candidates of the momentum sleeve of study 1 plus the 12-1 momentum run of the calibration. Study 15 is not counted
for stop-cascade-breakout (price-structure breakout, not taker-flow trigger).

Lessons the gates are built on (`gates.yaml`):

1. A well-known family can pass DSR in-sample and fail out of sample. The literature prior adds 20 trials to
   any hypothesis with references.
2. Isolated parameter peaks and profit concentrated in one sub-period kill a hypothesis (G2 neighbors and blocks).
3. A real effect that is invested 9 % of the time is not a strategy (minimum exposure and trade count).
4. Short-horizon ideas die on costs; costs x2 is a hard gate.
5. Crowded carry decays after it becomes popular (publication / regime check).
6. The strategy must beat random entries with the same turnover and exposure.
7. Daily lead-lag between liquid markets did not survive out of sample.

What has *not* been tried yet (not a recommendation, just the gap): anything on ETFs beyond trend, inverse
vol and sector momentum; cross-asset signals into crypto other than lead-lag; volatility-managed exposure;
calendar and flow effects (turn of month, option expiry, index rebalancing, ETF creation/redemption);
carry across ETF asset classes; anything that is long/short across classes with gross <= 1.
