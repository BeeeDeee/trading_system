# M2.2 Labeling sanity

Recorded after `tests/unit/test_labeling.py::test_no_edge_series_gives_near_zero_mean_r`
passed. Seed `20260827` throughout (config default / `bootstrap_ci` default).

## Random-walk check (mandatory)

A driftless geometric random walk, 4,000 sessions, overnight σ = 0.008 and
intraday σ = 0.008. Long setups placed every 8th bar from index 20, stop at 2% of
the decision close, target at 2R (`rr = 2.0`), `max_hold_bars = 40`. OPEN rows
excluded.

| Statistic | Value | Expectation |
|---|---|---|
| n resolved | 493 | — |
| `mean(realised_r_gross)` | −0.0082 | near 0 |
| bootstrap 90% CI of the mean | [−0.1215, 0.0998] | contains 0 |
| `win_rate` (`realised_r_gross > 0`) | 0.341 | `1/(1+2) ≈ 0.333` |
| `min(realised_r_gross)` | −1.885 | below −1.0 (gap-through) |
| stop / target / time rates | 0.657 / 0.337 / 0.006 | stop ≈ `rr/(1+rr) = 0.667` on a martingale; time-stops rare at hold 40 |

The mean sits inside the bootstrap CI of zero. Win rate is within 0.01 of
`1/(1+rr)`. Stop rate is the gambler's-ruin complement of a 2:1 target, not the
0.4–0.6 band quoted in `07-EDGE_AND_SCORING.md` §3 for *Donchian-detected* setups
(those have a time stop that eats some of the stop mass). This pass is labeling
geometry on a no-edge series, not strategy detection.

`min_r < −1` is the gap-through-stop rule firing. If it were clipped at −1.0 the
rule would be missing.

## Real data — recorded (M2.2a)

Tracked as [M2.2a](../15-ROADMAP.md#m22a-real-data-labeling-sanity--done).
Development window `config/development.yaml` (1998-01-01 → 2017-12-31), snapshot
`20260830-sharadar`, `config_hash=d36dea1f0fd93b5e0f0341de3f93affd3c54a500dbe799a06d6a8cda3fb40240`.
OPEN rows excluded, same as the random-walk table.

Labels are from `scout label` after the ADR-017 panel rebuild (contemporaneous
dividend cash). Written to `data/labels/setups_<id>.parquet`. The table below
replaces the pre-rebuild counts (pooled 0.086 R on 947 792 rows).

| Trigger file | Present? |
|---|---|
| `data/processed/panel/1d/*.parquet` | yes |
| `data/universe/snapshots.parquet` | yes |
| `data/reference/benchmark_1d.parquet` | yes — SPY OHLC + `vix_close` from Sharadar `^VIX`; `vix9d` / `vix3m` are NaN (not in Sharadar, unused until M4.3) |

| Check | `donchian_breakout_v1` | `xsec_momentum_v1` |
|---|---|---|
| n resolved | 475 455 | 469 668 |
| `mean(realised_r_gross)` | 0.0812 | 0.0328 |
| win rate | 0.435 | 0.523 |
| stop_rate | 0.492 | 0.129 |
| time_rate | 0.285 | 0.871 |
| `min(realised_r_gross)` | −230.80 | −4.951 |
| `mean(entry_gap_atr)` | −0.0032 | 0.0137 |

Pooled across both strategies: n = 945 123, `mean_r` = 0.057 R. **Not ~0.4 R.**
The 0.4 R stop does not fire. M3.6 trial 2 used this label set.
Donchian `min_r` is still GAHC (stop almost on entry, gap to 0.99), not CHKAQ.

Against [`07-EDGE_AND_SCORING.md` §3](../07-EDGE_AND_SCORING.md#3-the-resolved-setup-table)
expectations: Donchian stop_rate 0.49 sits in 0.4–0.6 and time_rate 0.29 in
0.1–0.3. Momentum time_rate 0.87 is above 0.85 and win rate is near 0.50; its
stop_rate 0.129 is a bit above the “under 0.10” note (5 ATR stop). That is not
the 0.4 R leak. `min_r < −1` on both sides is gap-through-stop, not clipping.
`mean(entry_gap_atr)` is near 0.

## Interpretations (docs were incomplete, not contradictory)

These are the places the documents did not specify a unique answer. They are
called out so they can be reversed if wrong, rather than becoming silent scoring
bugs.

1. **`vol_bucket` on `resolve_setup`.** `ResolvedSetup` requires it; `Setup` does
   not carry it; the `03-INTERFACES.md` signature does not pass it. Implemented as
   a keyword-only argument defaulting to `UNKNOWN`. The CLI passes
   `FeatureRow.vol_bucket`.
2. **OPEN with no next bar.** `entry_ts` is required and not nullable. With an
   empty forward window it is set to `setup.ts` and `entry_price` is NaN.
   `realised_r_gross` is NaN. OPEN after an actual entry (data ran out mid-trade)
   keeps the real `entry_ts` / `entry_price`.
3. **Parquet `stop_price` / `target_price`.** Written as the *re-anchored* levels
   used for resolution, not the decision-bar levels on `Setup`. Distances match
   the intended R/R; the decision-bar numbers remain on `reference_price`.
4. **`adv_usd_60`.** Not a `FeatureRow` field. Joined from the latest universe
   snapshot with `snapshot.ts <= setup_ts` (backward as-of, by `asset_id`).
5. **One file per strategy.** `data/labels/setups_<strategy_id>.parquet` is a
   single parquet file. Year partitioning is not implemented; 07's volume note
   is a performance hint, and the acceptance criterion names a `.parquet` file.
6. **`LabelConfig` in the interface is `LabelingConfig`** from
   `config/schema.py`. No second type.
7. **`had_earnings_in_window`** is `gates.eligibility.earnings_in_window` — i.e.
   whether the earnings gate *would* have fired, including the conservative
   “no earnings row ⇒ blocked” fallback for non-ETFs.
