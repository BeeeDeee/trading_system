# Fáze 2 – strategie a matice výnosů kandidátů

**Stav 2026-09-26: dokončeno, brána splněna** (PBO ≤ 0,30 a aspoň jedna rodina s kladným mediánem
Sharpe okolí). Období: vývojové 2000-01-03 … 2019-12-31. Holdout zůstává zamčený.

> Metriky přes celé vývojové období jsou **in-sample diagnostika**. Nic se podle nich nevybírá;
> výběr strategií proběhne walk-forward ve Fázi 3 uvnitř `select()`. Jakákoli změna mřížky nebo
> filtrů na základě těchto čísel je nový pokus v registru (spec §9.8).

## Běh

- 3 027 kandidátů (21 signálových variant × 144 portfoliových + 3 × regime), grid run
  `a03e443786b83e08`, zapsán v registru pokusů (`runs/registry/trials.sqlite`).
- ~1–1,6 s na kandidáta, celkem ~80 min na stroji s 3 GB RAM; matice `R` 5 031 dní × 3 027.
- Reprodukce: `uv run python scripts/run_grid.py sharadar_2026-09-25`,
  `uv run python scripts/analyze_grid.py sharadar_2026-09-25 a03e443786b83e08`.

## Výsledky po rodinách (po nákladech)

Laťka stejného období: SPY Sharpe 0,31 / CAGR 6,0 % / propad 55 %; EW LIQ1000 0,32 / 6,4 % / 61 %.

| Rodina | Kandidátů | Medián Sharpe | Medián Sharpe okolí | Medián CAGR | Medián propad | Medián obrat/rok | Medián náklady | Medián Sharpe při nákladech ×3 | Projde filtry |
|---|---|---|---|---|---|---|---|---|---|
| low_vol | 432 | **0,84** | **0,84** | 9,2 % | 16 % | 531 % | 59 bps | 0,70 | 396 |
| regime_market | 3 | 0,38 | 0,38 | 5,4 % | 21 % | 436 % | 23 bps | 0,34 | 0 |
| st_reversal | 432 | 0,15 | 0,15 | 2,3 % | 54 % | 3 806 % | 398 bps | −0,23 | 45 |
| xs_momentum | 864 | 0,12 | 0,12 | 0,8 % | 75 % | 1 226 % | 141 bps | 0,01 | 0 |
| ts_trend | 432 | 0,03 | 0,03 | −1,4 % | 80 % | 1 551 % | 174 bps | −0,11 | 0 |
| breakout | 864 | −0,09 | −0,13 | −1,2 % | 61 % | 3 012 % | 308 bps | −0,54 | 149 |

Vliv portfoliových voleb (medián Sharpe přes všechny kandidáty):

| Volba | Varianty |
|---|---|
| Overlay | none 0,08 · vol_target 0,12 · **trend_filter 0,27** |
| Rebalance | weekly 0,02 · **monthly 0,24** |
| Počet titulů | 10: 0,04 · 20: 0,14 · 30: 0,15 · 50: 0,19 |
| Vážení | equal 0,11 · inverse_vol 0,15 |
| Hystereze | 1,0: 0,12 · 1,5: 0,13 · 2,0: 0,14 |

Nejlepší kandidát, který projde tvrdými filtry, v každé rodině:

| Rodina | Konfigurace | Sharpe | CAGR | Propad | Náklady | Sharpe ×3 |
|---|---|---|---|---|---|---|
| low_vol | 252d, top 10, equal, monthly, hyst. 2, vol_target | 1,10 | 12,2 % | 12,7 % | 25 bps | 1,05 |
| st_reversal | 10d, top 50, inverse_vol, monthly, hyst. 2, trend_filter | 0,52 | 10,0 % | 37,6 % | 167 bps | 0,34 |
| breakout | 252d, top 30, inverse_vol, monthly, hyst. 2, trend_filter | 0,45 | 6,7 % | 33,1 % | 143 bps | 0,22 |

## Statistika přeučení

| Ukazatel | Hodnota | Poznámka |
|---|---|---|
| PBO (CSCV, 16 bloků) | **0,003** | Brána ≤ 0,30 splněna. Nízké hlavně proto, že low_vol vede konzistentně ve všech podobdobích; neříká nic o přeučení *uvnitř* low_vol. |
| Efektivní počet kandidátů | **39** shluků při korelaci 0,85 | 3 027 kandidátů je z velké části tatáž sázka (long akcie). |
| DSR nejlepšího kandidáta | 0,09 při N = 3 027 · **0,86** při N_eff = 39 | Správné je N_eff; ani tak nepřekročí 0,90 (brána Fáze 3 pro metodiku). |

## Kontrola věrohodnosti low_vol

Výsledek je nápadně dobrý, proto byl prověřen:
- Drží klasické defenzivní tituly (utility, spotřební zboží: D, ED, SO, JNJ, PG, WMT), ne firmy
  v probíhající akvizici ani SPAC.
- Bez overlaye: CAGR 13,1 %, propad 32 %, Sharpe 0,95. Rok 2000 +40 % (defenzivní sektory při
  splasknutí dot-com bubliny), 2008 −16 % proti −37 % u SPY. Nízký propad varianty s overlayem
  je daný snížením investovanosti v krizích (průměr 64 % v roce 2008).
- Odpovídá literatuře o low-volatility anomálii; 2000–2019 bylo pro defenzivní akcie navíc
  příznivé (klesající sazby). Koncentrace do 10 titulů = silná sektorová sázka na utility a
  citlivost na úrokové sazby – rizikový faktor pro holdout (2022).

## Chyby nalezené a opravené během Fáze 2

1. **Únik budoucnosti v rozvrhu rebalance** (test úniku): „poslední obchodní den týdne/měsíce"
   vyžaduje znát zítřejší datum → rozhodnutí po prvním obchodním dni týdne/měsíce.
2. **Znovunákup titulu se signálem k prodeji** (ruční test výběru).
3. **0/0 v enginu** při plné investici a zaokrouhleně záporné hotovosti (−1e-17) → NaN u 3
   kandidátů regime; regresní test.
4. **SPY bez pořadí likvidity** → účtována nejvyšší sazba nákladů (90–170 bps/rok místo ~20).

Ostatní kandidáti nebyli opravami 3 a 4 dotčeni (jen oni drží SPY / jen u nich vznikl NaN).

## Závěry pro Fázi 3

- Pouze low_vol je v in-sample pohledu jasně nad laťkou. Momentum, trend a breakout po nákladech
  v likvidním univerzu nefungují; krátkodobý reversal žije jen z nízkých nákladů (při ×3 záporný).
- Ansámbl K = 2 bude pravděpodobně low_vol + něco jiného. Dedup podle korelace zajistí, že to
  nebudou dvě varianty low_vol, ale druhý člen bude výrazně slabší.
- Týdenní rebalance je jasně horší (náklady, šum) – potvrzuje obavu ze spec.
- Rozhodující otázka Fáze 3: přežije výběr walk-forward (bez znalosti celého období) a projde DSR
  ≥ 0,90 a CI vůči EW_UNIV?
