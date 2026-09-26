# Výzkum 2 – výsledky (R2)

**Datum:** 2026-09-26 · **Období vyhodnocení:** 2010-01-04 … 2026-09-25 · **Pre-registrace:**
`PREREGISTRATION.md` v1.0 + upřesnění z §10 (vše commitnuto před prvním během, kód R2 také).

## Verdikt

**H1 předem daná kritéria nesplnila (1 ze 3). Hypotéza je zamítnutá.** Multi-asset alokace
s trendovým filtrem měla v letech 2010–2026 **nižší** Sharpe než prosté 60/40 a rozdíl je
statisticky významný v neprospěch strategie. H2 (walk-forward výběr ze 100 kandidátů) dopadla
o něco lépe, ale 60/40 také nepřekonala. Výzkum 2 se tím uzavírá negativně; forward test
nemá smysl spouštět (§5).

## Výsledky

| 2010–2026 | CAGR | Volatilita | Sharpe | Max. propad | Průměrná investovanost | Náklady |
|---|---|---|---|---|---|---|
| **H1** (iv126, SMA200, bez rel. momenta) | **4,0 %** | 4,9 % | **0,52** | 9,3 % | 70 % | 11 bps/rok |
| H2 (walk-forward výběr, K = 2) | 6,1 % | 7,1 % | 0,66 | 9,6 % | 78 % | 22 bps/rok |
| **60/40 (SPY/IEF)** | **9,8 %** | 9,8 % | **0,85** | 21,2 % | 100 % | – |
| SPY | 14,3 % | 17,1 % | 0,78 | 33,7 % | 100 % | – |
| Rovné váhy 9 ETF | 6,6 % | 9,2 % | 0,58 | 19,3 % | 100 % | – |

| H1 vs. 60/40 | Hodnota |
|---|---|
| ΔSharpe [90% CI] | −0,33 [−0,68; −0,03] |
| ΔCAGR [90% CI] | −5,8 p.b. [−8,8; −2,9] |
| DSR (N = 1) | 0,98 – Sharpe je kladný, jen nižší než u benchmarku |
| H2 ΔSharpe [90% CI] | −0,19 [−0,57; +0,16] |

**Kritéria H1 (pre-registrace §6):**

| Kritérium | Výsledek |
|---|---|
| Dolní mez 90% CI rozdílu Sharpe vs. 60/40 > 0 | ✘ (−0,03 … horní mez −0,03) |
| Max. propad ≤ 25 % | ✔ (9,3 %) |
| Sharpe při nákladech ×3 > 60/40 | ✘ (0,47 vs. 0,85) |

**Robustnost (§7):** Sharpe H1 při nákladech ×2 / ×3: 0,50 / 0,47; záměna ETF za alternativy 0,53;
SMA150 0,58, SMA250 0,52. Závěr na žádné z těchto voleb nezávisí.

| Podobdobí | H1 Sharpe / CAGR | H2 Sharpe / CAGR | 60/40 Sharpe / CAGR | SPY Sharpe / CAGR |
|---|---|---|---|---|
| 2010–2019 | 0,69 / 3,6 % | 0,71 / 4,8 % | 1,19 / 10,1 % | 0,89 / 13,5 % |
| 2020–2026 | 0,31 / 4,5 % | 0,63 / 8,1 % | 0,56 / 9,3 % | 0,68 / 15,5 % |

Jediný rok, kdy strategie splnila účel: **2022** (H1 −3,5 % vs. 60/40 −16,6 %).

## Proč to nevyšlo

1. **Nepákový risk parity má nízký výnos.** Váhy podle inverzní volatility dávají v průměru
   ~37 % dluhopisům a jen ~8 % SPY; trendový filtr drží dalších ~30 % v hotovosti. Literatura
   o risk parity počítá s pákou na cílovou volatilitu – ta je pro čistě long-only zadání mimo rozsah.
2. **2010–2026 bylo mimořádné období pro 60/40:** dlouhý býčí trh amerických akcií a (do 2020)
   klesající sazby. Diverzifikace do komodit, EM a zlata v tomto období stála výnos
   (rovné váhy 9 ETF: Sharpe 0,58).
3. **Trendový filtr pomáhá jen v dlouhých medvědích trzích** (2022). V rychlých propadech
   s rychlým odrazem (2020) a v klidných býčích letech stojí výnos.

## Poznámka k H2

Walk-forward výběr volil převážně konfigurace s relativním momentem a 12měsíčním trendovým
filtrem. Oproti H1 přidal ~2 p.b. ročně, ale ani to nestačilo na 60/40. N_meth = 2 (H1 + H2).

## Závěr pro forward test

Pre-registrace určila forward test jako rozhodující pro **nasazení**. Protože H1 historická
kritéria nesplnila, kandidát na nasazení neexistuje. Forward test by nepřinesl rozhodnutí,
proto se nespouští. Kód a data zůstávají, práh pro forward test (10. percentil 12měsíčního
rozdílu vůči 60/40 = −12,4 p.b.) je zapsán v `summary.json` pro případné budoucí použití.

Reprodukce: `uv run python scripts/r2_build_panel.py sharadar_2026-09-25`,
`uv run python scripts/r2_evaluate.py sharadar_2026-09-25` (~1 min).
