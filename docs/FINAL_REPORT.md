# Závěrečný report – finální holdout 2020–2026

**Datum:** 2026-09-26 · **Snapshot:** `sharadar_2026-09-25` · **Metodika:** hash `1f4ca3dc43e58550`,
commit `ee457b9` · **Trezor:** otevřen jednou, 2026-09-26T16:38:55Z (`docs/final/vault.log`).

## Verdikt

**Metodika v holdoutu neobstála.** Za 2020-01-02 … 2026-09-25 vydělala 17 % celkem
(2,4 % ročně) proti 164 % u SPY (15,5 % ročně), se Sharpe 0,01 proti 0,68. Rozdíl vůči SPY je
statisticky významně **záporný**. Hypotéza projektu – že kombinace jednoduchých strategií po
nákladech překoná pasivní investici – se pro tuto metodiku **zamítá**. Spec §1 s tímto výsledkem
počítala jako s legitimním a pravděpodobným.

## Postup (bez zásahů po otevření)

1. Kód finálního vyhodnocení commitnut a pushnut před během (`ee457b9`), čistý pracovní strom.
2. Metodika zmrazena (`runs/vault/frozen.lock`), trezor otevřen jednou pro snapshot.
3. Mřížka 3 027 kandidátů přes 2000–2026, walk-forward pokračující stejným pravidlem
   (roční refit, rozšiřující se okno) přes roky 2020–2026: 22 foldů.
4. **Replikace:** roky 2005–2019 se s předem registrovaným během Fáze 3 shodují přesně
   (max. rozdíl denních výnosů 0,0). Ledger vs. vektorový engine: 1,3·10⁻¹⁵.

## Výsledky

| 2020-01-02 … 2026-09-25 | CAGR | Volatilita | Sharpe | Max. propad | Celkem |
|---|---|---|---|---|---|
| **Primární (K = 2)** | **2,4 %** | 10,7 % | **0,01** | 28,5 % | +17 % |
| Sekundární (K = 3) | 3,5 % | 9,3 % | 0,11 | 21,6 % | +26 % |
| EW_UNIV | 11,2 % | 23,5 % | 0,45 | 41,4 % | +103 % |
| SPY_TR | 15,5 % | 20,1 % | 0,68 | 33,7 % | +164 % |

| Statistika (primární, holdout) | Hodnota |
|---|---|
| DSR (N_meth = 2) | 0,31 (vývoj: 0,999) |
| ΔSharpe vs. EW_UNIV [90% CI] | −0,44 [−0,96; +0,19] |
| ΔSharpe vs. SPY [90% CI] | −0,67 [−1,19; −0,10] |
| ΔCAGR vs. SPY [90% CI] | −13,1 p.b. [−23,6; −4,0] |
| Konzistence: percentil Sharpe holdoutu mezi ročními Sharpe 2005–2019 | 13 % (dolní okraj, těsně nad varovnou hranicí 10 %) |

| Rok | Primární | SPY | Poznámka |
|---|---|---|---|
| 2020 | −12,7 % | +18,4 % | Q1 −16,4 % vs. −19,4 %: low-vol v covidovém propadu **nechránil**. Q2 −2,6 % vs. +20,2 %: overlaye snížily investovanost (vol_target až na 25 %) těsně před odrazem. |
| 2021 | +14,2 % | +28,7 % | |
| 2022 | −3,0 % | −18,2 % | Jediný rok, kdy defenziva fungovala podle očekávání. |
| 2023 | −3,5 % | +26,2 % | Růst sazeb a rally velkých technologických firem – nejhorší prostředí pro utility a REIT. |
| 2024 | +17,4 % | +24,9 % | |
| 2025 | +5,9 % | +17,7 % | |
| 2026 (do 25. 9.) | +1,1 % | +14,0 % | |

Výběr v holdoutu dál vybíral jen varianty low_vol (od 2024 s kratším 63denním oknem).

## Proč to nevyšlo

1. **Výsledek 2005–2019 stál na jednom režimu.** Klesající sazby a opakované krize (2008,
   2011, 2015, 2018) přály defenzivním titulům. V letech 2020–2026 převažovaly prudké odrazy
   a trh tažený několika velkými růstovými firmami.
2. **Overlaye reagují se zpožděním.** 63denní volatilita a SMA200 snížily investovanost až po
   pádu a vrátily ji až po odrazu – klasický whipsaw. V letech 2008 a 2011 to pomohlo, v roce
   2020 to stálo většinu ztráty.
3. **Robustnost na S&P 500 varovala.** Už ve vývojovém období byl efekt mimo LIQ1000 slabší
   a statisticky neprůkazný (Fáze 3, robustnost).
4. **Statistika vývoje byla příliš optimistická.** DSR 0,999 s N_meth = 2 nepočítal s „lidskými"
   stupni volnosti: volba low-vol rodiny vycházela ze známé literatury a období 2000–2019 bylo
   pro ni mimořádně příznivé. Holdout tuto slabinu odhalil – přesně k tomu slouží.

## Co projekt dokázal

- **Infrastruktura funguje a je ověřená:** datová vrstva přesně sleduje SPY (≤ 0,95 p.b./rok),
  normalizace rekonstruuje total return na 1e-9, dva nezávislé enginy se shodují na 1e-15,
  test úniku budoucnosti našel dvě skutečné chyby, trezor a registr pokusů udržely holdout čistý.
- **Poctivý negativní výsledek za dny místo měsíců** – hlavní designový cíl spec §1.
- 7 opravených chyb během vývoje (viz reporty fází); žádná nebyla opravena po otevření holdoutu.

## Doporučení

1. **Tuto metodiku neobchodovat.** Ani v defenzivní roli: v holdoutu snížila výnos o 13 p.b.
   ročně a propad 28,5 % není výrazně menší než u SPY (33,7 %).
2. **Holdout je spotřebovaný.** Jakákoli nová verze postavená s vědomím těchto výsledků
   (např. bez vol_target overlaye, sektorové limity) se nesmí vyhodnocovat na 2020–2026. Jediný
   čistý test je **forward test** – paper trading od teď, s obnoveným předplatným dat.
3. Pokud má výzkum pokračovat, doporučuji novou výzkumnou otázku s vlastní pre-registrací,
   například:
   - diverzifikace přes třídy aktiv (akcie / dluhopisy / komodity přes ETF) místo výběru akcií,
   - faktorové strategie bez tržního načasování (overlaye byly hlavní zdroj škody v holdoutu),
   - delší horizont dat (fundamenty `SF1` jsou stažené a point-in-time, zatím nevyužité).
4. Framework (data, enginy, validace, trezor) je znovupoužitelný beze změn.
