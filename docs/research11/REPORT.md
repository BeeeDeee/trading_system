# Výzkum 11 – ML výběr akcií z fundamentů, insiderů a toků 13F (report)

Stav: **uzavřeno 2026-10-03, zamítnuto (0 ze 6 kritérií).**
Pre-registrace: [`PREREGISTRATION.md`](PREREGISTRATION.md) (commit `6add303`, před prvním výpočtem).
Výsledky: [`results/dev.json`](results/dev.json), [`results/final.json`](results/final.json).

## Otázka

Vybere LightGBM z bodových fundamentů, transakcí insiderů a změn institucionálního držení (13F) měsíčně
50 akcií z LIQ1000 tak, aby portfolio po nákladech překonalo SPY, rovné váhy LIQ1000 a stejný model jen
s cenovými příznaky?

## Verdikt v jedné větě

Ne. Model má slabou, ale kladnou predikční sílu (rank IC ≈ 0,02–0,03), ta se ale na výnos portfolia po
nákladech nepřenese. Fundamenty a toky 13F nepřidaly nic proti modelu jen z cen a ve vývoji ani ve finále
neporazil SPY žádný ze tří modelů.

## Výsledky

| | Vývoj 2010–2019 | | Finále 2020-01 → 2026-08 | | |
|---|---|---|---|---|---|
| | CAGR | Sharpe | CAGR | Sharpe | Max. propad |
| M_P (jen ceny) | 9,1 % | 0,53 | 17,5 % | 0,72 | 41,2 % |
| M_PFI (+ fundamenty, insideři) | 6,9 % | 0,43 | 12,5 % | 0,56 | 46,4 % |
| **M_ALL (+ 13F)** | 9,4 % | 0,55 | 12,8 % | **0,57** | 46,3 % |
| SPY | 13,3 % | 0,92 | 15,5 % | **0,82** | 33,7 % |
| EW LIQ1000 | 11,8 % | 0,74 | 11,7 % | 0,59 | 41,4 % |

**Rank IC (průměr přes měsíce):**

| | Vývoj | Finále |
|---|---|---|
| M_P | 0,022 (t 2,6) | 0,027 (t 2,0) |
| M_PFI | 0,017 (t 2,0) | 0,029 (t 2,2) |
| M_ALL | 0,018 (t 2,0) | 0,025 (t 1,7) |

Obrat 15–18× ročně, náklady 1,3–1,7 % ročně.

**Kritéria finále (M_ALL):**
1. ΔSharpe vs SPY −0,25, 90% CI [−0,67; +0,12] ✗
2. vs EW LIQ1000 −0,02 ✗
3. vs M_P −0,15 a IC 0,025 < 0,027 ✗
4. podúseky vs SPY −0,16 a −0,33 ✗
5. zátěž nákladů ✗
6. DSR 0,36 (N = 3) ✗; s N = 3 027 0,12

## Co jsme se naučili

1. **Predikce existuje, ale je slabá a drahá.** IC 0,02–0,03 je statisticky na hraně (t ≈ 2), podobně jako
   meta-vrstva výzkumu 4 (0,034). Měsíční top 50 z ~1 000 akcií ale znamená obrat 15–18× ročně a ten sní
   víc, než predikce přinese.
2. **Fundamenty a toky model nezlepšily.** LightGBM jim dal ~40 % (F), ~8 % (I) a ~10 % (H) důležitosti,
   takže je používal, ale IC ani výnos proti modelu jen z cen nevzrostly. Informace v nich je buď
   v cenách už obsažená (value, quality, momentum spolu korelují), nebo ji měsíční horizont nezachytí.
3. **Koncentrované portfolio 50 akcií má vyšší riziko než trh:** propad 46 % vs 34 % u SPY v roce 2022.
   Kladný IC neznamená vyšší Sharpe, když výběr systematicky tíhne k volatilnějším akciím.
4. **Roky 2020–2026 přály velkým firmám** (SPY 15,5 % vs EW LIQ1000 11,7 %). Model vybíral z širšího
   univerza a soutěžil spíš s EW, ale ani to neporazil.

## Co by se dalo zkusit (nový výzkum, ne úprava tohoto)

- **Nižší obrat:** držet akcii, dokud nevypadne z top 150–200 (hystereze), nebo čtvrtletní rebalance.
  Polovina problému jsou náklady.
- **Optimalizace proti benchmarku místo top 50:** naklonit váhy SPY podle predikce s omezeným tracking errorem.
  Slabý IC se lépe využije malými odchylkami od trhu než koncentrovaným portfoliem.
- Obojí je ale post hoc na stejných datech. Poctivé ověření je jen forward test.
