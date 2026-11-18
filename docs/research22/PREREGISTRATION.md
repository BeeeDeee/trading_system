# Výzkum 22 – Porazí evoluční hledání pravidel náhodné hledání? (akcie, pre-registrace v1.0)

**Stav: NÁVRH, nezmrazeno.** Zmrazením se rozumí: tento soubor + `docs/research22/results/frozen_{A,B}.json` + kód `src/qlab/research22/*.py` a `scripts/r22_*.py` jsou commitnuté, hash metodiky je uložený v trezoru (`scripts/r22_final.py freeze`) a existuje `docs/research22/run_lock.json`. Do té doby plné běhy odmítnou startovat. Podrobné zdůvodnění a historie návrhu: `reports/Experiment 1 specifikace` (v1.2, sekce 10).

## 1. Otázka a hypotézy

Evoluční hledání (typované genetické programování + NSGA-II) nad malou gramatikou cenových pravidel pro výběr akcií se obvykle přeučí. Tento výzkum se neptá „najdi nejlepší strategii“, ale **zda celý evoluční postup vyrobí něco, co přežije pozdější data lépe než stejně drahé náhodné hledání ze stejné gramatiky**, a zda se z přeživších dá vyčíst opakující se struktura.

- **H1:** nejlepší dev fitness evoluce převyšuje null celého searche (permutace dopředných výnosů).
- **H2:** pipeline GA poráží stejně drahé náhodné hledání na validaci.
- **H3:** ensemble validovaných reprezentantů nepropadne na pozdní validaci.
- **H4 (podmíněná G5):** jednoduchá strategie D odvozená z replikované „DNA“ nepropadne a není horší než náhodné strategie sladěné v obratu a složitosti.
- **H5:** verdikt nezávisí na volbě období (běh A a zrcadlový běh B dávají stejnou odpověď).

## 2. Data a přiznání spotřebovaných dat

Sharadar SEP `sharadar_2026-09-25` (1997-12-31 až 2026-09-25), panel `panel_liq1000` (point-in-time top 1 000 podle 63denního mediánu neupraveného dollar volume, cena ≥ 5 USD, bez SPAC, min. 252 dní historie; členství k rozhodovacímu dni), včetně 11 448 delistovaných titulů. Politika delistingu je `qlab.data.normalize.terminal_return` (akvizice přesná výplata, bankrot −100 %, výkonnostní −30 %, likvidace SPAC jako výplata trustu).

**Žádné období není nedotčené.** README: „Holdout 2020–2026 je spotřebovaný“; výzkumy 1, 4, 11 a 21 vyhodnotily akciové hypotézy v 2020+ a historii před 2020 vyhodnocovaly opakovaně jako vývoj. Proto:
- „pozdní validace“ (2020–2025 u A, 2023–2025 u B) **není čistý holdout** a gramatika je ovlivněná znalostí dřívějších výsledků;
- žádný výsledek na datech do 2026-09-25 není potvrzení; nejvýš **kandidát na forward** (forward po 2026-09-25 vyžaduje obnovu Sharadaru nebo jiný zdroj cen).

**Co už bylo s dev daty běhu A uděláno před touto pre-registrací** (nic z toho nebylo zapsáno do registru ani použito k volbě parametrů):
1. propustnost a paritní kontroly vyhodnocovače, včetně dvanácti ručně napsaných pravidel (jejich Q25 byl záporný, −0,17 až −2,86 při 2× nákladech) a statistik benchmarku (EW 9,9 % ročně po nákladech, obrat 3,2×/rok);
2. smoke běhy GA na dev A (populace 40×3, 100×8, 60×4 pro seedy 1 a 6), náhodného hledání (n = 300) a jednoho nulového běhu (20×2), jen kvůli časování a technické správnosti; výsledky se neinterpretují;
3. smoke běh selekce a DNA s **dev místo validace** (validační data nebyla čtena);
4. kontrola G0 na dev obou běhů (shoda s enginem pro tři kanonické strategie, bez vyhodnocení výkonnosti) a smoke běh pozdní stage s **dev místo pozdní validace**, při kterém se vypsala i IR tří kanonických strategií a tří ručně vložených stromů na dev A (např. `mom_12_1` ≈ 0,29 při 1× nákladech); to je informace o dev datech, ne o validaci ani pozdní validaci.

## 3. Běhy, stage a embargo

| | Běh A (primární) | Běh B (zrcadlový) |
|---|---|---|
| Warm-up | od 1997-12-31 | od 2005 |
| Dev | 1999-01-04 až 2014-12-31, 8 foldů po 2 letech | 2006-01-03 až 2019-12-31, 7 foldů po 2 letech |
| Validace | 2015-01-01 až 2019-12-31 | 2020-01-01 až 2022-12-31 |
| Pozdní validace | 2020-01-01 až 2025-12-31 | 2023-01-01 až 2025-12-31 |

Embargo 21 obchodních dní na začátku validace a pozdní validace. Terminály smějí číst historii před hranicí; týden, jehož držení by zasahovalo za konec stage, se nepočítá. A a B se překrývají (dev 2006–2014, test 2020–2025), takže nejsou nezávislé replikace.

## 4. Strategie, gramatika, náklady

- **Rozhodnutí** po close prvního obchodního dne týdne (`qlab.schedule.period_starts`), nákup EW top decilu (100 titulů) na open dalšího dne, držení do open po dalším rozhodnutí; váhy driftují a vracejí se na 1/k. Benchmark: EW způsobilý vesmír se stejným rebalancem a náklady. Metrika: týdenní čistý nadvýnos, IR = průměr / směrodatná odchylka × √52.
- **Gramatika** (typovaná, hloubka ≤ 5, uzlů ≤ 15): 23 terminálů (`ret_{1,5,20,60,120,250}`, `vol_{20,60}`, `dvr_20_250`, `dmax_250`, `dmin_250`, `z{20,60}_{ret_1,ret_5,ret_20,ret_60,vol_20,vol_60}`), stavy `mkt_trend`, `mkt_vol_high`, konstanty {−1; 0,5; 1; 2}, operátory `add sub mul div(chráněné) neg rank gt where`. Vyhodnocuje se normální forma stromu.
- **Pravidla remíz:** shodné hodnoty = průměrné pořadí; remíza na hranici top-100 = nejnižší ID titulu.
- **Náklady:** zmrazený `CostModel` projektu (`configs/frozen_defaults.yaml`; půl-spread 3/6/12 bp podle pořadí likvidity, min. 5 bp, před decimalizací × 2,5). Fitness při 2×, reporting při 1×, 2×, 3×. Obrat = nákupy + prodeje / NAV.

## 5. Fitness a hledání

NSGA-II (DEAP `selNSGA2`, `selTournamentDCD`), cíle: maximalizovat **f1** = Q25 přes dev foldy z čistého excess IR (každý fold zprůměrovaný přes K = 4 jitterované kopie: lookback ±1 krok v rodině terminálů, konstanty × U(0,8; 1,2); seed jitteru konstantní, SEED = 22), minimalizovat roční obrat, minimalizovat efektivní počet uzlů. Omezení: ≥ 90 % týdnů s alespoň dvěma různými hodnotami skóre. Populace 500, 60 generací, 10 seedů (1–10), křížení 0,7 na pár, mutace 0,3 na jedince, duplicitní genom se nahradí náhodným imigrantem. Zastavení výhradně rozpočtem.

## 6. Nully

- **N1:** stejná GA (stejný rozpočet) na R = 50 datasetech, v nichž se dopředné výnosy (a příznaky delistingu s nimi) permutují mezi způsobilými tituly v rámci každého týdne; seedy 10 000 + index.
- **N2:** 300 000 různých normálních forem z téže gramatiky (ramped half-and-half), stejná fitness.

## 7. Selekce

Kandidáti = finální populace a Pareto fronty seedů (strom patří nejnižšímu seedu). Dev série (1× náklady) → odstranění přesných duplicit → korelační clustery (average linkage, počet clusterů podle siluety) → reprezentant clusteru = nejlepší dev f1. Na validaci StepM (Romano–Wolf, FWER 10 %) a BY-FDR (q = 0,10) proti H0 „průměrný nadvýnos ≤ 0“ (studentizovaný stacionární bootstrap, blok 4 týdny, 2 000 replikací). **E** = EW reprezentantů, kteří projdou aspoň jedním testem, seřazených podle dev f1, nejvýše 20 (bez nettingu obchodů). **E_rand** = stejné pravidlo na náhodném poolu (stejný počet kandidátů, vybraných podle dev f1); pokud nikdo neprojde, nejlepších |E| podle dev f1 (příznak `random_ensemble_fallback`). **Šampion C** = reprezentant s nejvyšším validačním IR. **Kanonické strategie:** `mom_12_1 = ret_250 − ret_20`, `reversal_1w = −ret_5`, `lowvol_60d = −vol_60`. **Náhodné strategie sladěné** v obratu a velikosti: 1 000 nejbližších z N2. **D** (volitelně): jednoduchá strategie zapsaná před otevřením pozdní validace.

## 8. Brány a verdikt

Brány platí pro každý běh zvlášť.

- **G0 integrita (prošla 2026-11-07 pro A i B, `docs/research22/results/g0_{A,B}.json`):** `scripts/r22_g0.py` (shoda týdenního portfolia s `engine.vector.simulate` na skutečných datech dev při nulových intradenních výnosech (NAV po obchodu do relativní odchylky 10⁻⁹, obrat, počet pozic); přesný součin holdingového výnosu; delistované tituly jsou ve vesmíru) a zelené testy `tests/test_research22.py` (včetně kauzality terminálů). Selhání = oprava, ne výsledek.
- **G1 (H1):** medián nejlepšího dev f1 přes 10 seedů > 95. percentil nejlepšího f1 z 50 nulových běhů N1.
- **G2 (H2):** medián validačního IR (1× náklady) top-20 reprezentantů GA > stejná statistika pro náhodný pool.
- **G3:** aspoň jeden reprezentant projde StepM nebo BY-FDR.
- **G4:** PBO < 0,5 (CSCV, 16 bloků, dev + validace, deduplikovaný vzorek vše, co GA vyhodnotilo, až 100 000).
- **G5 (DNA):** komponenta s q < 0,10 (BY přes komponenty, permutace štítků uvnitř seedů, **5 000 permutací**) a poměrem šancí > 1,5 ve stejném směru v objevné (seedy 1–5) i replikační (6–10) polovině; „robustní“ = validační IR > 0 při 1× a ≥ 0 při 2× nákladech; kontrolní proměnné obrat, uzly, dev f1. Jinak se H4 netestuje.
- **G6 (pozdní validace, vyčerpaná):** IR(E) > 0 při 1× a ≥ 0 při 2×, IR(E) > IR(E_rand) při 1×, relativní max drawdown E vůči benchmarku (poměr bohatství) ≤ 1,5× nejhoršího dev foldu. Otevírá se jen u běhu, který prošel G0–G4, a **A i B se otevírají až poté, co jsou oba zmrazené**, každý jednou (trezor).

**Diagnostiky (nejsou brány):** DSR pro E s počtem pokusů N̂ = počet clusterů × {0,3; 1; 3} **a konzervativní řádek N = počet unikátních sérií**; výroky o významnosti se opírají o konzervativní řádek. MinTRL, break-even násobek nákladů šampiona, bootstrap CI (90 %) IR(E), srovnání s kanonickými strategiemi, E_rand a 1 000 sladěnými náhodnými strategiemi.

**Verdikt (H5):**
- *Kandidát na forward:* A projde G0–G6 a B projde G0–G4 a na své pozdní validaci má IR(E) > 0 při 1× nákladech. Víc než kandidát tento výzkum dát nemůže.
- *Negativní:* A i B selžou v G1–G4. Publikuje se negativní výsledek.
- *Nerozhodnuto:* cokoli jiného (např. A projde a B ne). Nic se nenasazuje; další krok je výslovné rozhodnutí, ne tichá úprava.

## 9. Postup, reprodukovatelnost a rozpočet

1. Commit této pre-registrace a kódu **před** prvním plným výpočtem; plné běhy hledání (`scripts/r22_search.py`) se zapisují do registru pokusů (`runs/registry/research22.sqlite`) před startem.
2. `r22_search.py` (ga ×10, null ×50, random) → `r22_select.py select` / `dna` → `r22_final.py prepare` → commit `frozen_{A,B}.json` → `freeze` (A i B) → `run` (A i B, jednou).
3. Pravidla pro pád: pád po registraci se zapíše jako opakování do registru a do deníku rozhodnutí.
4. Odhad rozpočtu na běh (z naměřených ~17 backtestů/s na proces při dvou souběžných, 21,6 samostatně; neměřeno celé): ≈ 150 core-hodin, ≈ 2,5–3,5 dne na běh při 3 paralelních procesech (špička 1,37 GB RAM na proces; nespouštět čtyři).

## 10. Deník rozhodnutí

| Datum | Verze | Rozhodnutí |
|---|---|---|
| 2026-11-07 | 1.0 | G0 na skutečných datech nejdřív selhala (kernel nepočítal s neprodejnými pozicemi, odchylka až 1,2 % NAV za týden); kernel přepsán podle `engine.vector`, G0 poté prošla na 10⁻¹⁴. Chyba ducha-pozice v nule N1 opravena. Návrh. Zrcadlový běh B přidán na žádost uživatele (obava, že 1999–2014 je pro dnešní trh staré). Pozdní validace přeznačena na vyčerpanou (holdout 2020–2026 je spotřebovaný). Rozhodnutí o N̂: počet clusterů × {0,3; 1; 3} plus konzervativní řádek s počtem unikátních sérií (souhlas uživatele). Odchylky od původní specifikace v1.1 (rozhodnutí v prvním obchodním dni týdne, nákladový model projektu, `zscore_ts` jako transformace terminálu s okny 20 a 60, 63denní likvidita, konstantní seed jitteru) viz specifikace v1.2, sekce 5, 6 a 10. |
| 2026-11-18 | 1.0 | **Verdikt: negativní (H5).** Plná hledání A i B doběhla (126 jobů, žádný pád). G1 prošla v A i B, G2 selhala v A a prošla v B, G3 selhala v A i B (0 ze 3 reprezentantů), G4 prošla, G5 selhala. Pozdní validace se neotevírá (žádný běh neprošel G0–G4), `prepare`/`freeze`/`run` se nespouští. Nula č. 45 v A (f1 0,688) ověřena jako přeučení jedné permutace, ne chyba generátoru. První `select` A byl zabit limitem paměti a spuštěn znovu bez limitu (výsledek stejný). Report: `REPORT.md`. |
