# Výzkum 22 – evoluční hledání strategií na akciích: výsledek

2026-11-18 · pre-registrace v1.0 (`51092572f`) · kód `20bdc4ec1` · zámek `a2acffbd9` · Sharadar `sharadar_2026-09-25`.
Plný rozpočet pro oba běhy (A: dev 1999–2014, validace 2015–2019; zrcadlový B: dev 2006–2019, validace 2020–2022), 126 jobů v registru pokusů, žádný nespadl.
**Pozdní validace (A 2020–2025, B 2023–2025) se neotevřela:** pre-registrace ji povoluje jen běhu, který prošel G0–G4, a žádný takový není. Veškerá historie je navíc už spotřebovaná, takže i kladný výsledek by byl jen kandidát na forward.

## Verdikt

**Negativní (H5): A i B selhaly v G1–G4 (v obou neprošla G3, v A i G2).** Typovaný GP s NSGA-II nenašel na týdenních top-decilových výběrech z likvidního univerza žádnou strategii, která by po korekci na vícenásobné testování přežila validaci. Žádný ensemble ani šampión nevznikl, nic se nenasazuje a nezmrazuje.

| Brána | Run A | Run B |
|---|---|---|
| G0 integrita (parita s `engine.vector`, přesný holdingový výnos, delistingy) | prošla (1e-14) | prošla (1e-14) |
| G1 medián nejlepšího dev f1 (10 seedů) > p95 z 50 nul | **prošla**: 0,265 vs 0,245 | **prošla**: 0,348 vs 0,241 |
| G2 validační IR GA > náhodný pool | **selhala**: −0,075 vs +0,045 | prošla: +0,221 vs −0,010 |
| G3 aspoň jeden reprezentant projde StepM/BY-FDR | **selhala**: 0 ze 3 | **selhala**: 0 ze 3 |
| G4 PBO < 0,5 | prošla: 0,095 | prošla: 0,040 |
| G5 DNA (replikace v obou polovinách seedů) | selhala, nic se nereplikovalo | selhala, nic se nereplikovalo |
| G6 pozdní validace | neotevřena | neotevřena |

Dev f1 je in-sample. Výsledek G1 jen říká, že GA na skutečných datech najde in-sample víc než ve většině případů na datech bez signálu, ne že nalezené pravidlo funguje.

## Co jsme udělali

- **Prostor:** malá gramatika (23 terminálů: výnosy 1–250 dní, volatility, z-skóre, vzdálenost od 250denního maxima/minima, 2 stavy trhu; aritmetika, `rank`, `where`, porovnání), hloubka ≤ 5, uzlů ≤ 15, normální forma stromů.
- **Realizace:** týdenní long-only top-decil (100 jmen) z point-in-time top 1000 podle likvidity, rozhodnutí po zavření prvního obchodního dne týdne, plnění na dalším openu, projektový nákladový model, fitness při dvojnásobných nákladech. Kernel odpovídá `engine.vector.simulate` včetně neprodejných pozic a výplat při delistingu.
- **Fitness:** f1 = 25. percentil excess IR přes dev foldy (A 8, B 7; proti EW benchmarku se stejnými náklady, průměr ze 4 jitterů), f2 = roční obrat, f3 = efektivní uzly.
- **Hledání:** 10 seedů GA (populace 500, 60 generací), 3 × 110 000 náhodných stromů (po sloučení 312 595 různých normálních forem, požadavek 300 000), 50 nul N1 (stejné GA na datech se zamíchanými budoucími výnosy mezi způsobilými jmény každého týdne). Nuly N1 + náhodné hledání N2 jsou kontroly, které rozhodují, zda je výsledek víc než šum a víc než slepé zkoušení.
- **Selekce:** deduplikace → korelační clustery → jeden reprezentant na cluster → StepM a BY-FDR na validaci (stacionární bootstrap, blok 4 týdny).

## Výsledky podrobně

**G1 a nuly.** Nejlepší dev f1 jednotlivých GA seedů: A 0,238–0,346, B 0,203–0,503. Nuly N1: medián A −0,071, B −0,113; nejvyšší A 0,688, 0,261, 0,260, 0,228, 0,161; nejvyšší B 0,365, 0,291, 0,243, 0,238, 0,237. Šum tedy dosahuje podobných hodnot jako skutečné běhy. Nula č. 45 v A (0,688, víc než dvojnásobek nejlepšího skutečného seedu) byla přezkoumána: její vítězné pravidlo na 10 dalších nezávislých permutacích dává f1 −0,14 až −1,27 a na skutečných datech −0,74, jde tedy o přeučení jedné konkrétní permutace, ne o chybu generátoru nul. V G1 zůstává (95. percentil se počítá ze všech 50 podle pre-registrace). Marže G1 v A je 0,02 a není robustní závěr.

**G2.** Medián validačního IR (1× náklady): A GA −0,075 vs náhodné +0,045, tedy GA je horší než slepé zkoušení se stejným rozpočtem; B GA +0,221 vs náhodné −0,010. Obě čísla jsou mediány ze 3 reprezentantů GA a 2 reprezentantů náhodného poolu, takže jde o velmi malý vzorek. B a A si protiřečí, což je samo o sobě informace: žádný z výsledků není reprodukovaný.

**G3.** Reprezentantů je po shlukování jen 3 (GA) a 2 (náhodný pool) v obou bězích, počet nezávislých kandidátů je tedy malý. Žádný z nich neprošel ani StepM (FWER 10 %), ani BY-FDR (q 0,10).

**G4.** PBO 0,095 (A) a 0,040 (B) přes 84 370 a 84 770 kandidátů. Pro tento výzkum to není argument pro strategii: ukazuje, že pořadí kandidátů je mezi bloky stabilní, ne že mají kladnou očekávanou hodnotu (G2 a G3 jsou přitom záporné).

**G5.** Žádná komponenta DNA se nereplikovala v obou polovinách seedů. Při počtu „přežilých" strategií 3 (A) a 3–16 (B) a nejmenším dosažitelném q 0,0074 až 0,017 nemá test sílu najít cokoli kromě velmi silného efektu. Tabulky frekvencí jsou v `runs/r22/{A,B}/dna.json`.

**Diagnostiky.** DSR, MinTRL, break-even násobek nákladů, srovnání s kanonickými strategiemi, E_rand a sladěné náhodné strategie se nepočítají: bez ensemblu E ani šampióna nemají co hodnotit.

## Co z toho plyne a co ne

- Na tomto univerzu, v této gramatice a s tímto rozpočtem GA nenašlo robustní strategii. To odpovídá tomu, co ukázala rešerše: po roce 1998 GP/GA pravidla na denních akciích po nákladech selhávají.
- Výsledek **není důkaz, že taková strategie neexistuje.** Test má malou sílu v G3 (2–3 klastry), gramatika je úzká (bez fundamentů, bez toků, bez intradenních dat) a rozpočet omezený (500 × 60 × 10 seedů).
- Proti sobě stojí dvě čísla z G2 (A záporné, B kladné). Obě jsou z 2–3 kandidátů a žádné z nich neprošlo G3, takže se nesmí číst jako „v B něco je".
- Hodnota experimentu je v kontrolách: nuly N1 ukázaly, že šum dosahuje dev f1 až 0,69; samotný dev f1 skutečných seedů (0,2–0,5) tedy nic nedokazuje.

## Omezení a odchylky

- Dev data A byla před zmrazením pravidel použita pro G0 a smoke běhy (jen časování, nezaregistrováno); popsáno v pre-registraci.
- Pozdní období (A 2020–2025, B 2023–2025) je vyčerpané. Čistý test existuje jen forward po 2026-09-25.
- G0 poprvé selhala (kernel nepočítal s neprodejnými pozicemi) a v nule N1 se našla chyba duchů-pozic; obojí opraveno před plným během, viz deník v pre-registraci a specifikace v1.2.
- První spuštění `select` pro A bylo zabito limitem paměti (cca 3 GB potřeba) a spuštěno znovu bez limitu; výsledek se tím nezměnil. `select` nevypisuje průběh, výpočet PBO trvá hodiny.
- Hodnoty IR jednotlivých reprezentantů a jejich stromy se do `selection.json` neukládají (jen mediány). Při případném dalším rozboru je třeba spočítat znovu.

## Soubory

`runs/r22/{A,B}/` (výstupy hledání, `selection.json`, `dna.json`), `docs/research22/results/g0_{A,B}.json`, `docs/research22/PREREGISTRATION.md`, `runs/registry/research22.sqlite` (126 pokusů). Specifikace: `/home/kapo/ccode/reports/Experiment 1 specifikace v1.1.md` (v1.2).
