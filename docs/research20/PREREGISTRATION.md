# Výzkum 20 – forward test H1b a H2 z výzkumu 19 (pre-registrace)

Verze 1.0 · 2026-11-02 · Stav: **schváleno vlastníkem, brána kvality dat splněna; čeká na implementaci vyhodnocení a zmrazení kódu (§11)**.
Navazuje na `docs/research19/REPORT.md` (jediné dvě hypotézy, které prošly všemi kritérii: H1b a H2) a na PLAN §7.6.
Po commitu verze 1.0 platí: změny = řádek v §12 s důvodem, **před** dalším vyhodnocením.
Značka **[R]** = rozhodnutí autora návrhu tam, kde výzkum 19 mlčí; vlastník ji potvrdí nebo změní před commitem 1.0.

## 1. Otázky

Dvě hypotézy, každá s **pravidly zmrazenými z výzkumu 19 v1.1** (§4.2 a §4.3 beze změny) a na datech po `T0`:

- **F1 – ETF trend (H1b).** Vydělá Donchianův ensemble 50–250 dní s inverzním volatilitním sizingem na 16 ETF
  (`SPY QQQ IWM EFA EEM VNQ TLT IEF TIP LQD HYG GLD SLV DBC USO UUP`) nad T-billem po nákladech 3 bps na stranu?
- **F2 – RSI(2) (H2).** Vydělá nákup RSI(2) < 10 nad SMA200 a výstup nad SMA5 (nebo po 10 dnech) na 13 indexových a
  sektorových ETF (`SPY QQQ IWM DIA XLB XLE XLF XLI XLK XLP XLU XLV XLY`, max. 4 pozice) kladný čistý výnos na obchod?

## 2. Poctivá omezení předem

- **Výběr 2 ze 7.** F1 a F2 jsou ty dvě hypotézy výzkumu 19, které prošly; ostatních pět ne. Pozorované Sharpe 0,77
  (F1) a 0,49 (F2) obsahuje výběrové zkreslení (DSR při N = 7: 0,988 a 0,898; při N = 34: 0,955 a 0,731). Plán
  počítá s haircutem 0,5×: očekávaný live Sharpe **F1 ≈ 0,38, F2 ≈ 0,25**. Testovat se proto má i to, jestli
  zbude aspoň polovina.
- **Síla testu je malá** (§8). Ani při pravdivém historickém edge nelze za 36 měsíců statisticky potvrdit víc než
  zhruba každý šestý (F1) až třetí (F2) případ. Test je především **screen na selhání a sběr čistých dat**, ne rychlý
  potvrzovač. Nejpravděpodobnější výsledek je „nepotvrzeno“.
- **Kontaminovaná je jen historie.** Data po `T0` jsou jediný čistý důkaz. Data mezi koncem Sharadaru (2026-09-25)
  a `T0` se používají jen jako rozběh indikátorů, nehodnotí se (na začátku se nedrží žádná pozice).
- **Jiný zdroj dat než ve výzkumu 19.** Sharadar skončil, forward používá Yahoo + Nasdaq (§3). Rozdíly ve vykreslení
  dividend, splitů a closů jsou riziko; proto brána kvality dat před `T0` (§3.3).
- **F2 plní na close (MOC).** V paper testu se rozhodnutí i plnění berou na oficiálním close, tedy horní mez toho, co
  jde reálně stihnout. Varianta s plněním na open (r1 ve výzkumu 19: průměrný obchod +0,258 %) se hlásí vedle.
- **F1 a F2 nejsou nezávislé** (stejná ETF; korelace sleevů B–C ve výzkumu 19: 0,42, ve dnech SPY < −2 %: 0,55).
- **Survivorship koše:** všechna ETF žijí dodnes. Forward ho nemá (koš je pevný od `T0`).

## 3. Data

### 3.1 Zdroje
- **Ceny:** Yahoo chart API (primární; open/high/low/close, dividendy a splity z `events=div,split`), Nasdaq historical
  API jako nezávislá kontrola (stejný přístup jako `evening_bot_server/server/evening/prices.py`). Close je
  **bez dividendové úpravy**, adjclose se nepoužívá (Yahoo ho při každé dividendě přepisuje zpětně).
- **Hotovost:** FRED DTB3 jako ve výzkumu 19 (`qlab.engine.costs.cash_returns`).
- **Kalendář:** obchodní dny SPY z Yahoo.
- Stažení denně po US close cronem (bez výpočtu). Každý snapshot se ukládá surový s sha256 do
  `data/raw/yahoo_forward20/<datum>/` (git-ignorováno).

### 3.2 Total return
Denní řada se skládá **z uložených surových snapshotů**, ne z adjclose. Sharadar-ekvivalent (`closeadj`) se sestaví
ze zpětně nepřepsatelných vstupů: split-adjusted close a cash dividend na ex-date `D`,
`TR_t / TR_{t−1} = (close_t + D_t) / close_{t−1}` a open se převede na stejný základ faktorem téhož dne
(`open_adj = open · closeadj_t / close_t`, přesně jako `qlab.data.normalize`, takže `ret_co` a `ret_oc` mají tu
samou definici jako ve výzkumu 19). Splity se berou z `events=split` pro den, kdy nastaly.
Za konec rozběhu se řada napojí násobně na úroveň TR indexu z `panel_r19_etf` k 2026-09-25.

### 3.3 Brána kvality dat (před commitem 1.0, ne výpočet strategie)
Na překryvu 2025-09-26 → 2026-09-25 (jeden rok, oba zdroje) se pro všech 26 tickerů porovná denní close-to-close TR
Yahoo-sestavený proti Sharadaru. **[R]** Požadavky: medián |rozdíl| ≤ 1 bp, 99. percentil ≤ 10 bps, roční kumulativní
rozdíl ≤ 0,5 %, shodná množina dnů a **maximum ≤ 50 bps**, kde se maximum bere z „rozdílu neodpuštěného sousedním dnem“:
`m_t = min(|d_t|, |w_{t−1,t}|, |w_{t,t+1}|)`, `d` = denní rozdíl a `w` = rozdíl složených dvoudenních výnosů. Dividenda
zaúčtovaná oběma zdroji v sousedních dnech (+x, −x) se tak vyruší, izolovaná chyba zůstane v plné výši. Denní maximum se
hlásí jako popisné. Nesplněný ticker → **stop a dotaz vlastníkovi**, nic se neladí tolerancemi.
Výsledek brány se uloží do `docs/research20/results/data_gate.json` jako první soubor výsledků.

### 3.4 Chybějící data
Den s chybějícím close/open u tickeru koše se doplní z Nasdaqu (shoda close do 0,3 % jako v `repair_ohlc`), jinak se
den **nehodnotí a doplní se zpětně**, jakmile zdroj data dodá. Výpočet je bezstavový: vždy se přepočítá celá historie
od `T0` z uložených dat. Chybějící den se **nikdy** nenahrazuje „neplatným oknem“ (jeden chybějící den by u L = 250
vynuloval signál na rok). Zrušené ETF z koše: pozice se vyplatí na posledním close, náhrada se nedělá.

## 4. Pravidla (beze změny po commitu)

Doslova `docs/research19/PREREGISTRATION.md` v1.1:
- **F1 = §4.2:** lookbacky 50, 100, 150, 200, 250; stav a trailing stop podle §4.0; `s_i = E_i · min(2, 0,10/σ_i)`,
  `σ` z 90 dní, √252; `w_i = s_i / 16` s pevným jmenovatelem; Σw > 1 → škálovat na 1; denní rozhodnutí, pásmo 0,20,
  plnění na open *t+1*; hotovost T-bill.
- **F2 = §4.3:** RSI(2) Wilder, SMA200, SMA5 z TR indexu; vstup MOC při `P > SMA200` a `RSI2 < 10` (nejnižší RSI2,
  shoda abecedně), `max_positions = 4`, velikost `NAV/4` v okamžiku vstupu, výstup MOC při `P > SMA5`, časový stop
  10 dní; hotovost T-bill.
- Náklady 3 bps na stranu (`ET.etf_cost`, ×1); bez ADV limitu.
- **Kód je tentýž:** `qlab.research19.{signals,etf_trend,rsi2}`, `qlab.engine.vector`, `qlab.research5.sim` pod
  zámkem (§11). Nový kód jen pro data a vyhodnocení.

## 5. Obchod a portfolio

Dvě oddělená paper portfolia (žádné sdílení kapitálu), NAV 1 na `T0`, žádné pozice na začátku. Denní výnos
portfolia včetně T-billu na volné hotovosti. **F1:** první rozhodnutí na close posledního obchodního dne před `T0`
(plnění na open `T0`). **F2:** první rozhodnutí na close `T0`.

## 6. Období

| Fáze | Období |
|---|---|
| Začátek forwardu | **první US obchodní den po commitu verze 1.0** (`T0`) |
| Rozběh indikátorů | historie do `T0` (Sharadar do 2026-09-25, pak Yahoo), nehodnotí se |
| Rozhodnutí 1 / 2 / 3 | konce 12., 24., 36. měsíce od `T0` (poslední obchodní den před výročím) |

Průběžné popisné reporty se smějí generovat, ale nic nezastavují, neprodlužují ani nemění.

## 7. Kritéria a rozhodnutí

**Statistiky** (denní mřížka od `T0`):
- **F1:** Sharpe = průměr denního přebytku nad T-billem / sd × √252.
- **F2:** průměrný čistý výnos uzavřeného obchodu (obchody uzavřené do data rozhodnutí; otevřené se nepočítají);
  doplňkově Sharpe denních výnosů stejně jako F1.
- **p-hodnota:** stacionární bootstrap, 10 000 vzorků, seed 0, jednostranně `p = (1 + #{vzorky se statistikou ≤ 0}) /
  10 001`. F1: dny, `mean_block = 21`. F2: obchody se převzorkují **po dnech vstupu** (obchody téhož dne spolu),
  `mean_block = 21` přes seřazené dny vstupu.
- Náklady ×1 = 3 bps (primární), ×1,5 = 4,5 bps (laťka), ×3 jen hlášení.

**Minimum dat pro rozhodnutí:** F1 ≥ 240 obchodních dnů; F2 ≥ 50 / 100 / 150 uzavřených obchodů při 1. / 2. / 3. rozhodnutí
(historicky ~66 ročně). **[R]**

**Stavy hypotézy** (vyhodnocuje se jen při rozhodnutích a jen pokud je dosaženo minima; jinak „málo dat“ = pokračuje):
1. **Zamítnuta**, právě když **horní mez 90% intervalu statistiky při ×1 je ≤ 0** (bootstrap jako výše), **nebo** (jen při 3. rozhodnutí) bodový odhad při ×1 je ≤ 0 (F2: Sharpe ≤ 0 nebo průměrný obchod ≤ 0).
   Zamítnutá se uzavírá. **[R]** Bodové pravidlo „odhad ≤ 0“ už po 12 měsících by při pravdivém edge zamítlo
   F1 s pravděpodobností ~25 % (~36 % při haircutu), proto se bodově zamítá až po 36 měsících.
2. **Potvrzena**, právě když **všechno** platí:
   a. **Laťka z PLAN §7.6:** při ×1,5 je Sharpe > 0 (F1) / Sharpe > 0 a průměrný obchod > 0 (F2);
   b. při ×1 je statistika > 0;
   c. **Holmova korekce** přes zbývající otevřené hypotézy při `α = 0,05 / 3 = 0,0167` na jedno rozhodnutí (menší z
      dvou p < 0,0083, větší < 0,0167; zbyla-li jedna, p < 0,0167). Tři rozhodnutí × Holm dává celkovou chybu
      prvního druhu ≤ 5 % (Bonferroni přes rozhodnutí).
3. **Konzistentní, nepotvrzeno**, právě když platí a) a b), ale ne c). Pokračuje k dalšímu rozhodnutí.
4. **Neprůkazné** v ostatních případech (např. kladné při ×1, nekladné při ×1,5). Pokračuje.

Po 3. rozhodnutí se z „konzistentní“ a „neprůkazné“ stává **nepotvrzeno** bez dalšího prodlužování (další pokus =
nová pre-registrace).

Poznámka: samotná laťka a)+b) není test; i bez jakéhokoli edge ji splní zhruba třetina až polovina běhů.
Nese proto jen „nezamítnuto“, ne důkaz. Důkazem je jen stav 2.

## 8. Síla testu (hrubý odhad, normální aproximace, předem)

Pravděpodobnost stavu „potvrzena“ při pravdivém edge. F1: SE Sharpe ≈ √((1 + 0,5·SR²)/roky). F2: sd obchodu ≈ 2,3 %
(z CI výzkumu 19), ~66 obchodů ročně. Práh p < 0,0167 (Holm, větší z dvou) / 0,0083 (menší).

| | pravdivý edge | 12 měs. | 24 měs. | 36 měs. |
|---|---|---|---|---|
| F1 | SR 0,77 (historie) | 4–7 % | 8–12 % | 11–17 % |
| F1 | SR 0,38 (haircut) | 2–4 % | 3–5 % | 4–7 % |
| F2 | +0,265 % na obchod | 7–12 % | 14–21 % | 22–30 % |
| F2 | +0,13 % na obchod | 3–5 % | 4–7 % | 5–9 % |

Důsledek: i bez chyby je statistické potvrzení v praxi nedosažitelné pro F1 a málo pravděpodobné pro F2. Hlavní užitek:
(i) rychle odhalit, že edge **není** (stav 1 u skutečně záporného proudu), (ii) ověřit, že data a implementace
dávají historii podobné výnosy (§9), (iii) nasbírat čistá data pro další pre-registraci.
**Otázka pro vlastníka:** je to pro vás dostatečný účel, nebo chcete laxnější práh (α = 0,05 jednostranně bez Bonferroniho
přes rozhodnutí, síla F2 po 36 měs. ~49 %, F1 ~32 %, za cenu až 15 % falešných potvrzení)?

## 9. Jen hlášení (nerozhoduje)

- Náklady ×3; F2 s plněním na open (r1); SPY-only varianta H2 (H2-SPY).
- Skutečný vs historický profil: úspěšnost F2 (hist. 69,5 %), čas v trhu F2 (24 %), investovanost F1 (45 %),
  korelace F1 se SPY (0,31), korelace F1–F2.
- Přesnost dat: průběžný rozdíl Yahoo vs Nasdaq close; počet dní doplněných z Nasdaqu; dny chybějících dat.
- Srovnání s 60/40 (SPY/IEF měsíčně, 3 bps) a SPY: Sharpe, výnos, propad. Kritérium to **není** (jako ve výzkumu 19).
- H4 (kombinace s kryptem), H1a a overlaye H3 se neprovádějí.

## 10. Co znamená výsledek

- **Potvrzena:** paper edge potvrzený na neviděných datech při 3 bps a 4,5 bps. Teprve pak smysl měřit reálný skluz
  a MOC proveditelnost a uvážit malý kapitál (nový dokument). Žádná živá alokace před tím.
- **Konzistentní, nepotvrzeno / nepotvrzeno:** nic se nemění, žádný kapitál; další pokračování nová pre-registrace.
- **Zamítnuta:** hypotéza se uzavírá a do README se zapíše, že forward neprošel.
- F1 a F2 se rozhodují **zvlášť**, ne jako kombinace.

## 11. Implementace a postup

1. Schválení, brána kvality dat (§3.3) splněna, verze 1.0, **commit před `T0`** (a před prvním snapshotem, ze
   kterého by se cokoli po `T0` počítalo).
2. `qlab.research20`: `data` (stažení, snapshoty, TR dle §3.2, doplnění z Nasdaqu, napojení na `panel_r19_etf`),
   `forward` (stavba panelu, běh F1 a F2 přes `research19`), `evaluate` (bootstrap, Holm, stavy). Skripty
   `scripts/r20_download.py` (cron) a `scripts/r20_forward.py freeze | status | final` podle vzoru `r18_forward.py`.
3. **Testy:** TR z dividendy na ručním příkladu (ex-date, split); napojení nemění úroveň TR; **regrese:** forward
   pipeline na Sharadar-only panelu zkráceném k libovolnému dni reprodukuje výnosy výzkumu 19 do toho dne (bitově
   nebo do 1e-12); bod v čase pro celou pipeline; bootstrap po dnech vstupu; Holm a stavy na ručních příkladech;
   `final` odmítne běžet mimo termíny §6; chybějící den nevynuluje signál.
4. `freeze`: otisk souborů (`research20/*.py`, `research19/*.py`, `engine/vector.py`, `research5/sim.py`, tento soubor,
   skripty) do `docs/research20/forward_lock.json`; registr `runs/registry/research20.sqlite`: F1 a F2 jako N = 2
   před prvním výpočtem (kumulativně na ETF datech 34 + 2).
5. `status` popisně průběžně (`results/forward_<datum>.json`), `final` jen při termínech.

## 12. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-11-02 | Vlastník schválil všechny návrhy autora (zamítnutí přes horní mez CI, Holm + Bonferroni přes 3 rozhodnutí, rozhodnutí po 12/24/36 měsících, prahy brány kvality dat a minima dat, TR z close + dividend, Yahoo + Nasdaq jako zdroj, účel testu jako screen na selhání). |
| 2026-11-02 | Verze 0.1, návrh. Počítalo se jen pro §8 (síla testu) z čísel výzkumu 19; na datech po 2026-09-25 se nic nepočítalo. |
| 2026-11-02 | Brána kvality dat: první běh (denní maximum) neprošel u XLE (60,7 bps; Yahoo má ex-dividendu 2026-09-21, Sharadar 2026-09-18, součet obou dnů se shoduje; 103 ze 106 dividendových dnů ≥ 0,2 % sedí na stejný den). Vlastník schválil úpravu statistiky maxima před commitem 1.0 (prahy beze změny). Pozn.: prosté maximum z klouzavých dvoudenních výnosů by časovou chybu nevyrušilo (okno obsahující jen jeden z obou dnů ji vidí), proto se používá `m_t` výše. Druhý běh: 26 z 26 prošlo, maximum `m_t` ≤ 0,23 bps. Výsledek v `results/data_gate.json`. |
| 2026-11-02 | Verze 1.0 commitnuta. Denní stahování surových snapshotů běží v cronu (`scripts/r20_download.sh`, 23:15 UTC). **`T0` = 2026-11-03** (první US obchodní den po commitu). Snapshot z 2026-11-02 je rozběh, nehodnotí se. Od 2026-09-26 do `T0` se nic nepočítalo kromě brány kvality dat. |
