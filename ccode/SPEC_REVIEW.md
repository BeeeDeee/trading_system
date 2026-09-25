# Revize PROJECT_SPECIFICATION.md

Datum: 2026-09-25. Předmět: kritická analýza stávající specifikace před přepsáním na v2.

## 0. Shrnutí

Specifikace dobře zachycuje obecné zásady (point-in-time, next-open execution, žádné náhodné dělení,
náklady v primárních výsledcích). Je ale psaná **obecně a opatrnicky** – většinu konkrétních rozhodnutí
odkládá do „Fáze 0" a právě na těch odložených místech leží skoro všechna reálná rizika projektu.

Největší problémy, seřazené podle dopadu:

1. **Vrstvená selekce bez kvantifikace overfittingu.** Generátor → filtr → ranking → dedup → výběr páru
   → ML nad párem je 5 vrstev výběru. Spec říká „zaznamenat počet pokusů", ale nikde tento počet
   nepoužije (Deflated Sharpe, PBO, SPA test chybí). Výsledek bude téměř jistě přeoptimalizovaný.
2. **ML meta-model nemá dost dat a trénuje na zkreslených datech.** Při `H = 20` je ve vývojové oblasti
   ~290 nepřekrývajících se labelů, v prvním WFO foldu ~60. Navíc A a B byly vybrány *proto, že* v minulosti
   vyšly dobře – model se učí na jejich in-sample glorifikované historii.
3. **Nákladový model (flat bps) + 10 000 USD + zlomkové akcie systematicky zvýhodní mikrocapy.**
   Mean reversion a breakout na nelikvidních titulech budou vypadat skvěle a jsou neobchodovatelné.
   Chybí minimální poplatek za příkaz, který je při 10k a desítkách pozic dominantní.
4. **Sharadar-specifické pasti nejsou zmíněny** (split-adjusted ceny v cenových filtrech = look-ahead,
   snapshot metadata v TICKERS = survivorship bias, chybějící delisting returns, recyklované tickery).
5. **WFO není definované pro pravidlové strategie.** Není jasné, co je „trénink" strategie s pevnými
   parametry a kde přesně leží hranice mezi globální a per-fold selekcí. To je nejčastější zdroj
   nechtěného úniku.
6. **Benchmark je nekonzistentní a degeneruje.** Neřeší, kam jdou peníze z delistingů, neobsahuje nové
   tituly, resetuje se po foldech a je extrémně citlivý na chybné printy mikrocapů.
7. **Finální OOS je statisticky slabý nástroj.** 5 let OOS má směrodatnou chybu Sharpe ≈ 0,5 – nerozliší
   strategii se Sharpe 0,5 od nuly. Spec na něm staví „nezávislé potvrzení".
8. **Scope.** Šest rodin + WFO + páry + ML + reporty je na první verzi hodně, bez jediného kill-kritéria.
   Není definováno, kdy projekt uznat za neúspěšný – a to je nejpravděpodobnější výsledek.

Upřímně k bodu 8: dlouhodobě publikované anomálie (momentum, low-vol, reversal) ztratily po publikaci
zhruba třetinu až polovinu výnosu (McLean & Pontiff 2016) a long-only verze nesou převážně tržní betu.
Nejpravděpodobnější poctivý výsledek je „po nákladech a daních nic nepřekoná SPY/EW univerzum přesvědčivě".
Framework má být postaven tak, aby k tomuto závěru – nebo k jeho vyvrácení – došel **rychle, levně a
věrohodně**. To je nejdůležitější designový cíl, který ve spec chybí.

---

## 1. Kritické nedostatky (K)

### K1 – Multiple testing je jen zaznamenán, ne korigován (§9.3, §10)

Spec vyžaduje zapsat počet kandidátů, ale nedefinuje, co se s ním stane. Bez korekce se ranking podle
Sharpe s tisíci kandidáty rovná výběru maxima šumu. Očekávané maximum Sharpe z N nezávislých
nulových strategií roste zhruba jako √(2 ln N) · σ(SR) – při 1000 kandidátech a 10 letech dat je to
kolem SR ≈ 1,2 **čistě náhodou**.

Chybí:
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) s efektivním počtem pokusů (po clusteringu
  korelovaných kandidátů, ne raw N).
- **PBO přes CSCV** (Probability of Backtest Overfitting) nad maticí výnosů kandidátů.
- **Test proti benchmarku s korekcí na data snooping** (Hansen SPA / White Reality Check, stationary
  block bootstrap).
- Počítání **všech** pokusů včetně těch, které dělá člověk (upravit rozsah parametrů po pohledu na
  výsledky = nový pokus). Spec to zmiňuje, ale nemá mechanismus – potřebujeme append-only registr pokusů.

### K2 – ML meta-model: málo dat + selection-induced bias (§12)

Čísla pro vývojovou oblast ~1998–2021 (≈ 23 let):

| Horizont H | Nepřekrývající se labely celkem | V 1. foldu (5 let tréninku) |
|---|---|---|
| 5 dní | ~1 160 | ~250 |
| 20 dní | ~290 | ~63 |

Denní rozhodování s překrývajícími se labely počet efektivních pozorování nezvyšuje. Klasifikace do tří
tříd (A/B/cash) nad pár stovkami silně autokorelovaných vzorků s nízkým poměrem signál/šum je v praxi
neodlišitelná od šumu.

Horší je **selection bias v trénovacích datech**: v čase refitu `T` jsou A a B vybrány, protože na
`[.., T]` vyšly nejlépe. Meta-model se trénuje na téže historii → vidí nadhodnocené výnosy obou
strategií a učí se vzory, které jsou artefaktem výběru. Spec tohle vůbec neřeší.

A věcně: A a B jsou obě long-only US akciové strategie, typicky s korelací 0,7–0,9. Volba „A vs B"
nese málo informace; skoro všechna hodnota (i riziko) je ve volbě „cash", tj. v **market timingu** –
disciplíně s mimořádně špatnou empirickou historií.

Návrh: meta-model degradovat na volitelnou fázi s kill-kritériem; trénovat ho (pokud vůbec) na
**všech kandidátech / rodinách** (model „režim → očekávaný výnos rodiny"), ne jen na vítězích; povinně
porovnat s jednoduchým pravidlem (např. trend filtr trhu 200 dní), které ML musí porazit po nákladech.

### K3 – Nákladový model zvýhodňuje neobchodovatelné tituly (§7.3)

Flat bps na obrat + zlomkové akcie + malý kapitál = backtest nevidí:
- spread mikrocapů (často 100–500 bps, flat model třeba 5–10 bps),
- kapacitu a market impact,
- **minimální poplatek za příkaz** (např. IBKR Pro 0,35 USD/příkaz → při obchodu 200 USD je to 17,5 bps
  jen na poplatku; broker s poplatkem ~1–2 EUR/příkaz → 50–100 bps),
- FX konverzi CZK/EUR → USD, pokud je relevantní.

Short-term reversal a mean reversion v mikrocapech jsou z velké části bid-ask bounce. Bez
likviditně závislých nákladů framework přesně tyto artefakty vybere jako „nejlepší strategie".

Návrh: od první verze
- per-order fixní/min poplatek + bps + **odhad spreadu z OHLC** (Corwin–Schultz nebo Abdi–Ranaldo)
  škálovaný podle dollar volume,
- tvrdý likviditní filtr univerza (min cena z **neupravené** ceny, min medián dollar volume, případně
  min market cap),
- no-trade band (neobchodovat drobné rebalance pod X % váhy nebo pod Y USD),
- citlivostní běh s 2× a 3× náklady jako povinná součást reportu.

### K4 – Sharadar pasti, které spec nezná (§5)

Spec se záměrně vyhýbá předpokladům o Sharadaru. To je rozumné pro audit, ale nebezpečné pro design –
tyto body se musí promítnout do architektury, ne jen „ověřit". Níže uvedené je známé chování Sharadar
SEP/SFP/TICKERS/ACTIONS; audit je má **potvrdit**, ne objevovat:

| Past | Důsledek | Opatření |
|---|---|---|
| `open/high/low/close` v SEP jsou zpětně **split-adjusted**, `closeadj` split+dividend, `closeunadj` neupravená | Filtr „cena > 5 USD" nad upravenou cenou je look-ahead (titul s pozdějším reverse splitem má historicky uměle vysokou cenu) | Cenové filtry jen z `closeunadj`; výnosy z TR faktoru `closeadj/close` aplikovaného i na open |
| Míchání `open` (bez dividend) a `closeadj` (s dividendami) | Falešné skoky výnosu v ex-div dny | Jednotná konstrukce adjusted OHLC z jednoho faktoru |
| TICKERS (`exchange`, `category`, `sector`, `isdelisted`, `scalemarketcap`) je **snapshot k dnešku** | Filtr typu „burza ∈ {NYSE, NASDAQ}" nebo cokoliv s `isdelisted` tiše vyhodí delistované tituly → survivorship bias zpět | Žádné snapshot pole jako filtr univerza bez posouzení; `isdelisted`/`lastpricedate` nikdy jako feature |
| Tickery se recyklují, starý ticker dostane číselný suffix | Join podle tickeru spojí dvě různé firmy | Primární klíč `permaticker` |
| Sharadar **nemá delisting return** (na rozdíl od CRSP DLRET) | Bankroty „končí" na posledním printu, např. 0,40 USD, místo ~0 → výnosy strategií kupujících propady (mean reversion) nadhodnocené | Explicitní politika terminálního výnosu podle typu delistingu z ACTIONS (akvizice ≈ poslední cena / známá hodnota; výkonnostní delisting bez údaje: konzervativně −30 % až −100 %, viz Shumway 1997) + citlivostní analýza |
| Spinoffy – nemusí být v `closeadj` | Falešná ztráta rodičovské firmy v den spinoffu | Audit: extrémní jednodenní výnosy vs. ACTIONS `spinoff` |
| Open = 0 / chybí u nelikvidních dnů | Fill za nulovou cenu | Stav „neobchodovatelné", příkaz se neprovede |
| SF1: dimenze `MR*` jsou restatované (look-ahead), `AR*` as-reported, dostupnost = `datekey` | Použití MRQ = look-ahead | Pouze `AR*` + `datekey` + zpoždění (pokud se fundamenty kdy použijí) |
| `lastupdated` – Sharadar zpětně opravuje data | Neopakovatelné běhy | Neměnný raw snapshot s hashem; nikdy nepřepisovat |
| ETF/ETN/CEF/preferred/warranty/ADR ve stejných tabulkách | Pákové ETF ovládnou momentum ranking; warranty a unity rozbijí statistiky | Explicitní allow-list kategorií (viz rozhodnutí níže) |

### K5 – WFO pro pravidlové strategie není definováno (§9.1)

Pravidlová strategie s pevnými parametry se „netrénuje". Co se v každém foldu reálně děje, je **výběr**
(které konfigurace, rodiny, pár, prahy). Spec to tuší (§4 „výběr jako součást modelu"), ale architektura
(§14) má generování, ranking a dedup jako lineární globální kroky – přesně místo, kde vznikne únik.

Návrh (klíčová architektonická změna): celý výběrový proces je **čistá funkce**
`select(data[..T], registr kandidátů) → politika`. WFO ji volá v každém refit datu `T` a vyhodnotí
politiku na `(T, T_next]`. Poctivý odhad výkonu celé metodiky = poskládané OOS segmenty.
Nic, co ovlivňuje výběr, nesmí být spočteno nad celou vývojovou oblastí najednou.

Důležitý praktický důsledek: výnosová řada kandidáta s pevnými parametry **nezávisí na foldu**. Stačí
ji spočítat jednou přes celou historii a uložit matici `R[dny × kandidáti]` čistých denních výnosů.
Veškerá WFO selekce, dedup, páry, CSCV/PBO pak pracují jen s řezy této matice – řádově rychlejší a
zároveň to vynucuje správnou strukturu. (Omezení: výnosy musí být scale-invariant, tj. náklady v bps;
fixní poplatky za příkaz se řeší v druhém, ledgerovém enginu – viz Z1.)

### K6 – Benchmark (§3, §13)

„Buy-and-hold rovnoměrně váženého PIT univerza k začátku období bez rebalancování":
- neřeší, kam jdou peníze z delistingu (cash s 0 % úrokem → benchmark se postupně „ředí"),
- nezahrnuje nové IPO → po 10+ letech je to reprezentant roku X, ne trhu,
- 3–5 tisíc titulů equal-weight včetně mikrocapů = maximální citlivost na chybné printy a na politiku
  delisting returns,
- při WFO se resetuje na začátku každého foldu → nekonzistentní s poskládanou equity křivkou,
- „stejné univerzum" – čí? Každá strategie má vlastní eligibility filtr.

Návrh: sada benchmarků s různým účelem:
1. **SPY total return** – „vyplatí se to vůbec?" Investovatelná alternativa. Zároveň **sanity check
   datové vrstvy**: rekonstruovaný cap-weighted top-500 ze Sharadaru se musí vejít do tolerance vůči SPY.
2. **EW univerzum strategie, měsíčně rebalancované** – „má výběr titulů nějakou hodnotu?" (= výnos
   náhodného výběru ze stejného univerza).
3. Tvůj **buy-and-hold EW** (s opravami: proceeds z delistingu se reinvestují proporčně) – ponechat jako
   požadované srovnání, ale ne jako jediné primární.

### K7 – Finální OOS a jeho síla (§9.1)

Směrodatná chyba ročního Sharpe ≈ √((1 + SR²/2)/T):

| Délka OOS | SR 0,5 | SR 1,0 |
|---|---|---|
| 3 roky | ±0,61 | ±0,71 |
| 5 let | ±0,47 | ±0,55 |
| 10 let | ±0,34 | ±0,39 |

Pětiletý OOS neodliší rozumnou strategii od nuly. Navíc pravidlo „po prohlédnutí spotřebován" je jen
organizační – v praxi se na OOS podívá, upraví a pustí znovu.

Návrh:
- **OOS vault v kódu**: datová vrstva odmítne vrátit data za hranicí vývoje, pokud běh nemá příznak
  `final_evaluation` a zmrazený hash metodiky. Každé otevření se zapíše do append-only logu.
- **Forward OOS**: skutečně čistý test je budoucnost. Stejná code path, která dělá backtest, má umět
  po close vygenerovat zítřejší příkazy (paper trading). Po zmrazení metodiky běží N měsíců dopředu.
- Rozhodovat podle **intervalů spolehlivosti** (block bootstrap), ne bodových odhadů.

---

## 2. Závažné nedostatky (Z)

**Z1 – Chybí volba simulačního paradigmatu.** Pro tisíce kandidátů je ledger engine (akcie, cash,
objednávky) pomalý; pro reporty a 10k-specifické jevy (min. poplatky, zlomky, zaokrouhlení) je
nutný. Návrh: dva enginy – *vektorový* (váhy × výnosy, bps náklady) pro screening a *ledger* pro finalisty
a reporty – a **paritní test**, že se na shodném nastavení shodují v toleranci. To je zároveň nejsilnější
test správnosti obou.

**Z2 – Detaily exekuce nejsou definované.** Váhy se počítají z close `t`, fill je na open `t+1` za jinou
cenu → co když na nákup nestačí cash? Pořadí (nejdřív prodeje)? Chybějící open? Titul neobchodovaný
v `t+1`? Delisting mezi signálem a fillem? Každý z těchto případů musí mít explicitní pravidlo.

**Z3 – Dividendy.** Ledger s dividendou v ex-date vs. automatické reinvestice přes TR ceny – spec nevybírá.
Pro českého rezidenta je navíc dividenda US titulu zdaněná srážkou 15 % (W-8BEN); backtest s hrubou
dividendou systematicky nadhodnocuje dividendové strategie (low-vol!).

**Z4 – Daně.** Pokud je cílem reálné nasazení z ČR: zisk z prodeje cenných papírů držených > 3 roky je
osvobozen (časový test), kratší držení se daní. Aktivní strategie platí daň, buy-and-hold po 3 letech ne.
**Po zdanění se závěr porovnání se SPY může otočit.** Spec daně vůbec nezmiňuje.

**Z5 – Úrok z hotovosti 0 %.** 1998–2026 byl průměrný výnos T-bills ~2 %, v letech 2000, 2006–07,
2023–25 kolem 4–5 %. Nulový úrok znevýhodňuje cash volbu, regime strategie i ML „cash". Sharpe má být
počítán z výnosu nad bezrizikovou sazbou. Řada DTB3 z FRED je zdarma – doporučuji ji mít od začátku.

**Z6 – Chybí atribuce rizika.** Long-only strategie s vysokým Sharpe bývá jen beta × small-cap tilt.
Povinné metriky: beta k trhu, alfa a information ratio vůči benchmarkům; volitelně regrese na
Fama–French faktory (Ken French data library, zdarma).

**Z7 – Výběr „páru" je libovolný a kombinatoricky drahý.** Proč právě 2? Po dedupu 500 kandidátů =
124 750 párů = další masivní multiple testing. Portfolio K dekorelovaných strategií s jednoduchým
vážením (EW, inverse-vol) je robustnější a pár je jen speciální případ K = 2. Navíc chybí pravidlo
rebalance 50/50 (denně? měsíčně? drift?) a netování shodných titulů mezi A a B (spec to řeší jen u ML).

**Z8 – Generátor náhodně vzorkuje parametry.** Pro analýzu citlivosti (plateau vs. osamělý peak) jsou
potřeba strukturované mřížky nebo Sobolovy sekvence. Robustní skóre = výkon **okolí** v parametrickém
prostoru, ne bodu.

**Z9 – Rodiny nemají společný rámec.** Každá rodina tak dostane vlastní kód pro výběr titulů, vážení
a rebalance. Návrh: jedna kompozice
`strategie = univerzum ∘ skóre ∘ výběr (top-N / práh) ∘ vážení ∘ rebalance/holding ∘ overlay (expozice)`.
Rodina = preset skórovací funkce. Trend vs. breakout se pak liší prokazatelně (jiné skóre), kód je
vektorizovatelný a nové rodiny jsou levné.

**Z10 – `decision_interval_days = 1` jako výchozí.** S labelem H = 20 a strategiemi rebalancovanými
týdně/měsíčně je denní rozhodování jen generátor obratu a šumu. Výchozí hodnota má odpovídat frekvenci
rebalance podkladových strategií (např. 5 nebo 21 dní); 1 den jen jako laditelná varianta.

**Z11 – Chybí testovací strategie proti look-ahead.** Návrh povinného testu: pro každý feature/signál
spočítat výsledek nad daty useknutými v `t` a nad kompletními daty – musí být identické („future
perturbation test"). Plus syntetická data se známou odpovědí (golden tests) pro engine.

**Z12 – Chybí kill-kritéria mezi fázemi.** Každá fáze má mít bránu: např. do fáze párů/ansámblu jen pokud
aspoň jedna rodina projde DSR > 0,95 a PBO < 0,3; do ML jen pokud ansámbl překoná benchmark v poskládaném
WFO OOS. Jinak stop a report „negativní výsledek" – což je legitimní výstup.

**Z13 – Výpočetní zdroje.** Tento stroj má 3 jádra a 3 GB RAM. Sharadar SEP je řádově desítky milionů
řádků; v pandas se do paměti nevejde. Nutné: Parquet + Polars/DuckDB (lazy), float32, práce jen nad
eligible podmnožinou. Počet kandidátů × foldů je potřeba dimenzovat podle HW.

---

## 3. Menší body (M)

- **M1** „US akcie a ETF" – ETF v univerzu strategií vs. jen jako benchmark/režimové vstupy není
  rozhodnuto. Doporučení: ETF mimo univerzum výběru, SPY/sektorová ETF jen jako vstupy a benchmark.
- **M2** Obchodní kalendář: odvodit z NYSE kalendáře (`exchange_calendars`), ne z „dnů, kdy jsou data".
- **M3** Metriky „podstatných tržních období" – předem definovat seznam (2000–02, 2007–09, 2020,
  2022), jinak se období vybírají dodatečně.
- **M4** Ranking skóre s konfigurovatelnými vahami = další stupeň volnosti k overfittingu. Méně metrik,
  pevná forma.
- **M5** Git: repozitář má kořen v `/home/kapo` a obsahuje smazaný předchozí projekt. Nový projekt by
  měl mít vlastní repo v `ccode/`, data mimo git.
- **M6** Reprodukovatelnost: lock závislostí (`uv.lock`), deterministický seed i při paralelizaci
  (seed per kandidát odvozený z hashe konfigurace).
- **M7** §7.1 bod 5 „delší forward label začíná až po okamžiku provedení" – správně, ale label musí
  začínat na **open t+1**, ne close t+1; jinak se ztrácí/přidává overnight gap.
- **M8** Spec neříká, co je finální „produkt". Návrh: zmrazená politika, která umí každý večer vydat
  seznam příkazů na zítřejší open.

---

## 4. Co ze stávající spec zachovat

- Point-in-time princip, signál po close / fill na open `t+1`, žádné intradenní SL/TP ve v1.
- Long-only, zlomkové akcie, 10 000 USD (pro ledger engine a reporty).
- Náklady v primárních metrikách, žádné náhodné dělení, purging/embargo.
- Datový audit jako první krok a manifest každého běhu.
- „Neověřená schopnost zdroje se nesmí tiše předpokládat."

## 5. Rozhodnutí, která potřebuji od tebe

Viz otázky v konverzaci; podle odpovědí vznikne `PROJECT_SPECIFICATION_v2.md`.
