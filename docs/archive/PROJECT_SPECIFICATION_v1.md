# PROJECT_SPECIFICATION.md

## 1. Účel dokumentu

Tento dokument popisuje požadavky a architekturu lokálního výzkumného frameworku pro tvorbu, backtestování, porovnávání a výběr long-only strategií na denních datech amerických akcií. Slouží jako výchozí specifikace pro následnou detailní dokumentaci, implementaci a ověřování.

Dokument záměrně rozlišuje potvrzená rozhodnutí, navržená výchozí nastavení a body, které je nutné uzavřít po prohlídce lokálních dat Sharadar. Žádná strategie ani model se nepovažuje za ziskový jen proto, že dobře vyšel na historických datech.

## 2. Cíl projektu

Framework má umožnit:

1. Načíst a ověřit lokální denní data US akcií, která uživatel získal od Sharadar a která podle uživatele obsahují také delistované společnosti.
2. Vytvářet konfigurovatelný počet reprodukovatelných variant strategií z omezené knihovny typů a parametrů.
3. Simulovat obchody po započtení provizí, slippage a dalších konfigurovatelných nákladů.
4. Vyhodnocovat strategie chronologicky pomocí walk-forward optimalizace (WFO) a chráněného finálního out-of-sample (OOS) období.
5. Řadit strategie, odmítat předem definované slabé varianty a odstraňovat prakticky duplicitní varianty.
6. Hledat dvojice strategií, jejichž kombinované portfolio má stabilnější chování než jednotlivé strategie.
7. Ověřit meta-model, který z informací dostupných v čase rozhodnutí odhaduje, která z vybraných strategií bude mít vyšší budoucí rizikově očištěný výsledek, a smí zvolit také hotovost.
8. Vytvářet reprodukovatelné reporty a grafy equity strategie nebo kombinace proti buy-and-hold portfoliu ze stejného akciového univerza.

## 3. Rozsah a potvrzená rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Trh | Americké akcie a ETF |
| Data | Denní data Sharadar přibližně od roku 1998; přesný obsah lokálních souborů se ověří před implementací datové vrstvy. |
| Jazyk a provoz | Python, lokální nástroj pro jednoho uživatele. |
| Počáteční kapitál | 10 000 USD pro každou samostatnou simulaci strategie nebo portfolia. |
| Pozice | Long-only; zlomkové akcie povoleny. Prodej vlastní pozice je výstup ze long pozice, nikoli short. |
| Short selling | Mimo první verzi. Architektura může připustit budoucí rozšíření, ale nepředstírá short simulaci bez údajů o půjčitelnosti, borrow fee a marži. |
| Signál a provedení | Signál po close dne `t`; nejčasnější provedení na open následující obchodní seance. |
| Intradenní příkazy | Take-profit a stop-loss v průběhu dne nejsou součástí první verze. Denní OHLC data nemusejí určit pořadí zasažení úrovní. |
| Základní dvojice | Výsledek dvojice se nejprve porovná při pevných vahách 50/50. |
| ML akce | Vybrat strategii A, strategii B nebo hotovost. |
| Frekvence rozhodování ML | Konfigurovatelný počet obchodních dní; doporučený výchozí interval je 1 den. Častější vyhodnocení samo o sobě neznamená povinnost přepnout. |
| Benchmark | Long-only buy-and-hold stejného historického univerza dostupného na začátku vyhodnocovaného období; rovnoměrné počáteční váhy, bez běžného rebalancování. |

## 4. Zásady návrhu

- **Časová korektnost:** každý signál, feature, filtr univerza a výběr modelu používá pouze informace dostupné do příslušného času rozhodnutí.
- **Výběr jako součást modelu:** generování variant, jejich filtrace, výběr dvojice i ladění ML patří do validačního procesu. Nelze je provést na finálním OOS období.
- **Reprodukovatelnost:** stejná data, konfigurace, verze kódu a seed musí umožnit zopakovat běh.
- **Náklady vždy v hlavním výsledku:** metriky po nákladech jsou primární; hrubé výsledky lze uvést jen jako doplňkovou diagnostiku.
- **Jednoduché modely jako první:** nejprve se porovnají jednoduché strategie a pravidlové baseline. ML se přidá až po ověření strategie a validačního procesu.
- **Žádné náhodné dělení časové řady:** panelové řádky různých titulů ze stejného období nesmí skončit současně v tréninku a testu jen kvůli náhodnému dělení.
- **Nezaměňovat počet řádků za nezávislá pozorování:** výnosy akcií ve stejný den sdílejí tržní riziko.

## 5. Datová vrstva a první kontrola Sharadar

### 5.1 Povinná průzkumná fáze

Před stavbou strategií musí implementace zmapovat lokální uložení dat a vytvořit datový inventář. Inventář má zachytit názvy souborů/tabulek, formáty, sloupce, rozsah dat, počet záznamů, chybějící hodnoty a vazby mezi cenami, instrumenty a corporate actions. Nesmí předpokládat konkrétní adresářovou strukturu ani konkrétní Sharadar produkt.

Ověřit zejména:

- zda data obsahují stabilní identifikátor společnosti/instrumentu nezávislý na tickeru;
- pokrytí aktivních i delistovaných titulů a dostupnost data delistingu;
- zda jsou dostupné surové a/nebo upravené OHLC ceny a jakou metodou jsou upravené;
- dividendy, splity a další corporate actions;
- historické listing/delisting intervaly a případné symbol changes;
- volume a případná metadata pro likviditní filtry;
- dostupnost fundamentálních údajů a jejich publication/filing date, pokud existují;
- časovou zónu, datum obchodní seance, duplicity, chybějící dny a neobvyklé cenové skoky.

Výstupem je verzovaný datový audit a rozhodnutí, která pole lze bezpečně použít. Pokud nějaká schopnost zdroje není doložena, systém ji označí jako neověřenou; nesmí ji tiše předpokládat.

### 5.2 Universe a survivorship bias

Univerzum musí být definováno jako point-in-time množina titulů. Způsobilost instrumentu v datu `t` vychází jen z údajů, které lze k `t` doložit. Pozdější přežití firmy na trhu nesmí zpětně rozhodovat, zda titul do historického univerza patřil.

Pravidla pro cenu, minimální objem/likviditu, historii po IPO a ostatní eligibility filtry budou konfigurovatelná a zaznamenaná v každém běhu. Delistované tituly zůstávají v datech až do skutečného ukončení obchodování a jejich konečný výsledek se nesmí zahodit. Zacházení s výplatou při delistingu se řídí dostupnými zdrojovými údaji a musí být auditovatelné.

Pokud budou použita fundamentální data, smějí vstoupit do feature až po historickém datu zveřejnění a po konzervativním konfigurovatelném zpoždění. Samotné fiskální období není datem dostupnosti.

### 5.3 Normalizace a audit

Normalizovaná datová vrstva poskytne alespoň denní OHLCV, identifikátor instrumentu, datum seance, stav obchodovatelnosti a potřebné corporate-action výnosy. Každý běh uloží manifest obsahující zdroj/otisk verze dat, čas auditu, pravidlo univerza, úpravy cen, rozsah, konfiguraci a počty vyřazených či chybějících řádků.

Chybějící hodnoty se nesmějí doplňovat budoucími hodnotami. IPO warm-up, suspendované obchodování, symbol change a chybějící cena mají explicitní stav. Výnos strategie musí správně zahrnovat split a dividendový výnos, pokud to zdrojová data umožňují.

## 6. Strategie a jejich generování

### 6.1 Knihovna v první verzi

Každá rodina je šablona se schématem parametrů, povolenými rozsahy, vstupními poli a pravidly portfolia. Generátor vybírá z těchto rozsahů; neskládá libovolný nový kód. Počet vygenerovaných kandidátů i seed jsou konfigurovatelné. Každý kandidát má stabilní identifikátor a uloženou kompletní konfiguraci.

| ID rodiny | Rodina | Výchozí popis |
|---|---|---|
| `time_series_trend` | Trend/momentum jedné akcie | Vstup a výstup podle trendu daného instrumentu, vypočteného pouze z jeho historie. |
| `cross_sectional_momentum` | Relativní momentum | Seřadí způsobilé tituly podle jejich minulého výkonu a vybírá horní část pořadí. |
| `mean_reversion` | Krátkodobý návrat k průměru | Vybírá krátkodobé odchylky ceny s předem definovaným filtrem likvidity a rizika. |
| `breakout` | Průraz cenového rozsahu | Vstupuje po průrazu hranice určené výhradně minulými cenami. |
| `low_volatility` | Defenzivní/nízkovolatilní výběr | Relativně upřednostňuje tituly s nižší trailing volatilitou. |
| `market_regime_exposure` | Tržní režim a expozice | Portfolio-level pravidlo mění investovanost nebo přechází do hotovosti podle minulého stavu trhu. |

Rodiny trend following a breakout se nesmějí odlišovat jen názvem. Specifikace jejich signálu musí určit, jak se liší. Strategie s fundamentálními vstupy ani sezónní rodiny nejsou ve výchozí sadě; mohou být přidány až po kontrole point-in-time dat a samostatném posouzení rizika multiple testing.

### 6.2 Parametry a konstrukce portfolia

Strategie musí oddělit:

- parametry signálu (např. délky trailing oken a vstupní/výstupní prahy);
- universe/eligibility filtry;
- počet držených titulů;
- způsob vážení pozic;
- maximální váhu jednotlivého titulu a portfolio-level expozici;
- pravidla držení, výstupu a rebalance;
- nákladové a kapitálové předpoklady.

Konkrétní rozsahy parametrů se definují v konfiguraci a před zahájením finální validace se zmrazí. V jednom běhu lze spouštět konfigurovatelný počet kandidátů, ale systém musí zaznamenat všechny vyzkoušené konfigurace včetně neúspěšných.

## 7. Časování, portfolio a simulované provedení

### 7.1 Časová smlouva

Pro denní backtest platí:

1. Denní bar pro den `t` je kompletní až po jeho close.
2. Strategie/feature/model mohou pro rozhodnutí po close `t` použít data nejpozději do tohoto close.
3. Objednávka z tohoto rozhodnutí se provede nejdříve na open následující obchodní seance.
4. Cena close dne `t` se nesmí použít jako současně známá cena i garantovaná cena provedení.
5. Delší forward label začíná až po okamžiku, kdy dané rozhodnutí mohlo být provedeno.

### 7.2 Kapitál, hotovost a pozice

Každá samostatná simulace začíná s 10 000 USD. Zlomkové akcie jsou povolené. Portfolio vede hotovost i tržní hodnotu pozic; equity je jejich součet po započtení realizovaných nákladů a dostupných dividend. Long-only portfolio nesmí vytvořit zápornou hotovost ani záporný počet akcií, pokud takové chování nebude někdy výslovně přidáno jako jiný režim.

Úrok z hotovosti je v první verzi nulový, pokud uživatel později neposkytne odpovídající časovou řadu a nedefinuje pravidlo její dostupnosti. Strategie nesmí investovat více kapitálu, než dovolí její alokovaný kapitál a konfigurované limity.

### 7.3 Náklady a slippage

Náklady se uplatní při každém nákupu i prodeji a jsou konfigurovatelné. První implementace má podporovat jednoduchý transparentní model v bazických bodech obratu na každý směr, s oddělenou komisí a slippage. Rozšíření o poplatek za akcii či model závislý na likviditě/velikosti obchodu je možné později.

Konkrétní číselné defaulty se nastaví až při implementaci konfigurace; nesmějí být vydávány za skutečné sazby konkrétního brokera bez jeho údajů. Report ukazuje použité hodnoty. Primární metriky jsou po nákladech.

### 7.4 Intradenní stop-loss a take-profit

Stop-lossy ani take-profity, které by se měly spustit během dne, nejsou ve verzi 1.0 podporovány. Denní OHLC může ukázat průchod více úrovněmi, ale ne jejich pořadí. Jejich budoucí podpora vyžaduje buď jemnější data, nebo předem stanovenou konzervativní fill konvenci s analýzou citlivosti.

## 8. Backtestovací běh a výsledky

Backtestovací běh načte verzovaná data a konfiguraci, vytvoří point-in-time univerzum, vypočte trailing features, vytvoří cílové pozice, naplánuje proveditelné obchody, uplatní náklady a uloží denní equity, cash, pozice, objednávky a důvody rozhodnutí.

Výsledek musí být reprodukovatelný a musí nést alespoň:

- identitu/verzi strategie a úplnou konfiguraci;
- identitu běhu, čas vytvoření, verzi kódu, seed a data manifest;
- časové hranice train/validation/test a informaci o OOS statusu;
- denní equity/cash/positions, obchody a náklady;
- agregované metriky za celek i za jednotlivé foldy.

## 9. Validace bez úniku budoucnosti

### 9.1 Časové vrstvy

Historie se rozdělí chronologicky na vývojovou oblast a finální, nedotčené OOS období. Konkrétní kalendářní hranice se stanoví až po datovém auditu. Ve vývojové oblasti probíhá WFO: každá simulace trénuje, vybírá nebo ladí pouze na minulosti a měří se na následujícím časovém okně.

Výběr rodin, parametrů, filtrů, dvojice strategií a modelu musí být součástí walk-forward postupu. Finální OOS se nepoužívá k úpravám, rozhodování o rodinách ani k ručnímu výběru grafu. Po jeho prohlédnutí se považuje za spotřebované.

### 9.2 Purging, embargo a panelové vzorky

U ML labelů s horizontem `H` musí být tréninkové labely, jejichž budoucí výnos zasahuje do validačního/testovacího intervalu, odstraněny (purging). Mezi intervaly se použije konfigurovatelné embargo odpovídající překryvu labelů a dalším závislostem.

Rozdělení probíhá podle času, nikoli náhodně po řádcích. Pokud se využívají data více titulů, všechny vzorky se stejným rozhodovacím datem zůstávají ve stejném časovém segmentu. Výchozí meta-model pracuje s výsledky portfoliových strategií a tržním stavem; více akcií může pomáhat s tvorbou trhu/features a kandidátních portfolií, ale nesmí uměle nafukovat počet nezávislých pozorování.

### 9.3 Ochrana před multiple testing

Framework zaznamená počet vygenerovaných a otestovaných kandidátů, výběrová kritéria i všechny iterace, které vedly k finální volbě. Hodnocení zohlední stabilitu napříč foldy, citlivost parametrů a koncentraci výnosů do několika titulů či období. Jedno nejlepší Sharpe ratio nestačí k přijetí strategie.

## 10. Metriky, filtrace, ranking a deduplikace

Každá varianta dostane metriky minimálně:

- CAGR, kumulativní výnos a volatilitu;
- Sharpe, Sortino a Calmar nebo jejich přesně definované ekvivalenty;
- maximální drawdown a délku zotavení;
- obrat portfolia, počet obchodů a zaplacené náklady;
- průměrnou expozici, podíl času v hotovosti a koncentraci pozic;
- rozdělení metrik po jednotlivých WFO foldů a podstatných tržních obdobích.

Prahy tvrdého vyřazení a váhy ranking skóre budou konfigurovatelné, verzované a stanovené před finálním OOS vyhodnocením. Konkrétní prahy se doplní po rozhodnutí o délce historie, foldů a prioritě rizika.

Deduplikace bude zohledňovat překryv časových řad čistých výnosů a podle dostupnosti podobnost signálů či držených pozic. Použije jen společná data v relevantním validačním období. Korelace není jediným kritériem; podobné strategie se mohou lišit v drawdownech a chování podle režimů. Pravidla a hranice duplicity budou konfigurovatelná a jejich použití auditované.

## 11. Výběr dvojice doplňujících se strategií

Pár se nevybírá pouze podle nízké korelace. Kandidátní dvojice se simuluje jako portfolio po nákladech a porovnává se s oběma složkami samostatně. Základní varianta používá 50/50 kapitálové váhy; každá složka obchoduje pouze v rámci své alokace.

Hodnocení páru zahrnuje:

- výnos a rizikově očištěné metriky kombinace;
- maximální drawdown, délku zotavení a stabilitu mezi WFO foldy;
- překryv ztrátových/drawdown období obou strategií;
- výkon kombinace v obdobích, kdy jedna strategie podle předem určeného pravidla selhává;
- obrat, náklady, koncentraci a chování napříč tržními režimy.

Definice „selhávání“ (např. drawdown přes práh, nebo zaostávání za benchmarkem po stanovenou dobu) musí být nastavena před porovnáním párů. Nelze zpětně vybrat právě ta období, která dělají konkrétní pár atraktivním. Výběr páru probíhá uvnitř vývojové/WFO oblasti; OOS pár pouze ověřuje.

## 12. ML meta-model a alokace

### 12.1 Úkol modelu

Model v rozhodovacím čase `t` odhaduje budoucí rizikově očištěný výsledek již známé strategie A a B pro konfigurovatelný horizont `H` (výchozí sada k vyhodnocení: 5, 10 a 20 obchodních dní). Z výstupů vzniká volba strategie A, strategie B nebo hotovost. Model nevyhledává nové strategie za běhu.

Výsledná cílová metrika musí být definována tak, aby ji šlo reprodukovat z budoucí cesty čistých portfoliových výnosů. Samotný krátký Sharpe může být nestabilní; před implementací se stanoví pevné skóre, které výslovně určí výnos, riziko a případně drawdown během `H`.

### 12.2 Vstupní informace

Features smí obsahovat pouze tržní a cross-sectional informace dostupné k času `t`, například trailing výnosy, volatilitu, šířku trhu, dispersion a stav portfolií. Každá feature má uvedený zdroj a nejpozdější čas dostupnosti. Fundamentální údaje se přidají pouze tehdy, pokud audit potvrdí historické datum zveřejnění.

### 12.3 Konfigurovatelné rozhodování a omezení překmitávání

Frekvence rozhodnutí je samostatný parametr `decision_interval_days`, vyjádřený v obchodních dnech. Výchozí doporučení je `1`; uživatel může zvolit například 2, 5, 10 nebo jiné kladné číslo. Výpočet frekvence je oddělen od délky labelu `H`.

Model může vyhodnotit volbu v každém rozhodovacím okamžiku, ale změna strategie se provede pouze tehdy, když očekávaný přínos nové volby překročí odhadované náklady na změnu a konfigurovatelný práh pro nejistotu/rezervu. `min_strategy_hold_days` může být samostatný konfigurovatelný limit; nesmí být zaměněn s frekvencí predikce. Hodnoty rozhodovacího intervalu, prahu a případného minimálního držení se ladí výhradně ve WFO.

Při přechodu se počítají skutečné čisté změny pozic a související náklady. Pokud strategie A a B drží stejný titul, výpočet objednávky má vycházet z rozdílu cílové expozice, ne z umělého prodeje a nákupu stejné pozice. Hotovostní volba uzavře nebo zredukuje pozice podle proveditelného plánu na nejbližším povoleném provedení; náklady uzavření se započítají.

### 12.4 Trénink a porovnání baseline

ML se porovnává minimálně s těmito variantami: držet A, držet B, pevně 50/50 a držet hotovost. Report uvádí přínos modelu po nákladech, počet změn alokace, obrat vyvolaný změnami a výsledky pro každé `H` a každé WFO okno. Použití dalšího akciového titulu jako dalšího vzorku neospravedlňuje náhodné časové rozdělení.

## 13. Benchmark a reporty

Primární benchmark pro každé hodnocené období je long-only buy-and-hold rovnoměrně váženého point-in-time univerza dostupného na začátku daného období. Akcie jsou drženy bez běžného rebalancování; dividendy, splity a delistingy se zohledňují podle auditovaných zdrojových dat. Benchmark se sestavuje z informací známých na začátku období, nikoli z pozdějších vítězů. Další benchmarky (například SPY) mohou být doplněny, ale nenahrazují potvrzené srovnání se stejným univerzem.

Každý report musí obsahovat minimálně:

- equity strategii/kombinace a equity buy-and-hold benchmarku ve stejném grafu;
- drawdown strategii a benchmarku;
- přehled výnosu, volatility, rizikových metrik, obratu a nákladů;
- rozdělení nebo anotace WFO testovacích oken;
- pozice/expozici a přechody ML mezi A, B a hotovostí;
- použité období, univerzum, kapitál, nákladový model a konfiguraci;
- samostatné označení výsledků vývojových, WFO a finálních OOS období.

Grafy musí být exportovatelné do běžného obrazového formátu; zdrojová časová řada zůstává dostupná pro vizuální i numerické srovnání běhů. OOS křivka nesmí být graficky spojena s tréninkovým úsekem způsobem, který skrývá hranici období.

## 14. Návrhová architektura

První implementace je lokální Python aplikace rozdělená do malých testovatelných modulů:

```text
Lokální Sharadar soubory
        |
        v
Inventář, normalizace a datové kontroly
        |
        v
Point-in-time universe + features
        |
        v
Generátor konfigurací -> knihovna strategií
        |
        v
Portfolio konstrukce -> plán provedení -> model nákladů
        |
        v
Backtest a audit rozhodnutí
        |
        +--> metriky, ranking a deduplikace
        +--> WFO výběr doplňující dvojice
        +--> ML meta-model / alokace
        |
        v
Reporty, equity a benchmark grafy
```

Data provider a konkrétní formát souborů zůstávají oddělené od logiky strategií. Strategie vrací cílové signály/pozice, nikoli předstírané vyplněné obchody. Exekuce a nákladový model jsou samostatné komponenty. Každý krok ukládá dost informací k auditu rozhodnutí.

Lokální perzistence a přesný konfigurační formát budou určeny při technickém návrhu. Požadavek je zachovat strojově čitelné výsledky a snadno porovnatelné běhy; není požadováno webové UI, API, cloud ani víceuživatelská služba v první verzi.

## 15. Implementační fáze

### Fáze 0 — Datový audit a uzavření metodických hodnot

- zmapovat lokální Sharadar soubory a potvrdit datové možnosti;
- zdokumentovat universe, corporate actions a delistingy;
- zvolit kalendářní hranice vývoj/WFO/final OOS podle pokrytí dat;
- určit konkrétní nákladové defaulty, eligibility a portfoliové limity;
- zmrazit definice metrik, rankingu, duplicity a cílového ML skóre.

### Fáze 1 — Datová vrstva a referenční long-only backtest

- normalizace, point-in-time universe, audit dat a časová pravidla;
- simulace kapitálu, zlomkových pozic, hotovosti, dividend/splitů a nákladů;
- jediná jednoduchá referenční strategie a buy-and-hold benchmark;
- equity, drawdown, obchody a audit rozhodnutí.

### Fáze 2 — Generátor a knihovna strategií

- zavést šest rodin strategií a verzované schéma parametrů;
- reprodukovatelné generování konfigurací a paralelizovatelné běhy;
- výsledkové metriky po foldů, ranking, slabé strategie a duplicity.

### Fáze 3 — WFO a výběr dvojic

- walk-forward běhy s odděleným finálním OOS;
- výběr strategií a dvojic pouze uvnitř vývojových foldů;
- vyhodnocení 50/50 páru, drawdown souběhu a stability.

### Fáze 4 — ML meta-model

- časově správné labely pro horizonty 5/10/20 dní;
- purging/embargo, baseline modely a výběr A/B/cash;
- konfigurovatelný rozhodovací interval, práh přepnutí a případné minimum držení;
- report turnoveru a čistého přínosu po nákladech.

### Fáze 5 — Závěrečné OOS vyhodnocení a dokumentace

- jednorázové spuštění zmrazené kompletní metodiky na finálním OOS;
- export všech reportů, konfigurací, výsledků a limitací;
- finální uživatelská a technická dokumentace podle skutečné implementace.

## 16. Akceptační kritéria

První verzi lze považovat za specifikačně a funkčně dokončenou, pokud:

1. Umí sestavit inventář Sharadar dat a zastaví/označí běh, když chybí pole nezbytné pro jeho předpoklady.
2. Umí vytvořit point-in-time historické univerzum a zahrnout dostupnou historii delistovaných titulů.
3. Signál po close nemůže být proveden za stejný close; provedení nastává nejdříve na další seanci.
4. Výpočet kapitálu podporuje zlomkové akcie, hotovost, long-only omezení a 10 000 USD počáteční kapitál.
5. Provize a slippage jsou explicitní, konfigurovatelné a zahrnuté v primárních výsledcích.
6. Každý běh lze reprodukovat z uložené konfigurace, seedu, verze kódu a manifestu dat.
7. Strategie, výběr páru i meta-model jsou validovány časově a final OOS se nepoužívá při vývoji.
8. ML labely nezasahují přes hranici testu a rozhodovací interval je konfigurovatelný v obchodních dnech.
9. Přepnutí meta-modelu zohledňuje odhad nákladů a skutečný obrat portfolia.
10. Výstupní graf vždy obsahuje equity vybrané strategie/kombinace i potvrzený buy-and-hold benchmark.

## 17. Otevřené hodnoty k rozhodnutí při implementaci

Tyto body nebrání vytvoření architektury, ale musejí být uzavřeny před příslušným kódem a finální validací:

- přesný obsah Sharadar balíku, úprava cen, schéma delistingů a data dostupnosti fundamentů;
- historické pravidlo eligibility a minimální likvidity pro universe;
- konkrétní komise/slippage a zda budou zpočátku nulové úroky na hotovost;
- přesné rozsahy parametrů šesti rodin a portfolio sizing;
- kalendářní hranice a délka WFO oken podle skutečného pokrytí dat;
- tvrdé prahy pro vyřazení a ranking skóre;
- prahy pro deduplikaci strategií;
- přesná formule „selhávání“ jedné strategie a párového komplementarity skóre;
- definice rizikově očištěného forward labelu pro horizonty 5/10/20 dní;
- výchozí nákladový práh přepnutí, případná rezerva nejistoty a `min_strategy_hold_days`;
- konkrétní výchozí `decision_interval_days` (doporučení: 1 obchodní den, konfigurovatelně).

Žádný z těchto parametrů se nesmí odhadnout z finálního OOS a potom prezentovat tentýž výsledek jako nezávislé potvrzení.
