# Výzkum 8 – funding carry na Binance (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **pre-registrace, žádná data zatím stažena ani spočtena**.
První výzkum mimo americké akcie. Navazuje na metodiku výzkumů 1–7 (registr pokusů, trezor na holdout,
stationary bootstrap, Deflated Sharpe). Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §10.

## 1. Otázka

Perpetual futures mají místo expirace **funding**: když je perp dražší než spot (převaha pákových longů),
longy platí shortům. Delta-neutrální pozice **long spot + short perp** stejného coinu nenese cenové riziko
a inkasuje funding.

**Vydělává tato pozice na Binance po nákladech víc než bezriziková sazba v USD (T-bill), stabilně v čase,
a přežije prodej a nákup v realistických cenách, likvidace a zátěžové náklady?**

Proč tato hypotéza a ne další predikce cen (výzkumy 1–7: 6× ne): nejde o předpověď, ale o **strukturální
prémii** s ekonomickým vysvětlením. Retail chce páku, arbitrážní kapitál je omezený, a kdo poskytuje
krátkou stranu, dostává zaplaceno.

## 2. Poctivé omezení předem

- **Historie je obecně známá.** Autor i Claude zhruba vědí, že funding byl vysoký v býčím trhu 2021
  a na začátku 2024 a záporný v části 2022. Historický test proto může hypotézu **zamítnout, nebo ji
  kvalifikovat na forward test** (§9). Sám nic nedokazuje.
- **Strategie je veřejně známá a od roku 2024 ji ve velkém dělá Ethena (USDe).** Prémie se může
  zmenšovat; holdout se proto hodnotí i po podúsecích (§6, kritérium 2).
- **Riziko protistrany se modelovat nedá.** Krach burzy (FTX 2022) by smazal kolaterál. Sharpe ratio
  toto riziko neobsahuje a v REPORTu bude uvedené vedle výsledku, ne pod čarou.
- Výsledek platí pro **taker** obchodování s malým kapitálem (do ~$1M), ne pro velikost Etheny.

## 3. Data

| Zdroj | Obsah |
|---|---|
| `data.binance.vision` (měsíční ZIPy + CHECKSUM) | USDT-M perpetual: funding (`fundingRate`), denní svíčky (`klines/1d`); spot: denní svíčky. **Všechny symboly včetně delistovaných** (seznam ze S3 listingu, ne z dnešního `exchangeInfo`). |
| FRED `DTB3` | 3měsíční T-bill, roční sazba → denní `(1 + r)^(1/365) − 1`; chybějící dny forward-fill. |

- Snapshot `binance_2026-10-03`. Data se nikdy needitují. Audit (mezery, duplicity, nekladné ceny,
  časové razítko fundingu mimo plán, kontrolní součty) proběhne nad celým obdobím a zapíše se do
  `docs/research8/DATA_AUDIT.md` **bez výnosových statistik**.
- Párování perp ↔ spot: `XUSDT` ↔ `XUSDT`, `1000XUSDT` (a `1000000X…`) ↔ `XUSDT` s multiplikátorem.
  Bez spotového páru na Binance se coin neobchoduje.
- Den = UTC. Kalendář 365 dní, anualizace √365.

## 4. Pravidla

### 4.1 Pozice (delta-neutrální „hedge“)

- Kapitál přidělený coinu `w·NAV` se dělí na **spot `s = 2/3`** (koupí se spot za `s·w·NAV`)
  a **kolaterál perp `1 − s = 1/3`** (short perp ve **stejné nominální hodnotě** jako spot → páka perpu 2×
  na jeho kolaterál).
- Funding: short dostane `rate × nominál perpu` za každou událost fundingu, kterou drží. Pozice otevřená
  na openu dne D (00:00 UTC) má nárok na události v intervalu **(D 00:00, E 00:00]**, kde E je den uzavření
  (otevření v 00:00:xx už událost 00:00 nestihne; kdo drží v E 00:00, dostane ji).
- P&L obou nohou z denních cen (spot close, perp close). Basis (rozdíl perp − spot) se tím promítá
  automaticky, včetně vstupu a výstupu.
- **Denní údržba:** když vlastní kapitál perp účtu (kolaterál + P&L perpu + funding) klesne pod 50 %
  nebo stoupne nad 150 % cílového kolaterálu, obě nohy se vyrovnají zpět na cílový poměr a stejný nominál
  (s náklady).
- **Likvidace:** když denní high perpu vůči ceně při posledním vyrovnání znamená ztrátu ≥ vlastní kapitál
  perp účtu − 0,5 % nominálu (udržovací marže), perp účet je ztracený celý. Spot se prodá na close
  téhož dne a coin je v hotovosti do dalšího rebalance. Počet likvidací se hlásí vždy.
- Hotovost (USDT) nenese nic.
- Měřítko: funding 10 % p. a. na nominál = **~6,7 % p. a. na kapitál** (nominál je 2/3 kapitálu), před
  náklady. Laťka T-bill byla v letech 2023–2026 zhruba 4–5 %.

### 4.2 Rozhodování

- Rebalance **týdně**, rozhodnutí z dat do pondělí 00:00 UTC (funding s časem < pondělí 00:00, svíčky
  uzavřené do neděle), plnění na **openu pondělí** + náklady. Coin, jehož váha se nemění, se neobchoduje.
- Trailing funding `F_L` = součet sazeb fundingu za posledních `L` dní × 365/`L` (roční sazba).
  Coin bez úplné historie `L` dní není způsobilý.
- **Univerzum k datu:** perp listovaný ≥ 30 dní, se spotovým párem, top 20 podle průměrného denního
  quote objemu perpu za 30 dní. Stablecoiny a wrapped tokeny (USDC, FDUSD, TUSD, BUSD, DAI, USDP, WBTC,
  WBETH, …) jsou vyřazené.
- Delisting perpu nebo spotu: pozice se zavře na posledním dostupném close obou nohou + 2 % slippage.

### 4.3 Kandidáti (mřížka; každý je pokus v registru)

| id | Popis | Parametry | Počet |
|---|---|---|---|
| `S0` | BTC + ETH 50/50, vždy zajištěno | – | 1 |
| `S1` | BTC + ETH po 50 %, coin zajištěn jen když `F_L > θ`, jinak hotovost | L ∈ {7, 30}, θ ∈ {0 %, 10 % p. a.} | 4 |
| `S2` | průřez univerzem: top K podle `F_L`, jen `F_L > θ`, rovné váhy 1/K (nevyužitá místa = hotovost) | K ∈ {3, 5, 10}, L ∈ {7, 30}, θ ∈ {0 %, 10 % p. a.} | 12 |

Celkem **N = 17**. Nic dalšího se na holdoutu nehodnotí. Pokud vývoj ukáže chybu v pravidlech
(ne slabý výsledek), oprava = nová verze dokumentu a nový pokus, N roste.

### 4.4 Náklady (za stranu, každá noha zvlášť)

| | Poplatek | Slippage podle 30d quote objemu perpu |
|---|---|---|
| Spot | 10 bps (taker) | ≥ $1 mld./den 2 bps · $200 mil.–1 mld. 5 bps · $50–200 mil. 10 bps · < $50 mil. 25 bps |
| Perp | 5 bps (taker) | stejné pásmo |

Vstup a výstup hedge u BTC tak stojí ~38 bps nominálu (~25 bps kapitálu) tam i zpět. **Zátěž:** poplatky i slippage ×2.

## 5. Období a trezor

| Fáze | Období | Použití |
|---|---|---|
| Vývoj | 2020-01-06 → 2022-12-31 | mřížka, výběr kandidáta, ladění kódu |
| Holdout | **2023-01-01 → 2026-09-30** | **jednou**, přes `Vault` (zámek metodiky, log otevření) |

Loader odmítne data po 2022-12-31 mimo finální běh. Audit dat (§3) počty a mezery vidí, výnosy ne.

## 6. Kritéria (holdout; nutné splnit **všechna**)

Hodnotí se dva kandidáti, každý zvlášť (Holm přes 2 u kritéria 1):

- **P1 = `S0`** (pevný, bez výběru, čistě strukturální prémie),
- **P2 = nejlepší z 17 podle Sharpe nadvýnosu ve vývoji** (při shodě nižší obrat).

Nadvýnos = denní čistý výnos strategie − denní T-bill.

1. Roční nadvýnos > 0 a dolní mez 90% intervalu (stationary bootstrap, blok 21 dní, 1000 vzorků,
   seed 0) > 0,
2. nadvýnos > 0 v obou podúsecích holdoutu: **2023-01 → 2024-06** a **2024-07 → 2026-09** (před a po
   nástupu Etheny ve velkém),
3. nadvýnos > 0 i při zátěžových nákladech,
4. P1: PSR nadvýnosu ≥ 0,95. P2: Deflated Sharpe ≥ 0,95 (N = 17, rozptyl Sharpe z vývojové mřížky).

Sharpe = průměr / sm. odchylka denního nadvýnosu × √365.

## 7. Robustnost (jen hlášení, nerozhoduje)

Výsledky po letech; maximální propad; počet likvidací a dní se zápornou sumou fundingu; `s` = 1/2 a 3/4;
plnění o den později; univerzum top 10 a top 40; bez hlídání delistingu; podíl výnosu z fundingu vs. z basis;
vývoj průměrného `F_7` BTC a ETH v čase (komprese prémie).

## 8. Implementace (před vyhodnocením, testy)

`qlab.research8`: stahování a ověření kontrolních součtů, denní panel (funding agregovaný po dnech
s přesnými časy událostí), simulátor hedge s likvidacemi, strategie, hranice trezoru. Testy: ruční
přepočet jednoho týdne hedge (funding, P&L nohou, náklady), likvidace na syntetických datech, nulový
funding = výnos −náklady, test bodu v čase (perturbace budoucích řádků nezmění rozhodnutí).

## 9. Co znamená výsledek

- **Nesplní-li P1 i P2 cokoli z §6:** zamítnuto. Funding carry na Binance pro malého obchodníka po
  nákladech T-bill spolehlivě nepřekonává.
- **Splní-li to P1 nebo P2:** kandidát na **forward paper test minimálně 6 měsíců**. Přidá se jako
  rukáv do krypto paper bota (nový release po skončení jeho pre-registrovaného okna 2026-11-30;
  bot už hodinový funding sbírá). Teprve úspěšný forward test je důkaz, se kterým jde za lidmi.

## 10. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-10-03 | Pre-registrace v1.0 | – |
| 2026-10-03 | Data: snapshot `binance_2026-10-03` (895 perpů, 389 kandidátů, 0 chyb, vše ověřeno sha256) | `DATA_AUDIT.md` |
| 2026-10-03 | **Pravidlo platnosti páru** (před jakýmkoli výsledkem): pár je platný v den *t*, jen když closy perpu a spotu (× multiplikátor) jsou v poměru ≤ 1,2. Univerzum vyžaduje platný pár všech 30 dní lookbacku; S0/S1 platný pár včera. Držená pozice s neplatným párem včera se zavře na dnešním openu **za skutečné ceny obou nohou** + 2 % slippage (ztráta z rozjetí basis se započítá). | Audit: dlouhé nesoulady FTT, RAY, STRAX, SC, CVC, TLM (krach FTX, migrace a relisting tokenů) a jednodenní výkyvy u ~50 coinů. Simulátor předpokládá, že obě nohy jsou stejné aktivum. |
| 2026-10-03 | Holdout se počítá jako souvislá simulace od začátku vývoje; metriky jen z výnosů 2023-01-01 → 2026-09-30 (pozice z konce 2022 se přenášejí). „Bez hlídání delistingu“ z §7 vypuštěno (bez cen nejde simulovat). | upřesnění §5, §7 |
| 2026-10-03 | Při sestavení parseru byly vidět 4 hodnoty fundingu BTC (2 z ledna 2020, 2 z června 2025) jako kontrola formátu. | transparentnost; bez vlivu na pravidla |
