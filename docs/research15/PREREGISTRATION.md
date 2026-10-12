# Výzkum 15 – Multi-timeframe 4h / 1h / 15m: směr, struktura, potvrzení (pre-registrace)

Verze 1.0 · 2026-10-12 · Stav: **schváleno, nic nespočteno**.
Navazuje na výzkum 14 (MTF pullback, 1h, zamítnut). Po commitu verze 1.0 platí: změny = nový pokus v registru
a řádek v §12.

## 1. Otázka

Klasické diskreční schéma „vyšší timeframe určuje směr, střední timeframe dá setup, nízký timeframe potvrdí
vstup“:

- **4h** – směr trendu (obchoduje se jen ve směru 4h),
- **1h** – setup: průraz struktury (*break*) nebo vybrání likvidity za swingem a návrat (*liquidity sweep*),
- **15m** – potvrzení: lokální průraz 15m struktury ve směru obchodu.

**Vydělává na BTC a ETH vstup podle 1h setupu potvrzeného na 15m ve směru 4h trendu, se stopem za strukturou
setupu a cílem v násobcích rizika, po nákladech a fundingu víc než nula, víc než náhodné vstupy ve stejném 4h
režimu se stejnou geometrií výstupu, a přidává každá z vrstev (4h filtr, 15m potvrzení) něco navíc?**

## 2. Poctivé omezení předem

- **Apriorní očekávání je negativní.** Výzkum 14 na stejných aktivech a stejném holdoutu ukázal, že edge byl
  ve vyšším trendu samotném (trvale long při W + D nahoru: Sharpe 0,74) a 1h načasování vstupu nepřidalo nic
  (P na úrovni náhodných vstupů). Tento výsledek je znám před touto pre-registrací.
- Období 2022–2026 už otevřely výzkumy 9, 10 C, 13 a 14. Na 15m datech a pro tuto hypotézu nebylo použito, ale
  „neviděné“ není. Historický test proto může hypotézu jen **zamítnout, nebo kvalifikovat na forward test** (§10).
- „Break“, „liquidity“ a „potvrzení“ jsou diskreční pojmy. Převod na pravidla (§4) je jeden z mnoha možných,
  volí se **před výpočtem** a během studie se nemění. Neúspěch zamítá tuto operacionalizaci, ne každou možnou
  podobu myšlenky.
- Short se v praxi dělá přes USDT perpetual. Ceny jsou ze spotu (dostupné od 2017), funding podle §5. Varianta
  s cenami perpu od 2020 je v robustnosti (§9).

## 3. Data

- 15m svíčky Binance **spot** `BTCUSDT`, `ETHUSDT` (data.binance.vision, měsíční zipy ověřené `.CHECKSUM`),
  `scripts/r15_download.py` → `data/raw/binance_2026-10-03/spot_15m/`. Rozsah 2017-08 → **2026-09-30 23:45 UTC**.
- 15m svíčky **USDT-M perp** pro BTC, ETH a sekundární koš (§9) → `perp_15m/`, od 2020-01.
- Funding perpetuálů z `data/raw/binance_2026-10-03/funding/`.
- Mřížka 15 min UTC bez mezer. Chybějící svíčka = close se přenese (OHLC = předchozí close), příznak
  `real = False`. Na nereálné svíčce nevzniká signál, potvrzení ani vstup a nekontroluje se na ní stop (otevřená
  pozice běží, stop/cíl se vyhodnotí na další reálné svíčce včetně mezery přes open).
- **1h a 4h svíčky se agregují z 15m** (open = první, high = max, low = min, close = poslední reálné svíčky;
  4h hranice 00, 04, 08, 12, 16, 20 UTC). Svíčka vyššího TF s méně než 75 % reálných 15m svíček je neúplná a
  nevytváří signál. Svíčka vyššího TF je použitelná až po svém uzavření.
- Audit (mezery, duplicity, OHLC konzistence, posunuté časy, SHA256 sady zipů) do `DATA_AUDIT.md`.
- Časové značky od 2025 v mikrosekundách se převedou.

## 4. Signál (vše jen z dat uzavřených do konce 15m svíčky *k*)

**4h režim** podle poslední uzavřené 4h svíčky: EMA20 a EMA50 z 4h close.
- nahoru: `C4 > EMA50` a `EMA20 > EMA50`; dolů: `C4 < EMA50` a `EMA20 < EMA50`; jinak neutrální.

**1h setup** na uzavřené 1h svíčce *j* s okénkem *N* předchozích 1h svíček (`HH = max H1[j−N … j−1]`,
`LL = min L1[j−N … j−1]`):
- `BRK` long: `C1_j > HH` (close nad strukturou). Short: `C1_j < LL`.
- `SWP` long: `L1_j < LL` a `C1_j > LL` (knot pod swing low, vybrání likvidity, close zpět nad úrovní).
  Short: `H1_j > HH` a `C1_j < HH`.
- Setup vzniká jen tehdy, když 4h režim při uzavření *j* souhlasí se směrem (long v režimu nahoru).
- **Kotva stopu** = low svíčky *j* (long), high svíčky *j* (short). U `SWP` je to přímo low knotu.
- Setup je aktivní **16 svíček 15m (4 h)** po uzavření *j*. Nový setup téhož směru nahradí starý, setup opačného
  směru starý zruší. Pokud během aktivity 15m low ≤ kotva (long), setup se ruší (zneplatněn).

**15m potvrzení:** reálná 15m svíčka *k* v době aktivního setupu, pro kterou `C15_k > max H15[k−4 … k−1]`
(long; short zrcadlově `C15_k < min L15[k−4 … k−1]`) a 4h režim na *k* stále souhlasí.

**Vstup:** za close svíčky *k*.

## 5. Obchod

- **Stop** S = kotva. Riziko `r = |vstup − S| / vstup`. Obchod se **nevstoupí**, je-li `r < 0,15 %` (stop
  menší než náklady obchodu) nebo `r > 5 %`.
- **Cíl** T = vstup ± R · |vstup − S|.
- **Časový stop** 48 h (192 svíček 15m) → výstup za close.
- Vyhodnocení na každé reálné 15m svíčce po vstupu (long; short zrcadlově), v tomto pořadí:
  1. open ≤ S → výstup za open (mezera přes stop),
  2. low ≤ S → výstup za S (**stop má přednost, i když svíčka zasáhla i cíl**),
  3. open ≥ T → výstup za open,
  4. high ≥ T → výstup za T.
- Na aktivum nejvýš jedna pozice. Během pozice se nové setupy ignorují, nový vstup je možný od svíčky po výstupu.
- **Portfolio:** dva rukávy BTC a ETH, každý 50 % kapitálu; pozice = +1, −1 nebo 0 nominálu rukávu (bez páky,
  bez sizingu podle rizika). Výnos rukávu po 15m svíčkách × pozice.
- **Náklady:** 7,5 bps na stranu z nominálu (taker perp 5 bps + skluz 2,5 bps), u stopů i cílů stejně; zátěž ×2.
- **Funding:** v každém vypořádání (00, 08, 16 UTC), kdy je pozice otevřená, platí long `+rate`, short `−rate`.
  Před začátkem dat paušálně 0,01 % / 8 h.
- Denní výnos portfolia = součet 15m výnosů dne (UTC) po nákladech a fundingu.

**Mřížka (N = 16):** setup `∈ {BRK, SWP}` × `N ∈ {12, 48}` × `R ∈ {1,5; 3}` × strana `∈ {LS, LONG}`.

Pevné, mimo mřížku: EMA 20/50 na 4h, aktivita setupu 4 h, potvrzení 4 svíčky 15m, časový stop 48 h,
meze rizika 0,15–5 %.

## 6. Benchmarky a nulové hypotézy

- **B_RND (hlavní nula):** 500 běhů, seed 0..499. Pro každé aktivum, stranu a kalendářní měsíc stejný počet
  obchodů jako P, vstup v náhodné reálné 15m svíčce se **stejným 4h režimem** bez otevřené pozice. Riziko `r` se
  losuje (s vracením) z obchodů P téhož aktiva ve stejné fázi, stejné R, časový stop, náklady a funding. Měří, zda
  načasování 1h setupu a 15m potvrzení přidává něco nad náhodný vstup v 4h trendu se stejnou geometrií výstupu.
- **B_NO15:** stejný 1h setup a stop, vstup hned za close svíčky *j* bez 15m potvrzení. Měří přínos 15m vrstvy.
- **B_NO4H:** stejný setup a potvrzení bez 4h filtru, obě strany. Měří přínos 4h vrstvy.
- **B_4H:** trvalá pozice ve směru 4h režimu (+1 / −1 / 0; u `LONG` jen +1 / 0), stejné náklady a funding.
- **B_HOLD:** 50/50 BTC/ETH, denní rebalance, bez nákladů.

## 7. Období a trezor

| Fáze | Období |
|---|---|
| Vývoj | 2018-02-01 → 2021-12-31 (výběr P = nejvyšší Sharpe z 16 kandidátů) |
| Holdout | **2022-01-01 → 2026-09-30**, jednou, přes `Vault` (`runs/vault_r15/`, zámek metodiky, log otevření) |

Indikátory se mohou počítat z dat před začátkem fáze, na začátku fáze se nic nedrží. Obchod otevřený na konci
fáze se uzavře za close poslední svíčky fáze. Při shodě Sharpe vyhrává první v pořadí mřížky.

## 8. Kritéria (holdout, P; nutné splnit **všechna**)

Sharpe = průměr / sm. odchylka denních výnosů × √365, bez bezrizikové sazby.

1. Sharpe P > 0 a průměrný čistý výnos obchodu > 0 s dolní mezí 90% intervalu > 0 (bootstrap obchodů,
   2000 vzorků, seed 0).
2. Sharpe P > **95. percentil** Sharpe B_RND.
3. Průměrný čistý výnos obchodu P > B_NO15 **a** > B_NO4H se stejnými parametry (obě vrstvy pomáhají).
4. Sharpe P > 0 při nákladech ×2.
5. Sharpe P > 0 v obou podúsecích **2022-01 → 2023-12** a **2024-01 → 2026-09**.
6. Průměrný čistý výnos obchodu > 0 zvlášť na BTC i na ETH.

Deflated Sharpe (N = 16, rozptyl Sharpe z vývojové mřížky) se hlásí, není kritériem.

## 9. Robustnost (jen hlášení, nevybírá se podle ní)

Všech 16 kandidátů na holdoutu; výsledky po letech; plnění o svíčku později (close *k*+1); bez fundingu; long a
short zvlášť; statistiky obchodů (počet, úspěšnost, průměr v R, medián, rozpad podle důvodu výstupu: stop / cíl /
čas); srovnání s B_4H a B_HOLD; **ceny perpu** místo spotu (2020-01 → 2026-09).

**Sekundární koš** (jen popisně): USDT-M perpy, které byly v daném měsíci v **top 10 podle objemu za 30 dní
před začátkem měsíce** (bez BTC a ETH, jen kryptoměny s protějškem na spotu, bez stablecoinů a tokenizovaného
zlata, historie ≥ 60 dní; `qlab.research15.universe`). Výběr je podle data, bez survivorship biasu. Pravidla P,
ceny perpu, rovné rukávy členů; nový vstup jen do členů v daném měsíci, otevřená pozice doběhne i po vypadnutí.
Období **2020-05 → 2026-09** (první měsíc s plnými 10 členy; perpy altcoinů Binance spustil většinou až v lednu 2020).

## 10. Co znamená výsledek

- Nesplní-li P cokoli z §8: **zamítnuto**, forward test se nespouští.
- Splní-li vše: kandidát na **forward paper test ≥ 6 měsíců** s vlastní pre-registrací.

## 11. Implementace a postup

1. Schválení vlastníkem, verze 1.0, commit této pre-registrace (před jakýmkoli výpočtem nad cenami).
2. `scripts/r15_download.py` (hotovo, data staženo před commitem; čte jen objemy, ne výnosy),
   `scripts/r15_build_panel.py`: 15m panel (OHLC, `real`, funding na svíčku) do
   `data/derived/binance_2026-10-03/r15_15m/`, `DATA_AUDIT.md`.
3. `qlab.research15`: agregace 1h/4h, režim, setupy, potvrzení, simulátor se stopem/cílem/časem, B_RND, B_NO15,
   B_NO4H, B_4H. Testy: bod v čase (`check_point_in_time`), uzavřené svíčky vyššího TF, ruční přepočet obchodu
   (náklady, funding, mezera přes stop, stop a cíl v jedné svíčce), nepřekrývání, zneplatnění setupu,
   přenesené svíčky.
4. Commit kódu. Registr `runs/registry/research15.sqlite`: všech 16 kandidátů zapsat před výpočtem.
5. `scripts/r15_dev.py` (vývoj, výběr P) → `scripts/r15_final.py freeze|run` (jednorázový holdout).
6. `REPORT.md`, řádek v tabulce README, verdikt do §12.

## 12. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-10-12 | Návrh v0.1. 15m data stažena před commitem pre-registrace (jen stažení a výběr koše podle objemu, žádné výnosy ani signály). Spot BTC/ETH 2017-08-17 → 2026-09-30: 319 294 svíček, 562 chybějících, bez duplicit, 81 svíček mimo 15m mřížku (výpadek Binance 2018-02, jako ve výzkumu 14). Perp BTC/ETH 2020-01 → 2026-09 bez mezer. Sekundární koš: 105 perpů za 79 měsíců (`r15_manifest.json`). | stažení (§3) |
| 2026-10-12 | Pre-registrace v1.0 schválena vlastníkem, beze změn pravidel oproti v0.1. | – |
| 2026-10-12 | Upřesnění implementace (nemění mřížku ani kritéria; rozhodnuto před jakýmkoli výpočtem nad cenami, `qlab.research15.strategy`): (a) EMA se zahájí první platnou 4h close, režim je 0, dokud nebylo 150 platných 4h close; chybějící 4h svíčka přenáší EMA, ale režim je v ní 0. (b) 1h svíčka, která by byla zároveň long a short `SWP` setup, nedává žádný setup. (c) Setup nahrazuje jen další *platný* setup (souhlasí s 4h režimem a stranou); okno potvrzení končí nejpozději na svíčce vzniku dalšího setupu včetně. (d) Potvrzovací svíčka, jejíž low (long) / high (short) zasáhne kotvu, setup ruší (nevstupuje se na ní). (e) Setup je spotřebován prvním potvrzením, i když je obchod kvůli mezi rizika 0,15–5 % odmítnut. (f) Setupy vzniklé do svíčky výstupu otevřené pozice včetně se zahazují; setupy vzniklé před začátkem fáze se nepoužijí. (g) Plnění o svíčku později (§9) se přeskočí, pokud mezitím real svíčka zasáhla kotvu. (h) Funding se přiřadí svíčce, v níž leží `calc_time`; vypořádání nese pozice držená přes začátek svíčky `t % 32 == 0`. (i) B_RND: pro každý (měsíc, strana) se losují permutace kandidátních svíček, přijme se první, jejíž skutečně simulovaný průběh (od vstupu po výstup) nepřekrývá už přijatý obchod, do dosažení počtu; riziko se losuje z rizik obchodů P téhož aktiva a strany. (j) Sekundární koš: vstup jen v měsících, kdy je symbol členem (kontrola na svíčce vstupu), každý rukáv váha 1/10 kapitálu, nevyužité rukávy drží hotovost; ceny a funding perpu příslušného symbolu. (k) Robustnost „ceny perpu“ se hlásí na holdoutu i na celé historii perpu 2020-01 → 2026-09. | implementace, před výpočtem |
| 2026-10-12 | Před vývojem žádný smoke běh na skutečných cenách (jen syntetická data a testy, 20 testů + 4 vložené chyby, které testy zachytily). Kód commitnut (`9e717d5e`), 16 kandidátů zapsáno do `runs/registry/research15.sqlite` před výpočtem. | §11 |
| 2026-10-12 | Vývoj spočten (`results/dev.json`, 2018-02 → 2021-12): **P = `MTF_BRK_N48_R30_LONG`**, Sharpe 0,79 (CAGR 15,5 %, propad 16 %, 372 obchodů, průměr 0,345 %/obchod, 90% CI [0,022; 0,682] %). B_RND medián 0,30, p95 0,93, P nad 90 % náhodných běhů. B_NO15 Sharpe 0,58 (průměr 0,175 %), B_NO4H 0,40 (0,135 %), B_4H 1,20, B_HOLD 0,89. Kladný Sharpe má 9 z 16 kandidátů (7 z 8 `BRK`, kromě `BRK_N12_R15_LONG` −0,02, a obě `SWP_N12_R30` jen 0,12–0,13); záporných je 7: všechny 4 `SWP` s N = 48, obě `SWP_N12_R15` a `BRK_N12_R15_LONG`. Pravidla se nemění. | výběr podle §7 |

