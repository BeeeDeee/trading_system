# PROJECT_SPECIFICATION v2

Verze: 2.1-draft · Datum: 2026-09-25 · Nahrazuje: `archive/PROJECT_SPECIFICATION_v1.md` · Podklad: `SPEC_REVIEW.md`

Značky u rozhodnutí:
- **[D]** potvrzené rozhodnutí,
- **[N]** navržený default; zmrazí se na konci Fáze 1 po datovém auditu (přehled v §19),
- **[A]** musí ověřit datový audit.

---

## 1. Účel a hlavní cíl

Lokální výzkumný framework pro vývoj, backtest a poctivé vyhodnocení long-only strategií na denních
datech US akcií (Sharadar).

**Hlavní designový cíl:** co nejrychleji, nejlevněji a nejvěrohodněji zjistit, zda nějaká kombinace
jednoduchých strategií po nákladech překoná investovatelné alternativy (SPY, rovnoměrně vážené
univerzum), nebo tuto hypotézu věrohodně zamítnout. Negativní výsledek je legitimní a očekávatelný výstup.

Z toho plynou tři pravidla, která mají přednost před ostatními:
1. Každá vrstva výběru se započítává do multiple testing a do odhadu overfittingu.
2. Výsledky se prezentují s intervaly spolehlivosti, ne jen jako bodové odhady.
3. Každá fáze končí bránou (§15). Kdo neprojde, dál nepokračuje.

## 2. Rozsah a potvrzená rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Účel | **[D]** Čistý výzkum. Daně, konkrétní broker a FX mimo rozsah. |
| Trh | **[D]** US akcie. ETF jen jako benchmark a vstupy pro režim trhu, ne jako tituly k výběru. |
| Data | **[D]** Sharadar Bundle, Full History, statický dump (předplatné neaktivní). **[A]** přesný seznam tabulek a koncové datum. |
| Jazyk a provoz | **[D]** Python, lokálně, jeden uživatel, CLI. Bez webového UI a API. |
| HW | **[D]** Zatím slabý server (3 jádra, 3 GB RAM). Návrh musí škálovat na silnější stroj bez přepisu. |
| Kapitál | **[D]** 10 000 USD na simulaci (pro ledger engine a reporty; vektorový engine pracuje s výnosy). |
| Pozice | **[D]** Long-only, zlomkové akcie povoleny, žádná záporná hotovost ani páka. |
| Signál a fill | **[D]** Signál po close `t`, fill na open nejbližší obchodní seance `t+1`. |
| Intradenní SL/TP | **[D]** Mimo v1. |
| Kombinace strategií | **[D]** Primárně pár (K = 2, 50/50); K = 3 (EW) jako předem registrovaná sekundární varianta vyhodnocená ve stejném běhu. |
| ML | **[D]** Volitelná poslední fáze za bránou. Musí porazit jednoduché pravidlové overlaye. |
| Úrok z hotovosti | **[N]** 3M T-bill (FRED `DTB3`), posunutý o 1 den. Sharpe se počítá z výnosu nad touto sazbou. |

## 3. Zásady

- **Časová korektnost (point-in-time).** Každá hodnota v čase `t` je funkcí dat dostupných do close `t`.
  Vynucuje se testem (§14.2), ne jen dobrou vůlí.
- **Selekce je součást modelu.** Celý výběrový proces je čistá funkce času řezu (§9.2).
- **Náklady v primárním výsledku.** Hrubé výsledky jen jako diagnostika.
- **Registr pokusů.** Každý spuštěný kandidát, konfigurace i ruční zásah se zapisuje do append-only
  registru. Počet pokusů vstupuje do statistik (§10).
- **Nejdřív jednoduché.** Každá složitější vrstva musí porazit jednodušší baseline po nákladech.
- **Reprodukovatelnost.** Neměnný snapshot dat + hash konfigurace + commit kódu + lock závislostí =
  stejný výsledek bit po bitu (v toleranci float32).

---

## 4. Data

### 4.1 Zdroj a raw snapshot

- Raw dump se zkopíruje do `data/raw/<snapshot_id>/` a už se nemění. `snapshot_id` = datum exportu +
  SHA-256 manifest souborů.
- Konverze CSV → Parquet (partitionovaný podle tabulky, u SEP případně podle roku). Raw CSV zůstávají.
- Data nikdy nejsou v gitu.

### 4.2 Očekávané tabulky a použití

Seznam vychází ze známé struktury Sharadar Core US Equities Bundle; audit ho musí potvrdit **[A]**.

| Tabulka | Klíčová pole | Použití |
|---|---|---|
| `SEP` | ticker, date, open, high, low, close, volume, closeadj, closeunadj, lastupdated | Ceny akcií |
| `SFP` | stejné jako SEP | SPY a sektorová ETF: benchmark, režim trhu |
| `TICKERS` | table, permaticker, ticker, category, exchange, isdelisted, firstpricedate, lastpricedate, sector | Identita a typ titulu |
| `ACTIONS` | date, action, ticker, value, contraticker | Dividendy, splity, delistingy, akvizice, změny tickeru, spinoffy |
| `DAILY` | ticker, date, marketcap, ev, pe, pb, ... | Denní market cap (filtr velikosti, sanity check) |
| `SP500` | date, action, ticker | Historické složení S&P 500 → alternativní PIT univerzum |
| `SF1` | ticker, dimension, datekey, reportperiod, ... | Fundamenty. **Ve v1 se nepoužívají**, jen audit dostupnosti. |
| `funds` kategorie IDX | `^VIX`, `^GSPC`, `^RUT`, `^IXIC`, `^DJI` (od 1997-12-31) | VIX pro ML, indexy pro sanity check |
| mimo Sharadar | FRED `DTB3` | Bezriziková sazba |

### 4.3 Známé pasti a povinná opatření

| # | Past | Pravidlo |
|---|---|---|
| P1 | SEP OHLC jsou zpětně split-adjusted | Cenové a likviditní filtry jen z neupravených hodnot (`closeunadj`, objem přepočtený zpět) |
| P2 | `closeadj` obsahuje dividendy, `open` ne | TR faktor `f = closeadj / close`; adjusted open = `open × f` (faktor platný pro daný den, **[A]** ověřit chování v ex-div den) |
| P3 | TICKERS jsou snapshot k dnešku | Pole `exchange`, `isdelisted`, `lastpricedate`, `scalemarketcap`, `sector` se nesmí použít jako filtr historického univerza ani jako feature. Výjimka: `category` pro vyřazení ne-akciových instrumentů, s auditem stability. |
| P4 | Recyklované tickery | Primární klíč `permaticker`. Ticker jen jako popisek. |
| P5 | Chybí delisting return | Politika terminálního výnosu (§4.5) |
| P6 | ~~Spinoffy nemusí být v `closeadj`~~ | **Vyřešeno:** `closeadj` spinoffy obsahuje (viz `DATA_FINDINGS.md` §2). |
| P10 | SPAC schránky: cena ~10 USD, nulová volatilita; jejich likvidace je v ACTIONS jako `bankruptcyliquidation` | Vyloučit z univerza (SIC 6770 k datu `t`); likvidace SPAC = výplata poslední ceny (§4.5) |
| P11 | Dvě třídy akcií jedné firmy (Primary/Secondary Class) | V univerzu jen Primary Class |
| P12 | `metrics` je jen aktuální snapshot | Nepoužívat |
| P13 | ACTIONS obsahuje budoucí data (ohlášené splity) | Ignorovat akce po posledním obchodním dni snapshotu |
| P7 | Open = 0 / chybí | Den je pro daný titul neobchodovatelný. |
| P8 | Zpětné opravy dat | Řeší se neměnným snapshotem. |
| P9 | SF1 `MR*` dimenze jsou restatované | Pokud se v budoucnu použijí fundamenty: jen `AR*`, dostupnost od `datekey` + zpoždění. |

### 4.4 Normalizovaná vrstva

Tabulka `bars` v long formátu, klíč `(permaticker, date)`:

| Pole | Popis |
|---|---|
| `open_u, high_u, low_u, close_u, volume_u` | Neupravené hodnoty |
| `tr_factor` | Kumulativní faktor pro total return (splity + dividendy + případně spinoffy) |
| `ret_co` | Overnight výnos close `t−1` → open `t` (TR) |
| `ret_oc` | Intradenní výnos open `t` → close `t` (TR) |
| `dollar_volume` | `close_u × volume_u` |
| `tradable` | bool (open > 0, volume > 0, titul je listovaný) |
| `status` | `active`, `no_open` (bez použitelného open, vč. výplňových dní s nulovým objemem), `halted`, `delisted` |
| `terminal_ret` | Terminální výnos v den delistingu (§4.5), jinak null |

Obchodní kalendář: data SEP (audit: shodné se SPY, 7 228 dní, bez mimořádných uzávěr). Den bez
záznamu pro listovaný titul = `halted`; Sharadar však dny bez obchodu vyplňuje kopií ceny s nulovým
objemem, ty jsou `no_open`.
Nic se nedoplňuje budoucími hodnotami; forward-fill je povolen jen pro zobrazení, nikdy pro výnos.

### 4.5 Politika delistingu [N]

Poslední platný close = `P_last`. Terminální výnos se připíše v den delistingu, proceeds jdou do hotovosti.

| Typ z ACTIONS | Terminální výnos vůči `P_last` |
|---|---|
| Akvizice / fúze s `acquisitioncash` a/nebo `acquisitionstock` | `(cash + stock_ratio × P_acquirer) / P_last − 1`, kde `P_acquirer` je close kupujícího v den delistingu. Pole `value` u delistingových akcí je tržní kapitalizace, ne výplata. |
| Akvizice / fúze bez údaje o protihodnotě | 0 % |
| Likvidace SPAC (SIC 6770 k datu delistingu) | 0 % (výplata hodnoty trustu ≈ poslední cena) |
| Bankrot / likvidace (ostatní) | −100 % |
| Regulatorní / výkonnostní delisting, nebo neznámý typ při `P_last < 1 USD` | −30 % (Shumway 1997) |
| Ostatní / neznámý | 0 % |

Povinná citlivostní analýza: scénáře „optimistický" (vše 0 %) a „pesimistický" (neznámé −50 %,
výkonnostní −100 %). Pokud se závěr mezi scénáři liší, report to musí uvést.

### 4.6 Datový audit (výstup Fáze 1)

Report `audit/<snapshot_id>/` obsahuje:
- inventář tabulek, sloupců, rozsahů dat a počtů řádků,
- pokrytí delistovaných titulů po letech (počet aktivních vs. později delistovaných),
- rozložení typů akcí v ACTIONS, zejména delistingů,
- kontroly P1–P7 s konkrétními příklady,
- extrémní výnosy (|r| > 50 % denně) s vazbou na ACTIONS,
- duplicity `(permaticker, date)`, díry v kalendáři,
- **sanity check SPY:** rekonstruovaný cap-weighted index top-500 (z DAILY market cap) a EW
  S&P 500 (ze SP500) proti SPY a RSP. Roční tracking difference cap-weighted verze musí být pod
  ~1 p.b. **[N]**, jinak je chyba v datové vrstvě.

---

## 5. Univerzum

### 5.1 Primární univerzum „LIQ1000" [N]

V každém rebalančním dni `t` (point-in-time):
1. `category` ∈ {`Domestic Common Stock`, `Domestic Common Stock Primary Class`}; tím se vyřadí
   ETF/fondy (jsou v jiné tabulce), preferred, ADR, kanadské akcie a sekundární třídy akcií.
   Vyřadit SPAC: SIC kód 6770 platný k datu `t` (z `tickers` + historie `sicchangefrom/to`).
2. `close_u(t) ≥ 5 USD`.
3. Alespoň 252 obchodních dní historie (kvůli lookbackům).
4. Seřadit podle mediánu `dollar_volume` za 63 dní a vzít **top 1000**.

Hranice podle pořadí místo absolutních dolarů nepodléhá inflaci ani růstu trhu. Zároveň drží paměť pod
kontrolou: ~1000 titulů × ~7000 dní.

### 5.2 Alternativní univerza

- `SP500_PIT` – historické složení S&P 500 z tabulky SP500. Robustnostní test: strategie, která funguje
  jen v LIQ1000 a v SP500 selže, je podezřelá z efektu malých firem nebo likvidity.
- `LIQ3000` – jen jako citlivostní analýza, ne pro výběr.

Univerzum je součástí konfigurace strategie, ale **pro výběr kandidátů se ve v1 používá jen LIQ1000**,
aby se nenásobil počet pokusů.

---

## 6. Strategie

### 6.1 Jednotná kompozice

Každá strategie je složení šesti bloků. Rodina = volba skóre + přípustné rozsahy ostatních bloků.

```
univerzum → skóre → výběr → vážení → rebalance/držení → overlay expozice
```

| Blok | Možnosti v1 |
|---|---|
| Skóre `s_i(t)` | Dle rodiny (§6.2), vždy jen z dat do close `t` |
| Výběr | top-N podle skóre, N ∈ {10, 20, 30, 50}; nebo práh `s ≥ θ` s max N |
| Vážení | EW; inverse-vol (63d); cap na váhu titulu 10 % |
| Rebalance | týdně (poslední obchodní den týdne), měsíčně; hystereze: titul vypadne až pod pořadím `N × b`, b ∈ {1, 1.5, 2} |
| Overlay | žádný; tržní trend filtr (SPY nad SMA_L); volatility targeting na úrovni portfolia (max 100 %) |
| Hotovost | Nevyužitá váha = hotovost s úrokem T-bill |

### 6.2 Rodiny v1

| ID | Skóre | Klíčové parametry | Odlišnost |
|---|---|---|---|
| `xs_momentum` | Výnos za `L` dní s vynecháním posledních `S` dní | L ∈ {63, 126, 252}, S ∈ {0, 21} | Relativní pořadí napříč tituly |
| `ts_trend` | Tituly s `close/SMA_L > 1`, řazené podle vzdálenosti od SMA | L ∈ {50, 100, 200} | Absolutní trend titulu; počet pozic se mění (zbytek je cash) |
| `breakout` | Titul uzavřel na `L`-denním maximu v posledních `k` dnech; řazení podle blízkosti k maximu | L ∈ {55, 126, 252}, k ∈ {1, 5} | Událostní vstup (Donchian), výstup při poklesu pod `L_exit`-denní minimum |
| `st_reversal` | Záporný výnos za `L` dní, jen tituly nad SMA_200 | L ∈ {3, 5, 10} | Krátkodobý návrat; nejvíc citlivý na náklady |
| `low_vol` | Záporná volatilita za `L` dní | L ∈ {63, 126, 252} | Defenzivní |
| `regime_market` | Neobchoduje tituly: drží SPY nebo cash podle overlaye | SMA_L ∈ {100, 150, 200} | Baseline pro market timing |

`ts_trend` a `breakout` se liší definicí vstupu (stav vs. událost) i výstupu; na úrovni kódu jsou to
různé skórovací a výstupní funkce.

### 6.3 Generování kandidátů

- Plná mřížka nebo Sobolova sekvence nad parametry z §6.1 a §6.2. Náhodné vzorkování ne; potřebujeme
  okolí bodů pro analýzu citlivosti.
- Předpokládaná velikost: ~1–3 tisíce kandidátů celkem **[N]**. Konkrétní mřížka se zmrazí na konci
  Fáze 1 a každá pozdější změna je nový záznam v registru pokusů.
- `candidate_id` = hash kanonické konfigurace.

---

## 7. Simulace

### 7.1 Časová smlouva

1. Bar `t` je kompletní po close `t`.
2. Rozhodnutí po close `t` používá data do close `t` včetně.
3. Příkazy se provedou na open `t+1` (první obchodní seance po `t`).
4. Výnos portfolia za den `t+1`:
   `r_{t+1} = Σ w_i^{old} · ret_co_i(t+1) + Σ w_i^{new} · ret_oc_i(t+1) − cost_{t+1}`
   Staré váhy nesou overnight gap, nové váhy intradenní pohyb.
5. Label jakéhokoliv horizontu `H` začíná na open `t+1` a končí na open `t+1+H`.

### 7.2 Dva enginy

**Vektorový engine (screening).** Pracuje s váhami a výnosy (§7.1 bod 4), náklady v bps obratu. Je
scale-invariant, takže výsledkem je řada čistých denních výnosů na kandidáta. Výstup: matice
`R[den × kandidát]` (float32, Parquet) + matice obratu + denní počet pozic. Počítá se jednou pro celou
historii; strategie s pevnými parametry nezávisí na foldu.

**Ledger engine (finalisté a reporty).** Pracuje s kusy akcií, hotovostí a příkazy při kapitálu 10 000 USD:
- prodeje před nákupy,
- množství nákupu z open ceny `t+1` a dostupné hotovosti; při nedostatku se nákupy škálují pro-rata,
- neobchodovatelný titul: příkaz propadne a zopakuje se při dalším rebalanci, držená pozice zůstává,
- delisting: pozice se uzavře terminálním výnosem (§4.5), proceeds jdou do hotovosti,
- volitelný fixní poplatek za příkaz a minimální velikost obchodu,
- no-trade band: neobchodovat změnu váhy pod 0,5 p.b. **[N]**.

**Paritní test:** při nulových fixních poplatcích a vypnutém no-trade bandu se denní výnosy obou enginů
shodují v toleranci (např. |Δ| < 1 bp denně, drift CAGR < 0,1 p.b.) **[N]**. Parita je akceptační
kritérium (§16).

### 7.3 Nákladový model [D]

Pro každý obchod titulu `i`:

```
cost_i = |Δw_i| × ( commission_bps + max(floor_bps, half_spread(rank_i, t)) )
```

- `rank_i` = pořadí 63denního mediánu dollar volume titulu mezi všemi titulů daného dne.
- `half_spread`: rank ≤ 200 → 3 bps, ≤ 500 → 6 bps, ≤ 1000 → 12 bps, jinak 25 bps; před
  decimalizací (do 2001-04-09) × 2,5. `commission_bps = 0`, `floor_bps = 5`.
- Původně navržený odhad spreadu z OHLC (Abdi–Ranaldo) byl ověřen a zamítnut: pod ~30 bps ho
  přehluší volatilita (AAPL/MSFT vycházely 0–50 bps), viz `DATA_FINDINGS.md` §7.
- Povinné citlivostní běhy: náklady × 2 a × 3.
- **Kapacita (diagnostika):** maximální kapitál, při kterém žádný obchod nepřekročí 1 % 21denního
  mediánu dollar volume. Výzkumný projekt nepotřebuje kapacitu 10 USD mil., ale strategie s kapacitou
  pod 100k USD je podezřelá.

### 7.4 Úrok z hotovosti

Denní úrok = `DTB3(t−1) / 360`. Aplikuje se na hotovost ve vektorovém i ledger enginu.

---

## 8. Benchmarky

| ID | Definice | Otázka, na kterou odpovídá |
|---|---|---|
| `SPY_TR` | SPY total return (ze SFP) | Vyplatí se to vůbec oproti pasivní investici? |
| `EW_UNIV` | EW univerza strategie (LIQ1000), měsíčně rebalancované, stejný nákladový model | Má výběr titulů hodnotu, nebo jde jen o vlastnosti univerza? |
| `BH_UNIV` | Buy-and-hold EW univerza k začátku období, proceeds z delistingu reinvestovány proporčně do zbývajících pozic | Původně požadované srovnání (v1 §13) |
| `CASH` | T-bill | Spodní hranice |

Pro poskládané WFO OOS se benchmark počítá **nepřetržitě přes celý interval**, ne s resetem na začátku
každého foldu. `BH_UNIV` se jako výjimka reportuje per-fold i souvisle.

---

## 9. Validace

### 9.1 Časové rozdělení [N]

Předpokládaná data 1998-01 → konec dumpu (**[A]**, pravděpodobně 2025/2026):

| Úsek | Období | Účel |
|---|---|---|
| Warm-up | 1998-01 → 1999-12 | Jen lookbacky a spread odhady, žádné výnosy |
| Vývoj (WFO) | 2000-01 → 2019-12 | Veškerý výběr, ladění a porovnávání |
| Finální holdout | 2020-01 → konec dat | Jednorázové ověření zmrazené metodiky |

Holdout ~6 let zahrnuje covid crash, rychlé zotavení, medvědí rok 2022 i následný býčí trh. Jeho
statistická síla je omezená (SE Sharpe ≈ 0,45), proto se v něm posuzuje hlavně **konzistence** s WFO
výsledkem (§9.6), ne samostatná „významnost".

### 9.2 Výběrová politika jako čistá funkce

```
select(T, R[..T], metadata[..T], config) → Policy
```

`Policy` určuje, které strategie tvoří ansámbl, s jakými vahami a (ve Fázi 4) jaký ML overlay.
Uvnitř `select` probíhá všechno, co v1 dělala globálně: tvrdé filtry, ranking, dedup, výběr ansámblu.
Funkce smí číst jen řádky matice `R` do `T` včetně. Vynucuje se tím, že dostane již oříznutý pohled
(`R.slice(end=T)`), ne celou matici s indexem.

### 9.3 Walk-forward [N]

- Expanding (anchored) okno, trénink od 2000-01, minimální délka 5 let.
- Refit každých 12 měsíců (na konci roku). Test = následujících 12 měsíců.
- Foldy: test 2005, 2006, …, 2019 → 15 foldů.
- Embargo 21 obchodních dní mezi koncem tréninku a začátkem testu pro ML labely; pro pravidlový výběr
  není potřeba (žádný label nepřesahuje `T`), ale kvůli jednotnosti se aplikuje také.
- Přechod mezi politikami na hranici foldu: ledger engine přejde ze starého portfolia do nového s
  plnými náklady na čistý rozdíl pozic.
- **Poctivý odhad výkonu metodiky** = poskládané výnosy testovacích úseků 2005–2019.

### 9.4 Tvrdé filtry (uvnitř `select`) [N]

Nad tréninkovým oknem kandidát vypadne, pokud:
- průměrný počet pozic < 8,
- roční obrat > 2000 %,
- maximální drawdown > 45 % **[D]** (viz poznámka pod seznamem),
- Sharpe (nad T-bill) < 0,
- méně než 60 % tréninkových let s kladným výnosem nad T-bill.

Poznámka k drawdownu: SPY měl v letech 2007–09 propad ~55 % a 2000–02 ~49 %. Plně investovaná
long-only strategie bez overlaye tedy limit 45 % typicky nesplní, jakmile tréninkové okno obsahuje
rok 2008. Limit v praxi upřednostní strategie s trend filtrem, volatility targetingem nebo defenzivním
výběrem. To je zamýšlený důsledek, ne chyba. Report výběru uvádí, kolik kandidátů limit vyřadil.

### 9.5 Ranking a dedup (uvnitř `select`)

**Robustní skóre kandidáta:**
1. `SR_i` = Sharpe čistých výnosů nad T-bill v tréninkovém okně.
2. `SR_neigh_i` = medián `SR` kandidáta a jeho sousedů v mřížce (±1 krok v každém parametru).
3. Skóre = `min(SR_i, SR_neigh_i)` – osamělý peak se tím potrestá.

**Dedup:** hierarchické shlukování (average linkage) nad vzdáleností `1 − ρ` denních čistých výnosů v
tréninkovém okně, řez na `ρ = 0,85` **[N]**. Ze shluku postupuje kandidát s nejvyšším skóre. Počet
shluků ze všech kandidátů = **efektivní počet kandidátů `N_eff`** (diagnostika in-sample výběru a
DSR jednotlivých kandidátů).

**Ansámbl [D]:** dvě předem registrované varianty, obě vyhodnocené ve stejném běhu:
- **primární:** K = 2, váhy 50/50,
- **sekundární:** K = 3, váhy 1/3.

Členové = K nejlépe hodnocených reprezentantů různých shluků. Rebalance vah měsíčně, pozice přes
strategie se **netují** (stejný titul v A i B = jedna pozice se součtem vah). Obě varianty se počítají
do `N_meth` (= 2). Finální závěr projektu stojí na primární variantě; sekundární se reportuje vedle ní
a v holdoutu se mezi nimi nevybírá. Další K (např. 4) je nový pokus v registru.

### 9.6 Statistické testy a overfitting

| Nástroj | Nad čím | Použití |
|---|---|---|
| **Deflated Sharpe Ratio – metodika** | Poskládané WFO OOS výnosy metodiky; počet pokusů = `N_meth` (počet variant metodiky vyhodnocených na poskládaném OOS, z registru §9.8) | Brána Fáze 3: DSR ≥ 0,90 **[N]** |
| **Deflated Sharpe Ratio – kandidát** | In-sample výnosy kandidáta v tréninkovém okně, s `N_eff` a rozptylem SR kandidátů | Diagnostika v reportu výběru |
| **PBO přes CSCV** | Matice `R` vývojového období, 16 bloků | Brána Fáze 2: PBO ≤ 0,30 **[N]** |
| **Stationary block bootstrap** | Rozdíl výnosů metodika − benchmark | 90% CI pro rozdíl CAGR a Sharpe; v každém reportu |
| **Hansen SPA** | Metodika vs. `EW_UNIV` a `SPY_TR` | Doplňkový test v závěrečném reportu |
| **Konzistence holdoutu** | Sharpe holdoutu vs. rozdělení ročních Sharpe z WFO | Holdout mimo 10.–90. percentil = varování |

### 9.7 OOS vault

- Datová vrstva vrací data po 2019-12-31 jen běhu s příznakem `final_evaluation` a s hashem
  metodiky zapsaným v `frozen_methodology.lock`.
- Každý přístup se zapíše do `vault.log` (čas, commit, hash konfigurace). Druhý přístup se zamítne,
  pokud se nezmění snapshot dat. Obejít to jde (je to lokální kód), ale ne omylem a ne bez stopy.
- Diagnostické moduly (datový audit, sanity check SPY) data po 2019 vidí; neprodukují žádné metriky
  strategií.

### 9.8 Registr pokusů

Append-only tabulka (DuckDB/SQLite): `trial_id, čas, commit, config_hash, typ (kandidát / změna mřížky
/ změna filtrů / změna metodiky), poznámka`. Každá změna čehokoliv, co ovlivňuje `select`, je nový
pokus. Každé vyhodnocení varianty metodiky na poskládaném WFO OOS zvyšuje `N_meth` o 1.

Poznámka: poskládané WFO OOS je mimo vzorek vůči výběru kandidátů, proto se jeho DSR nedeflatuje
počtem kandidátů, ale počtem variant metodiky, které na něm byly vyzkoušeny. Kandidáti se počítají
v `N_eff` uvnitř `select`.

---

## 10. Metriky

Pro každou řadu výnosů (kandidát, ansámbl, benchmark), za celek, po foldech a po letech:

- **Výnos:** CAGR, kumulativní výnos, výnos nad T-bill.
- **Riziko:** volatilita, max. drawdown, délka nejdelšího drawdownu, CVaR 5 %.
- **Poměry:** Sharpe (nad T-bill), Sortino, Calmar.
- **Relativně:** beta k SPY, alfa, tracking error, information ratio vůči `SPY_TR` a `EW_UNIV`.
- **Obchodní:** roční obrat, počet obchodů, zaplacené náklady (v bps/rok), průměrný počet pozic,
  průměrná investovanost, podíl času v hotovosti, max. váha titulu.
- **Koncentrace výnosu:** podíl výnosu z top 10 titulů a z top 10 % dní.
- **Režimy (předem pevně dané):** 2000–02, 2007-10 → 2009-03, 2011, 2015–16, 2020-02 → 2020-04,
  2022; poslední dva jsou v holdoutu a v reportech se objeví až po finálním vyhodnocení.
- **Nejistota:** 90% bootstrap CI pro CAGR, Sharpe a rozdíl vůči benchmarku.

## 11. Fáze 4: ML overlay (volitelné, za bránou)

### 11.1 Úloha

Model v rozhodovacím čase `t` určuje **míru investovanosti** zmrazeného ansámblu `e(t) ∈ {0, 0.5, 1}`
a volitelně **náklon mezi rodinami** v ansámblu. Nevybírá nové strategie. Cash je `e = 0`.

### 11.2 Data a label

- Trénink ze **všech reprezentantů shluků** (ne jen z vybraných strategií), aby se odstranil selection
  bias. Jeden vzorek = (datum, rodina); features jsou tržní a rodinné, label je výkon rodiny.
- Label pro horizont `H` ∈ {5, 21} **[N]**: součet log čistých výnosů nad T-bill mezi open `t+1` a
  open `t+1+H`, dělený trailing volatilitou 63d (vol-normalizovaný výnos).
- Purging a embargo = `H` dní na hranicích foldů. Všechny vzorky se stejným datem jsou ve stejném foldu.
- Rozhodovací interval `decision_interval_days` ∈ {5, 21}, default = frekvence rebalance ansámblu.
  Hodnota 1 den není ve v1 povolena.

### 11.3 Features

Každá feature má zdroj a čas dostupnosti (close `t`):
- trh: výnosy SPY za 21/63/252 dní, SPY vs. SMA_200, realizovaná volatilita 21/63d, VIX (**[A]**)
  a jeho změna, šíře trhu (podíl LIQ1000 nad SMA_200), cross-sectional dispersion,
- rodina: trailing výnos a volatilita rodiny za 21/63/252 dní, její drawdown,
- sazby: `DTB3` a jeho změna za 63 dní.

### 11.4 Modely a baseline

Modely: regularizovaná lineární regrese (ridge), mělký gradient boosting (max. hloubka 3, silná
regularizace). Nic složitějšího ve v1.

Model musí po nákladech porazit **všechny** tyto baseline v poskládaném WFO OOS:
1. ansámbl vždy investovaný (`e = 1`),
2. trend overlay: `e = 1`, pokud SPY > SMA_200, jinak 0,
3. volatility targeting: `e = min(1, σ_target / σ_63d)`.

Přepnutí `e` se provede, jen když odhadovaný přínos překročí náklady přepnutí + rezervu (laděno ve WFO).

### 11.5 Brána

ML se zařadí do finální metodiky, pouze pokud:
- Sharpe OOS vyšší než nejlepší baseline a dolní hranice 90% bootstrap CI rozdílu > 0,
- výsledek konzistentní pro oba horizonty `H` (ne jen jeden),
- obrat vyvolaný overlayem < 200 % ročně **[N]**.

Jinak ML ve finální metodice nebude a report uvede negativní výsledek.

---

## 12. Reporty

Každý report (HTML + PNG grafy + Parquet se zdrojovými řadami) obsahuje:
- equity metodiky a benchmarků (`SPY_TR`, `EW_UNIV`, `BH_UNIV`) v log měřítku,
- drawdowny,
- barevně odlišené úseky warm-up / WFO test / holdout; OOS křivka nesmí být vizuálně spojena s
  tréninkovým úsekem,
- tabulku metrik s CI (§10),
- složení ansámblu po foldech (co `select` vybral a proč, včetně vyřazených kandidátů a důvodů),
- expozici a u ML overlaye průběh `e(t)`,
- citlivostní analýzu: náklady ×2/×3, politika delistingu, alternativní univerzum `SP500_PIT`,
- statistiky overfittingu: `N`, `N_eff`, `N_meth`, DSR, PBO,
- manifest: snapshot dat, commit, config hash, seed, čas.

---

## 13. Technická architektura

### 13.1 Stack [N]

| Oblast | Volba | Proč |
|---|---|---|
| Python | 3.12, správa přes `uv` + `uv.lock` | Reprodukovatelnost |
| Data | Parquet + **Polars** (lazy) + **DuckDB** pro ad-hoc dotazy | Funguje na 3 GB RAM a škáluje |
| Numerika | NumPy, případně Numba pro ledger engine | Rychlost bez složité infrastruktury |
| Konfigurace | Pydantic modely + YAML | Validace, kanonický hash |
| ML | scikit-learn, LightGBM | Stačí pro v1 |
| Statistika | vlastní implementace DSR, PBO, block bootstrap (malé, testovatelné) | Kontrola nad definicemi |
| Grafy | matplotlib (PNG) + jednoduchý HTML report | Offline, bez serveru |
| CLI | Typer | |
| Testy | pytest + hypothesis | |
| Paralelismus | `concurrent.futures` přes kandidáty, chunkování | Na silnějším stroji se jen zvýší počet workerů |

### 13.2 Struktura repozitáře

Git repozitář `trading_system`, branch `strategy_backtester_2026_sep`, kořen v
`/home/kapo/ccode/strategy_backtester_2026_sep/`.

```
strategy_backtester_2026_sep/
  AGENTS.md, README.md, .gitignore
  pyproject.toml, uv.lock
  docs/               # specifikace, review, audit reporty, archive/
  configs/            # universe, costs, grid, wfo, ml – YAML
  src/qlab/
    data/             # raw → parquet, normalizace, audit, kalendář, vault
    universe/
    strategies/       # bloky: score, select, weight, rebalance, overlay; families.py
    engine/           # vector.py, ledger.py, costs.py
    selection/        # filters, ranking, dedup, ensemble, select()
    validation/       # wfo, stats (dsr, pbo, bootstrap, spa), trials registry
    ml/
    benchmarks/
    report/
    cli.py
  tests/
  data/               # mimo git
  runs/               # mimo git; runs/<run_id>/manifest.json + výstupy
```

### 13.3 Tok dat

```
raw snapshot ─► parquet ─► normalizace (bars, actions, calendar) ─► audit
                                   │
                                   ▼
                        univerzum (PIT maska)
                                   │
                grid kandidátů ─► vektorový engine ─► R[den × kandidát]
                                   │
                  WFO: pro každý T: select(T, R[..T]) ─► Policy_T
                                   │
                 ledger engine: poskládaný OOS běh politik ─► equity, obchody
                                   │
                     statistiky (DSR, PBO, CI) + benchmarky ─► report
```

### 13.4 Paměťový rozpočet (3 GB RAM)

- LIQ1000: ~7000 dní × ~1000 aktivních titulů; unie přes celou historii ~6–8 tis. permatickerů.
  Features se počítají po titulech v long formátu, lazy.
- `R` pro 3000 kandidátů × 7000 dní × float32 ≈ 84 MB. OK.
- Nejdražší je výpočet skóre přes celé univerzum. Počítá se jednou pro každou unikátní kombinaci
  (rodina, lookback), ne pro každého kandidáta, a ukládá se do Parquetu.

---

## 14. Testování

### 14.1 Jednotkové a golden testy

- Syntetická data se známým výsledkem: konstantní výnos, split, dividenda, delisting, halt, open = 0.
- Buy-and-hold jednoho titulu v ledger enginu = TR řada titulu (po nákladech na vstup).

### 14.2 Test úniku budoucnosti (povinný)

Pro každou feature, skóre a pro `select`: výsledek pro čas `t` spočtený nad daty oříznutými v `t` musí
být identický s výsledkem nad kompletními daty. Navíc test s perturbací: náhodná změna dat po `t` nesmí
změnit nic do `t` včetně. Testuje se na náhodném vzorku dat `t` a titulů.

### 14.3 Parita enginů

Viz §7.2.

### 14.4 Sanity check dat

Viz §4.6 (rekonstrukce vs. SPY/RSP).

---

## 15. Fáze a brány

**Vstupní podmínka dat:** Sharadar dump na serveru v `data/raw/` je potřeba až od Fáze 1. Fáze 0 se
dělá celá na syntetických datech a nečeká na přenos dat.

| Fáze | Obsah | Brána (podmínka pokračování) |
|---|---|---|
| **0 – Základ bez dat** | Repozitář, stack, konfigurační modely, generátor syntetických dat (splity, dividendy, halty, delistingy), vektorový i ledger engine, nákladový model, test úniku, paritní test, statistická knihovna (DSR, PBO, bootstrap), registr pokusů, mechanika vaultu, kostra reportu | Golden testy, test úniku a parita enginů zelené na syntetických datech |
| **1 – Data a audit** | raw → Parquet, datový audit (§4.6), normalizace, univerzum, politika delistingu, benchmarky, sanity check SPY, uzavření [A], zmrazení [N] do `frozen_defaults.yaml` | Audit bez nevysvětlených anomálií; sanity check SPY v toleranci; parita enginů na reálných datech |
| **2 – Strategie a matice R** | Bloky, 6 rodin, grid, vektorový běh, metriky, PBO | Aspoň jedna rodina má medián Sharpe nad T-bill > 0 napříč okolím; PBO ≤ 0,30 |
| **3 – WFO a ansámbl** | `select`, filtry, ranking, dedup, ansámbl, poskládaný OOS, DSR, reporty | DSR ≥ 0,90 **a** dolní hranice CI rozdílu vůči `EW_UNIV` > 0. Jinak STOP s negativním reportem (Fáze 5 se spustí s negativním závěrem) |
| **4 – ML overlay** (volitelná) | Features, labely, modely, baseline overlaye | §11.5 |
| **5 – Finální holdout** | Zmrazení metodiky, jednorázový běh vaultu, závěrečný report | – |
| **6 – Forward test** (volitelná, mimo v1) | Denní generování příkazů z externího EOD zdroje | Vyžaduje nový datový zdroj; Sharadar se neaktualizuje |

Bez ohledu na výsledek fáze 3 má smysl pustit fázi 5 pro nejlepší pravidlovou metodiku: negativní
holdout výsledek je informace, ne ztráta času.

---

## 16. Akceptační kritéria v1

1. Datový audit existuje pro aktuální snapshot a běh se zastaví, pokud chybí pole, na kterém závisí.
2. Sanity check rekonstruovaného indexu vs. SPY je v toleranci.
3. Univerzum je point-in-time, obsahuje delistované tituly a nepoužívá snapshot pole z TICKERS (§4.3 P3).
4. Politika delistingu je konfigurovatelná a citlivostní analýza je součástí reportu.
5. Signál z close `t` nemůže být vyplněn dříve než na open `t+1`; výnos dne `t+1` je rozdělen na
   overnight a intradenní část podle §7.1.
6. Test úniku budoucnosti prochází pro všechny features, skóre a `select`.
7. Vektorový a ledger engine jsou v paritě.
8. Náklady závislé na likviditě jsou v primárních výsledcích; citlivost ×2/×3 v reportu.
9. Výběr strategií probíhá výhradně uvnitř `select(T, R[..T])`; registr pokusů je úplný.
10. Report obsahuje DSR, PBO, `N`, `N_eff`, `N_meth` a bootstrap CI.
11. Holdout je chráněn vaultem a v logu je nejvýše jeden finální přístup na snapshot.
12. Každý běh je reprodukovatelný z manifestu (snapshot, commit, config hash, seed, lock).
13. Report vždy obsahuje metodiku a benchmarky `SPY_TR`, `EW_UNIV`, `BH_UNIV` ve stejném grafu.

---

## 17. Otevřené body

Uzavřou se ve Fázi 1 (vše, co je výše označeno [A] nebo [N]):

- chování `closeadj` v ex-div den (spinoffy potvrzeny, viz `DATA_FINDINGS.md`),
- stabilita `category` (Primary/Secondary Class je snapshot) v čase,
- typy delistingů v ACTIONS a jejich mapování na §4.5,
- finální mřížka parametrů a počet kandidátů podle rychlosti enginu na dostupném HW,
- případná úprava hranice holdoutu podle koncového data dat.

Doladí se ve Fázi 2 (před implementací rodin):

- `ts_trend` a `breakout` mají stavové držení (vstup/výstup), které se nevejde do čistého
  „skóre → top-N". Vektorový engine pro ně potřebuje stavový krok (stav pozice per titul), který
  musí projít testem úniku stejně jako skóre.
- Výkon vektorového enginu na dostupném HW určí reálnou velikost mřížky.

## 19. Přehled navržených hodnot [N]

Hodnoty mají čtyři druhy původu. Jen poslední skupina je věc preference; ostatní určují data nebo
zavedená praxe. Podstatné je, že se **zmrazí dřív, než se uvidí výsledky strategií**. Předem pevný
a „dost dobrý" práh má větší hodnotu než „optimální" práh zvolený po pohledu na výsledky.

**Určí je data (audit, Fáze 1)**

| Hodnota | Návrh | Co ji může změnit |
|---|---|---|
| Warm-up | 1998–1999 | Skutečný začátek pokrytí |
| Vývojové období | 2000–2019 | Kvalita dat na začátku (pokud je 1998–2001 řídké, začátek se posune) |
| Holdout | 2020 → konec dat | Koncové datum dumpu; cíl je holdout alespoň 5 let |
| Kategorie akcií, typy delistingů, VIX | viz §4.2, §4.5 | Skutečné hodnoty v tabulkách |
| Tolerance sanity checku SPY | ~1 p.b. ročně | Pokud rozdíl vysvětlí dokumentovaná vlastnost dat |

**Zavedená praxe (měnit jen s důvodem)**

| Hodnota | Návrh | Zdůvodnění |
|---|---|---|
| Min. cena | 5 USD (neupravená) | Běžný filtr penny stocks v akademické literatuře |
| Univerzum | top 1000 podle 63d mediánu dollar volume | Likvidní trh, nezávislé na inflaci, vejde se do paměti |
| Delisting bez údaje | −30 % | Shumway (1997) |
| Náklady | max(5 bps, polovina odhadnutého spreadu), citlivost ×2/×3 | Konzervativní pro likvidní US akcie |
| WFO | expanding, min. 5 let, refit a test po 12 měsících, 15 foldů | Dost foldů na stabilitu, dost dat na výběr |
| Embargo | 21 obchodních dní | Nejdelší label v ML (≈ 1 měsíc) |
| Dedup | korelace výnosů ≥ 0,85 = duplicita | Běžný řez; citlivost ověřit na 0,80/0,90 |
| Úrok z hotovosti | 3M T-bill (FRED `DTB3`) | Standard |

**Parametry strategií (mřížka)**

| Blok | Hodnoty | Počet |
|---|---|---|
| Signál | xs_momentum 6, ts_trend 3, breakout 6, st_reversal 3, low_vol 3 | 21 |
| Počet titulů N | 10, 20, 30, 50 | 4 |
| Vážení | EW, inverse-vol | 2 |
| Rebalance | týdně, měsíčně | 2 |
| Hystereze | 1; 1,5; 2 | 3 |
| Overlay | žádný, trend filtr, vol targeting | 3 |
| **Celkem** | 21 × 144 + 3 (regime_market) | **≈ 3 000 kandidátů** |

Rozsahy odpovídají běžně publikovaným variantám (např. momentum 12-1 měsíců, SMA 200). Záměrně nejsou
jemné: hustší mřížka zvyšuje počet pokusů, ale nepřidává informaci.

**Preference a brány (tady je tvůj hlas nejvíc relevantní)**

| Hodnota | Návrh | Otázka, kterou odpovídá |
|---|---|---|
| Max. drawdown kandidáta | **45 % [D]** | Rozhodnutí uživatele; důsledky v §9.4 |
| Max. roční obrat | 2000 % | Pojistka proti strategiím žijícím z nákladového modelu |
| Min. průměrný počet pozic | 8 | Pojistka proti koncentraci |
| Ansámbl | **K = 2 (50/50) primárně, K = 3 sekundárně [D]** | Rozhodnutí uživatele; obě varianty předem registrované |
| Brána Fáze 2 | PBO ≤ 0,30 | Jak velkou pravděpodobnost přeučení tolerujeme |
| Brána Fáze 3 | DSR ≥ 0,90 a CI rozdílu vůči `EW_UNIV` > 0 | Jak silný důkaz chceme, než pokračujeme |
| ML brána | §11.5 | Totéž pro ML |

## 18. Rozhodovací log

| Datum | Rozhodnutí | Zdůvodnění |
|---|---|---|
| 2026-09-25 | Čistý výzkum, bez daní a konkrétního brokera | Rozhodnutí uživatele |
| 2026-09-25 | Pár + ML → ansámbl K strategií, ML jako volitelná fáze za bránou (varianta 1) | Méně multiple testingu; ML bez selection bias; výsledek i při selhání ML |
| 2026-09-25 | Sharadar Bundle Full History, statický dump na jiném stroji | Rozhodnutí uživatele; forward test mimo v1 |
| 2026-09-25 | Vývoj na současném serveru, návrh škálovatelný | Rozhodnutí uživatele |
| 2026-09-25 | ETF mimo univerzum výběru | Pákové/inverzní ETF by zkreslily cross-sectional rankingy |
| 2026-09-25 | Benchmark rozšířen o `SPY_TR` a `EW_UNIV`; `BH_UNIV` zachován | Různé otázky potřebují různé benchmarky; SPY slouží i jako kontrola dat |
| 2026-09-25 | DSR metodiky se deflatuje počtem variant metodiky (`N_meth`), ne počtem kandidátů | Poskládané WFO OOS je mimo vzorek vůči výběru kandidátů; oprava chyby ve v2.0 |
| 2026-09-25 | Projekt v `ccode/strategy_backtester_2026_sep/`, repo `trading_system`, branch `strategy_backtester_2026_sep`; `/home/kapo` bez gitu | Rozhodnutí uživatele |
| 2026-09-25 | Fáze 0 bez reálných dat; Sharadar dump je vstupní podmínkou až Fáze 1 | Práce nečeká na přenos dat; engine se nejdřív ověří na syntetice |
| 2026-09-25 | Max. drawdown kandidáta 45 % | Rozhodnutí uživatele; upřednostní strategie s řízením rizika |
| 2026-09-25 | Ansámbl: K = 2 primárně, K = 3 sekundárně, obě předem registrované | Rozhodnutí uživatele; zkoušet K = 3 až po výsledcích K = 2 by byl pokus podmíněný výsledkem |
| 2026-09-26 | Stažen kompletní Sharadar bundle (snapshot 2026-09-25) + FRED `DTB3`; předběžná zjištění v `DATA_FINDINGS.md` | Předplatné končí; data dostupná dřív, než je Fáze 1 potřebuje |
| 2026-09-26 | Univerzum bez SPAC a sekundárních tříd; delisting akvizic podle skutečné protihodnoty; likvidace SPAC ≠ bankrot | Zjištění z dat (P10–P13) |
| 2026-09-26 | Náklady podle pořadí likvidity (tiers) místo odhadu spreadu z OHLC | Estimátor Abdi–Ranaldo je pro likvidní tituly šum (ověřeno simulací i na AAPL/MSFT) |
| 2026-09-26 | Kalendář = data SEP, bez `exchange_calendars` | Audit: shoda se SPY na všech 7 228 dnech |
| 2026-09-26 | Výchozí hodnoty zmrazeny v `configs/frozen_defaults.yaml` (konec Fáze 1) | Spec §15; změny od teď = nový pokus v registru |
| 2026-09-26 | Rebalance po prvním obchodním dni týdne/měsíce (místo posledního) | Test úniku: poslední den periody vyžaduje znát zítřejší datum |
| 2026-09-26 | SPY v panelu jako neuniverzové aktivum s nejvyšší likviditní úrovní nákladů | Regime strategie a trend overlay; bez pořadí likvidity dostával nejvyšší sazbu |
| 2026-09-26 | Vol-target overlay: cíl 15 % p.a. na realizované 63d volatilitě SPY; trend filtr SPY nad SMA200 | Upřesnění §6.1 (spec neurčovala parametry) |
| 2026-09-26 | Brána Fáze 3: CI se počítá pro rozdíl **Sharpe** metodika − EW_UNIV (90 %, stationary bootstrap, blok 21 d); rozdíl CAGR jen reportován. DSR metodiky: N = N_meth z registru, rozptyl SR nulové hypotézy 1/T | Zapsáno před prvním WFO během |
| 2026-09-26 | Ansámbl: členové = nejlepší reprezentanti různých korelačních shluků mezi kandidáty, kteří projdou filtry; každý 1/K kapitálu, chybějící člen = hotovost. Pozice členů se sčítají (netují) a portfolio běží jedním enginem; změna členů na hranici foldu stojí plné náklady na čistý rozdíl pozic | Upřesnění §9.3/§9.5 před během |
