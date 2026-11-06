# Výzkum 21 – mean reversion na jednotlivých akciích a proxy filtr „šum, nebo informace?“ (pre-registrace)

Verze 1.1 · 2026-11-06 · Stav: **schváleno vlastníkem; nic není spočítáno; čeká na implementaci (§8)**.
Podle CLAUDE.md platí po commitu verze 1.0: změny jen řádkem v §11 s důvodem, **před** dalším vyhodnocením.
Značka **[R]** = rozhodnutí autora návrhu tam, kde zadání (Trading systém: MR + TF + AI vrstva) mlčí; vlastník ji potvrdí nebo změní.
Značka **[ověřit]** = předpoklad o datech nebo kódu, který jsem při psaní nekontroloval; ověří se ve fázi F0.

## 1. Otázka a poctivá omezení předem

**Otázka.** Pokud Connors-styl reversal (RSI(2) < 10 nad SMA200) na likvidních US akciích rozdělíme proxy filtry na
„pravděpodobný šum“ (povolit) a „pravděpodobná informace“ (blokovat), vydělávají povolené vstupy po nákladech
systematicky víc než blokované? To je jádro AI vrstvy ze zadání, otestované zatím bez LLM (zadání: fáze 4, „proxy filtr“).

**Co od studie nečekat.**
- **Prior je negativní.** Základní pravidlo na akciích už bylo zamítnuto: výzkum 5 (z-skóre reversal: vývoj Sharpe 1,02,
  validace 0,32, DSR 0,52 při N = 993; IBS < 0,2: Sharpe −0,41 po nákladech; RSI(2) < 10: 0,43), výzkumy 6 a 7
  (reversal jen při stresu, po započtení trhu bez přínosu vůči SPY) a lab K-0079 (past-return sorty v likvidních akciích
  jsou pod náklady). Nejpravděpodobnější výsledek: S0 je slabý a filtry ho nezachrání.
- **Žádný čistý holdout.** Celá historie (Sharadar do 2026-09-25) je spotřebovaná výzkumy 1–20 i předtrénováním LLM.
  Studie je **post hoc odhad velikosti efektu**, ne potvrzení. Potvrdit může jen forward test (§9).
- **Co studie přinese i při negativu:** (a) odhad, zda proxy filtry vůbec rozdělují vstupy na dvě různé populace,
  (b) rozhodnutí, zda má smysl LLM úroveň 2 (zadání: nemá, pokud proxy nic nepřidá), (c) pevná pravidla pro forward.

## 2. Data

- **Akcie:** Sharadar `sharadar_2026-09-25`, panel `panel_r5` (průběžně survivorship-free, delistingy se zaplatí na openu, viz CLAUDE.md).
  Univerzum v čase *t*: nekorigovaná cena ≥ 5 USD, ≥ 252 barů, ne SPAC a **ADV20 ≥ 50 mil. USD** **[R]** (large capy; výzkum 5
  měl nejlepší výnos v nejlikvidnějším kvintilu). Období simulace: 1999-01-04 → 2026-09-25.
- **Sektory:** pole `sector` z `tickers.parquet` (11 sektorů) → sektorový ETF: Technology XLK, Healthcare XLV, Financial Services XLF,
  Energy XLE, Industrials XLI, Basic Materials XLB, Consumer Cyclical XLY, Consumer Defensive XLP, Utilities XLU, Real Estate VNQ.
  Communication Services nemá ETF před 2018, takže bez sektorového filtru (filtr I se na ně nevztahuje, viz §4.4). **Omezení:** pole `sector`
  je stav k datu snapshotu, ne point-in-time (firma mohla sektor změnit). Zkreslení je malé, ale je to skutečné look-ahead v klasifikaci.
- **Sektorové ETF:** panel `panel_r19_etf` (výzkum 19). **Ověřeno ve F0:** stejný kalendář jako `panel_r5` (7 228 dnů, 1997-12-31 → 2026-09-25, 100 % shoda).
- **Earnings:** `extra_earn8k` v `panel_r5` (**ověřeno ve F0**: bool matice 7 228 × 8 866, 0,5 % buněk True): 8-K Item 2.02 podaný v daný obchodní den
  (víkendové podání přiřazeno následujícímu dni, viz `scripts/r5_build_panel.py`). Je to **datum podání, ne reakční den** (podání po close = reakce až další den;
  čas přijetí v panelu není). Okno ±2 dny to pokrývá. Žádný export z labu není potřeba.
- **Objem a gap (ověřeno ve F0):** panel nemá surový objem akcií, má `dollar_volume`. Abnormální objem se proto měří jako dollar_volume_t / průměr dollar_volume za předchozích 20 dní **[R]**
  (cenový efekt se v poměru nevykrátí, ale pro práh 3× je to přijatelná aproximace). Gap = `ret_co` (close→open, total return, dividendy zanedbatelné). Matice `volratio20`,
  `gapz60` se stavějí v `scripts/r21_build_panel.py`.

## 3. Základní pravidlo S0 (kontrola, pevné)

| Prvek | Pravidlo |
|---|---|
| Signál | RSI(2, Wilder) v *t* < 10 **a** cena (TR index) > SMA(200) v *t* |
| Plnění | rozhodnutí po close *t*, **vstup na open *t*+1** (primární, konzervativní); MOC jen jako robustnost r1 |
| Výstup | close *t'* > SMA(5) → výstup na open *t'*+1; jinak časový stop po 10 obchodních dnech |
| Sizing | rovný podíl NAV / 10, max. 10 pozic, při přebytku kandidátů nižší RSI(2) má přednost, remízy dle tickeru |
| Náklady | model nákladů výzkumu 5 (`qlab.research5.costs`, tiery podle ADV); varianta ×2 v kritériích |
| Short | ne (long-only) |

Parametry (10, 2, 200, 5, 10 dní, 10 pozic) jsou z literatury a z H2 výzkumu 19 a **neladí se**.

## 4. Proxy filtry (každý rozděluje vstupy S0 na „povolené“ a „blokované“)

Rozdělení je definováno jen z dat do close *t* (point-in-time kontrakt qlab), kromě E s poznámkou níže.

| Filtr | Blokuje vstup („pravděpodobná informace“), když … | Povoluje („pravděpodobný šum“), když … |
|---|---|---|
| **E – earnings** | reakční den výsledků (8-K 2.02) leží v okně *t*−2 … *t*+2 | jinak |
| **V – abnormální objem** | dollar_volume v *t* / průměr dollar_volume za 20 dní předtím > 3 | ≤ 3 |
| **G – gap** | gap *t* (open_t / close_{t−1} − 1), standardizovaný σ gapů za 60 dní, < −2 | ≥ −2 |
| **I – idiosynkratický pohyb** | výnos akcie za *t* minus výnos sektorového ETF za *t*, děleno σ rozdílu za 60 dní, < −2 | ≥ −2 (pokles sdílený se sektorem) |

- **E a look-ahead.** Okno *t*+1 … *t*+2 používá skutečné datum výsledků, které by živý systém znal z kalendáře firmy, ne z 8-K. Je to **mírný
  look-ahead** (datum bývá oznámené týdny předem, ale ne vždy). Proto je primární E1 okno *t*−2 … *t*+2 (jako zadání), **robustnost r2** používá
  jen striktně point-in-time okno *t*−2 … *t*. Obě se registrují.
- Pokud filtr nelze spočítat (chybí sektor, chybí ETF, málo historie), vstup se **klasifikuje jako „blokovaný“ pro daný filtr**
  (konzervativní; a hlásí se podíl).
- **C – kombinace** = povolen jen vstup, který povolí všechny čtyři filtry (pre-registrovaná úroveň 1 „proxy“ ze zadání).

## 5. Hypotézy a počet pokusů

| ID | Co se testuje |
|---|---|
| S0 | základní pravidlo (kontrola, nepočítá se jako hypotéza, ale registruje se) |
| HE, HV, HG, HI | spread povolené − blokované pro jednotlivý filtr |
| HC | spread kombinace C |

**N = 6 primárních konfigurací** (S0 + 4 filtry + C). Robustnost (r1–r5, §7) se registruje také a dělá se pro každou, ale **nikdy se z ní nevybírá**.
Registr: `runs/registry/research21.sqlite`; DSR se hlásí pro N = 6 a pro kumulativní počet pokusů na akciích **[R]** (výzkum 5: N = 993).

## 6. Kritéria (mechanická, pevná)

Pro každý filtr F ∈ {E, V, G, I, C} musí platit **všech 5**, aby byl verdikt „náznak → forward“:

1. **Spread.** Průměrný čistý výnos obchodu (povolené − blokované) > 0, dolní mez 90% intervalu (stacionární block bootstrap po měsících) > 0.
2. **Povolená populace vydělává.** Průměrný čistý obchod povolených vstupů > 0 po nákladech a **nad p95 náhodných vstupů** (kontrola c2: stejné dny a počty,
   náhodní kandidáti ze stejného univerza, stejné držení a náklady).
3. **Stabilita v čase.** Spread kladný v každé ze tří stejných třetin historie (1999–2007, 2008–2016, 2017–2026). Lab opakovaně našel úpadek efektů v poslední třetině.
4. **Náklady ×2.** Průměrný čistý obchod povolených vstupů > 0.
5. **Podíl.** Blokované i povolené vstupy tvoří každé ≥ 15 % obchodů S0 (jinak filtr nic nerozděluje a verdikt je „nelze posoudit“).

**S0 (kontrola)** se hlásí popisně: průměrný čistý obchod s CI, Sharpe, CAGR, max. DD, úspěšnost, držení, čas v trhu, obrat, SPY jako benchmark.
Navíc: **kontrola C0 = vždy long v režimu** (cena > SMA200, rovné váhy 10 pozic), aby bylo vidět, kolik dělá reversal a kolik jen drift akcií.

**Rozhodnutí.** (a) Žádný filtr neprojde → proxy úroveň nepřidává informaci, **LLM klasifikátor (úroveň 2) se podle zadání nestaví nad tímto signálem**; logger zpráv
běží dál (data jsou užitečná i pro jiné hypotézy). (b) Některý projde → kandidát na forward (§9), ne na živé obchodování.

## 7. Robustnost (jen hlášení)

r1 plnění MOC (horní mez) · r2 E bez budoucího okna (*t*−2 … *t*) · r3 prahy 2σ → 1,5σ a 3σ u G/I, objem 2× a 4× · r4 ADV ≥ 20 mil. a ≥ 100 mil. ·
r5 bez SMA200 · r6 bez let 2000–2002 a 2008–2009 (závislost na krizích) · r7 náklady ×3. Tabulka v reportu: průměrný obchod a Sharpe každé varianty.

## 8. Implementace (pořadí je závazné, ze stylu výzkumu 19)

- **F0 – orientace (bez výpočtů): provedeno 2026-11-06** (§11). Zbývá ověřit mapování `assets` → `permaticker` → `sector` a podíl akcií bez sektoru.
- **F1 – schválení a commit pre-registrace v1.0.**
- **F2 – data:** `scripts/r21_build_panel.py` (matice `volratio20`, `gapz60`, `ea_reaction_day`, sektor→ETF, idiosynkratické z), test point-in-time (`check_point_in_time`).
- **F3 – signály a filtry:** `src/qlab/research21/{setup,signals,filters,strategy,evaluate}.py`. S0 přes `research5.sim.simulate` (opakované použití, bez změny chování).
  Testy: pokaždé point-in-time, filtr E1 vs. r2 (rozdíl jen ve výhledu), rozdělení povolené + blokované = všechny vstupy S0 (nic se neztratí ani nezdvojí).
- **F4 – kontroly:** c2 (náhodné vstupy), C0 (always long), bootstrap, třetiny, DSR.
- **F5 – smoke a zámek:** smoke run do scratch registru (neznečistit N), pak `run_lock.json` (hash pravidel). **Jeden plný běh.**
- **F6 – report:** `REPORT.md`, řádek do README, verdikt do §11.

**Stop-body (zastavit a zeptat se vlastníka):** 1) pokud mapování sektorů nebo něco jiného z F0 neplatí; 2) pokud S0 ve smoke runu vyjde výrazně mimo výzkum 5/19
(např. průměrný čistý obchod < −0,5 % nebo > +1 %): nejdřív hledat chybu, ne měnit pravidla; 3) před plným během.

## 9. Forward (navazuje na news logger)

Pokud některý filtr projde: pravidla se zmrazí a přidají do forward testu vedle F1/F2 z výzkumu 20 (paper, bez ladění). News logger (`/srv/newslog`) sbírá
od 2026-11-05 titulky s `first_seen_at`; až bude dost dat (řádově měsíce), lze ke každému vstupu přiřadit zprávy z okna a teprve pak má smysl LLM úroveň 2,
**měřená jako přírůstek nad proxy filtrem na stejném období** (zadání, §Validace).

## 10. Co záměrně není v této studii

Žádné další MR varianty (IBS, N-day low, z-score, short-term reversal: výzkum 5 je pokryl a zamítl), žádná alokace s TF (H4 výzkumu 19 neprošla, počká na forward),
žádný LLM, žádná krypta (BTC MR je mimo rozsah; zadání ji řeší jen jako srovnání).

## 11. Deník rozhodnutí

| Datum | Verze | Rozhodnutí | Důvod |
|---|---|---|---|
| 2026-11-06 | 0.9 | Návrh: S0 + filtry E, V, G, I, C; N = 6; bez dalších MR variant | vlastník zvolil variantu B (MR na akciích + proxy filtr) po zjištění, že samotné MR na akciích je opakovaně zamítnuté |
| 2026-11-06 | 0.9 | F0 provedeno: kalendáře `panel_r5` a `panel_r19_etf` identické; `extra_earn8k` v panelu (datum podání); objem jen jako `dollar_volume`; gap z `ret_co` | stop-bod se zdrojem 8-K odpadl, `volratio20` je z dollar_volume [R] |
| 2026-11-06 | 1.0 | Schváleno vlastníkem beze změn, commit před jakýmkoli výpočtem | postup podle CLAUDE.md |
| 2026-11-06 | 1.1 | **Filtr E a kombinace C se hodnotí jen na vstupech se signálem od 2005-01-01** (obě skupiny, povolené i blokované, tedy spread je srovnání na stejném období). Třetiny pro E a C: 2005–2011, 2012–2018, 2019–2026. Pro S0, V, G, I platí celá historie 1999-01-04 → 2026-09-25 a třetiny 1999–2007, 2008–2016, 2017–2026. Podíl ≥ 15 % (kritérium 5) se počítá na období daného filtru. | F0 (druhá část): `extra_earn8k` je prázdný do 2003 a v 2004 částečný (4 286 podání; od 2005 ~14 000 ročně), protože Item 2.02 vznikl až 2003. Pravidlo „nelze spočítat → blokovat“ by zablokovalo celých 1999–2004. Volba vlastníka. N zůstává 6. |
| 2026-11-06 | 1.1 | **Upřesnění operacionalizace (ne změna pravidel):** (a) σ u G, I a průměr u V se počítají z 60, resp. 20 obchodních dnů **končících v *t*−1** (událost sama se do vlastního jmenovatele nepočítá); (b) požadavek na historii: aspoň 40 platných z posledních 60 dnů u G a I, aspoň 15 z 20 u V, jinak „nelze spočítat → blokovat“; (c) okno E je v indexech obchodních dnů, E1 = [*t*−2, *t*+2], r2 = [*t*−2, *t*]; (d) idiosynkratický výnos = výnos akcie za *t* (total return, (1+ret_co)(1+ret_oc)−1) minus výnos sektorového ETF za *t*; (e) kontrola c2 pro filtr F = pro každý obchod povolené skupiny jeden náhodný obchod ve stejný signální den, náhodná akcie z univerza s cenou > SMA200, stejné držení a náklady, 1 000 simulací; (f) bootstrap intervalu: stacionární block bootstrap přes kalendářní měsíce vstupu, průměrný blok 3 měsíce, 1 000 vzorků, seed 21, 90% interval (5.–95. percentil). | Operacionalizace, kterou pre-registrace nechala otevřenou. Dělá se před jakýmkoli výpočtem. |
