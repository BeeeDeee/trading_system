# Výzkum 19 – trend ensemble, RSI(2) a overlaye jako portfolio streamů (implementační plán)

Verze plánu 1.0 · 2026-11-01 · Autor plánu: Claude Opus 5.5 · **Implementuje slabší model podle tohoto souboru.**
Zdroj: rešerše vlastníka „Co skutečně používají dlouhodobě ziskoví tradeři…“, doporučení Priorita 1, 2 a 3.

Tento soubor je **specifikace i postup**. Implementující model nesmí nic z §4–§7 měnit, domýšlet ani
„vylepšovat“. Kde plán něco neříká, implementující model se **zastaví a zeptá vlastníka** (seznam stop-bodů v §9).

---

## 0. Pro implementující model: jak s tímto souborem pracovat

1. Nejdřív si přečti `CLAUDE.md` a `AGENTS.md` v kořeni repa. Platí beze změny (Czech docs, English code,
   žádná data v gitu, registr pokusů, bod v čase = rozhodnutí po close *t*, plnění na open *t+1*, pokud §4 neříká jinak).
2. Pracuj **po fázích v pořadí §8**. Každá fáze má akceptační kritéria. Další fázi začni až ve chvíli, kdy předchozí
   prošla a je commitnutá (lokálně; push jen na pokyn vlastníka).
3. Čísla v §4–§7 jsou **pevná**. Pokud ti test nebo výsledek nedává smysl, neměň parametr. Zapiš problém a zastav se.
4. Nikdy nespouštěj plný běh (`r19_run.py full`) dřív, než je commitnutá pre-registrace, kód a zámek (§8, fáze F6).
5. Při smoke bězích přesměruj registr do scratch cesty (monkeypatch `setup.REGISTRY`), jinak se nafoukne N.
6. Odpovědi vlastníkovi piš česky, stručně, bez přikrášlování.

---

## 1. Co jsem z rešerše převzal a co je potřeba vědět předem (pro vlastníka)

**Rešerše je solidní v tom hlavním:** edge na denních datech má smysl hledat jen v pomalém trendu, v krátkodobém
mean reversion indexů jako placené likviditě a v diverzifikaci přes nekorelované streamy. Dále že backtest se má
škálovat dolů o 50–60 %.

**Co rešerše neví a co mění očekávání:** část doporučení už v tomto repu padla.

| Doporučení rešerše | Co už bylo testováno | Co je v tomto výzkumu nové |
|---|---|---|
| P1 trend na koši ETF | **Výzkum 2** (9 ETF, inv. volatilita + SMA200): Sharpe 0,52 vs 60/40 0,85, zamítnuto. **Výzkum 10B** (TSMOM na 9 ETF jako rukáv k SPY): zamítnuto. | Donchian ensemble 50–250 dní místo jednoho SMA a širší koš (16 ETF). **Stejná rodina, čekám stejný verdikt vůči 60/40.** Proto se H1b nesoudí proti 60/40, ale jen jako diverzifikátor uvnitř H4. |
| P1 krypto trend | **Výzkum 9** (SMA/momentum filtr BTC+ETH): ΔSharpe +0,02, zamítnuto. Altcoiny rovné váhy 2022–26: −45 % p. a. | Ensemble 9 Donchianů s trailing stopem, vol sizing a rotace top 20. Tohle tu ještě nebylo. **Nejsilnější kandidát.** |
| P2 RSI(2) na indexových ETF | **Výzkum 5** (STR-TF, reversal s trendovým filtrem na **jednotlivých akciích**): validace Sharpe 0,32, 53 % zisku z let 1999–2002. | Indexová a sektorová ETF, literární pravidla bez ladění, vstup MOC. Nové, ale výzkum 5 varuje před koncentrací zisku do 1999–2002, proto je kritériem povinný post-publikační úsek 2009+. |
| P3 overlaye (TOM, vol targeting, crash filtr) | Nic přímo. Lekce z výzkumu 1 a 9: časování podle zpožděných signálů škodí při rychlých odrazech. | Vše nové, ale každý overlay je jeden pokus navíc a čekám, že většina neprojde. |

**Kontaminace dat (nejdůležitější omezení):** veškerá historie (Sharadar do 2026-09-25, Binance do 2026-08-31) už byla
použita ve výzkumech 1–18 a je i v tréninkových datech LLM. Navíc vzorek Zarattiniho studie (do 2025-03) se s našimi
krypto daty překrývá celý. **Tento výzkum proto nemá čistý holdout.** Je to *replikace literárních pravidel s pevnými
parametry* (N je malé a parametry neladíme), vyhodnocená jednou na celé historii. Informativní jsou hlavně
**post-publikační úseky** (RSI(2) po 2008, TOM po 2001, trend po 2009) a **forward test**. Ten je jediný čistý důkaz
a spouští se jen pro to, co projde (§7.6).

**Realistické očekávání:** H1a (krypto) má šanci projít. H2 (RSI(2)) možná projde na celém období a bude slabá po 2009.
H1b a overlaye nejspíš neprojdou. H4 (kombinace) skoro jistě nepřekoná 60/40 na Sharpe, protože 60/40 bylo v letech
2010–2026 výjimečné (výzkum 2). Hodnota výzkumu je v tom, že dostaneš **poctivě změřené streamy a jejich korelace**,
ne v tom, že se najde nový stroj na peníze.

**Pravidla Zarattiniho studie:** plné PDF (SSRN 5209907) jsem neměl k dispozici. Pravidla níže vycházejí z abstraktu
a ze souhrnu CXO Advisory (lookbacky 5–360, vstup při close nad maximem close, trailing stop = max(předchozí stop,
střed kanálu), velikost = 25 % / σ90, páka ≤ 200 %, náklady 0,10/0,25/0,50 %). Kde zdroj mlčí (definice top 20,
složení portfolia, pásmo rebalance), rozhodl jsem já. Rozhodnutí jsou v §4.1 označená **[R]**. Pokud vlastník PDF
dodá a liší se, mění se pre-registrace **před** během, ne po něm.

---

## 2. Architektura: co se znovu použije a co vznikne

Vše se staví v `qlab` (tento repozitář). Research Lab se nepoužije, protože ten je pro LLM-generované hypotézy.
Tyto tři strategie jsou lidsky pre-registrovaná studie stejného typu jako výzkumy 1–18.

### 2.1 Znovupoužité (neměnit chování)

| Co | Kde | K čemu |
|---|---|---|
| Krypto denní panel spot USDT párů vč. delistingů | `data/derived/binance_2026-10-03/panel_r9` (`qlab.research9.setup.panel_dir`, `symbols`) | H1a. Zavřené ceny, `extra_qv` (quote volume), delisting s haircutem −2 % |
| Filtr symbolů (bez stablecoinů, pákových tokenů) | `qlab.research9.data.tradable_spot` (panel už je jím filtrovaný) | – |
| Univerzum top-N podle 30denního průměru quote volume | `qlab.research9.strategy.universe(p, qv, t, n)` | H1a, rozšířit o podmínku stáří (§4.1) v novém kódu, ne úpravou této funkce |
| Krypto náklady podle likvidity (10 bps fee + 2–25 bps slippage) | `qlab.research9.panel.cost_rate(qv, mult)` | H1a primární náklady |
| ETF panel builder (Sharadar funds → Panel, T-bill) | `scripts/r2_build_panel.py` | vzor pro nový `scripts/r19_build_etf_panel.py` |
| Vektorový engine (cílové váhy → denní výnosy) | `qlab.engine.vector.simulate` | H1a, H1b; rozšíří se o `band` (F2) |
| Event-driven engine (pozice s pevnou velikostí, exit signál, časový stop, MOC vstup) | `qlab.research5.sim.simulate`, `SimSpec`, `Candidates` | H2; rozšíří se o `exit_at_close` (F2) |
| Náhodný benchmark pro event engine | `qlab.research5.strategy.random_benchmark` (vzor), `qlab.research13.strategy.random_benchmark` | H2 kontrola c2 |
| Měsíční mix streamů | `qlab.research10.sim.monthly_mix(returns, w, dates)` | H4 |
| TR index | `qlab.features.basic.tr_index(panel)` | všechny signály počítat z TR indexu, ne z `close_u` |
| Statistika | `qlab.validation.stats`: `sharpe`, `deflated_sharpe`, `bootstrap_ci`, `spa_pvalue`, `stationary_bootstrap_indices` | §7 |
| Metriky | `qlab.validation.metrics.summary`, `max_drawdown` | ETF (252 dní). **Krypto anualizuj √365** jako `scripts/r9_final.py` (`ANN = np.sqrt(365)`) |
| Test bodu v čase | `qlab.validation.leakage.check_point_in_time` | test každého signálu |
| Registr pokusů | `qlab.validation.registry.TrialRegistry` | `runs/registry/research19.sqlite` |
| Zámek pravidel | vzor `scripts/r18_forward.py` (`check_lock`, `freeze`) a `qlab.research17.forward.rule_hash` | F6 |

### 2.2 Nové soubory

```
docs/research19/PLAN.md                 # tento soubor
docs/research19/PREREGISTRATION.md      # F1, česky, commit PŘED jakýmkoli výpočtem na reálných datech
docs/research19/REPORT.md               # F7
docs/research19/run_lock.json           # F6, hash pravidel (commitnout)
docs/research19/results/*.json          # F6/F7, commitnuté výsledky (malé JSONy, žádné parquet/csv)
scripts/r19_build_etf_panel.py          # F3
scripts/r19_audit.py                    # F3
scripts/r19_run.py                      # F6: smoke | freeze | full
src/qlab/research19/__init__.py
src/qlab/research19/setup.py            # konstanty, cesty, období, registr (vše z §4–§7)
src/qlab/research19/signals.py          # F4: donchian_state, ensemble, rsi_wilder, sma, realized_vol, tom_window
src/qlab/research19/crypto_trend.py     # F5: H1a (+ BTC-only replikace)
src/qlab/research19/etf_trend.py        # F5: H1b
src/qlab/research19/rsi2.py             # F5: H2 (koš + SPY-only), kontroly c1, c2
src/qlab/research19/overlays.py         # F5: H3a TOM, H3b vol targeting, H3c crash filtr
src/qlab/research19/combine.py          # F5: H4, zarovnání kalendáře krypto → US obchodní dny
src/qlab/research19/evaluate.py         # F5/F6: kritéria §7, bootstrap, null, JSON výstupy
tests/test_research19.py                # F2–F5
```

### 2.3 Úpravy frameworku (F2), obě zpětně kompatibilní

**(a) `qlab.engine.vector.simulate(..., band: float | None = None)`.** Pásmo rebalance, aby vol sizing neobchodoval
každý den drobné rozdíly.
- `band=None` → chování přesně jako dnes (bitově stejné výstupy).
- Jinak: v okamžiku rozhodnutí po close *t* spočti aktuální váhy `w_close = v / nav[t]` (hodnoty po close *t*, ne po
  open *t+1*, aby nevznikl pohled do budoucnosti). Pro každé aktivum rozhodni, zda se obchoduje:
  - obchoduje se, pokud `(target > 0) != (w_close > WEIGHT_TOL)` (vstup nebo úplný výstup), nebo
  - pokud `target > 0` a `abs(w_close - target) > band * target`.
  - Ostatní aktiva se na open *t+1* **neobchodují** a drží svou hodnotu.
- Implementace: v `pending` ulož cílové váhy a masku `trade`; v bloku plnění použij
  `delta = np.where(panel.tradable[t] & trade, pending * nav_open - v, 0.0)`. Zbytek (prodeje první, nákupy škálované
  hotovostí) beze změny.
- Testy: (1) `band=None` a `band=0.0` dávají stejné `nav` jako dosavadní kód na existujících testovacích datech
  (`tests/test_vector_engine.py` musí projít beze změny); (2) ruční příklad se 2 aktivy: cíl 0,50/0,50, drift na
  0,55/0,45, `band=0.2` → žádný obchod; drift na 0,65/0,35 → obchod obou (0,35 je od 0,50 vzdáleno o 30 %);
  (3) přechod cíle na 0 vždy prodá.

**(b) `qlab.research5.sim.SimSpec.exit_at_close: bool = False`.** Connorsova pravidla vystupují na close.
- `False` → beze změny (exit signál a časový stop z close *t* se plní na open následujícího obchodovatelného dne).
- `True` → exit signál a časový stop vyhodnocené na close *t* se plní **na close *t*** za `close` cenu
  (pozice už nedostane `ret_co`/`ret_oc` dne *t+1*). Náklady `exit_cost(t, a)`. Pokud aktivum na close *t*
  není `tradable`, plní se podle stávající logiky na dalším open.
- Pořadí v rámci dne při `exit_at_close=True` a `entry_delay=0`: nejdřív výstupy na close, pak vstupy na close
  (uvolněný slot se smí ten den znovu obsadit jiným aktivem, ale **ne stejným aktivem**, které právě vystoupilo).
- Testy: (1) všechny dosavadní testy `tests/test_research5.py` projdou beze změny; (2) ruční obchod: vstup MOC den 0,
  exit signál na close dne 2 → čistý výnos = (1−c)·TR(0→2)/(1+c) − 1 s c = náklad na stranu; (3) časový stop
  `max_hold=10` → obchod drží přesně 10 close-to-close úseků.

---

## 3. Data

### 3.1 Krypto (H1a)
- Panel `panel_r9` snapshotu `binance_2026-10-03`, 2017-08-01 → 2026-08-31, kalendářní dny UTC.
- Ceny pro signály: `tr = tr_index(panel)`. Okno signálu je platné jen tehdy, když je aktivum `listed` a
  `tradable` ve **všech** dnech okna (kvůli mezerám a znovu-listingům; panel drží TR konstantní, když symbol chybí).
- Náklady: `cost_rate(extra["qv"], mult)`.
- Hotovost: 0 % (USDT bez výnosu). Citlivost: T-bill (stejně jako výzkum 9).

### 3.2 ETF (H1b, H2, H3, benchmarky)
- Nový panel `data/derived/sharadar_2026-09-25/panel_r19_etf/` skriptem `scripts/r19_build_etf_panel.py`
  (kopie `r2_build_panel.py`, jiný seznam tickerů a výstupní adresář, `extra_cash_ret` z FRED DTB3, `tickers.json`).
- Tickery (pevně):
  - H1b koš (16): `SPY QQQ IWM EFA EEM VNQ TLT IEF TIP LQD HYG GLD SLV DBC USO UUP`
  - H2 univerzum (13): `SPY QQQ IWM DIA XLB XLE XLF XLI XLK XLP XLU XLV XLY`
  - benchmark: `SPY IEF` (60/40) – už obsaženo
  - sjednocení = 26 unikátních tickerů. Seznam ulož v `research19/setup.py` jako konstanty `ETF_TREND`, `RSI_UNIVERSE`.
- Ověřené začátky historie v Sharadaru: SPY 1997-12-31, DIA 1998-01-20, sektory XL* 1998-12-22, QQQ 1999-03-10,
  IWM 2000-05-26, EFA 2001-08-27, TLT/IEF/LQD 2002-07-30, EEM 2003-04-14, TIP 2003-12-05, VNQ 2004-09-29,
  GLD 2004-11-18, DBC 2006-02-06, USO 2006-04-10, SLV 2006-04-28, UUP 2007-03-01, HYG 2007-04-11.
- Survivorship: koš je vybraný ex post (všechna ETF dodnes žijí, QQQ je známý vítěz). U širokých indexových ETF je
  zkreslení malé, ale zapiš ho do pre-registrace jako omezení.

### 3.3 Audit (F3, `scripts/r19_audit.py`)
Vypiš a ulož do `docs/research19/results/audit.json`: první/poslední den každého tickeru, počet dní bez
`tradable` po prvním dni, denní TR výnosy |r| > 25 % (ETF) a > 60 % (krypto, jen členové top 20 universa), počet
členů krypto univerza po měsících, seznam všech symbolů, které kdy byly v top 20. **Stop-bod:** pokud je v top 20
stablecoin, tokenizovaná akcie (např. `AAPLBUSDT`) nebo wrapped token, zastav a ukaž seznam vlastníkovi.

---

## 4. Pravidla strategií (pevná)

Společné: rozhodnutí po close *t* jen z dat ≤ *t*. Pokud není řečeno jinak, plnění na open *t+1* (vektorový engine).
`P` = TR index aktiva. Rok = 365 dní pro krypto a 252 dní pro ETF.

### 4.0 Donchianův stav pro jeden lookback L (sdílené H1a a H1b)

Pro aktivum a lookback `L` na close *t* (smyčka přes čas, stav se nese):
```
U_t = max(P[t-L .. t-1])               # předchozích L close, BEZ dne t
M_t = (max(P[t-L+1 .. t]) + min(P[t-L+1 .. t])) / 2    # střed kanálu VČETNĚ dne t
platné okno: P[t-L .. t] všechny platné (listed & tradable), jinak stav := 0 a stop := NaN

if stav == 0:
    if P_t > U_t:  stav = 1;  stop = M_t
else:
    stop = max(stop, M_t)
    if P_t < stop: stav = 0;  stop = NaN
```
Ensemble: `E_t = průměr stavů přes lookbacky` (hodnoty 0, 1/K, …, 1).
Realizovaná volatilita: `σ_t = std(denní TR výnos za posledních 90 dní včetně t, ddof=1) · √(rok)`;
platná, jen pokud je všech 90 výnosů platných, jinak se aktivum nesmí držet (váha 0).

### 4.1 H1a – krypto Donchian ensemble, rotace top 20 (primární)
- Lookbacky: `5, 10, 20, 30, 60, 90, 150, 250, 360` (K = 9).
- Univerzum **[R]**: přepočítává se na close posledního dne každého měsíce (den *t*, kde měsíc(*t+1*) ≠ měsíc(*t*))
  a platí pro všechna rozhodnutí následujícího měsíce, včetně rozhodnutí na close dne přepočtu. Členem je top 20 podle
  `research9.strategy.universe(p, qv, t, 20)` **s dodatečnou podmínkou** stáří: aktivum má ≥ 365 dní `listed` před *t*
  (kandidáti se filtrují před výběrem top 20, ne po něm, tj. napiš vlastní funkci `universe_aged` v `crypto_trend.py`,
  která aplikuje stejnou logiku jako `universe` plus stáří).
- Stav signálů se počítá pro **všechna** aktiva nezávisle na členství (stavový automat běží celou historii).
  Člen s `E_t > 0` v den vstupu do univerza se tedy koupí hned.
- Velikost **[R]**: `s_i = E_i · min(2, 0.25 / σ_i)` (vol cíl 25 %, páka aktiva ≤ 200 %, jak uvádí zdroj).
  Portfolio: `w_i = s_i / 20` pro členy, 0 pro nečleny. Pokud `Σ w_i > 1`, všechny váhy se proporcionálně zmenší na
  součet 1 (spot, bez páky na úrovni portfolia; **odchylka od studie**, zapsat). Průměrná investovanost bude nízká
  (řádově 10–30 %). To je v pořádku: Sharpe vůči nulové hotovosti na škále nezávisí.
- Rozhodnutí každý den (`Decisions` s řádkem každý den od začátku), engine s `band=0.20` **[R]** (zdroj zmiňuje
  pásmo 20 % jako způsob snížení nákladů; přesnou definici neznám, platí §2.3a).
- Náklady: `cost_rate(qv, 1.0)`.
- Období vyhodnocení: **2019-01-01 → 2026-08-31** (před tím rozběh signálů: 360 + 90 dní a dost členů s historií).
  Simulace začíná rozhodnutím na close 2018-12-31 bez držených pozic.
- **H1a-BTC (replikace, sekundární):** jen BTCUSDT, `w = E · min(1, 0.25/σ)`, stejné lookbacky, `band=0.20`, stejné období.

### 4.2 H1b – ETF Donchian ensemble (primární)
- Koš `ETF_TREND` (16 ETF), lookbacky `50, 100, 150, 200, 250` (K = 5). Rychlé lookbacky < 50 se záměrně nepoužívají
  (CFM 2026: rychlý trend na indexech/FX po 2009 nefunguje).
- Velikost: `s_i = E_i · min(2, 0.10 / σ_i)` (vol cíl 10 % na aktivum, σ z 90 dní, √252), `w_i = s_i / 16`,
  pevný jmenovatel 16 i v době, kdy některá ETF ještě nemají historii (jejich slot je v hotovosti). Pokud `Σ w > 1`,
  proporcionálně zmenšit na 1.
- Denní rozhodnutí, `band=0.20`, plnění open *t+1*.
- Náklady: 3 bps na stranu (commission 0), ×2,5 před 2001-04-09 (to se tu neuplatní, start je později).
  Implementace: `CostModel(floor_bps=3.0, tiers=(), default_bps=3.0, pre_decimal_multiplier=2.5).rate(...)` nebo
  konstantní matice; ověř testem, že vychází 0,0003.
- Hotovost: T-bill (`extra_cash_ret`).
- Období vyhodnocení: **2007-07-02 → 2026-09-25** (aby rok 2008 byl celý v testu).

### 4.3 H2 – RSI(2) mean reversion na indexových a sektorových ETF (primární)
- Univerzum `RSI_UNIVERSE` (13 ETF).
- `RSI2` (Wilder, n = 2) z TR indexu: `d = P_t − P_{t−1}`, `g = max(d,0)`, `l = max(−d,0)`; inicializace
  `AG, AL = průměr g, l z prvních 2 dnů`, dál `AG_t = (AG_{t−1}·(n−1) + g_t)/n`, totéž pro `AL`.
  `RSI = 100 − 100/(1 + AG/AL)`; `AL = 0` → `RSI = 100`. Přes mezery v historii se nepočítá (reset).
- `SMA200`, `SMA5` z TR indexu (včetně dne *t*).
- **Vstup:** na close *t*, pokud `P_t > SMA200_t` a `RSI2_t < 10` a aktivum není drženo → kandidát se skóre `RSI2_t`
  (nižší = lepší). Plnění **na close *t*** (MOC, `entry_delay=0`).
- **Výstup:** na close *t*, kde `P_t > SMA5_t` → prodej na close *t* (`exit_at_close=True`).
  Časový stop: po 10 obchodních dnech držení prodej na close (`max_hold=10`). Žádný cenový stop.
- Portfolio: `max_positions=4`, velikost `NAV/4` v okamžiku vstupu, pevná do výstupu (žádný rebalance).
  Při víc kandidátech než volných slotů rozhoduje nejnižší RSI2, při shodě abecední pořadí tickeru.
- `adv_cap`: vypnout (`adv_cap=1e9` nebo `capital` tak, aby cap nikdy nevázal; zkontroluj, že se v `trades` nikde
  neuplatnil).
- Náklady: 3 bps na stranu, ×2,5 před 2001-04-09. Hotovost: T-bill.
- Období vyhodnocení: **1999-11-01 → 2026-09-25**. Podúseky: před publikací knihy **1999-11-01 → 2008-12-31**,
  po publikaci **2009-01-01 → 2026-09-25**.
- **H2-SPY (replikace, sekundární):** jen SPY, `max_positions=1`, velikost 100 % NAV, jinak stejné.

### 4.4 H3 – overlaye (každý je samostatný pokus)
Overlaye se aplikují na **výnosové řady** sleevů (ne na úrovni jednotlivých obchodů), náklady overlaye se počítají
zjednodušeně, jak je uvedeno. To je vědomé zjednodušení, zapsat do pre-registrace.

- **H3a – turn-of-month (TOM).**
  - Okno TOM pro US obchodní dny: poslední obchodní den měsíce (den −1) a první tři obchodní dny dalšího měsíce
    (+1, +2, +3). Definice z kalendáře panelu SPY. Funkce `tom_window(dates) -> bool[T]` je „známá dopředu“
    (kalendář obchodních dnů), to je jediná výjimka z pravidla bodu v čase a je přípustná.
  - (i) **Statistický test efektu:** průměrný denní TR výnos SPY ve dnech TOM minus mimo TOM, období
    **2002-01-02 → 2026-09-25** (primární, po decimalizaci a po Han, Han & Tian 2025), 1998–2001 jen popisně.
    90% interval stacionárním bootstrapem (`mean_block=21`, 10 000 replik) rozdílu průměrů.
  - (ii) **Overlay na H2:** ve dnech TOM se nevyužitá hotovost sleevu H2 drží v SPY:
    `r_H3a(t) = r_H2(t) + (1 − expo_H2(t−1)) · TOM(t) · (r_SPY(t) − r_cash(t)) − náklady`,
    náklady = 3 bps × |změna investované části SPY| ve dnech vstupu do okna a výstupu z něj.
    `expo_H2` je `exposure` z `SimOutput`.
- **H3b – vol targeting celého portfolia** (na řadě H4, §4.5):
  `k_t = min(1, 0.10 / σ20_t)`, kde `σ20_t` je realizovaná vol. denních výnosů H4 za 20 dní do *t* včetně, √252.
  `r_H3b(t+1) = k_t · r_H4(t+1) + (1 − k_t) · r_cash(t+1) − 3 bps · |k_t − k_{t−1}|`. Bez páky (k ≤ 1).
- **H3c – filtr momentum crash** (na trendových sleevech H1a a H1b zvlášť, pak znovu složit H4):
  stav „crash“ na close *t*, když benchmark (BTC pro H1a, SPY pro H1b) má `P_t < SMA200_t` **a**
  `σ20_t > 1.5 · medián(σ20 za posledních 252 dní)`. V crash stavu se cílové váhy sleevu násobí 0,5.
  Implementace na úrovni vah (před enginem), ne výnosů. Očekávání: long-only trend je v medvědím trhu většinou mimo
  trh, takže filtr bude redundantní. Je to pokus, ne doporučení.

### 4.5 H4 – kombinované portfolio streamů (primární)
- Sleevy: A = H1a (krypto), B = H1b (ETF trend), C = H2 (RSI(2) koš). Váhy **1/3 každý**, reset na začátku každého
  měsíce, mezi tím drift (`monthly_mix`).
- Zarovnání kalendáře: krypto má kalendářní dny UTC, ETF obchodní dny US. Denní výnos krypto sleevu přiřazený
  US obchodnímu dni *d* = složený výnos krypto dnů s datem v intervalu (*d_prev*, *d*], kde *d_prev* je předchozí
  US obchodní den. (Krypto close je 00:00 UTC, US close 20:00/21:00 UTC; posun < 1 den je přijatelný pro měsíční mix
  a korelace, zapsat.) Funkce `align_to_calendar(crypto_dates, crypto_ret, us_dates) -> ret_us` v `combine.py`
  s testem na ručním příkladu (víkend se složí do pondělí).
- Náklady rebalance mezi sleevy: 3 bps × |změna vah| při každém měsíčním resetu (zjednodušení).
- Období: **2019-01-02 → 2026-08-31** (průnik dostupnosti).
- Benchmarky: 60/40 SPY/IEF měsíčně, SPY, rovné váhy A/B/C bez krypta (B+C po 1/2) jako popis.

### 4.6 Kontroly (nulové modely) a benchmarky
- **H1a:** BTC HOLD, 50/50 BTC/ETH měsíčně (`research9.strategy.monthly_5050`), rovné váhy stejného top 20
  (`ew_universe_decisions` se stejným univerzem). **Náhodné načasování:** pro každé aktivum se řada `E_i` cyklicky
  posune o náhodný offset v rámci jeho listovaného úseku (σ zůstává skutečná), 500 replik, seed 19.
- **H1b:** rovné váhy 16 ETF (měsíčně), 60/40, SPY. Stejný cyklický posun `E_i`, 500 replik.
- **H2:**
  - c1 = „vždy long, když `P > SMA200`“ na stejném koši (rovné váhy přes aktiva v režimu, denně, open *t+1*). Odpovídá
    na otázku z výzkumu 14, zda je edge v režimu, nebo ve vstupu.
  - c2 = náhodné vstupy: pro každý skutečný obchod náhodný den v **témž aktivu** s `P > SMA200`, stejná délka držení,
    stejný engine. 1 000 replik, seed 19. Statistika = průměrný čistý obchod a Sharpe.
- **H3a (i):** permutační kontrola není potřeba, stačí bootstrap.

---

## 5. Varianty robustnosti (pre-registrované, nikdy se z nich nevybírá)

Všechny se zapíší do registru jako `kind="robustness"` a počítají se do rozptylu Sharpe pro DSR, ale **nesmí nahradit
primární konfiguraci**, ani když vyjdou lépe.

| Hypotéza | Varianta |
|---|---|
| H1a | `band=None`; náklady 25 bps/stranu konstantně; náklady ×2 (`cost_rate(qv, 2.0)`); hotovost T-bill; top 10; top 40; bez podmínky stáří 365 dní; plnění o den později (rozhodnutí *t*, plnění open *t+2*: posuň `Decisions.days` o +1) |
| H1a-BTC | `band=None`; náklady ×2 |
| H1b | `band=None`; náklady ×2 a ×3; vol cíl 15 %; jen 9 ETF výzkumu 2 (`SPY EFA EEM IEF TLT TIP GLD DBC VNQ`) |
| H2 | (r1) vstup i výstup na open *t+1* (`entry_delay=1`, `exit_at_close=False`); (r2) výstup `RSI2 > 70` místo `SMA5`; (r3) vstup podle součtu RSI2 za 2 dny < 35; (r4) náklady ×2 a ×3; (r5) bez filtru SMA200; (r6) jen 4 indexová ETF (`SPY QQQ IWM DIA`); (r7) `max_positions=2` |
| H4 | sleevy s náklady ×2; váhy inverzní volatilitou za 252 dní měsíčně |

---

## 6. Registr, N a zámek

- Registr: `runs/registry/research19.sqlite` (`TrialRegistry`), `kind` ∈ {`primary`, `secondary`, `robustness`,
  `control`}.
- Primární pokusy (N_meth): **H1a, H1b, H2, H3a, H3b, H3c, H4 = 7**. Sekundární (replikace): H1a-BTC, H2-SPY.
- Každá konfigurace se zapíše **před** výpočtem (`registry.record(...)`), config = dataclass → `as_dict()`
  s `"study": "research19"`.
- Kumulativní N na krypto datech (výzkumy 8–18) je podle paměti ≥ 89. V reportu uveď DSR dvakrát: s N tohoto výzkumu
  a s kumulativním N = 89 + počet krypto konfigurací tohoto výzkumu (konzervativní pohled).
- Zámek: `scripts/r19_run.py freeze` spočítá `rule_hash` souborů `src/qlab/research19/*.py`,
  `src/qlab/engine/vector.py`, `src/qlab/research5/sim.py`, `docs/research19/PREREGISTRATION.md` (vzor
  `qlab.research17.forward.rule_hash`) a zapíše `docs/research19/run_lock.json` s commitem. `full` odmítne běžet,
  pokud hash nesedí nebo pracovní strom není čistý v těchto souborech, nebo pokud už existuje
  `docs/research19/results/full.json` (jednorázovost). Opakovaný běh po pádu: jen s přepínačem `--rerun "<důvod>"`,
  který zapíše řádek do registru (`kind="rerun"`) a do decision logu pre-registrace.

---

## 7. Vyhodnocení a kritéria (pevná, rozhodují mechanicky)

Sharpe = průměr denního přebytku nad hotovostí / sd · √rok (krypto √365, hotovost 0; ETF √252, T-bill).
Intervaly = stacionární bootstrap, `mean_block=21` (ETF) / `30` (krypto), 10 000 replik, seed 19, 90 % oboustranně.
DSR = `deflated_sharpe(returns, n_trials, sr_variance)` s rozptylem Sharpe přes všechny registrované konfigurace
výzkumu. „Průměrný obchod“ u H2 = průměr čistých výnosů obchodů, interval bootstrapem po **dnech vstupu**
(obchody vstoupené stejný den se převzorkují spolu, protože sektory padají současně).

### 7.1 H1a (všech 6 musí projít)
1. Dolní mez 90% CI Sharpe > 0 (2019-01 → 2026-08).
2. Sharpe H1a > Sharpe BTC HOLD (bodově) **a** alfa z regrese denních výnosů na BTC (Newey–West, 30 lagů) > 0
   s t > 2. Funkci `alpha_hac` napiš v `evaluate.py` s testem na syntetických datech.
3. Sharpe v podúseku 2022-01-01 → 2026-08-31 > 0.
4. Sharpe > 0 při nákladech 25 bps/stranu **i** při `cost_rate ×2`.
5. Sharpe > 95. percentil náhodného načasování (§4.6).
6. DSR (N výzkumu) > 0,95.
Popisně: max. propad vs BTC, investovanost, obrat, náklady p. a., roky, post-publikační úsek 2025-04 → 2026-08,
podíl zisku od 5 % nejlepších obchodů (pozitivní šikmost).

### 7.2 H1b (všech 5)
1. Dolní mez 90% CI Sharpe > 0 (2007-07 → 2026-09).
2. Sharpe 2010-01-01 → 2026-09-25 > 0.
3. Výnos v kalendářním roce 2008 **i** 2022 > výnos 60/40 v témže roce.
4. Sharpe při nákladech ×3 > 0.
5. Korelace denních výnosů s SPY < 0,5.
(Neporovnává se se 60/40 na Sharpe. To už udělal výzkum 2 a H1b tu má roli diverzifikátoru.)

### 7.3 H2 (všech 6)
1. Průměrný čistý obchod > 0, dolní mez 90% CI > 0 (celé období).
2. Post-publikace 2009-01 → 2026-09: průměrný čistý obchod > 0 **a** Sharpe > 0.
3. Průměrný čistý obchod > 95. percentil kontroly c2 (náhodné vstupy).
4. Sharpe H2 > Sharpe c1 (vždy long v režimu) **nebo** max. propad H2 < 0,5 × propad c1. Jinak je edge jen v režimu.
5. Při nákladech ×2 průměrný čistý obchod > 0.
6. Varianta r1 (plnění na open) průměrný čistý obchod > 0.
Popisně: úspěšnost, průměrné držení, čas v trhu, roky, podíl zisku z let 1999–2002 (varování z výzkumu 5),
H2-SPY vedle čísel z rešerše (82 % úspěšnost, 3,7 dne držení, 10 % času v trhu, Sharpe 0,62).

### 7.4 H3 (každý overlay zvlášť)
- H3a (i): dolní mez 90% CI rozdílu (TOM − mimo TOM) > 0 na 2002–2026. Pokud ne, H3a (ii) se počítá, ale overlay
  je zamítnut bez ohledu na výsledek (ii).
- H3a (ii), H3b, H3c: dolní mez 90% CI **párového** rozdílu Sharpe (overlay − bez overlaye, převzorkovat obě řady
  spolu) > 0 a max. propad overlaye ≤ max. propad bez overlaye.

### 7.5 H4 (všechny 4)
1. Sharpe H4 > Sharpe každého jednotlivého sleevu A, B, C (diverzifikace funguje).
2. Dolní mez 90% CI párového rozdílu Sharpe H4 − 60/40 > 0.
3. Max. propad H4 < max. propad 60/40.
4. Při nákladech ×2 všech sleevů Sharpe H4 > Sharpe 60/40.
Popisně: korelační matice sleevů (celkem, v dnech, kdy SPY < −2 %, v roce 2022), příspěvky k riziku.
**Očekávám, že bod 2 neprojde.**

### 7.6 Rozhodnutí
- Hypotéza, která projde všemi kritérii → **kandidát na forward test**. Forward je samostatná pre-registrace
  (výzkum 20, vzor výzkumu 18): pevná pravidla z tohoto výzkumu, T0 = první den po commitu, minimální délka
  12 měsíců, laťka Sharpe > 0 a průměrný obchod > 0 při nákladech ×1,5. Krypto data forward: Binance REST/
  `data.binance.vision` (vzor `qlab.research17.forward`). ETF data forward: Sharadar už není. Zdroj je potřeba
  ověřit (Yahoo + Nasdaq ve večerním botu `evening_bot_server/server/evening/prices.py`, rozdíly od Sharadaru jsou tam
  změřené). **Forward test v tomto výzkumu neimplementuj**, jen ho navrhni v reportu.
- Nic neprojde → výzkum se uzavírá negativně, řádek do README, report. Žádné „zkusím ještě variantu“.
- Haircut v reportu: u každé hypotézy, která projde, uveď i „očekávaný live Sharpe = 0,5 × backtest“ (rešerše:
  McLean–Pontiff).

---

## 8. Fáze implementace (pořadí je závazné)

### F0 – orientace (bez kódu)
- Přečti: `CLAUDE.md`, tento plán, `src/qlab/engine/vector.py`, `src/qlab/research5/sim.py`,
  `src/qlab/research9/{strategy,panel,setup}.py`, `src/qlab/research10/sim.py` (`monthly_mix`),
  `scripts/r2_build_panel.py`, `scripts/r9_final.py`, `scripts/r18_forward.py`, `docs/research18/PREREGISTRATION.md`
  (formát pre-registrace).
- Spusť `uv run pytest` a zapiš počet testů a čas (baseline). Pokud něco padá už teď, **stop**, nahlas vlastníkovi.

### F1 – pre-registrace
- Napiš `docs/research19/PREREGISTRATION.md` česky podle formátu výzkumu 18: §1 otázky (H1a, H1b, H2, H3a–c, H4),
  §2 poctivá omezení (zkopíruj podstatu §1 tohoto plánu: kontaminace, překryv s výzkumy 2/5/9/10, odchylky [R]),
  §3 data, §4 pravidla (doslova §4 plánu), §5 robustnost, §6 registr a N, §7 kritéria, §8 rozhodnutí, §9 decision log
  (první řádek: „v1.0 vytvořena podle PLAN.md v1.0“).
- **Stop-bod:** ukaž vlastníkovi, počkej na schválení. Po schválení commit `research19: pre-registration v1.0`.
- Akceptace: commit existuje dřív než jakýkoli soubor v `docs/research19/results/`.

### F2 – úpravy frameworku
- `vector.simulate(band=...)` a `SimSpec.exit_at_close` podle §2.3, včetně testů v `tests/test_vector_engine.py`
  a `tests/test_research5.py` (přidej, neměň existující).
- Akceptace: celý `uv run pytest` prochází, počet testů = baseline + nové. Commit `engine: optional rebalance band; r5 sim: exit at close`.

### F3 – data
- `scripts/r19_build_etf_panel.py sharadar_2026-09-25` → `panel_r19_etf`. `scripts/r19_audit.py` → `audit.json`.
- `research19/setup.py`: konstanty z §3–§7 (tickery, lookbacky, období, náklady, seed, cesty), `load_etf()`,
  `load_crypto()` (bez vaultu: výzkum explicitně deklaruje celou historii jako viděnou, stejně jako `load_full`
  ve výzkumu 5; napiš to do docstringu).
- Akceptace: audit vypíše začátky tickerů shodné s §3.2 (±1 den), žádné neočekávané symboly v top 20 (§3.3 stop-bod).
  Commit (bez dat!).

### F4 – signály
- `signals.py`: `donchian_state(P, valid, L) -> (state, stop)`, `ensemble(P, valid, lookbacks)`,
  `rsi_wilder(P, valid, n=2)`, `sma(P, valid, n)`, `realized_vol(P, valid, n, ann)`, `tom_window(dates)`.
  Vše vrací matice (T, N), NaN/0 podle §4.0. Smyčka v čase je v pořádku (krypto: jen sloupce, které kdy byly
  v top 40, aby to bylo rychlé).
- Testy (synthetic, ručně spočitatelné):
  - Donchian L=3 na řadě `[1,2,3,4,3.4,3.6,2,5]`: ručně spočti stavy a stopy a porovnej. Ověř, že vstup nenastane,
    když `P_t` jen vyrovná maximum (ostrá nerovnost), a že stop nikdy neklesne.
  - RSI2 proti ruční tabulce 6 dnů a proti `AL=0 → 100`.
  - `check_point_in_time` pro `ensemble`, `rsi_wilder`, `realized_vol` (vzor `tests/test_research5.py`).
  - `tom_window` na kalendáři s víkendem a svátkem (poslední obchodní den měsíce není poslední kalendářní).
- Commit.

### F5 – strategie, kontroly, vyhodnocení
- `crypto_trend.py`: `universe_aged`, `monthly_members`, `weights_h1a(p, qv, cfg) -> Decisions`,
  `weights_btc(...)`, `run(cfg) -> SimResult`. Config dataclass (`frozen=True`, `as_dict()`), parametry z `setup`.
- `etf_trend.py`, `rsi2.py` (vč. c1, c2), `overlays.py`, `combine.py` (`align_to_calendar`, `h4`), `evaluate.py`
  (`sharpe_ann`, `ci_sharpe`, `paired_ci`, `alpha_hac`, `trade_ci_by_entry_day`, `circular_shift_null`, `criteria_*`
  vracející dict `{kritérium: (hodnota, práh, prošlo)}`).
- Testy: H1a na syntetickém panelu 3 aktiv se známým trendem (jedno roste lineárně, jedno klesá, jedno šum) →
  rostoucí má `E→1`, klesající 0, gross ≤ 1; váhy s `band` nemění počet vstupů; `align_to_calendar` ruční příklad;
  `alpha_hac` na řadě `r = 0.001 + 0.5·r_btc + šum` vrátí alfu ≈ 0,001; c2 zachovává délky držení.
  `check_point_in_time` na celém pipeline H1a a H2 (vzor `test_whole_pipeline_is_point_in_time`).
- Akceptace: `uv run pytest` zelený. Commit.

### F6 – harness, smoke, zámek, jeden plný běh
- `scripts/r19_run.py smoke`: všechny hypotézy na **syntetických datech** (`qlab.data.synthetic`) nebo na reálných
  datech zkrácených na 2018-01 → 2018-12 (krypto) a 1999–2000 (ETF), registr přesměrovaný do
  `runs/scratch/research19_smoke.sqlite`. Smoke výsledky se **necommitují** a nikam nereportují.
  Účel: běží to, rychlost, tvary. Čísla z reálných smoke dat nečti a nekomentuj.
- `freeze`: zámek §6, registrace všech konfigurací z §4–§5 (`record` před výpočtem; `full` ověří, že registr obsahuje
  přesně tyto hashe).
- **Stop-bod:** ukaž vlastníkovi `git log` (pre-registrace → engine → kód → zámek) a počkej na „spusť“.
- `full`: jeden běh, výstupy `docs/research19/results/full.json` (všechna čísla, kritéria, provenance: commit,
  rule_hash, snapshoty, čas) a menší `h1a.json`, `h1b.json`, `h2.json`, `h3.json`, `h4.json`, `nulls.json`.
  Odhad času: H1a null 500 × engine ~ jednotky minut, H2 c2 1 000 × event sim ~ do 15 min. Pokud by běh trval > 1 h,
  zastav a nahlas, nic nezkracuj sám.
- Commit výsledků JSON.

### F7 – report
- `docs/research19/REPORT.md` česky, formát jako `docs/research17/REPORT.md`: verdikt na začátku, tabulky kritérií
  (✔/✘ s hodnotami), podúseky, robustnost, kontroly, korelace sleevů, haircut, co z toho plyne, návrh forward testu
  (jen pokud něco prošlo).
- Řádek do tabulky výsledků v `README.md` a řádek verdiktu do decision logu pre-registrace.
- Verdikt plyne z kritérií mechanicky. Žádné „skoro prošlo, tak je to slibné“ jako závěr. Post hoc pozorování
  smí být jen v sekci „Post hoc (není důkaz)“.
- **Stop-bod:** ukaž report vlastníkovi před commitem.

---

## 9. Stop-body (vždy se zastav a zeptej vlastníka)

1. Baseline testy padají (F0).
2. Pre-registrace hotová (F1), před commitem.
3. V top 20 krypto univerza je stablecoin, tokenizovaná akcie nebo wrapped token (F3).
4. Ticker z §3.2 v Sharadaru chybí nebo začíná o víc než týden jinak.
5. Existující test po úpravě enginu (F2) změnil výsledek. To znamená, že úprava není zpětně kompatibilní.
6. Plán je nejednoznačný nebo vnitřně rozporný. **Nedomýšlej, zeptej se.**
7. Před `full` (F6).
8. Plný běh spadl po registraci.
9. Report hotový (F7), před commitem.

## 10. Časté chyby, kterých se vyvaruj

- **Off-by-one v Donchianu:** `U_t` je maximum **bez** dne *t*, jinak vstup nikdy nenastane. `M_t` je **s** dnem *t*.
- **Anualizace:** krypto √365 a hotovost 0, ETF √252 a T-bill. `qlab.validation.metrics.summary` má natvrdo 252,
  pro krypto ho nepoužívej na Sharpe/CAGR (nebo mu předej přepočet). Vzor `scripts/r9_final.py`.
- **`close_u` vs TR:** signály vždy z `tr_index`. `close_u` u ETF obsahuje splity a dividendy (skoky).
- **Pásmo rebalance** se vyhodnocuje proti vahám po close *t*, ne po open *t+1*.
- **Gross > 1:** engine vyhodí výjimku. Normalizuj váhy před vytvořením `Decisions`.
- **NaN v řádku rozhodnutí:** engine nepovolí mix NaN a čísel. Neplatné aktivum = 0.
- **Smoke běh do produkčního registru** nafoukne N. Vždy přesměruj.
- **Ladění po pohledu na výsledky** je zakázané. Chyba v kódu po plném běhu → oprava, `--rerun "<důvod>"`, zápis do
  decision logu, a v reportu uvést obě čísla.
- **Delisting v kryptu:** engine vyplatí pozici na open řádku `delisting`. Nesnaž se to obcházet.
- **Paměť:** stroj má teď ~8 GB RAM. Panely jsou malé (krypto 3 318 × 663), přesto načítej `mmap_mode="r"`
  a nedělej zbytečné float64 kopie v nullových smyčkách.
- **Commity:** anglické zprávy, žádná data (`data/`, `runs/`, `*.parquet`, `*.csv`, `*.sqlite` jsou ignorované),
  push jen na pokyn vlastníka.
