# Výzkum 19 – trend ensemble, RSI(2) a overlaye jako portfolio streamy (pre-registrace)

Verze 1.0 · 2026-11-02 · Stav: **návrh ke schválení vlastníkem** (necommitnuto).
Vychází z `docs/research19/PLAN.md` v1.0 (2026-11-01). Pravidla v §4–§7 jsou z plánu převzata **doslova**.
Po commitu verze 1.0 platí: změny = řádek v §9 s důvodem, **před** dalším vyhodnocením.

## 1. Otázky

Sedm primárních hypotéz (N_meth = 7), dvě sekundární replikace:

- **H1a – krypto Donchian ensemble.** Má ensemble 9 Donchianových trendů (lookbacky 5–360 dní, trailing stop na středu kanálu,
  velikost podle volatility) s rotací top 20 spotových USDT párů pozitivní, nenáhodný a po nákladech udržitelný výnos
  nad BTC buy & hold? (Sekundárně H1a-BTC: totéž jen na BTC.)
- **H1b – ETF Donchian ensemble.** Je ensemble na koši 16 ETF (lookbacky 50–250) pozitivním, s SPY nekorelovaným
  diverzifikátorem, který v letech 2008 a 2022 překonal 60/40? (Nesoudí se proti 60/40 na Sharpe.)
- **H2 – RSI(2) mean reversion na 13 indexových a sektorových ETF** s trendovým filtrem SMA200, vstup MOC. Je průměrný čistý
  obchod kladný, i po publikaci (2009+), a lepší než náhodné vstupy a než pouhý režim „nad SMA200“? (Sekundárně H2-SPY.)
- **H3a – turn-of-month** (efekt na SPY a overlay na H2), **H3b – vol targeting** kombinovaného portfolia, **H3c – filtr
  momentum crash** na trendových sleevech.
- **H4 – kombinace** streamů A (H1a), B (H1b), C (H2) po 1/3, měsíční reset.

## 2. Poctivá omezení předem

- **Kontaminace dat, žádný čistý holdout.** Veškerá historie (Sharadar do 2026-09-25, Binance do 2026-08-31) byla použita
  ve výzkumech 1–18 a je i v tréninkových datech LLM. Vzorek Zarattiniho studie (do 2025-03) se s našimi krypto daty
  překrývá celý. Výzkum je **replikace literárních pravidel s pevnými parametry** (N je malé, neladí se),
  vyhodnocená jednou na celé historii. Informativní jsou hlavně post-publikační úseky (RSI(2) po 2008, TOM po 2001,
  trend po 2009) a **forward test**, který je jediným čistým důkazem a spouští se jen pro to, co projde (§8).
- **Překryv s dřívějšími výzkumy.**
  - P1 ETF trend: výzkum 2 (9 ETF, inv. volatilita + SMA200; Sharpe 0,52 vs 60/40 0,85, zamítnuto) a výzkum 10B
    (TSMOM na 9 ETF jako rukáv k SPY, zamítnuto). H1b je stejná rodina; **čekám stejný verdikt vůči 60/40**, proto se
    H1b nesoudí proti 60/40, jen jako diverzifikátor uvnitř H4.
  - Krypto trend: výzkum 9 (SMA/momentum filtr BTC+ETH: ΔSharpe +0,02, zamítnuto; altcoiny rovné váhy 2022–26: −45 % p. a.).
    H1a (ensemble, trailing stop, vol sizing, rotace top 20) tu ještě nebyla.
  - RSI(2): výzkum 5 (STR-TF na jednotlivých akciích: validace Sharpe 0,32, 53 % zisku z let 1999–2002). Proto je
    kritériem povinný post-publikační úsek 2009+.
  - Overlaye: nic přímo; lekce z výzkumů 1 a 9: časování podle zpožděných signálů škodí při rychlých odrazech.
- **Pravidla Zarattiniho studie** jsou z abstraktu a souhrnu CXO Advisory (plné PDF, SSRN 5209907, nebylo k dispozici).
  Rozhodnutí tam, kde zdroj mlčí, jsou v §4.1 označena **[R]**. Pokud vlastník dodá PDF a liší se, mění se
  pre-registrace **před** během, ne po něm.
- **Odchylky od studie:** portfolio bez páky (součet vah ≤ 1, proporcionální zmenšení); definice top 20 s podmínkou
  stáří 365 dní; pásmo rebalance 20 % je moje interpretace.
- **Survivorship:** ETF koš je vybraný ex post (všechna ETF dodnes žijí, QQQ je známý vítěz). U širokých indexových ETF
  je zkreslení malé, ale existuje.
- **Zjednodušení:** overlaye H3 se aplikují na výnosové řady sleevů s approximovanými náklady (§4.4); krypto a US kalendář
  se zarovnávají posunem < 1 den (§4.5); měsíční reset sleevů nese náklad 3 bps × |Δ vah|.
- **Očekávání (předem):** H1a má šanci projít, H2 možná projde celkově a bude slabá po 2009, H1b a overlaye nejspíš
  neprojdou, H4 skoro jistě nepřekoná 60/40 na Sharpe (60/40 bylo v 2010–2026 výjimečné). Hodnota výzkumu je v poctivě
  změřených streamech a jejich korelacích.

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


## 8. Rozhodnutí

- Hypotéza, která projde všemi kritérii → **kandidát na forward test**. Forward je samostatná pre-registrace
  (výzkum 20, vzor výzkumu 18): pevná pravidla z tohoto výzkumu, T0 = první den po commitu, minimální délka
  12 měsíců, laťka Sharpe > 0 a průměrný obchod > 0 při nákladech ×1,5. Krypto data forward: Binance REST/
  `data.binance.vision` (vzor `qlab.research17.forward`). ETF data forward: Sharadar už není. Zdroj je potřeba
  ověřit (Yahoo + Nasdaq ve večerním botu `evening_bot_server/server/evening/prices.py`, rozdíly od Sharadaru jsou tam
  změřené). **Forward test v tomto výzkumu neimplementuj**, jen ho navrhni v reportu.
- Nic neprojde → výzkum se uzavírá negativně, řádek do README, report. Žádné „zkusím ještě variantu“.
- Haircut v reportu: u každé hypotézy, která projde, uveď i „očekávaný live Sharpe = 0,5 × backtest“ (rešerše:
  McLean–Pontiff).

## 9. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-11-02 | v1.0 vytvořena podle PLAN.md v1.0. Před commitem se na reálných datech nic nepočítalo. |
