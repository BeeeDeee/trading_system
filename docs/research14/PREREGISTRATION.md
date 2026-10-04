# Výzkum 14 – Multi-timeframe pullback: vstup proti 1h pohybu ve směru týdenního a denního trendu (pre-registrace)

Verze 1.0 · 2026-10-04 · Stav: **pre-registrováno, nic nespočteno**.
Navazuje na výzkumy 9 a 13 (krypto, období vývoj/holdout). Změny po commitu = nový pokus v registru a řádek v §12.

## 1. Otázka

Když vyšší timeframy (týden a den) ukazují trend jedním směrem a nejnižší timeframe (1 h) se právě silně
pohnul opačně, vsadit na návrat do směru vyššího trendu („buy the dip in an uptrend“, zrcadlově prodej
odrazu v downtrendu).

Terminologie: jde o **mean reversion na nízkém timeframe filtrovanou trendem z vyšších timeframů**
(pullback / multi-timeframe trend entry). Čistá mean reversion by sázela proti 1h pohybu bez ohledu na
vyšší trend. Obě varianty se měří (§6, `B_NOHTF`).

**Vydělává na BTC a ETH vstup proti silnému 1h pohybu, pokud týdenní i denní trend ukazují opačně než
tento pohyb, po nákladech a fundingu víc než nula a víc než náhodné vstupy ve stejném režimu vyššího
trendu se stejnou dobou držení?**

## 2. Poctivé omezení předem

- Historie BTC/ETH 2018–2026 je obecně známá a období 2022–2026 už otevřely výzkumy 9, 10 C a 13 (denní
  data). Na hodinových datech a pro tuto hypotézu nebylo použito, ale „neviděné“ není. Historický test
  může hypotézu jen **zamítnout, nebo kvalifikovat na forward test** (§10).
- Literatura (krátkodobý reversal na intradenních horizontech v kryptu je slabý, často je tam spíš
  momentum; time-series momentum na týdnech funguje lépe) dává apriorní očekávání spíš negativní nebo
  nevýrazné. Výzkum 13 ukázal, že nákup poklesu v kryptu chytá padající nože.
- Short se v praxi dělá přes USDT perpetual. Ceny jsou ze spotu (spot a perp BTC/ETH se liší o jednotky
  bps); funding se účtuje podle §5.

## 3. Data

- Hodinové svíčky Binance spot `BTCUSDT`, `ETHUSDT` (data.binance.vision, měsíční zipy), staženo pro
  projekt `market_relations_2026_oct` do `data/raw/binance_1h/spot_1h/` – čte se **jen ke čtení**, SHA256
  zipů se zapíše do `DATA_AUDIT.md`. Rozsah 2017-08 → **2026-08-31 23:00 UTC**.
- Funding perpetuálů z `data/raw/binance_2026-10-03/funding/` (od 2020-01).
- Hodinová mřížka UTC bez mezer; chybějící hodina = cena se přenese (nulový výnos) a v hodině se
  **nevstupuje** (stávající pozice běží dál). Audit mezer, duplicit a OHLC konzistence do `DATA_AUDIT.md`.
- Časové značky: od 2025 jsou ve zdroji v mikrosekundách, převedou se.

## 4. Signál (vše jen z close do konce hodiny *t*; *c_t* = close hodiny *t*)

- **Denní trend** `D_t = sign(c_t − SMA50_D)`, kde SMA50_D je průměr posledních 50 **uzavřených** denních
  close (close hodiny 23:00 UTC).
- **Týdenní trend** `W_t = sign(c_t − SMA20_W)`, SMA20_W = průměr posledních 20 uzavřených týdenních close
  (close neděle 23:00 UTC).
- Režim vyššího trendu: **nahoru**, když `D_t = W_t = +1`; **dolů**, když `D_t = W_t = −1`; jinak neutrální.
- **1h pohyb**: `z_t = ln(c_t / c_{t−L}) / (σ_t · √L)`, kde σ_t = sm. odchylka hodinových log výnosů za
  posledních 720 h (min. 600 platných). *c_t* i *c_{t−L}* musí být skutečné (ne přenesené) svíčky.
- **Vstup long:** režim nahoru a `z_t ≤ −z0`. **Vstup short:** režim dolů a `z_t ≥ +z0` (jen varianta `LS`).

## 5. Obchod

- Rozhodnutí po close hodiny *t*, plnění za *c_t* (open hodiny *t*+1 se v kryptu od *c_t* liší zanedbatelně;
  robustnost s plněním o hodinu později, §9).
- **Pevná doba držení H hodin**, pak výstup za close. Během otevřené pozice se nové signály téhož aktiva
  ignorují. Žádný stop, žádné prodlužování.
- **Portfolio:** dva rukávy BTC a ETH, každý 50 % kapitálu; pozice v rukávu = +1, −1 nebo 0 (konstantní
  expozice, hodinové výnosy × pozice). Hotovost nenese nic.
- **Náklady:** 7,5 bps na stranu z nominálu (taker perp 5 bps + skluz 2,5 bps), zátěž ×2 = 15 bps.
- **Funding:** v každém vypořádání (00, 08, 16 UTC), kdy je pozice otevřená, platí long `+rate`, short
  `−rate` (záporný rate obráceně). Před 2020-01 (bez dat) paušálně 0,01 % / 8 h.
- Denní výnos portfolia = součet hodinových výnosů dne (UTC) po nákladech a fundingu.

**Mřížka (N = 16):** `L ∈ {6, 24}` × `z0 ∈ {1,5; 2,5}` × `H ∈ {12, 48}` × strana `∈ {LS, LONG}`.

## 6. Benchmarky a nulové hypotézy

- **B_RND (hlavní nula):** 500 běhů, seed 0..499. Pro každé aktivum a stranu se v každém kalendářním měsíci
  otevře stejný počet obchodů jako u strategie, v náhodných hodinách se **stejným režimem vyššího trendu**
  (long v režimu nahoru, short v režimu dolů), se stejným H, stejným pravidlem nepřekrývání a stejnými
  náklady i fundingem. Měří, zda 1h pullback přidává něco nad samotný vyšší trend.
- **B_NOHTF:** stejné L, z0, H bez filtru vyššího trendu, obě strany (čistá 1h mean reversion).
- **B_HTF:** trvalá pozice ve směru režimu (+1 nahoru, −1 dolů, 0 neutrální; u `LONG` jen +1/0), stejné náklady
  a funding – čisté sledování vyššího trendu.
- **B_HOLD:** 50/50 BTC/ETH, rebalance denně, bez nákladů.

## 7. Období a trezor

| Fáze | Období |
|---|---|
| Vývoj | 2018-02-01 → 2021-12-31 (výběr P = nejvyšší Sharpe z 16 kandidátů) |
| Holdout | **2022-01-01 → 2026-08-31**, jednou, přes `Vault` (`runs/vault_r14/`, zámek metodiky, log otevření) |

Indikátory se mohou počítat z dat před začátkem fáze; na začátku fáze se nic nedrží. Shoda Sharpe →
první v pořadí mřížky.

## 8. Kritéria (holdout, P; nutné splnit **všechna**)

Sharpe = průměr / sm. odchylka denních výnosů × √365, bez bezrizikové sazby.

1. Sharpe P > 0 a průměrný čistý výnos obchodu > 0 s dolní mezí 90% intervalu > 0 (bootstrap obchodů,
   2000 vzorků, seed 0).
2. Sharpe P > **95. percentil** Sharpe B_RND.
3. Průměrný čistý výnos obchodu P > průměr B_NOHTF se stejnými L, z0, H (filtr vyššího trendu pomáhá).
4. Sharpe P > 0 při nákladech ×2.
5. Sharpe P > 0 v obou podúsecích **2022-01 → 2023-12** a **2024-01 → 2026-08**.
6. Průměrný čistý výnos obchodu > 0 zvlášť na BTC i na ETH.

Deflated Sharpe (N = 16, rozptyl Sharpe z vývojové mřížky) se hlásí, není kritériem.

## 9. Robustnost (jen hlášení, nevybírá se podle ní)

Všech 16 kandidátů na holdoutu; výsledky po letech; plnění o hodinu později; bez fundingu; long a short
strany zvlášť; statistiky obchodů (počet, úspěšnost, průměr, medián); srovnání s B_HTF a B_HOLD;
**sekundární koš** 9 altcoinů s 1h daty od 2018 (BNB, LTC, ADA, XRP, ETC, TRX, XLM, NEO, EOS; velké coiny
roku 2018, výběr má survivorship bias, proto jen popisně) s pravidly P.

## 10. Co znamená výsledek

- Nesplní-li P cokoli z §8: **zamítnuto**, forward test se nespouští.
- Splní-li vše: kandidát na **forward paper test ≥ 6 měsíců** s vlastní pre-registrací.

## 11. Implementace a postup

1. Commit této pre-registrace (před jakýmkoli výpočtem).
2. `scripts/r14_build_panel.py`: hodinový panel (close, flag skutečné svíčky, funding na hodinu) do
   `data/derived/binance_2026-10-03/r14_1h/`, `DATA_AUDIT.md`.
3. `qlab.research14`: signál, simulátor obchodů s pevným H, B_RND, B_NOHTF, B_HTF. Testy: bod v čase
   (`check_point_in_time`), ruční přepočet jednoho obchodu (náklady, funding), nepřekrývání, uzavřené
   denní/týdenní svíčky, přenesené hodiny.
4. Commit kódu. Registr `runs/registry/research14.sqlite`: všech 16 kandidátů zapsat před výpočtem.
5. `scripts/r14_dev.py` (vývoj, výběr P) → `scripts/r14_final.py freeze|run` (jednorázový holdout).
6. `REPORT.md`, řádek v tabulce README, verdikt do §12.

## 12. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-10-04 | Pre-registrace v1.0 | – |
