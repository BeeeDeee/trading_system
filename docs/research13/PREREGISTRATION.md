# Výzkum 13 – Smart Zones: nákup v discount zóně swingového rozpětí (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **uzavřeno, zamítnuto** (`REPORT.md`).
Navazuje na výzkum 9 (data Binance, univerzum, náklady) a na event-driven simulátor výzkumu 5.
Změny po commitu = nový pokus v registru a řádek v §11.

## 1. Otázka

Indikátory typu „Smart Zones“ (TradingView: LuxAlgo / MILLION MEN Smart Zones a podobné; koncept premium/discount
ze „smart money concepts“) vyznačí rozpětí mezi posledním potvrzeným swing high a swing low a dělí ho na
**discount zónu** (spodní část, nákupy), **equilibrium** (50 %) a **premium zónu** (horní část, prodeje).
Doporučené obchodování: vstup v zóně, stop za opačným okrajem rozpětí, cíl na další úrovni.

**Vydělává long-only nákup likvidních kryptoměn v discount zóně (stop pod swing low, výstup na equilibriu
nebo v premium zóně) po nákladech víc než (a) rovné váhy likvidního univerza, (b) BTC HOLD a (c) náhodné
vstupy se stejným profilem expozice?**

## 2. Poctivé omezení předem

- **Holdout 2022–2026 už byl otevřen** výzkumy 9 a 10 C na stejných datech a historie kryptoměn je obecně
  známá. Pro tuto hypotézu nebyl použit, ale není „neviděný“. Historický test proto může hypotézu jen
  **zamítnout, nebo kvalifikovat na forward test** (§10), nic víc.
- Obsahově jde o **mean-reversion v rámci rozpětí**, příbuzné STR-TF (výzkum 5, zamítnuto) a reversalu
  (výzkum 6). Apriorní očekávání je negativní výsledek.
- Indikátor je určen hlavně pro intraday (5 min – 1 h). Zde se testuje **jen denní** varianta (swingové
  obchodování 4H–1D je podle autorů také zamýšlené použití). Intradenní stopy a limitní příkazy engine
  neumí: stop i cíl se kontrolují na close a provádějí na openu dalšího dne (konzervativní k cíli,
  optimistické ke stopu při gapu – obojí se hlásí).
- Jen long (spot), bez páky a bez shortu z premium zóny. Hotovost (USDT) nenese nic.
- Parametry indikátorů z TradingView nejsou veřejně ukotvené; mřížka v §5 pokrývá rozumný rozsah, ne
  přesnou kopii konkrétního skriptu.

## 3. Data

- Snapshot `binance_2026-10-03` výzkumu 9 (denní svíčky všech spotových párů `XUSDT`, včetně delistovaných;
  stejné vyřazení stablecoinů, wrapped, fiat a pákových tokenů). Konec dat **2026-08-31**.
- Panel výzkumu 9 nemá high a low. Nový panel `panel_r13` = panel výzkumu 9 + `extra_high`, `extra_low`
  ze stejných raw svíček a se stejnými filtry řádků. Panel výzkumu 9 se nemění. Kontrola: všechny ostatní
  matice `panel_r13` jsou bitově shodné s `panel_r9`; `low ≤ min(open, close)`, `high ≥ max(open, close)`.
- Delisting, mezery a výnosy (`ret_co`, `ret_oc`) přesně podle výzkumu 9.

## 4. Definice zón (vše jen z dat do close dne *t*)

Parametr *k* (lookback pivotu).

- **Pivot high** v den *j*: `high_j > max(high_{j−k..j−1})` a `high_j ≥ max(high_{j+1..j+k})`.
  **Potvrzen na close dne *j*+*k***, dřív neexistuje. Pivot low zrcadlově (`<`, `≤`). Mezera ve svíčkách
  uvnitř okna = pivot se nepočítá.
- K close dne *t*: `H` = poslední potvrzený pivot high, `L` = poslední potvrzený pivot low.
  **Dynamické rozšíření** (jako indikátor): `top_t = max(H, max high od dne pivotu H do t)`,
  `bottom_t = min(L, min low od dne pivotu L do t)`.
- Rozpětí je **platné**, když existují oba pivoty, oba jsou staré ≤ **90 dní** (expirace zón)
  a `(top − bottom) / bottom ≥ 5 %`.
- Pro podíl *z*: discount `[bottom, bottom + z·R]`, equilibrium `bottom + 0,5·R`,
  premium `[top − z·R, top]`, kde `R = top − bottom`.

## 5. Pravidla obchodu

- Univerzum k datu: top 20 párů podle 30denního průměrného quote objemu, obchodované ≥ 60 dní (výzkum 9).
- **Vstup:** po close *t* coin v univerzu, bez otevřené pozice, platné rozpětí a
  `bottom_t < close_t ≤ bottom_t + z·R_t`. Varianta `bounce` navíc vyžaduje `close_t > close_{t−1}`.
  Plnění na openu *t*+1.
- **Zóna se při vstupu zmrazí** (bottom, top, R ze dne *t*); stop i cíl se dál počítají z ní.
- **Výstupy** (kontrola na close, plnění na openu dalšího dne; při více současně platí pořadí stop → cíl → čas):
  - stop: `close < bottom_entry · 0,99`,
  - cíl `EQ`: `close ≥ equilibrium_entry`; cíl `PREM`: `close ≥ top_entry − z·R_entry`,
  - časový stop: 60 obchodních dní držení,
  - delisting: podle výzkumu 9.
- **Velikost:** max. 5 pozic, každá 20 % NAV v den vstupu (pevně do výstupu), zbytek USDT. Více kandidátů než
  volných míst: nejhlouběji v zóně (`(close − bottom)/R` vzestupně), shoda → vyšší 30denní quote objem.
- Náklady za stranu podle výzkumu 9 (10 bps + slippage podle objemu), zátěž ×2.

**Mřížka (N = 24):** `k ∈ {5, 10, 20}` × `z ∈ {0,25; 0,5}` × cíl `∈ {EQ, PREM}` × vstup `∈ {touch, bounce}`.

## 6. Benchmarky

- **B_EW:** rovné váhy univerza top 20, týdně v pondělí, stejné náklady (benchmark X výzkumu 9).
- **B_BTC:** BTC HOLD.
- **B_RND (nulová hypotéza bez informace o zónách):** 500 běhů, seed 0..499. V každý den, kdy strategie
  otevře *m* pozic, se otevře *m* pozic v náhodných coinech univerza bez otevřené pozice; doba držení se
  losuje z empirického rozdělení dob držení strategie na stejném období; stejná velikost, sloty a náklady.
  Měří, zda výběr okamžiku a coinu přidává něco nad profil expozice.

## 7. Období a trezor

| Fáze | Období |
|---|---|
| Vývoj | 2018-04-01 → 2021-12-31 (výběr P = nejvyšší Sharpe z 24 kandidátů) |
| Holdout | **2022-01-01 → 2026-08-31**, jednou, přes `Vault` (`runs/vault_r13/`, zámek metodiky, log otevření) |

Na začátku každé fáze se nic nedrží; pivoty se mohou počítat z dat před začátkem fáze.

## 8. Kritéria (holdout, P; nutné splnit **všechna**)

Sharpe = průměr / sm. odchylka denních výnosů × √365, bez bezrizikové sazby.

1. Rozdíl Sharpe P − B_EW > 0 a dolní mez 90% intervalu > 0 (stationary bootstrap párově, blok 21,
   1000 vzorků, seed 0).
2. Sharpe P > Sharpe B_BTC.
3. Sharpe P > **95. percentil** Sharpe B_RND.
4. Rozdíl Sharpe P − B_EW > 0 v obou podúsecích **2022-01 → 2023-12** a **2024-01 → 2026-08**.
5. Rozdíl Sharpe P − B_EW > 0 při zátěžových nákladech ×2.
6. Deflated Sharpe aktivního výnosu vůči B_EW ≥ 0,95 (N = 24, rozptyl Sharpe z vývojové mřížky).

## 9. Robustnost (jen hlášení, nevybírá se podle ní)

Všech 24 kandidátů na holdoutu; výsledky po letech; plnění o den později; univerzum top 10 a top 40;
pivoty jen z close místo high/low; stop provedený na úrovni stopu místo openu (horní odhad, pokud by šel
dát stop příkaz); statistiky obchodů (počet, úspěšnost, průměrný čistý výnos, podíl výstupů stop / cíl /
čas, průměrná doba držení); srovnání s `T_sma20` z výzkumu 9.

## 10. Co znamená výsledek

- Nesplní-li P cokoli z §8: **zamítnuto**, forward test se nespouští.
- Splní-li vše: kandidát na **forward paper test ≥ 6 měsíců** s vlastní pre-registrací (stejná pravidla,
  zámek rozhodnutí před plněním jako u krypto paper bota). Historický výsledek sám nic nedokazuje (§2).

## 11. Implementace a postup

1. Commit této pre-registrace (před jakýmkoli výpočtem).
2. `scripts/r13_build_panel.py` (panel + high/low, kontrola shody s `panel_r9`).
3. `qlab.research13`: pivoty, rozpětí a zóny, kandidáti, výstupy se zmrazenou zónou, B_RND. Simulátor
   výzkumu 5 se rozšíří o cílový výstup a stop vztažený k ceně (ne k hodnotě pozice), jen pokud to nejde
   vyjádřit stávajícím rozhraním; parita se stávajícími testy zůstává.
4. Testy: bod v čase (`check_point_in_time`, perturbace budoucnosti; pivot nesmí být vidět před *j*+*k*),
   ruční přepočet jednoho obchodu (vstup, stop, cíl, náklady), shody v pivotech, mezera ve svíčkách,
   expirace zóny, delisting během držení.
5. Commit kódu. Registr `runs/registry/research13.sqlite`: všech 24 kandidátů zapsat před výpočtem
   (smoke testy do dočasného registru).
6. `scripts/r13_dev.py` (vývoj, výběr P) → `scripts/r13_final.py freeze|run` (jednorázový holdout).
7. `REPORT.md`, řádek v tabulce v README, verdikt do §12.

## 12. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-10-03 | Pre-registrace v1.0 | – |
| 2026-10-03 | Panel `panel_r13` postaven (`DATA_AUDIT.md`): všechny matice bitově shodné s `panel_r9`, close z raw svíček shodný. Jediná nekonzistentní svíčka (high < max(open, close)) je `AUDUSDT` 2020-11-13. Nic se neopravuje. | audit (§3) |
| 2026-10-03 | Filtr výzkumu 9 propustil dva fiat páry: `AUDUSDT` (australský dolar) a `BKRWUSDT` (stablecoin vázaný na KRW). Podle §3 se fiat vyřazuje, takže se vyřadí před seřazením univerza. V top 20 nebyly nikdy, `AUDUSDT` byl 26 dní v top 40 (2022-10 → 2023-01, jen robustnost). Výzkum 9 se zpětně nemění. | dodržení §3, zjištěno před výpočtem |
| 2026-10-03 | Při z = 0,5 je cíl `PREM` (top − 0,5·R) totožný s `EQ`: 6 dvojic kandidátů je identických, různých konfigurací je 18. Mřížka a **N = 24 pro DSR zůstávají** (konzervativně). Shoda Sharpe ve vývoji → první kandidát v pořadí mřížky (`EQ`). | chyba návrhu mřížky, zjištěno před výpočtem |
| 2026-10-03 | Simulátor výzkumu 5 rozšířen (§11.3): cenový stop a cíl pro každého kandidáta (`stop_px`, `target_px`, pořadí stop → cíl → signál → čas) a `reentry_same_open=False` (coin prodaný na openu se na stejném openu nekupuje, „bez otevřené pozice“ v §5). Výchozí chování beze změny, testy výzkumu 5 procházejí. B_RND otevírá méně pozic než strategie, protože jeho pozice nekončí cílem a déle drží sloty; to odpovídá §6 („stejné sloty“). | implementace |
| 2026-10-04 | Vývoj spočten (`results/dev.json`, N = 24 zapsáno do registru před výpočtem): **P = `Z_k10_z50_EQ_touch`**, Sharpe 0,28 (CAGR −14,8 %, propad 91 %, náklady 15 % p. a., 501 obchodů, medián čistého výnosu −4,7 %). Všech 24 kandidátů má Sharpe pod B_EW (0,77) i B_BTC (1,06); P je pod 96 % náhodných běhů B_RND (medián 0,73, 95. percentil 1,18). Kontrola: ceny všech obchodů P sedí s openy panelu. Pravidla se nemění; holdout podle §8 se spustí jen jednorázovým skriptem `r13_final.py`. | výběr podle §7 |
| 2026-10-04 | Metodika zamčena (`b035f81d…`), holdout otevřen jednou (commit `0a885a6`), bez pádu. Předtím smoke běh kopie skriptu na vývojových datech ve scratchpadu (bez trezoru a registru). | §11 |
| 2026-10-04 | **0 ze 6 kritérií → zamítnuto.** P na holdoutu Sharpe −0,92 (CAGR −66,7 %) vs B_EW −0,42, B_BTC 0,48, B_RND p95 0,05; P pod všemi 500 náhodnými běhy. Forward test se nespouští. | `REPORT.md` |
