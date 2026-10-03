# Výzkum 9 – trend a momentum v kryptu (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **uzavřeno, zamítnuto** (`REPORT.md`).
Navazuje na výzkum 8 (data Binance, trezor, registr) a na vektorový engine výzkumů 1–4.
Změny po commitu = nový pokus v registru a řádek v §10.

## 1. Otázky

- **T (časová řada):** Zlepší jednoduchý trendový filtr (držet coin jen v rostoucím trendu, jinak USDT)
  rizikově vážený výnos BTC a ETH proti jejich držení?
- **X (průřez):** Vydělává týdenní výběr likvidních coinů s nejsilnějším minulým výnosem víc než rovné váhy
  celého likvidního univerza?

Literatura (např. Liu & Tsyvinski 2021) hlásí v kryptu časové i průřezové momentum. Výzkum 1 ukázal,
že na amerických akciích momentum po nákladech nepřežilo. Výsledek slouží i jako **laťka pro krypto paper
bota**: když jednoduchý filtr bez LLM poráží BTC HOLD, LLM musí porazit tento filtr, ne BTC.

## 2. Poctivé omezení předem

- **Historie kryptoměn je obecně známá** (medvědí trhy 2018 a 2022, býčí 2020–21 a 2024). Že trendové
  filtry v takových letech pomáhají, je známé a publikované, a autor i Claude to vědí. Holdout proto není
  „neviděný“ v pravém smyslu. Historický test může hypotézu **zamítnout, nebo kvalifikovat na forward test**
  (§9), nic víc.
- Jen long (spot), bez páky a bez shortu. Hotovost (USDT) nenese nic; s T-billem jen jako robustnost.
- Jediná burza (Binance), jen páry proti USDT.

## 3. Data

- `data.binance.vision`: denní svíčky **všech** spotových párů `XUSDT`, které kdy existovaly (S3 listing,
  včetně delistovaných), měsíční ZIPy ověřené proti `.CHECKSUM`. Snapshot `binance_2026-10-03`
  (spot za 2026-09 v něm ještě není, viz výzkum 8 → holdout končí **2026-08-31**).
- Vyřazeno: stablecoiny, wrapped tokeny a fiat (seznam z výzkumu 8), pákové tokeny (`…UP`, `…DOWN`,
  `…BULL`, `…BEAR`).
- Výnosy z cen open/close. Delisting: pozice se vyplatí za poslední close − 2 % na openu dalšího dne.
- Audit dat bez výnosových statistik, včetně kontroly posledního dne dat. Mezera ve svíčkách uprostřed
  historie = coin ten den nejde obchodovat.

## 4. Pravidla

- Rozhodnutí po close dne *t* (00:00 UTC), plnění na openu *t*+1 (engine `qlab.engine.vector`).
- Náklady za stranu: poplatek 10 bps + slippage podle průměrného denního quote objemu za 30 dní
  (≥ $1 mld. 2 bps · $200 mil.–1 mld. 5 bps · $50–200 mil. 10 bps · < $50 mil. 25 bps). Zátěž ×2.
- **Univerzum k datu:** top 20 párů podle 30denního průměrného quote objemu, obchodované ≥ 60 dní.

### 4.1 Kandidáti T (8): BTC 50 % + ETH 50 %, každý coin filtrován zvlášť, denně

| id | Držet coin, když | Parametr |
|---|---|---|
| `T_sma{n}` | close > SMA(n) | n ∈ {20, 50, 100, 200} |
| `T_mom{L}` | výnos za L dní > 0 | L ∈ {7, 14, 30, 90} |

Obchoduje se jen při změně signálu (ne denní rebalance). Když se oba coiny drží, nově vstupující dostane
50 % NAV. **Benchmark T:** 50/50 BTC/ETH, rebalance první den v měsíci, stejné náklady.

### 4.2 Kandidáti X (18): týdně v pondělí

Top K coinů univerza podle výnosu za L dní (jen výnos > 0), rovné váhy 1/K (nevyužité místo = USDT).
K ∈ {3, 5, 10}, L ∈ {7, 30, 90}, tržní filtr ∈ {vypnutý, zapnutý: když BTC close < SMA100, vše USDT}.
**Benchmark X:** rovné váhy celého univerza top 20, týdně, stejné náklady. Hlásí se i BTC HOLD.

Celkem **N = 26** pokusů.

## 5. Období a trezor

| Fáze | Období |
|---|---|
| Vývoj | 2018-04-01 → 2021-12-31 (BTC/ETH spot od 2017-08; SMA200 potřebuje ~200 dní) |
| Holdout | **2022-01-01 → 2026-08-31**, jednou, přes `Vault` (zámek metodiky, log otevření) |

## 6. Kritéria (holdout; nutné splnit **všechna**)

Hodnotí se **P_T** = nejlepší kandidát T podle Sharpe ve vývoji a **P_X** = nejlepší kandidát X podle
Sharpe ve vývoji, každý proti svému benchmarku. Aktivní výnos = denní výnos strategie − benchmarku.

1. Rozdíl Sharpe (strategie − benchmark) > 0 a dolní mez 90% intervalu (stationary bootstrap párově,
   blok 21, 1000 vzorků, seed 0) > 0; Holm přes P_T a P_X (jednostranné bootstrap p),
2. rozdíl Sharpe > 0 v obou podúsecích **2022-01 → 2023-12** a **2024-01 → 2026-08**,
3. rozdíl Sharpe > 0 při zátěžových nákladech,
4. P_T: max. propad menší než benchmark. P_X: CAGR vyšší než benchmark,
5. Deflated Sharpe aktivního výnosu ≥ 0,95 (N = 26, rozptyl Sharpe z vývojové mřížky).

Sharpe = průměr / sm. odchylka denních výnosů × √365, bez odečtu bezrizikové sazby.

## 7. Robustnost (jen hlášení)

Výsledky po letech; všichni kandidáti T a X na holdoutu (jen popisně, nevybírá se podle nich); hotovost
nesoucí T-bill; plnění o den později; univerzum top 10 a top 40; samostatně BTC a ETH u T.

## 8. Implementace

`qlab.research9`: stažení spotu přes `qlab.research8.download`, panel ve formátu `qlab.data.panel.Panel`
(`ret_co` = open/předchozí close, `ret_oc` = close/open, delisting řádek), signály, kandidáti, nákladová
matice. Testy: bod v čase (perturbace budoucnosti), pákové tokeny vyřazené, delisting, ruční přepočet
jednoho obchodu.

## 9. Co znamená výsledek

- Nesplní-li P_T i P_X cokoli z §6: zamítnuto.
- Splní-li to P_T nebo P_X: kandidát na **forward paper test ≥ 6 měsíců** jako nová varianta krypto paper
  bota (nový release až po skončení jeho pre-registrovaného okna 2026-11-30).

## 10. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-10-03 | Pre-registrace v1.0 | – |
| 2026-10-03 | Data: 664 spotových párů USDT (26 013 souborů, 0 chyb), panel 2017-08-01 → 2026-08-31 (`DATA_AUDIT.md`). Univerzum je ve vývoji malé (2018-04-01: 7 obchodovatelných párů; na Binance se tehdy obchodovalo hlavně proti BTC), od 2022 stovky párů. Extrémní denní pohyby (> 90 %) zkontrolovány ručně: skutečné události (DOGE 2021-01, LUNA 2022-05, pumpy malých coinů). Nový LUNA pod stejným tickerem je díky mezeře > 7 dní nový listing. Nic se nemění. | audit |
| 2026-10-03 | Vývoj spočten (`results/dev.json`): **P_T = `T_sma20`**, **P_X = `X_K10_L30_mf`**, N = 26. | výběr podle §6 |
| 2026-10-03 | Oprava kódu před zamčením: `max_drawdown` vrací kladnou velikost propadu, kritérium 4 pro P_T v `r9_final.py` porovnávalo obráceně. Pravidlo se nemění. | chyba kódu |
| 2026-10-03 | Metodika zamčena (`c885d5d9…`), holdout otevřen (commit `1522234`). Skript spadl **po otevření** při zápisu JSON (numpy bool). Rerun téhož zamčeného skriptu přes `scripts/r9_final_rerun.py` (jen převod typů pro JSON, druhé otevření zapsané jako `rerun` ve `vault.log`). | pravidlo frameworku: pád po otevření → zalogovaný rerun |
| 2026-10-03 | **P_T i P_X nesplnily kritéria → zamítnuto.** Forward test se nespouští. | `REPORT.md` |
