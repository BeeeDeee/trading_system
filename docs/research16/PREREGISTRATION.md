# Výzkum 16 – Dávka 1: osm rodin intradenních strategií na 15m datech (pre-registrace)

Verze 1.0 · 2026-10-14 · Stav: **vývoj hotov, čeká na holdout**.
Navazuje na výzkum 15 (MTF 4h/1h/15m, zamítnut 4 ze 6). První dávka „vymyšlených“ strategií: osm rodin
s jednoduchým mechanismem, každá s malou mřížkou, všechny kandidáty se počítají do N. Po commitu verze 1.0
platí: změny = nový pokus v registru a řádek v §12.

## 1. Otázka

Má některá z osmi rodin (§4) na BTC a ETH čistý edge po nákladech, který **nevysvětlí náhodné vstupy se
stejnou expozicí** (stejná strana, měsíc, délka držení a časový kontext) a který přežije dvojnásobné náklady
a plnění o svíčku později?

## 2. Poctivé omezení předem

- **Holdout 2022-01 → 2026-09 už je spotřebovaný** (výzkumy 9–15, naposledy výzkum 15 na stejných 15m datech).
  Před touto pre-registrací vím: BTC/ETH v holdoutu rostly (B_HOLD Sharpe 0,37), long průrazy v trendu byly slabě
  kladné, shorty a liquidity sweep ztrácely, 15m vstupy jsou citlivé na zpoždění plnění. Historický test proto
  může rodinu jen **zamítnout, nebo kvalifikovat na forward test** (§10). Potvrzení dá jen forward.
- Proti známému růstu trhu chrání nula se **stejnou stranou a expozicí** (§6): drift zaplatí i náhodný vstup.
  Volba LONG/LS je v mřížce a rozhoduje o ní vývojové období, ne moje znalost holdoutu.
- Rodiny jsem vybral z literatury a tradingové praxe (seasonalita, ORB, squeeze, funding, CME gap, lead-lag).
  Je to výběr z ideí, které „někde fungovaly“, tedy s publikačním biasem. Mřížky jsou malé a pevné, aby výběr
  uvnitř rodiny nepřidal další.
- **Apriorní očekávání:** náklady 15 bps na kolo jsou u rodin s držením 1–4 h (SHOCK, RANGE, LEAD) velmi
  pravděpodobně smrtelné. Realistický výsledek je 0, nanejvýš 1 rodina přes holdout.

## 3. Data

Panel z výzkumu 15 beze změny (`data/derived/binance_2026-10-03/r15_15m/`, `docs/research15/DATA_AUDIT.md`):
spot BTCUSDT a ETHUSDT 15m 2017-08-17 → 2026-09-30 (OHLC, `real`), funding perpu od 2020-01-01. Objem v panelu
není a žádná rodina ho nepoužívá.

## 4. Rodiny a mřížka (vše jen z dat uzavřených do konce 15m svíčky *k*, vstup za close *k*)

Společné: σ = směrodatná odchylka 15m log výnosů za posledních 672 reálných svíček (7 dní); σ₁ₕ = 2σ.
1h a 4h bary, 4h režim (EMA 20/50, `regime_4h`) a 15m potvrzení (close nad maximem / pod minimem 4 předchozích
svíček) přesně podle výzkumu 15. Bollinger 1h = SMA 20 uzavřených 1h close ± m · sm. odchylka.

| Rodina | Mechanismus | Vstup (long; short zrcadlově) | Výstup | Mřížka | N |
|---|---|---|---|---|---|
| **ORB** | otevření UTC dne | po uzavření rozpětí prvních *H* hodin UTC dne první close nad jeho maximem; max 1 obchod denně | stop = opačný okraj rozpětí; jinak close 23:45 UTC | *H* ∈ {1, 4} h × {LS, LONG} | 4 |
| **SQZ** | stlačení volatility → expanze | poslední uzavřený 1h bar má šířku Bollingera (m = 2) ≤ *q*-kvantil šířek 720 předchozích 1h barů; 15m close nad horním pásmem | stop = střed pásma; cíl 2R; čas 48 h | *q* ∈ {0,1; 0,2} × {LS, LONG} | 4 |
| **SHOCK** | likvidační kaskáda → návrat | 1h pohyb `c[k]/c[k−4] − 1 < −K·σ₁ₕ` (5 reálných svíček) | cíl = `c[k−4]`; stop = vstup − ½ pohybu; čas 4 h | *K* ∈ {4, 6} × {LS, LONG} | 4 |
| **FUND** | přeplněné pozice ve fundingu | ve vypořádání (00/08/16 UTC) je sazba ≤ (1−*q*)-percentil posledních 270 sazeb (90 dní) → long; ≥ *q* → short | čas 72 h, bez stopu | *q* ∈ {0,90; 0,95} × {LS, LONG} | 4 |
| **CME** | víkendová mezera CME futures | ve svíčce znovuotevření CME (ne 17:00 Chicago) mezera `g = c[k]/F − 1`, F = close svíčky končící zavřením v pá 16:00 Chicago; \|g\| ≥ *G* → obchod směrem k F | cíl = F; stop = vstup ± ½ \|vstup − F\| dál od F; čas 72 h | *G* ∈ {1 %, 2 %}, vždy obě strany | 2 |
| **PULL** | návrat v trendu | 4h režim nahoru, RSI(14) posledního 1h baru ≤ *L* → setup aktivní 16 svíček; vstup na 15m potvrzení | stop = minimum posledních 32 svíček; cíl 2R; čas 48 h | *L* ∈ {30, 40} × {LS, LONG} | 4 |
| **LEAD** | BTC vede ETH | BTC 1h pohyb `|rB| ≥ K·σ₁ₕ(BTC)` a ETH zaostává (`rE/rB < 0,5`) → ETH ve směru rB | čas 4 h, bez stopu | *K* ∈ {2, 3} × {LS, LONG} | 4 |
| **RANGE** | návrat k průměru bez trendu | 4h režim neutrální (0); 15m close pod dolním 1h pásmem (m) | cíl = střed pásma při vstupu; stop ve stejné vzdálenosti pod vstupem (1R); čas 4 h | *m* ∈ {2; 2,5} × {LS, LONG} | 4 |

**N = 30.** Pro DSR se hlásí N = 30 i kumulativně N = 46 (s 16 kandidáty výzkumu 15 na stejných datech).

Upřesnění:
- `LONG` = jen long signály. `LS` = long i zrcadlové short signály. U FUND je short při sazbě ≥ *q*-percentil,
  long při ≤ (1−*q*). U CME je strana dána směrem k F, `LONG/LS` se nevolí.
- CME časy se počítají v `America/Chicago` (letní čas podle USA). Svátky se ignorují (pevný týdenní rozvrh).
  BTC od 2018-02 (futures od 2017-12-17), ETH až od **2021-02-08** (spuštění ETH futures).
- FUND má data od 2020-01 a 90denní warm-up, signály tedy od 2020-04. Vývoj FUND je kratší (21 měsíců).
- LEAD obchoduje jen ETH. Portfolio LEAD = rukáv ETH 50 % + 50 % hotovost (stejná váha jako jinde).
- Stop/cíl s rizikem mimo 0,15–5 % → obchod se nevstoupí (jako výzkum 15). Netýká se FUND a LEAD (bez stopu).

## 5. Obchod

Simulátor výzkumu 15 (`trade_book`, pořadí stop → cíl, mezery za open, stop má přednost ve stejné svíčce),
rozšířený o výstup v pevném čase a obchody bez stopu. Na aktivum a rodinu nejvýš jedna pozice; signály během
pozice se ignorují. Portfolio 50/50 BTC/ETH, pozice ±1 nominálu rukávu. **Náklady 7,5 bps na stranu**, zátěž ×2.
**Funding** podle výzkumu 15 (sazba perpu, před daty 1 bp / 8 h). Ceny ze spotu.

## 6. Nula a benchmarky

- **B_RND (hlavní nula, 1000 běhů, seed 0..999):** pro každé aktivum, stranu a kalendářní měsíc stejný počet
  obchodů jako P_f, **délka držení** losovaná (s vracením) z obchodů P_f téhož aktiva a strany, výstup jen časem.
  Vstupy v náhodných reálných svíčkách bez otevřené pozice a se **stejným kontextem** rodiny:

  | Rodina | Kontext náhodného vstupu |
  |---|---|
  | ORB | svíčky téhož UTC dne po konci rozpětí, výstup nejpozději 23:45 |
  | FUND | svíčky vypořádání (00/08/16 UTC) |
  | CME | **stejné svíčky znovuotevření, náhodná strana** (testuje směr, ne načasování) |
  | PULL | stejný 4h režim jako strana |
  | RANGE | 4h režim 0 |
  | SQZ, SHOCK, LEAD | libovolná reálná svíčka (LEAD jen ETH) |

  p-hodnota = (1 + #{běhy se Sharpe ≥ P_f}) / 1001.
- **B_HOLD** (50/50 BTC/ETH, bez nákladů) a **B_4H** (trvale ve směru 4h režimu) pro srovnání.

## 7. Období, výběr a trezor

| Fáze | Období |
|---|---|
| Vývoj | 2018-02-01 → 2021-12-31 (FUND 2020-04-01 →; CME u ETH 2021-02-08 →) |
| Holdout | **2022-01-01 → 2026-09-30**, jednou, přes `Vault` (`runs/vault_r16/`) |

- **Výběr v rodině:** P_f = nejvyšší vývojový Sharpe mezi variantami s ≥ 30 obchody. Shoda → první v pořadí mřížky.
- **Vývojová brána** (rozhoduje, které rodiny jdou do holdoutu): P_f má na vývoji Sharpe > 0 při nákladech ×2
  **a** p-hodnotu B_RND < 0,05 (bez korekce). Rodiny bez P_f nebo bez brány se uzavírají ve vývoji, holdout je
  nevidí. *k* = počet rodin, které bránou projdou.

## 8. Kritéria (holdout, každá rodina zvlášť, nutné splnit **všechna**)

1. Sharpe P_f > 0 a průměrný čistý výnos obchodu > 0 s dolní mezí 90% intervalu > 0 (bootstrap, 2000, seed 0).
2. p-hodnota B_RND ≤ **Holmova hranice** přes *k* rodin na hladině 0,05.
3. Sharpe > 0 při nákladech ×2.
4. Sharpe > 0 při plnění o svíčku později (close *k*+1). *Nové proti výzkumu 15, kde to bylo jen v robustnosti.*
5. Sharpe > 0 v obou podúsecích 2022-01 → 2023-12 a 2024-01 → 2026-09.
6. Průměrný čistý výnos obchodu > 0 zvlášť na BTC i ETH (u LEAD neplatí, má jen ETH).

DSR (N = 30 a N = 46) se hlásí, není kritériem.

## 9. Robustnost (jen hlášení)

Všech 30 variant na holdoutu (i rodin zastavených branou, **až po** vyhodnocení §8), výsledky po letech, bez
fundingu, long a short zvlášť, statistiky obchodů podle důvodu výstupu, ceny perpu 2020-01 → 2026-09,
sekundární koš altcoinů z výzkumu 15 (top 10 podle objemu, jen u rodin, které splní §8; LEAD a CME vynechány).

## 10. Co znamená výsledek

- Rodina, která nesplní cokoli z §8: **zamítnuta**.
- Rodina, která splní vše: **forward paper test** na datech od 2026-10-01 s pevnými pravidly P_f. Vyhodnocení
  na datech do 2027-03-31: Sharpe > 0 a průměrný obchod > 0 při nákladech ×1. Kritérium je pevné už teď.
- Dávka jako celek slouží i jako šablona: další dávky přidávají rodiny do stejného rámce, N se sčítá.

## 11. Implementace a postup

1. Schválení vlastníkem, verze 1.0, commit (před jakýmkoli výpočtem signálů nad cenami).
2. `qlab.research16`: `engine` (obecný simulátor nad `qlab.research15.strategy`: pole vstupů, stran, stopů, cílů
   a nejzazších výstupů na obchod), `families` (8 generátorů signálu se společným rozhraním), `null` (B_RND
   s kontextem). Testy: bod v čase pro **každou** variantu (oříznutí a perturbace budoucích svíček), ruční
   přepočet jednoho obchodu na rodinu, CME časy přes letní/zimní čas, nepřekrývání, kontext nuly.
3. Commit kódu. Registr `runs/registry/research16.sqlite`: všech 30 variant zapsat před výpočtem.
4. `scripts/r16_dev.py` (vývoj, výběr P_f, brána) → `scripts/r16_final.py freeze|run` (jednorázový holdout).
5. `REPORT.md`, řádek v README, verdikt do §12.

## 12. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-14 | Verze 0.1, návrh. |
| 2026-10-14 | Verze 1.0 schválena vlastníkem beze změn (dotaz na rozdělení období zodpovězen, §7 platí). |
| 2026-10-14 | Upřesnění implementace (rozhodnuto při psaní kódu, před jakýmkoli výpočtem nad cenami; pravidla §4–8 se nemění): (a) σ = sm. odchylka (ddof 1) 15m log výnosů mezi dvěma reálnými svíčkami v okně 672 svíček končícím *k*, nutno ≥ 75 % platných, jinak bez signálu. (b) ORB: rozpětí jen ze dnů, kde jsou všechny svíčky rozpětí reálné; kandidát dne je první průraz (u LONG první průraz nahoru); nevstoupí-li (meze rizika), den je vyčerpán. (c) SQZ, RANGE, SHOCK, PULL, LEAD: kandidátem je každá svíčka splňující podmínku; během pozice se ignorují, po výstupu platí další. PULL: setup = 1h bar s RSI ≤ *L* (≥ 100 − *L*) a režimem ve směru při jeho uzavření, aktivní svíčky 4j+4 … 4j+19. (d) FUND: ostré nerovnosti (> *q*-kvantil, < (1−*q*)-kvantil 270 předchozích záznamů, lineární interpolace), protože sazby se kupí na 0,01 %; „neostré“ by v klidu otevíralo short při každé základní sazbě. (e) CME: F = close svíčky končící pá 16:00 Chicago, *k* = svíčka začínající ne 17:00 Chicago; obě musí být reálné; ETH od víkendu, jehož pátek je ≥ 2021-02-08. (f) B_RND vystupuje jen časem (bez stopu a cíle); u CME stejné vstupní svíčky jako P_f, náhodná strana, délka z obchodů P_f; ORB nejpozději 23:45 téhož dne. Kvóty (měsíc, strana) nemusí jít vždy naplnit (FUND: 72 h na 3 svíčkách denně), pak se naplní, co jde. (g) Minimálně 30 obchodů platí pro výběr P_f na vývoji. (h) Neprojde-li bránou žádná rodina, holdout se neotevírá. (i) Mřížka FUND „LS“ = long i short podle §4, „LONG“ = jen long při sazbě pod (1−*q*)-kvantilem. |
| 2026-10-14 | **Zveřejnění:** před commitem kódu proběhl smoke test `r16_dev.py` na vývojových datech (registr do scratchpadu, B_RND jen 20 běhů), viděl jsem tedy vývojová čísla všech 30 variant. Kód ani pravidla jsem po něm neměnil. Ostrý vývojový běh (1000 běhů nuly) následuje po commitu kódu. |
| 2026-10-14 | **Vývoj** (`results/dev.json`, 1000 běhů B_RND): bránou prošly **ORB** (P = `ORB_H4_LONG`, Sharpe 1,67, ×2 náklady 0,85, p = 0,001, 1569 obchodů), **FUND** (`FUND_Q95_LONG`, 1,88 / 1,80, p = 0,001, jen 47 obchodů za 2020-04 → 2021-12) a **LEAD** (`LEAD_K3_LONG`, 1,28 / 0,93, p = 0,003, 113 obchodů). Neprošly: SQZ (Sharpe −0,07), SHOCK (−1,11), CME (0,24, ×2 náklady −0,01, p = 0,21), PULL (−0,47), RANGE (−1,90). Všechna tři P_f jsou jen long a vývoj (B_HOLD Sharpe 0,89) byl býčí; holdout tedy rozhodne hlavně nula se stejnou stranou. *k* = 3, Holmovy hranice 0,0167 / 0,025 / 0,05. |
