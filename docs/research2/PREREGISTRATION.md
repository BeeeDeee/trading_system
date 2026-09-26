# Výzkum 2 – multi-asset alokace s trendem (pre-registrace)

Verze 1.0 · 2026-09-26 · Stav: **pre-registrace, žádný výsledek strategie na ETF zatím spočten**.
Znovu používá framework výzkumu 1 (data, enginy, test úniku, statistika, registr, trezor).
Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §10.

## 1. Otázka

Překoná diverzifikovaná alokace přes třídy aktiv s trendovým filtrem na každé třídě zvlášť po
nákladech portfolio 60/40 v poměru výnosu k riziku – a obstojí ve forward testu?

Účel: **čistý výzkum** (bez daní a konkrétního brokera), stejně jako výzkum 1.

## 2. Poctivé omezení předem

- **Historie 2007–2026 není čistý holdout.** Období 2020–2026 bylo viděno ve výzkumu 1 a je
  obecně známé (covid, dluhopisový krach 2022, inflace). Návrh níže je jím nevyhnutelně
  ovlivněn – konkrétně **vynechání overlaye celého portfolia podle volatility** vychází
  z poučení z holdoutu výzkumu 1. Historický walk-forward je proto jen výzkumná evidence.
- **Jediný čistý test je forward test** (§8): metodika se zmrazí a sleduje se 12 měsíců na datech,
  která dnes neexistují.
- **Výběr ETF** je výběr dnešních „přeživších" vlajkových fondů. Zmírnění: všechna byla už v roce
  2006 hlavními zástupci své třídy; robustnost záměnou za alternativní ETF (§7).

## 3. Univerzum

| Třída aktiv | ETF | Alternativa (robustnost) |
|---|---|---|
| Akcie USA | SPY | IWM není substitut → bez alternativy |
| Akcie rozvinuté mimo USA | EFA | VEU (od 2007-03) |
| Akcie rozvíjející se trhy | EEM | VWO |
| Státní dluhopisy USA 7–10 let | IEF | AGG |
| Státní dluhopisy USA 20+ let | TLT | – |
| Dluhopisy chráněné proti inflaci | TIP | – |
| Zlato | GLD | IAU |
| Komodity | DBC | GSG |
| Nemovitosti (REIT) | VNQ | IYR |

- Data: Sharadar `funds` (total return z `closeadj`), společná historie od 2006-02-06.
- Hotovost: 3M T-bill (FRED `DTB3`), stejně jako výzkum 1.
- Náklady: 5 bps na stranu pro všechna ETF (nejlikvidnější úroveň modelu), citlivost ×2 a ×3.

## 4. Hypotézy

**H1 (primární, bez výběru):** předem pevně daná strategie
*váhy podle inverzní volatility (126 dní), trendový filtr SMA200 na každém ETF zvlášť
(pod SMA → podíl do hotovosti), měsíční rebalance, max. váha 40 %*
má vyšší Sharpe než 60/40.

H1 nemá žádný výběr parametrů → N_meth = 1, žádná deflace výběrem. Závěr výzkumu stojí na H1.

**H2 (sekundární, s výběrem):** walk-forward výběr z mřížky (§5) stejnou funkcí `select()`
jako ve výzkumu 1 (K = 2, rovné váhy) překoná 60/40. Slouží k odhadu, zda výběr parametrů
přidává hodnotu nad pevnou volbu H1.

## 5. Mřížka kandidátů (H2)

| Blok | Varianty |
|---|---|
| Vážení | rovné (1/9); inverzní volatilita 63 / 126 / 252 dní |
| Trendový filtr na ETF | žádný; SMA 100; SMA 200; 6měsíční TS momentum; 12měsíční TS momentum (výnos nad T-bill > 0) |
| Relativní momentum | žádné; držet 3 nebo 5 nejlepších ETF podle 6m / 12m výnosu |
| Rebalance | měsíční (týdenní vyloučeno – výzkum 1: horší ve všech rodinách) |
| Max. váha | 40 % |

Celkem 4 × 5 × 5 = 100 kandidátů. Overlay celého portfolia se nepoužívá (§2).

## 6. Validace a kritéria

- Warm-up 2006-02 → 2007-01 (12 měsíců pro nejdelší signál), první rozhodnutí 2007-01-31.
- Walk-forward: testovací roky **2010 … 2026** (poslední do 2026-09-25), trénink rozšiřující se
  od 2007-02, embargo 21 dní. H1 se nevybírá – běží celé období 2007-02 … 2026-09.
- Benchmarky: **60/40 = 60 % SPY + 40 % IEF, měsíčně rebalancované** (primární); SPY; rovné váhy
  všech 9 ETF měsíčně.
- Statistika: stationary bootstrap (blok 21 dní, 90% CI), DSR (H1: N = 1; H2: N_meth z registru),
  SPA vůči 60/40.

**Kritérium úspěchu (výzkumná evidence) pro H1, období 2010–2026:**
1. dolní mez 90% CI rozdílu Sharpe H1 − 60/40 > 0,
2. maximální propad H1 ≤ 25 %,
3. Sharpe při nákladech ×3 stále vyšší než 60/40.

**Forward test (jediné rozhodující kritérium, §8):** H1 za 12 měsíců forward nesmí zaostat za
60/40 o víc, než odpovídá 10. percentilu rozdělení 12měsíčních rozdílů z walk-forward období
2010–2026 (předem spočtené a zapsané při zmrazení).

## 7. Robustnost (předem dané)

1. Náklady ×2, ×3.
2. Záměna ETF za alternativy z §3 (všechny najednou), společná historie od 2007-03.
3. H1 s SMA 150 a SMA 250 (citlivost na jediný parametr trendu).
4. Období: zvlášť 2010–2019 a 2020–2026.

## 8. Forward test

- Zmrazení metodiky (trezor, hash, commit) po dokončení historické části.
- Období 2026-10-01 … 2027-09-30. Každý měsíc: po close prvního obchodního dne měsíce spočítat
  cílové váhy H1 z nových dat a zapsat je do append-only logu **před** dalším obchodním dnem.
- Zdroj dat: volně dostupné denní ceny ETF (bez Sharadaru); adaptér se ověří proti Sharadaru
  na překryvu do 2026-09-25 (tolerance TR rozdílu ≤ 0,1 p.b. ročně).
- Vyhodnocení po 12 měsících bez jakékoli úpravy metodiky.

## 9. Postup prací

| Krok | Obsah |
|---|---|
| R1 | Panel ETF (funds → normalizace → panel), audit: TR vs. známé hodnoty, kalendář, rozdělení/distribuce |
| R2 | Signály a portfolio (inverzní vol., trend, relativní momentum), test úniku, H1 běh, mřížka H2, walk-forward, statistika, robustnost, report |
| R3 | Zmrazení, adaptér forward dat, protokol a skript měsíčních vah |

## 10. Rozhodovací log

| Datum | Rozhodnutí | Zdůvodnění |
|---|---|---|
| 2026-09-26 | Pre-registrace v1.0 | Výzkum 1 uzavřen negativně; nová otázka schválena uživatelem, čistý výzkum |
| 2026-09-26 | R1: panel 15 ETF, audit v `DATA_AUDIT.md` | Bez výsledků strategií |
| 2026-09-26 | Upřesnění před R2 (žádný výsledek strategie ještě nespočten): (a) rozhodnutí po close prvního obchodního dne měsíce (jako výzkum 1); ETF je způsobilé až po 252 dnech historie → první rozhodnutí 2007-03-01; (b) pořadí kroků: způsobilost → relativní momentum (top-k) → základní váhy mezi vybranými (EW / inverzní vol.) → strop 40 % s přerozdělením → trendový filtr (neprojde-li ETF, jeho váha jde do hotovosti); (c) H2 tvrdé filtry = výzkum 1 kromě min. počtu pozic (zde 1); dedup 0,85; K = 2; (d) okolí v mřížce: liší se jedna dimenze o jeden krok v pořadí z §5; (e) registr pokusů výzkumu 2 je samostatný (`runs/registry/research2.sqlite`) | Pre-registrace tyto detaily nepokrývala |
