# Výzkum 5 – krátkodobý reversal s trendovým filtrem, STR-TF (pre-registrace)

Verze 1.0 · 2026-09-27 · Stav: **pre-registrace, žádný výsledek STR-TF zatím spočten**.
Zdroj: zadání „Strategie: Short-Term Reversal s trendovým filtrem (STR-TF)“ (dále *zadání*),
převedené na data a framework výzkumů 1–4. Odchylky od zadání jsou v §12 i s důvodem.
Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §13.

## 1. Otázka

Mají likvidní US akcie v dlouhodobém uptrendu, které za 1–5 dní prudce spadly (vůči vlastní
volatilitě), po nákladech využitelnou tendenci k návratu v řádu dní? Nákupem poskytujeme
likviditu netrpělivým nebo nuceným prodejcům (Jegadeesh 1990, Lehmann 1990, Nagel 2012).
Druhý cíl zadání: ověřit, že engine nemá lookahead ani survivorship bias.

## 2. Poctivé omezení předem

- **Autor zná negativní předchozí výsledek.** Ve výzkumu 1 (in-sample 2000–2019) měla rodina
  `st_reversal` (měsíční rebalance, top-N podle −výnosu 3–10 d nad SMA 200) medián Sharpe 0,15
  a při nákladech ×3 zápornou hodnotu. STR-TF se liší: denní rozhodování, událostní vstup
  normalizovaný volatilitou, rychlý výstup. Rodina přesto není „nová“ a období 1999–2019 bylo viděno.
- **Validace 2015–2019 není čistá** (byla součástí vývojového období výzkumu 1).
- **Období 2020–2026 není čistý holdout** (spotřebovaly ho výzkumy 1–4). Spustí se jednou na konci
  přes samostatný trezor výzkumu 5 a hlásí se zvlášť jako sekundární evidence.
- **Jediný čistý test je forward (paper) test** (§11), minimálně 3 měsíce na datech po 2026-09-25.
  Vyžaduje obnovu předplatného Sharadaru (končí 2026-09-27) nebo jiný zdroj denních dat.

## 3. Data a univerzum

Snapshot `sharadar_2026-09-25`, normalizované `bars` výzkumu 1 (total-return výnosy
`ret_co`/`ret_oc`, neupravené ceny, politika delistingu §4.5 specifikace výzkumu 1).

Univerzum STR-TF k datu *t* (vše z dat do close *t* včetně):

| Filtr | Hodnota |
|---|---|
| Typ | `Domestic Common Stock` + `… Primary Class` (jako výzkum 1), bez SPAC (SIC 6770 k datu) |
| Cena | neupravený close ≥ 5 USD |
| Historie | ≥ 252 barů |
| Likvidita | ADV20 = průměr dolarového objemu za 20 dní ≥ `min_adv` |
| Delistované | v univerzu až do posledního obchodního dne |

Nový panel `panel_r5` obsahuje všechny tituly, které někdy splnily filtr s `min_adv` = 5 mil. USD
(nejširší bod mřížky). Panel LIQ1000 nestačí: hranice top-1000 je od roku 2008 přísnější než 20 mil. USD.
Panel navíc obsahuje neupravené `high_u`/`low_u` (IBS, ATR), SPY a SIC k datu (sektory).

Termíny výsledků hospodaření: 8-K Item 2.02 (Sharadar `events`, kód 22) podle data podání.
Použitelné až **od 2005**, protože dřívější pokrytí je neúplné.

## 4. Signál (základní verze)

Všechny indikátory se počítají z total-return indexu (upravené ceny). Cenový filtr univerza
bere neupravený close.

```
trend_ok     = TR > SMA(TR, trend_sma)
ret_L        = TR / TR.shift(lookback_ret) − 1
vol_20       = std(denní TR výnosy, 20)
z_rev        = ret_L / (vol_20 · sqrt(lookback_ret))
entry_signal = v univerzu ∧ trend_ok ∧ z_rev ≤ −entry_z
exit_signal  = TR > SMA(TR, exit_sma)
```

Alternativní signály jen pro srovnání (bez ladění): RSI(2) < 10 a IBS < 0,2 ∧ close < předchozí close.
RSI je z TR indexu, IBS z neupravených H/L/C téhož dne. Pořadí kandidátů u obou: RSI, resp. IBS
vzestupně.

## 5. Exekuce a portfolio

- Signál po close dne *t*, **vstup na open *t*+1**. Kandidáti se seřadí podle `z_rev` vzestupně
  a obsadí volné sloty. Titul netradovatelný na open (halt, bez open) se nekoupí.
- Výstup, podle toho, co nastane dřív:
  1. `exit_signal` na close → prodej na další open,
  2. time stop: po `max_hold` close v pozici → prodej na další open,
  3. delisting → výplata terminálního výnosu na open dne delistingu, bez nákladů.

  Pokud titul na open plánovaného prodeje nelze obchodovat, prodá se na nejbližší možný open.
- Stop-loss v základní verzi není.
- Velikost pozice: `equity_open / max_positions`, bez páky, zastropovaná na 1 % ADV20(*t*) v USD.
  Kapitál 1 mil. USD, citlivost 10 mil. USD. Titul drží nejvýš jednu pozici, bez přikupování
  a bez rebalance během držení.
- Hotovost v základu bez úroku (zadání), varianta s T-bill (DTB3).

## 6. Náklady (za stranu, zadání §6)

| Období | Slippage + ½ spreadu | Komise |
|---|---|---|
| do 2001-12-31 | 25 bps | 5 bps |
| 2002–2007 | 10 bps | 1 bp |
| od 2008 | 5 bps | 0,5 bps |

Slippage + ½ spreadu × `open_spread_mult` = 1,5 (všechny obchody jsou na open). Povinné běhy s náklady ×2 a ×3.
Doplňková citlivost: model výzkumu 1 (tiery podle pořadí likvidity; hlásí se, není kritériem).

## 7. Rozdělení dat

| Sada | Období | Použití |
|---|---|---|
| Rozehřátí | 1998-01 → 1998-12 | jen historie indikátorů (první možný signál po 252 barech) |
| Vývoj | 1999-01-04 → 2014-12-31 | mřížka, výběr, robustnost, kritéria §10 |
| Validace | 2015-01-01 → 2019-12-31 | jeden běh vybrané konfigurace |
| Pozdní období | 2020-01-01 → 2026-09-25 | jeden běh přes trezor výzkumu 5, sekundární (§2) |

- Vývojový kód čte panel jen do 2014-12-31. Validační skript čte do 2019-12-31. Data po 2019-12-31
  otevře jen trezor výzkumu 5 (`runs/vault_r5/`, zámek s hashem metodiky, log, jednou na snapshot).
  Každé jiné čtení vyhodí výjimku.
- Každý běh se zapíše do registru pokusů `runs/registry/research5.sqlite` (append-only)
  a do čitelného `runs/research5/experiments.log`.

## 8. Mřížka a výběr (vývoj)

| Parametr | Default | Mřížka |
|---|---|---|
| `entry_z` | 2,0 | 1,5 · 2,0 · 2,5 |
| `lookback_ret` | 3 | 2 · 3 · 5 |
| `trend_sma` | 200 | 100 · 200 |
| `exit_sma` | 5 | 3 · 5 · 10 |
| `max_hold` | 5 | 3 · 5 · 10 |
| `max_positions` | 10 | 10 · 20 |
| `min_adv` (mil. USD) | 20 | 5 · 20 · 100 |

Celkem **972** konfigurací (3·3·2·3·3·2·3; zadání uvádí 648, což je početní chyba, viz §13). **Pravidlo výběru, pevné předem.** Okolí konfigurace tvoří ona sama a
sousedé, kteří se liší o jeden krok v jednom parametru. Skóre = medián Sharpe (po nákladech,
vývoj) přes okolí. Vybírá se konfigurace s nejvyšším skóre. Při shodě na 2 desetinná místa
rozhoduje menší počet parametrů odlišných od defaultu. Default konfigurace se hlásí vždy vedle vybrané.

## 9. Robustnost (vývoj, vybraná konfigurace)

1. Heatmapy Sharpe: `entry_z × max_hold`, `lookback_ret × exit_sma`, `min_adv × max_positions`
   (ostatní parametry na vybrané hodnotě).
2. Vstup o den později (open *t*+2) a horní odhad MOC (vstup na close *t*, nikdy hlavní výsledek).
3. Náhodný benchmark, 1000 simulací. V každý den vstupu strategie se nakoupí stejný počet náhodných
   titulů z univerza s `trend_ok`. Každý obchod dostane délku držení odpovídajícího obchodu strategie
   (stejné pořadí). Sizing a náklady jsou stejné. Porovnává se čistý Sharpe. Seed 0.
4. Rozpady: rok, režim (2000–02, 2008–09, zbytek; ve validaci a pozdním období navíc 2020, 2022),
   ADV kvintily (k datu vstupu), sektor (SIC divize k datu), před a po 2002.
5. Varianty: filtr SPY > SMA 200, ATR stop (2 × ATR14, spouštěč na close, prodej na open),
   volatility targeting 12 % p. a. bez páky, RSI(2), IBS, hotovost s T-bill, kapitál 10 mil. USD,
   scénáře delistingu optimistic/pessimistic, ETF univerzum (24 ETF výzkumu 4), short strana
   (jen na úrovni obchodů, zrcadlové pravidlo: `z_rev ≥ +entry_z` pod SMA; bez nákladů na výpůjčku,
   proto jen horní odhad).
6. Vyřazení signálů s 8-K Item 2.02 v okně *t*−2…*t* (od 2005), porovnání se stejným obdobím bez vyřazení.

## 10. Kritéria zamítnutí (kill criteria, zadání §11)

Strategie se zamítá, pokud platí **cokoli** z tohoto:

1. průměrný hrubý výnos na obchod < 2 × průměrný náklad na obchod (tam i zpět), vývoj,
2. Sharpe ve validaci 2015–2019 < 0,7 (po nákladech),
3. CAGR ve vývoji při nákladech ×2 ≤ 0,
4. více než 50 % součtu ročních čistých log-výnosů za 1999–2019 pochází z let 1999–2002,
5. Deflated Sharpe (vývoj) < 0,95; N = všechny konfigurace v registru výzkumu 5, rozptyl Sharpe
   z mřížky (hlásí se i s N + 432 kandidáty `st_reversal` z výzkumu 1),
6. čistý Sharpe (vývoj) ≤ 95. percentil náhodného benchmarku,
7. izolovaná špička: méně než 2/3 přímých sousedů vybrané konfigurace má Sharpe ≥ 70 % jejího Sharpe.

Validace se spustí jednou, i když vývoj už něco zamítl (informace). Pozdní období se spustí jednou
na konci. Na výsledek se nic nemění.

## 11. Forward test

Jen pokud žádné kritérium §10 neplatí: zmrazení metodiky (hash, commit), minimálně 3 měsíce
paper tradingu na nových datech, porovnání realizované slippage s modelem §6.

## 12. Odchylky od zadání

| Zadání | Zde | Důvod |
|---|---|---|
| Holdout 2020 → dnes, zamčený | Pozdní období, sekundární | Spotřebován výzkumy 1–4 (§2) |
| Vývoj od 1998-01 | Od 1999-01-04 | Data začínají 1997-12-31, signál potřebuje 252 barů |
| Delisting bez výnosu: bankrot −30 % | Bankrot −100 %, výkonnostní −30 %, akvizice přesná protihodnota | Politika výzkumu 1 (konzervativnější), citlivost optimistic/pessimistic |
| „common stock“ | Domestic Common Stock (+ Primary Class), bez ADR, bez SPAC | Kategorie Sharadaru, audit výzkumu 1 |
| `experiments.log` | Append-only SQLite registr + čitelný log | Registr nejde zpětně upravit |
| `ALLOW_HOLDOUT=False` | Trezor (`qlab.validation.vault`) s hranicemi 2014-12-31 / 2019-12-31 | Existující, otestovaný mechanismus |
| Short strana | Jen úroveň obchodů, horní odhad | Engine je long-only, data o výpůjčkách chybí |
| „Střed plošiny“ | Mechanické pravidlo §8 | Výběr nesmí být úsudkem po zhlédnutí heatmap |

## 13. Plán a log rozhodnutí

| Krok | Obsah |
|---|---|
| R5.0 | Tato pre-registrace (commit před prvním výpočtem) |
| R5.1 | `panel_r5`, univerzum, testy správnosti enginu (zadání §12) |
| R5.2 | Signály, událostní simulátor, náklady, seznam obchodů, běh defaultu (vývoj) |
| R5.3 | Mřížka, výběr, robustnost, kritéria §10 (vývoj) |
| R5.4 | Validace 2015–2019 (jednou), pozdní období přes trezor (jednou), report |

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-09-27 | Pre-registrace v1.0 | – |
| 2026-09-27 | Mřížka §8 má 972 konfigurací, ne 648. Hodnoty parametrů beze změny, opraven jen počet. Před prvním výpočtem strategie. | Test `test_grid_and_neighbors` |
