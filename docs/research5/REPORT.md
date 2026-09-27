# Výzkum 5 – krátkodobý reversal s trendovým filtrem (STR-TF): report

2026-09-27 · Pre-registrace: [PREREGISTRATION.md](PREREGISTRATION.md) · Výsledky:
[dev](results/dev_summary.json) · [validace](results/validation_summary.json) ·
[pozdní období](results/late_summary.json)

## Verdikt

**Zamítnuto.** Strategie nesplnila 4 ze 7 předem stanovených kritérií (§10). Forward test se nespouští.

| # | Kritérium | Hodnota | Hranice | Výsledek |
|---|---|---|---|---|
| K1 | Hrubý výnos / náklady na obchod (vývoj) | 3,06 | ≥ 2 | prošlo |
| K2 | Sharpe validace 2015–2019 | **0,32** | ≥ 0,7 | **zamítá** |
| K3 | CAGR vývoj při nákladech ×2 | 5,6 % | > 0 | prošlo |
| K4 | Podíl zisku 1999–2019 z let 1999–2002 | **53 %** | ≤ 50 % | **zamítá** |
| K5 | Deflated Sharpe (N = 993) | **0,52** | ≥ 0,95 | **zamítá** |
| K6 | Sharpe vs. 95. percentil náhodného benchmarku | 1,02 vs. 0,21 | nad | prošlo |
| K7 | Sousedé s Sharpe ≥ 70 % vybrané | **62,5 %** | ≥ 2/3 | **zamítá** (těsně) |

## Co se stalo

Vybraná konfigurace (pravidlo §8, medián Sharpe okolí): `entry_z` 2,0, `lookback_ret` 5,
`trend_sma` 200, `exit_sma` 3, `max_hold` 10, `max_positions` 10, `min_adv` 100 mil. USD.

| Období | Sharpe | CAGR | Max. propad | Obchodů | Hrubě/obchod | Náklad/obchod | CAGR při ×2 |
|---|---|---|---|---|---|---|---|
| Vývoj 1999–2014 | 1,02 | 11,6 % | 14 % | 2 993 | 92 bps | 30 bps | 5,6 % |
| Validace 2015–2019 | 0,32 | 2,6 % | 18 % | 1 516 | 26 bps | 16 bps | −2,2 % |
| Pozdní 2020–2026* | 0,37 | 4,4 % | 26 % | 2 396 | – | – | −1,4 % |
| SPY 2020–2026 | 0,82 | 15,5 % | – | – | – | – | – |

\* Sekundární evidence, období bylo spotřebováno výzkumy 1–4.

1. **Edge je z velké části rok 2000.** Rok 2000 vynesl +113 % při průměrné investovanosti
   17 %. Šlo o skutečné odrazy po krachu Nasdaqu (např. 17. 4. 2000 BRCM +31 %, ORCL +24 % za den),
   ne o chybu dat. Období 1999–2002 dává 53 % zisku 1999–2019 a Sharpe před 2002 je 1,87,
   po 2002 jen 0,72.
2. **Výběr je špička, ne plošina.** Na mřížce 972 konfigurací je medián Sharpe 0,29. Hodnota 1,02
   odpovídá tomu, co by čekalo nejlepší z ~1000 pokusů bez edge (DSR 0,52). V heatmapě je
   `entry_z` = 2,0 úzký hřeben, při 1,5 i 2,5 Sharpe klesá na zhruba 0,6.
3. **Mimo vývoj se hrubý edge smrskl** z 92 na 26 bps na obchod. Při nákladech ×2 je CAGR
   ve validaci i v pozdním období záporný.
4. **Citlivost na exekuci je extrémní:** vstup o den později (open *t*+2) sráží Sharpe
   z 1,02 na 0,31. Horní odhad MOC dává 1,32. Edge žije v první noci po signálu.

## Robustnost (vývoj, vybraná konfigurace)

| Varianta | Sharpe | CAGR | Poznámka |
|---|---|---|---|
| Základ | 1,02 | 11,6 % | |
| Náklady ×2 / ×3 | 0,54 / 0,04 | 5,6 % / −0,2 % | |
| Model nákladů výzkumu 1 (tiery) | 1,31 | 15,5 % | levnější pro tituly s ADV ≥ 100M |
| Vstup open *t*+2 | 0,31 | 2,7 % | edge rychle mizí |
| MOC (horní odhad) | 1,32 | 14,2 % | nikdy hlavní výsledek |
| Filtr SPY > SMA 200 | 0,87 | 7,3 % | |
| ATR stop 2× | 0,98 | 11,0 % | stop nepomáhá (jak čekáno) |
| Vol. targeting 12 % | 0,95 | 9,6 % | |
| RSI(2) < 10 | 0,43 | 7,4 % | |
| IBS < 0,2 | −0,41 | −11,4 % | náklady ho zničí |
| Hotovost s T-bill | 1,17 | 13,5 % | |
| Kapitál 10 mil. USD | 1,01 | 9,4 % | limit 1 % ADV začíná brzdit |
| Delisting optimistic / pessimistic | 1,03 / 1,02 | | bez vlivu |
| ETF univerzum (24 ETF) | 0,14 | 0,3 % | 253 obchodů |
| Bez signálů do 2 dnů po 8-K 2.02 (2005–2014) | 0,77 vs. 0,71 | | vyřazení earnings mírně pomáhá |
| Default konfigurace ze zadání | 0,42 | 6,3 % | při ×2: −0,38 |
| Short strana (jen obchody, bez výpůjčky) | – | – | 91 bps hrubě, 55 bps čistě na obchod |

Náhodný benchmark (1000 simulací, stejné dny vstupu, délky držení, sizing i náklady):
medián Sharpe 0,02, 95. percentil 0,21. **Výběr titulů podle z-skóre tedy informaci nese**, ale na
obchodovatelnou strategii je příliš slabý, závislý na režimu a citlivý na parametry.

Rozpad podle likvidity: nejlikvidnější kvintil (Q5) má nejvyšší čistý výnos na obchod (118 bps),
což souhlasí s Nagelem (2012): reversal vydělává hlavně ve stresu vysoce likvidních titulů.

## Testy správnosti enginu (zadání §12)

Všechny prošly (`tests/test_research5.py`):

| Test | Výsledek |
|---|---|
| Lookahead: změna dat po D nemění signály, pozice ani equity do D (celý pipeline) | OK |
| Survivorship: univerzum k 2000-01-03 (ADV20 ≥ 20M) | 624 titulů, z toho 411 později delistovaných |
| Split 2:1 v držené pozici | equity beze skoku |
| Delisting v držené pozici (bankrot / výkonnostní / akvizice) | −100 % / −30 % / přesná protihodnota |
| Nulová strategie (náhodné vstupy, nulové náklady) | každý obchod = open-to-open TR výnos; průměr ≈ univerzum |
| Reprodukovatelnost | identické výsledky |
| Parita s vektorovým enginem výzkumu 1 (jeden obchod) | shoda 10⁻¹² |

## Poučení

- Zadání mělo správně nastavená kill kritéria. Bez K5 a K7 by vývojový Sharpe 1,02 vypadal
  přesvědčivě a validace by ho vyvrátila až dodatečně.
- Laťka je pro tuto rodinu stejná jako ve výzkumu 1: krátkodobý reversal v likvidních US akciích
  po roce 2002 po realistických nákladech nestačí.
- Pokud by se k myšlence někdo vracel, jediný směr s oporou v datech je **podmíněný reversal
  v obdobích stresu likvidity** (Nagel 2012), například aktivace jen při vysokém VIX. Šlo by
  o novou hypotézu, kterou lze poctivě ověřit jen forward testem.

## Odchylky a incidenty

- Mřížka má 972 konfigurací, ne 648 jako v zadání (početní chyba zadání, opraveno před výpočtem).
- Validační běh jednou spadl po zápisu do registru a před zobrazením výsledku. Opakování je zapsané
  v registru i v logu pre-registrace.

## Reprodukce

```
uv run python scripts/r5_build_panel.py sharadar_2026-09-25
uv run python scripts/r5_grid.py sharadar_2026-09-25
uv run python scripts/r5_evaluate.py sharadar_2026-09-25 1000
uv run python scripts/r5_final.py sharadar_2026-09-25 validation   # jednou
uv run python scripts/r5_final.py sharadar_2026-09-25 late         # jednou, přes trezor
```

Seznam obchodů (`ticker, entry_date, entry_px, exit_date, exit_px, reason, gross_ret, net_ret`):
`data/derived/sharadar_2026-09-25/research5/trades_dev_selected.csv`. Registr pokusů:
`runs/registry/research5.sqlite`, čitelný log: `runs/research5/experiments.log`.
